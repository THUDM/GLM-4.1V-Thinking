# -*- coding: utf-8 -*-


import re
from collections.abc import Sequence
from typing import Any, Optional, cast, Union

import editdistance

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_list
from glmv_reward.utils.text import find_boxed_content, protect_template

from ._base_verifier import Verifier

_logger = get_logger(__name__)


class OCRVerifier(Verifier):
    def __init__(
        self,
        strict_boxed_extraction: bool = True,
        edit_distance_upper_bound: float = 1.0,
        edit_distance_lower_bound: float = 0.0,
        ignore_case: bool = False,
        enable_llm_judge_fallback: bool = True,
        llm_api_key: Optional[Union[Sequence[str], str]] = None,
        llm_judge_url: Optional[Union[Sequence[str], str]] = None,
        llm_judge_prompt_template: Optional[str] = None,
        llm_model: Optional[Union[Sequence[str], str]] = None,
        llm_max_tokens: int = 10,
        llm_temperature: float = 0.1,
        llm_top_p: float = 1.0,
    ) -> None:
        self.strict_boxed = strict_boxed_extraction
        # >= upper bound, score will be 1.0
        self.edit_distance_upper_bound = edit_distance_upper_bound
        # <= lower bound, score will be 0.0
        self.edit_distance_lower_bound = edit_distance_lower_bound
        self.ignore_case = ignore_case
        self.enable_llm_judge_fallback = enable_llm_judge_fallback
        self.llm_api_key = llm_api_key
        self.llm_judge_url = llm_judge_url
        self.llm_judge_prompt_template = llm_judge_prompt_template
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p

        self.think_answer_pattern = re.compile(r"^<think>(.*?)</think>(.*)$", re.DOTALL | re.IGNORECASE)

    def extract_answer(self, response: str, question: Optional[str] = None) -> Any:
        del question
        match = self.think_answer_pattern.search(response)
        answer_content_to_check = None

        if match:
            think_part = cast(str, match.group(1)).strip()
            answer_part = cast(str, match.group(2)).strip()

            # Basic validation against nested tags
            avoid_tags = ["<think>", "</think>"]
            if any(tag.lower() in think_part.lower() for tag in avoid_tags):
                return None

            answer_content_to_check = answer_part

        if answer_content_to_check is None or len(answer_content_to_check) == 0:
            return None

        boxed_matches = find_boxed_content(answer_content_to_check)

        if len(boxed_matches) == 1:
            return boxed_matches[0].strip()

        if not self.strict_boxed and not boxed_matches:
            return answer_content_to_check

        return None

    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        if debug:
            breakpoint()

        if not isinstance(extracted_answer, str) or not isinstance(ground_truth, str):
            _logger.warning(
                "%s: Judge expects string inputs, but got `%s` and `%s`.",
                self.__class__.__name__,
                type(extracted_answer),
                type(ground_truth),
            )
            return self.min_reward

        if self.ignore_case:
            extracted_answer = extracted_answer.lower()
            ground_truth = ground_truth.lower()

        extracted_answer = extracted_answer.strip().replace("\n", " ").replace(" ", "")
        ground_truth = ground_truth.strip().replace("\n", " ").replace(" ", "")

        similarity = 1 - editdistance.eval(extracted_answer, ground_truth) / max(
            len(extracted_answer), len(ground_truth)
        )

        if similarity >= self.edit_distance_upper_bound:
            return 1.0
        if similarity <= self.edit_distance_lower_bound:
            return 0.0

        if (
            self.enable_llm_judge_fallback
            and self.llm_api_key
            and self.llm_judge_url
            and self.llm_judge_prompt_template
        ):
            llm_reward = self._llm_judge_fallback(extracted_answer, ground_truth, question, image_file)
            if llm_reward == 1.0:
                return 1.0

        return similarity

    def _llm_judge_fallback(
        self,
        extracted_answer: str,
        ground_truth: str,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
    ) -> float:
        """
        Fallback logic for LLM-based judgment.
        Applies placeholder protection and formats the prompt safely.
        """
        if self.llm_api_key is None:
            err_msg = "`llm_api_key` is required when calling `_llm_judge_fallback`"
            raise ValueError(err_msg)
        if self.llm_judge_url is None:
            err_msg = "`llm_judge_url` is required when calling `_llm_judge_fallback`"
            raise ValueError(err_msg)

        verifier_template = self.llm_judge_prompt_template or ""

        if len(verifier_template.strip()) == 0:
            return self.min_reward

        if not all(k in verifier_template for k in ("{question}", "{predict}", "{label}")):
            err_msg = "Template missing required placeholders: {question}, {predict}, {label}"
            raise ValueError(err_msg)

        verifier_template = protect_template(verifier_template, allowed=("question", "predict", "label"))
        try:
            prompt = verifier_template.format(question=question or "", predict=extracted_answer, label=ground_truth)
        except Exception as e:
            _logger.warning("[LLM Verifier] Prompt formatting failed: %s. Template: '%s'.", repr(e), verifier_template)
            return self.min_reward
        else:
            api_key_lst: list[str] = ensure_list(self.llm_api_key)
            reward_url_lst: list[str] = ensure_list(self.llm_judge_url)
            model_lst = ["glm-4-flash"] * len(api_key_lst)
            if self.llm_model is not None:
                model_lst = ensure_list(self.llm_model)

            reward_score = 0.0
            for api_key, reward_url, model in zip(api_key_lst, reward_url_lst, model_lst, strict=True):
                response_json = post_query_llm(
                    prompt,
                    api_key,
                    url=reward_url,
                    model=model,
                    image_file=image_file,
                    max_tokens=self.llm_max_tokens,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                )

                if response_json:
                    content = response_json.strip()
                    if "1.0" in content:
                        reward_score += 1.0
                        continue
                    if "0.0" in content:
                        continue
                    try:
                        reward_score += float(content)
                        continue
                    except ValueError:
                        pass
                _logger.warning(
                    "%s: LLM fallback judge failed or gave unexpected response for ('%s', '%s'). Raw response: %s",
                    self.__class__.__name__,
                    extracted_answer,
                    ground_truth,
                    response_json,
                )

            return float(reward_score > len(reward_url_lst) / 2)
