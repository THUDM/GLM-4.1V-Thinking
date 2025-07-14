# -*- coding: utf-8 -*-


import json
import re
from collections.abc import Sequence
from typing import Any, Optional, Union, cast

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_list
from glmv_reward.utils.text import find_boxed_content, protect_template

from ._base_verifier import Verifier

_logger = get_logger(__name__)


class GeoQuestVerifier(Verifier):
    def __init__(
        self,
        llm_api_key: Union[Sequence[str], str],
        llm_judge_url: Union[Sequence[str], str],
        llm_judge_prompt_template: str,
        llm_model: Optional[Union[Sequence[str], str]] = None,
        llm_max_tokens: int = 4096,
        llm_temperature: float = 0.8,
        llm_top_p: float = 0.6,
        strict_boxed_extraction: bool = True,
    ):
        self.llm_api_key = llm_api_key
        self.llm_judge_url = llm_judge_url
        self.llm_judge_prompt_template = llm_judge_prompt_template
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.strict_boxed = strict_boxed_extraction
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
        del question
        if debug:
            breakpoint()

        if not isinstance(extracted_answer, str) or not isinstance(ground_truth, (str, dict)):
            _logger.warning(
                "%s: Judge expects a string and a dict-like value, but got `%s` and `%s`.",
                self.__class__.__name__,
                type(extracted_answer),
                type(ground_truth),
            )
            return self.min_reward
        # Hard limit
        if len(extracted_answer) > 50:
            # print(f'overlong answer:', extracted_answer)
            return self.min_reward

        verifier_template = self.llm_judge_prompt_template

        # Ensure template is a valid non-empty string
        if len(verifier_template.strip()) == 0:
            err_msg = f"Invalid verifier template: {verifier_template}"
            raise ValueError(err_msg)

        # Ensure all required placeholders exist before formatting
        if not all(k in verifier_template for k in ("{question}", "{predict}", "{label}")):
            err_msg = "Template missing required placeholders: {question}, {predict}, {label}"
            raise ValueError(err_msg)

        # Protect any unintended format tokens before applying .format()
        verifier_template = protect_template(verifier_template, allowed=("predict", "place_name", "address"))

        if type(ground_truth) is str:
            ground_truth = json.loads(ground_truth)
        place_name = ground_truth["place_name"]
        address = ground_truth["address"]

        try:
            prompt = verifier_template.format(predict=extracted_answer, place_name=place_name, address=address)
        except Exception as e:
            _logger.warning(
                "[%s] Prompt formatting failed due to exception: %s. Template: '%s'",
                self.__class__.__name__,
                repr(e),
                verifier_template,
            )
            return 0.0

        api_key_lst: list[str] = ensure_list(self.llm_api_key)
        reward_url_lst: list[str] = ensure_list(self.llm_judge_url)
        model_lst = ["glm-4-flash"] * len(api_key_lst)
        if self.llm_model is not None:
            model_lst = ensure_list(self.llm_model)

        if len(reward_url_lst) == 0:
            err_msg = "Empty llm_judge_url"
            raise ValueError(err_msg)
        reward_score = 0.0
        for api_key, reward_url, model in zip(api_key_lst, reward_url_lst, model_lst, strict=True):
            response_text = post_query_llm(
                prompt,
                api_key,
                url=reward_url,
                model=model,
                image_file=image_file,
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
            )
            # Initialize content with a default value to avoid referencing it later
            if response_text and type(response_text) is str:
                verifier_think_pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
                response_text = verifier_think_pattern.sub("", response_text).strip()
                try:
                    grade_response = re.findall(r"(\{.*?})", response_text, re.S)[0]
                    judgment = json.loads(grade_response)
                    judge_score = float(judgment["score"])
                except Exception:
                    _logger.exception("Failed parsing json: %s, trying re...", response_text)
                    try:
                        judge_score = float(re.findall(r'"score": (.*?),', response_text)[0])
                        # hurdle 0.7
                        judge_score = 1.0 if judge_score > 0.7 else 0.0
                    except Exception:
                        _logger.exception("Error: Could not parse response as JSON: %s", response_text)
                        judge_score = 0.0
                try:
                    reward_score += judge_score
                    continue
                except ValueError:
                    pass

            _logger.warning(
                "%s: LLM fallback judge failed or gave unexpected response for ('%s', '%s'). Raw response: %s",
                self.__class__.__name__,
                extracted_answer,
                ground_truth,
                response_text,
            )

        # average reward score
        return reward_score / len(reward_url_lst)
