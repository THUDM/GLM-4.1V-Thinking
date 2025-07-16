# -*- coding: utf-8 -*-


import re
from typing import Any, Optional, cast

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_list
from glmv_reward.utils.text import protect_template

from .math_verifier import MathVerifier

_logger = get_logger(__name__)


def _normalize_list(text: str) -> Optional[set[str]]:
    """
    Normalizes a string that might represent a list of items into a set.
    Handles various delimiters like commas, semicolons, spaces, and Chinese commas.
    Example: "长江, 黄河, 珠江" -> {"长江", "黄河", "珠江"}

    Returns a set of strings for order-independent comparison, or None if input is invalid.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    delimiters = r"[,;\s、，；]"
    items = {item.strip() for item in re.split(delimiters, text) if item.strip()}

    if len(items) > 1:
        return items

    if len(items) == 1:
        return items

    return None


class GeographyVerifier(MathVerifier):
    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        """
        Judges the correctness of a geography answer.

        The logic is as follows:
        1.  Basic validation of inputs.
        2.  Check for exact string match after stripping whitespace.
        3.  Attempt to compare as normalized, order-insensitive lists (for "list the..." questions).
        4.  Attempt to compare as numbers (for questions about altitude, coordinates, etc.).
        5.  For all other cases (concepts, descriptions, locations), fall back to an LLM
            for robust semantic and factual evaluation.
        """
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

        if extracted_answer.strip().lower() == ground_truth.strip().lower():
            return 1.0

        list_extracted = _normalize_list(extracted_answer)
        list_gt = _normalize_list(ground_truth)
        if list_extracted is not None and list_gt is not None:
            if list_extracted == list_gt:
                return 1.0

        try:
            from sympy import Basic, sympify

            extract_answer_number = cast(Basic, sympify(extracted_answer, strict=True))
            extract_gt_answer_number = cast(Basic, sympify(ground_truth, strict=True))
        except Exception:
            _logger.debug("Failed to convert the answer to a numeric value. Skip number match.")
        else:
            if extract_answer_number.is_real and extract_gt_answer_number.is_real:
                if (
                    abs(extract_answer_number - extract_gt_answer_number) / (abs(extract_gt_answer_number) + 1e-6)
                    < self.sympy_tolerance
                ):
                    return 1.0

        if self.enable_llm_judge_fallback:
            return self._llm_judge_fallback(extracted_answer, ground_truth, question, image_file)

        return 0.0

    def _llm_judge_fallback(
        self, extracted_answer: str, ground_truth: str, question: Optional[str] = None, image_file: Optional[str] = None
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

            content = None
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

        # at least > half of the reward_url_list return 1.0
        return float(reward_score > len(reward_url_lst) / 2)
