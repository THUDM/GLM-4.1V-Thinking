# -*- coding: utf-8 -*-


import re
from typing import Any, Optional, cast

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.text import find_boxed_content, protect_template

from ._base_verifier import Verifier

_logger = get_logger(__name__)


class ChartVerifier(Verifier):
    def __init__(
        self,
        answer_extraction_regex: str,
        sympy_tolerance: float = 1e-5,
        strict_boxed_extraction: bool = True,
        enable_llm_judge_fallback: bool = True,
        llm_api_key: Optional[str] = None,
        llm_judge_url: Optional[str] = None,
        llm_judge_prompt_template: Optional[str] = None,
        llm_model: str = "glm-4-flash",
        llm_max_tokens: int = 10,
        llm_temperature: float = 0.1,
        llm_top_p: float = 1.0,
    ) -> None:
        self.extraction_pattern = re.compile(rf"{answer_extraction_regex}", re.DOTALL | re.IGNORECASE)

        self.sympy_tolerance = sympy_tolerance
        self.strict_boxed = strict_boxed_extraction
        self.enable_llm_judge_fallback = enable_llm_judge_fallback
        self.llm_api_key = llm_api_key
        self.llm_judge_url = llm_judge_url
        self.llm_judge_prompt_template = llm_judge_prompt_template
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p

    def extract_answer(self, response: str, question: Optional[str] = None) -> Any:
        del question  # unused

        match = self.extraction_pattern.search(response)
        answer_content_to_check = None

        if match:
            think_part = cast(str, match.group(1)).strip()
            answer_part = cast(str, match.group(2)).strip()

            # Basic validation against nested tags
            avoid_tags = ["<think>", "</think>", "<answer>", "</answer>"]
            if any(tag.lower() in think_part.lower() for tag in avoid_tags) or any(
                tag.lower() in answer_part.lower() for tag in avoid_tags
            ):
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

        if extracted_answer == ground_truth:
            return 1.0

        # ! seems hacky
        def is_year_question(question: str, answer1: str, answer2: str) -> bool:
            """check if the question's answer is about year"""
            del question
            # ? question is not used?
            # year_keywords = ['year', 'when', 'date', 'time']
            # question_lower = question.lower()
            # has_year_keyword = any(keyword in question_lower for keyword in year_keywords)

            try:
                num1 = float(answer1)
                num2 = float(answer2)
            except Exception:
                return False
            else:
                return (
                    1000 <= abs(num1) <= 3000 and num1 == int(num1) and 1000 <= abs(num2) <= 3000 and num2 == int(num2)
                )

        try:
            from sympy import Basic, sympify

            extract_answer_number = cast(Basic, sympify(extracted_answer, strict=True))
            extract_gt_answer_number = cast(Basic, sympify(ground_truth, strict=True))
        except Exception:
            _logger.debug("Failed to convert the answer to a numeric value. Skip number match.")
        else:
            if extract_answer_number.is_real and extract_gt_answer_number.is_real:
                if question is not None and is_year_question(question, extracted_answer, ground_truth):
                    return 1.0 if extract_answer_number == extract_gt_answer_number else 0.0
                if (
                    abs(extract_answer_number - extract_gt_answer_number) / (abs(extract_gt_answer_number) + 1e-6)
                    < self.sympy_tolerance
                ):
                    return 1.0
                return self.min_reward

        if self.enable_llm_judge_fallback:
            return self._llm_judge_fallback(extracted_answer, ground_truth, question, image_file)

        return self.min_reward

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
        # Ensure template is a valid non-empty string
        if len(verifier_template.strip()) == 0:
            return self.min_reward

        # Ensure all required placeholders exist before formatting
        if not all(k in verifier_template for k in ("{question}", "{predict}", "{label}")):
            err_msg = "Template missing required placeholders: {question}, {predict}, {label}"
            raise ValueError(err_msg)

        # Protect any unintended format tokens before applying .format()
        verifier_template = protect_template(verifier_template, allowed=("question", "predict", "label"))
        try:
            prompt = verifier_template.format(question=question or "", predict=extracted_answer, label=ground_truth)
        except Exception as e:
            _logger.warning("[LLM Verifier] Prompt formatting failed: %s. Template: '%s'.", repr(e), verifier_template)
            return self.min_reward
        else:
            response_json = post_query_llm(
                prompt,
                self.llm_api_key,
                url=self.llm_judge_url,
                model=self.llm_model,
                image_file=image_file,
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
            )
            if response_json:
                content = response_json.strip()
                if "1.0" in content:
                    return 1.0
                if "0.0" in content:
                    return 0.0
                try:
                    return float(content)  # LLM directly returns a number
                except ValueError:
                    pass
            _logger.warning(
                "%s: LLM fallback judge failed or gave unexpected response for ('%s', '%s'). Raw response: %s",
                self.__class__.__name__,
                extracted_answer,
                ground_truth,
                response_json,
            )
            return self.min_reward
