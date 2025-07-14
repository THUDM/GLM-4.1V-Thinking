# -*- coding: utf-8 -*-


import re
from typing import Any, Optional, cast

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.text import find_boxed_content, protect_template

from ._base_verifier import Verifier

_logger = get_logger(__name__)


class GeneralVerifier(Verifier):
    def __init__(
        self,
        llm_api_key: str,
        llm_judge_url: str,
        llm_judge_prompt_template: str,
        llm_model: str = "glm-4-flash",
        llm_max_tokens: int = 10,
        llm_temperature: float = 0.1,
        llm_top_p: float = 1.0,
        answer_extraction_regex: Optional[str] = None,
    ) -> None:
        self.llm_api_key = llm_api_key
        self.llm_judge_url = llm_judge_url
        self.llm_judge_prompt_template = llm_judge_prompt_template
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p

        self.extraction_pattern = None
        if answer_extraction_regex is not None:
            try:
                self.extraction_pattern = re.compile(rf"{answer_extraction_regex}", re.DOTALL | re.IGNORECASE)
            except re.error as e:
                err_msg = f"Invalid regex for 'answer_extraction_regex': {answer_extraction_regex}. Error: {e}"
                raise ValueError(err_msg) from e

    def extract_answer(self, response: str, question: Optional[str] = None) -> Any:
        del question
        if self.extraction_pattern is not None:
            match = self.extraction_pattern.search(response)
            if match:
                # 1. Prioritize extracting the named group 'answer', recommended to use
                if "answer" in match.groupdict():
                    return cast(str, match.group("answer")).strip()
                # 2. If there are no named groups, extract the last capturing group
                return cast(str, match.group(len(match.groups()))).strip()
        # 3. If regex does not match, return the entire response by default
        return response.strip()

    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        """
        Applies placeholder protection and formats the prompt safely.
        """
        if debug:
            breakpoint()

        if not isinstance(extracted_answer, str) or not isinstance(ground_truth, str):
            _logger.warning(
                "%s: Judge expects string inputs, got %s and %s.",
                self.__class__.__name__,
                type(extracted_answer),
                type(ground_truth),
            )
            return self.min_reward

        if extracted_answer == ground_truth:
            return 1.0

        verifier_template = self.llm_judge_prompt_template

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
            _logger.warning(
                "[GeneralVerifier] Prompt formatting failed: %s. Template: '%s'", repr(e), verifier_template
            )
            return self.min_reward
        else:
            _logger.info(
                "[GeneralVerifier] question: %s, extracted_answer: %s, ground_truth: %s",
                question,
                extracted_answer,
                ground_truth,
            )
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

            if len(response_json) > 0:
                content = find_boxed_content(response_json.strip())

                # Robust parsing of LLM response for score
                if "Correct" in content:
                    return 1.0
                if "Incorrect" in content:
                    return self.min_reward
            _logger.warning(
                "GeneralVerifier: Invalid or empty response from LLM judge. URL: %s. Prompt: %s...",
                self.llm_judge_url,
                prompt[:100],
            )
            return self.min_reward
