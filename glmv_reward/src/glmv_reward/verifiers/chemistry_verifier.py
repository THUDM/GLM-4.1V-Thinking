# -*- coding: utf-8 -*-


import re
from typing import Any, Optional, cast

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_list
from glmv_reward.utils.text import protect_template

from .math_verifier import MathVerifier

_logger = get_logger(__name__)


def _has_unit(text: str) -> bool:
    units = [
        "mol",
        "L",
        "ml",
        "M",
        "N",
        "g/mol",
        "mol/L",
        "N_A",
        "atm",
        "kPa",
        "mmHg",
        "pH",
        "pOH",
        "K_w",
        "K_a",
        "K_b",
        "K_sp",
        "K_c",
        "K_p",
        "ΔH",
        "ΔS",
        "ΔG",
        "E°",
        "cal",
        "kcal",
        "kJ",
        "eV",
        "ppm",
        "ppb",
        "摩尔",
        "升",
        "毫升",
        "标准大气压",
        "氢离子浓度",
        "平衡常数",
        "摩尔浓度",
        "阿伏伽德罗常数",
    ]
    return any(unit in text for unit in units)


class ChemistryVerifier(MathVerifier):
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

        if _has_unit(extracted_answer) or _has_unit(ground_truth):
            return self._llm_judge_fallback(extracted_answer, ground_truth, question, image_file)

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
                return self.min_reward

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

                    score_matches = re.findall(r"(?:1\.0|0\.0|(?<!\.)1(?!\.)|(?<!\.)0(?!\.))", content)
                    if score_matches:
                        last_score = score_matches[-1]
                        if last_score == "1.0":
                            reward_score += 1.0
                            continue
                        if last_score == "0.0":
                            continue
                    try:
                        reward_score += float(content)  # LLM directly returns a number
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
