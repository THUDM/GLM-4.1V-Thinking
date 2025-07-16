# -*- coding: utf-8 -*-


import re
from typing import Any, Optional

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_list
from glmv_reward.utils.text import protect_template

from .math_verifier import MathVerifier

_logger = get_logger(__name__)


def _normalize_genotype(genotype_str: str) -> Optional[str]:
    """
    Normalizes a genotype string for consistent comparison.
    Example: "bBAa" -> "AaBb"

    Handles genotypes like AaBb, XxYy, etc. It sorts alleles within a gene
    (case-insensitively, e.g., 'aA' -> 'Aa') and then sorts the gene pairs.
    Returns None if the format is not a simple paired genotype.
    """
    if not isinstance(genotype_str, str):
        return None

    genotype_str = genotype_str.strip()

    if not re.fullmatch(r"[A-Za-z\s]*", genotype_str):
        return None

    # Regex to find pairs of letters (e.g., 'Aa', 'bB')
    # This is a simplified assumption for common genetics problems.
    pairs = re.findall("[A-Za-z][A-Za-z]", genotype_str)

    # If the whole string is not composed of just these pairs, it's not a simple genotype
    if "".join(pairs) != genotype_str.replace(" ", ""):
        return None

    # Normalize each pair by sorting its characters
    normalized_pairs = ["".join(sorted(p)) for p in pairs]

    # Sort the pairs themselves
    sorted_normalized_pairs = sorted(normalized_pairs)

    return "".join(sorted_normalized_pairs)


class BiologyVerifier(MathVerifier):
    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        """
        Judges the correctness of a biological answer.

        The logic is as follows:
        1.  Basic validation of inputs.
        2.  Check for exact string match after stripping whitespace.
        3.  Attempt to compare as normalized genotypes (e.g., for genetics problems).
        4.  If none of the above rule-based checks pass, fall back to a powerful LLM
            for semantic evaluation, which is the primary method for most biology answers.
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

        if extracted_answer.strip() == ground_truth.strip():
            return 1.0

        if self.enable_llm_judge_fallback:
            return self._llm_judge_fallback(extracted_answer, ground_truth, question, image_file)

        return self.min_reward

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

            # Initialize content with a default value to avoid referencing it later
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
