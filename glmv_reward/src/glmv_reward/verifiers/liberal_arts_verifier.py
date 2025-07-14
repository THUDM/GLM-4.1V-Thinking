# -*- coding: utf-8 -*-


from typing import Any, Optional

from glmv_reward.utils.llm import post_query_llm
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.misc import ensure_list
from glmv_reward.utils.text import protect_template

from .math_verifier import MathVerifier

_logger = get_logger(__name__)


def _preprocess_text(text: str) -> str:
    """
    Preprocesses text to help LLM comparison by removing common noise.
    - Removes leading/trailing whitespace.
    - Removes common list markers like (1), 1., A., ①.
    - Normalizes whitespace.
    """
    if not isinstance(text, str):
        return ""

    # 移除首尾空格
    processed_text = text.strip()

    # 将多个空格或换行符替换为单个空格
    # processed_text = re.sub(r'\s+', ' ', processed_text)

    return processed_text.strip()


class LiberalArtsVerifier(MathVerifier):
    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        """
        Judges the correctness of a liberal arts answer.

        The logic is heavily reliant on LLM judgment:
        1.  Basic validation of inputs.
        2.  Check for an exact string match (a quick win).
        3.  Preprocess both answers to remove structural noise (like numbering).
        4.  Directly pass the preprocessed answers to the LLM for a robust
            semantic, factual, and logical comparison.
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

        # 规则 1: 简单的字符串完全匹配（最快、最可靠的检查）
        if extracted_answer.strip() == ground_truth.strip():
            return 1.0

        # 对于文科综合，几乎所有非完全匹配的情况都需要复杂的语义理解。
        # 因此，我们直接进入 LLM 判断流程，而不是添加脆弱的规则。

        if self.enable_llm_judge_fallback:
            # 在送入 LLM 前，对答案进行预处理，帮助模型聚焦核心内容
            processed_extracted = _preprocess_text(extracted_answer)
            processed_gt = _preprocess_text(ground_truth)

            # 如果预处理后为空，则认为无有效内容
            if not processed_extracted or not processed_gt:
                return self.min_reward

            return self._llm_judge_fallback(processed_extracted, processed_gt, question, image_file)

        # 如果没有启用 LLM，对于非完全匹配的情况，只能判为错误
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
