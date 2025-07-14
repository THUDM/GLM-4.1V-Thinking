# -*- coding: utf-8 -*-


from typing import Any, Optional

from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.text import detect_long_paragraph_mixing

from ._base_verifier import Verifier

_logger = get_logger(__name__)


class LanguageMixVerifier(Verifier):
    def extract_answer(self, response: str, question: Optional[str] = None) -> Any:
        del question
        return response

    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        del ground_truth, question, image_file
        if debug:
            breakpoint()

        if not isinstance(extracted_answer, str):
            _logger.warning(
                "%s.judge: `extracted_answer` should be a string, but got a `%s`.",
                self.__class__.__name__,
                extracted_answer.__class__.__name__,
            )
            return self.min_reward
        # reward 1 means no mixing, 0 means fully mixing
        return 1 - detect_long_paragraph_mixing(extracted_answer)
