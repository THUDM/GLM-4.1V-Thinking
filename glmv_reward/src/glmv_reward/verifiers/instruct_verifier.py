# -*- coding: utf-8 -*-


import re
from typing import Any, Optional, cast

from glmv_reward.utils.llm import query_rm_mean

from ._base_verifier import Verifier


class InstructVerifier(Verifier):
    def __init__(self, llm_judge_url: str, llm_model: str) -> None:
        self.llm_judge_url = llm_judge_url
        self.llm_model = llm_model

        self.think_answer_pattern = re.compile(r"^<think>(.*?)</think>(.*)$", re.DOTALL | re.IGNORECASE)

    @property
    def min_reward(self) -> float:
        return float("-inf")

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

        return answer_content_to_check

    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        del ground_truth, image_file
        if debug:
            breakpoint()

        return query_rm_mean(
            self.llm_judge_url, self.llm_model, prompt=question or "", response=extracted_answer, num_query=1
        )
