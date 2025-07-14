# -*- coding: utf-8 -*-


import random
import re
from typing import Any, Optional, cast

from glmv_reward.utils.llm import query_rm_mean
from glmv_reward.utils.logging import get_logger
from glmv_reward.utils.text import detect_long_paragraph_mixing, detect_repeat

from ._base_verifier import Verifier

_logger = get_logger(__name__)


class PostRLFormatVerifier(Verifier):
    def __init__(
        self,
        forbid_long_paragraph_mixing: bool = False,
        forbid_box: bool = False,
        forbid_repeat: bool = False,
        use_instruct_reward_model: bool = False,
        instruct_llm_judge_url: Optional[str] = None,
        instruct_llm_model: Optional[str] = None,
        use_fluency_reward_model: bool = False,
        fluency_llm_judge_url: Optional[str] = None,
        fluency_llm_model: Optional[str] = None,
    ) -> None:
        self.think_answer_pattern = re.compile(r"^<think>(.*?)</think>(.*)$", re.DOTALL | re.IGNORECASE)

        self.forbid_long_paragraph_mixing = forbid_long_paragraph_mixing
        self.forbid_box = forbid_box
        self.forbid_repeat = forbid_repeat

        self.use_instruct_reward_model = use_instruct_reward_model
        self.instruct_llm_judge_url = instruct_llm_judge_url
        self.instruct_llm_model = instruct_llm_model

        self.use_fluency_reward_model = use_fluency_reward_model
        self.fluency_llm_judge_url = fluency_llm_judge_url
        self.fluency_llm_model = fluency_llm_model

    @property
    def min_reward(self) -> float:
        return float("-inf")

    def extract_answer(self, response: str, question: Optional[str] = None) -> Any:
        del question

        match = self.think_answer_pattern.search(response)
        think_content_to_check = None
        answer_content_to_check = None

        if match:
            think_part = cast(str, match.group(1)).strip()
            answer_part = cast(str, match.group(2)).strip()

            # Basic validation against nested tags
            avoid_tags = ["<think>", "</think>"]
            if any(tag.lower() in think_part.lower() for tag in avoid_tags):
                return None

            think_content_to_check = think_part
            answer_content_to_check = answer_part

        # check answer part, should be some content in it
        if answer_content_to_check is None:
            return None

        # check think part, should be some content in it
        if think_content_to_check is None:
            return None

        return {"think": think_content_to_check, "answer": answer_content_to_check}

    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        image_file: Optional[str] = None,
        debug: bool = False,
    ) -> float:
        del image_file, ground_truth
        if not isinstance(extracted_answer, dict):
            _logger.warning(
                "%s: Judge expects dict inputs, but got `%s`.",
                self.__class__.__name__,
                type(extracted_answer),
            )
            return self.min_reward

        think_content = cast(str, extracted_answer.get("think"))
        answer_content = cast(str, extracted_answer.get("answer"))

        # should be no <begin_of_box> or <end_of_box>,
        # `\\boxed`` would be validated in `reward_system.check_answer_format`
        # checking answer content
        if self.forbid_box:
            # checking think content
            begin_box_count_in_think = think_content.lower().count("<|begin_of_box|>")
            end_box_count_in_think = think_content.lower().count("<|end_of_box|>")
            if begin_box_count_in_think > 0 or end_box_count_in_think > 0:
                if debug:
                    print("[DEBUG] detect_box in think")
                return self.min_reward

            begin_box_count_in_answer = answer_content.lower().count("<|begin_of_box|>")
            end_box_count_in_answer = answer_content.lower().count("<|end_of_box|>")
            if begin_box_count_in_answer > 0 or end_box_count_in_answer > 0:
                if debug:
                    print("[DEBUG] detect_box in answer")
                return self.min_reward

        if self.forbid_long_paragraph_mixing:
            if detect_long_paragraph_mixing(think_content):
                if debug:
                    print("[DEBUG] detect_long_paragraph_mixing in think")
                return self.min_reward

            if detect_long_paragraph_mixing(answer_content):
                if debug:
                    print("[DEBUG] detect_long_paragraph_mixing in answer")
                return self.min_reward

        if self.forbid_repeat:
            if detect_repeat(think_content):
                if debug:
                    print("[DEBUG] detect_repeat in think")
                return self.min_reward

            if detect_repeat(answer_content):
                if debug:
                    print("[DEBUG] detect_repeat in answer")
                return self.min_reward

        instruct_reward = None
        fluency_reward = None
        full_answer = f"<think>{think_content}</think>{answer_content}"
        if self.use_instruct_reward_model:
            if self.instruct_llm_judge_url is None:
                err_msg = "`instruct_llm_judge_url` is required when `use_instruct_reward_model` is `True`."
                raise ValueError(err_msg)
            if self.instruct_llm_model is None:
                err_msg = "`instruct_llm_model` is required when `use_instruct_reward_model` is `True`."
                raise ValueError(err_msg)
            instruct_reward = query_rm_mean(
                url=self.instruct_llm_judge_url,
                model=self.instruct_llm_model,
                prompt=question or "",
                response=full_answer,
                box_prompt=not self.forbid_box,
            )

        if self.use_fluency_reward_model:
            if self.fluency_llm_judge_url is None:
                err_msg = "`fluency_llm_judge_url` is required when `use_fluency_reward_model` is `True`."
                raise ValueError(err_msg)
            if self.fluency_llm_model is None:
                err_msg = "`fluency_llm_model` is required when `use_fluency_reward_model` is `True`."
                raise ValueError(err_msg)

            fluency_reward = query_rm_mean(
                url=self.fluency_llm_judge_url,
                model=self.fluency_llm_model,
                prompt=question or "",
                response=full_answer,
                box_prompt=not self.forbid_box,
            )

        # If neither reward model is used, return 1.0
        if instruct_reward is None:
            if fluency_reward is None:
                return 1.0
            # If only one reward model is used, return that one
            return fluency_reward

        if fluency_reward is None:
            return instruct_reward

        # If both reward models are used, select one based on question hash or random
        if question:
            # Use hash of question to create a deterministic random number generator
            rng = random.Random(hash(question))  # noqa: S311
            # Use the RNG to make a deterministic selection
            if rng.random() < 0.25:  # 25% chance for instruct reward
                return instruct_reward
            # 75% chance for fluency reward
            return fluency_reward
        _logger.warning("No question provided, using fluency reward")
        _logger.warning(
            "question: %s, answer: %s, instruct_reward: %s, fluency_reward: %s, full_answer: %s",
            question,
            answer_content,
            instruct_reward,
            fluency_reward,
            full_answer,
        )
        return fluency_reward
