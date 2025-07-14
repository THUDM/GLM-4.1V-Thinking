# -*- coding: utf-8 -*-


from abc import ABC, abstractmethod
from typing import Any, Optional


class Verifier(ABC):
    """
    Abstract Base Class for all verifiers.
    Each verifier is responsible for extracting an answer from a response
    and judging its correctness against a ground truth.
    """

    @abstractmethod
    def extract_answer(self, response: str, question: Optional[str] = None) -> Any:
        """
        Extracts the relevant answer part from the model's full response.

        Args:
            response (str): The full response string from the model.
            question (Optional[str]): The question/prompt, for context if needed.

        Returns:
            Any: The extracted answer. Type can vary (e.g., str, float, dict).
                 Returns None if no answer can be reliably extracted.
        """
        pass

    @abstractmethod
    def judge(
        self,
        extracted_answer: Any,
        ground_truth: Any,  # This will also be an "extracted" ground truth
        question: Optional[str] = None,
        image_file: Optional[str] = None,
    ) -> float:
        """
        Judges the correctness of the extracted answer against the (extracted) ground truth.

        Args:
            extracted_answer (Any): The answer extracted from the model's response.
            ground_truth (Any): The answer extracted from the ground truth response.
            question (Optional[str]): The question/prompt, for context.
            image_files (Optional[Sequence[str]]): Paths to images, if relevant for judging.

        Returns:
            float: A score, typically 0.0 for incorrect and 1.0 for correct.
                   Can also be a continuous score if applicable.
        """
        pass

    @property
    def min_reward(self) -> float:
        return 0.0

    @property
    def is_batch_verifier(self) -> bool:
        return False
