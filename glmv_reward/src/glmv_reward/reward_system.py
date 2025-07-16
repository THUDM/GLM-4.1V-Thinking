# -*- coding: utf-8 -*-


import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import msgspec

from .configs import RewardSystemConfig
from .configs.verifiers import VerifierConfig
from .utils.logging import get_logger
from .utils.misc import ensure_list
from .utils.path import mkdir
from .utils.serialization import load_yaml
from .verifiers import LanguageMixVerifier, Verifier, get_verifier_from_config

_logger = get_logger(__name__)


class RewardSystem(object):
    def __init__(self, config_file: Union[Path, str]) -> None:
        """
        Initialize the RewardSystem with configurations from a YAML file.

        Args:
            config_file (Optional[str]): Path to YAML configuration file containing reward model settings
        """

        # Load configuration from YAML file if provided
        _logger.info(f"> Loading reward config file: {config_file}")
        reward_config = msgspec.convert(load_yaml(config_file), RewardSystemConfig)

        self.reward_log_dir = reward_config.reward_log_dir

        # Set default configurations for each model if not provided
        self.reward_configs: dict[str, VerifierConfig] = {}
        for model_name, config in reward_config.reward_configs.items():
            if model_name in self.reward_configs:
                err_msg = f"duplicate model name: {model_name} in reward_configs"
                raise ValueError(err_msg)
            self.reward_configs[model_name] = config

        self.datasource_reward_configs: dict[str, VerifierConfig] = {}
        for datasource, model_name in reward_config.datasource_reward_config_mapping.items():
            if datasource in self.datasource_reward_configs:
                err_msg = f"duplicate datasource: {datasource} in datasource_reward_configs"
                raise ValueError(err_msg)
            if model_name not in self.reward_configs:
                err_msg = f"Cannot find the configuration of `{model_name}`."
                raise ValueError(err_msg)
            self.datasource_reward_configs[datasource] = self.reward_configs[model_name]

        self.language_mix_verifier = None
        if reward_config.enable_mix_verifier:
            self.language_mix_verifier = LanguageMixVerifier()

        self.think_answer_pattern = re.compile(
            r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$", re.DOTALL | re.IGNORECASE
        )

    def _process_single_item(
        self,
        prompt: str,
        answer: Any,
        gt_answer: Any,
        image_file: Optional[str],
        verifier: Verifier,
        debug: bool = False,
    ) -> tuple[float, Any, Any]:
        min_reward = getattr(verifier, "min_reward", float("-inf"))

        try:
            # if it is not a correct answer format, return -inf
            if isinstance(answer, str) and not self.check_answer_format(answer):
                return min_reward, None, None

            # if it is not a correct gt_answer format, return -inf
            if isinstance(gt_answer, str) and not self.check_answer_format(gt_answer):
                _logger.warning("> Receive bad format gt_answer: %s, please check your data", gt_answer)
                return min_reward, None, None

            if self.language_mix_verifier is not None and not self.language_mix_verifier.judge(answer, gt_answer):
                return min_reward, None, None

            # Extract ground truth
            extracted_gt = verifier.extract_answer(gt_answer, question=prompt)
            if extracted_gt is None:
                _logger.warning(f"> Receive bad gt_answer: {gt_answer}, please check your data")
                return min_reward, None, None

            # Extract and judge answer
            extracted_ans = verifier.extract_answer(answer, question=prompt)
            if extracted_ans is None or not isinstance(extracted_ans, (str, list, dict)):
                return min_reward, extracted_ans, extracted_gt

        except Exception as e:
            _logger.warning("> Error in verifier extract_answer due to exception: %s", repr(e))
            return min_reward, None, None
        else:
            if debug:
                print("--- Verifier Debug ---")
                print(f"Verifier class: {verifier.__class__.__name__}")
                print(f"Raw Model Answer: {answer[:200]}...")
                print(f"Extracted Model Answer: {extracted_ans}")
                print(f"Raw GT Answer: {gt_answer[:200]}...")
                print(f"Extracted GT Answer: {extracted_gt}")
                print("----------------------")
                breakpoint()

            try:
                # Get reward
                reward = verifier.judge(extracted_ans, extracted_gt, question=prompt, image_file=image_file)
            except Exception as e:
                _logger.warning("> Error in verifier judge: %s", repr(e))
                reward = min_reward

            try:
                reward = float(reward)
            except Exception:
                _logger.warning("> reward from verifier judge should be able to convert to float, but got: %s.", reward)
                reward = min_reward

            return reward, extracted_ans, extracted_gt

    @classmethod
    def from_yaml(cls, config_file: Union[Path, str]) -> "RewardSystem":
        """
        Create a RewardSystem instance from a YAML configuration file.

        Args:
            config_path (str): Path to YAML configuration file

        Returns:
            RewardSystem: Initialized RewardSystem instance
        """
        return cls(config_file)

    def check_answer_format(self, response: str) -> bool:
        match = self.think_answer_pattern.search(response)
        answer_part = None

        if match:
            think_part = match.group(1).strip()
            answer_part = match.group(2).strip()

            # Check for multiple box tags in answer part
            begin_box_count = answer_part.lower().count("<|begin_of_box|>")
            end_box_count = answer_part.lower().count("<|end_of_box|>")

            # don't support legacy \\boxed{}
            legacy_boxed_count = answer_part.lower().count("\\boxed{")

            if begin_box_count > 1 or end_box_count > 1 or legacy_boxed_count > 0:
                return False

            # Basic validation against nested tags
            avoid_tags = ["<think>", "</think>", "<answer>", "</answer>"]
            if any(tag.lower() in think_part.lower() for tag in avoid_tags) or any(
                tag.lower() in answer_part.lower() for tag in avoid_tags
            ):
                return False

        if answer_part is None:
            return False

        return True

    def get_reward_config_from_datasource(self, datasource: str) -> VerifierConfig:
        """
        Get the reward model name based on the datasource.
        """
        if datasource not in self.datasource_reward_configs:
            err_msg = f"No reward config found for datasource: {datasource}"
            raise ValueError(err_msg)

        return self.datasource_reward_configs[datasource]

    def get_verifier_from_datasource(self, datasource: str) -> Verifier:
        """
        Get the verifier from the datasource.
        """
        reward_config = self.get_reward_config_from_datasource(datasource)
        return get_verifier_from_config(reward_config, datasource)

    def get_reward(
        self,
        prompts: Union[Sequence[str], str],
        answers: Union[Sequence[str], str],
        gt_answers: Union[Sequence[str], str],
        uuids: Optional[Union[Sequence[str], str]] = None,
        image_files: Optional[Union[Sequence[str], str]] = None,
        answer_lengths: Optional[Union[Sequence[int], int]] = None,
        datasources: Optional[Sequence[str] | str] = None,
        log_reward_judge: bool = False,
        save_dir: Optional[str] = None,
        current_iteration: int = 0,
        debug: bool = False,
        return_extracted_answers: bool = False,
    ) -> Union[list[float], tuple[list[float], list, list]]:
        # TODO: revises the typing hints in the docstring
        """
        Get reward from reward model.

        Args:
            prompts (Union[Sequence[str], str]): List of prompts
            answers (Union[Sequence[str], str]): List of model answers
            gt_answers (Union[Sequence[str], str]): List of ground truth answers
            uuids (Optional[Union[Sequence[str], str]]): List of uuids
            image_files (Optional[Sequence[str]]): List of image paths
            answer_lengths (Optional[Sequence[int]]): List of answer lengths
            datasources (Optional[Sequence[str]]): List of datasource identifiers
            log_reward_judge (bool): Whether to log reward judgments
            save_dir (Optional[str]): Path to save logs
            current_iteration (int): Current iteration number
            debug (bool): Whether to enable debug mode
            return_extracted_answers (bool): If True, returns tuple (rewards, extracted_ans_list, extracted_gt_list)

        Returns:
            A list of rewards or a tuple of consisting of a list of rewards, a list of extracted answers,
            and a list of extracted ground truth.
        """
        if debug:
            breakpoint()

        log_save_dir = save_dir if save_dir else self.reward_log_dir

        # Ensure all inputs are lists
        prompt_lst: list[str] = ensure_list(prompts)
        answer_lst: list[str] = ensure_list(answers)
        gt_answer_lst: list[str] = ensure_list(gt_answers)

        uuid_lst: list[Optional[str]] = [None] * len(prompt_lst)
        if uuids is not None:
            uuid_lst = ensure_list(uuids)
        image_file_lst: list[Optional[str]] = [None] * len(prompt_lst)
        if image_files is not None:
            image_file_lst = ensure_list(image_files)
        datasource_lst = ["default"] * len(prompt_lst)
        if datasources is not None:
            datasource_lst = ensure_list(datasources)
        answer_length_lst = [-1] * len(prompt_lst)
        if answer_lengths is not None:
            answer_length_lst = ensure_list(answer_lengths)

        if len(set(datasource_lst)) != 1:
            err_msg = "all datasources should be the same"
            raise ValueError(err_msg)
        datasource = datasource_lst[0]

        # Process each prompt-answer-gt triplet using threads
        all_rewards: list[float] = []
        all_extracted_ans = []
        all_extracted_gt = []

        reward_config = self.get_reward_config_from_datasource(datasource)
        verifier = get_verifier_from_config(reward_config, datasource)

        if verifier.is_batch_verifier:
            # ! mypy issue, invalid signature and return type
            batch_rewards = verifier.judge(
                prompts=prompt_lst, answers=answer_lst, gt_answers=gt_answer_lst, image_files=image_file_lst
            )

            all_rewards = batch_rewards

            for answer, gt_answer, prompt in zip(answer_lst, gt_answer_lst, prompt_lst):  # noqa: B905
                extracted_ans = verifier.extract_answer(answer, question=prompt)
                extracted_gt = verifier.extract_answer(gt_answer, question=prompt)
                all_extracted_ans.append(extracted_ans)
                all_extracted_gt.append(extracted_gt)

        else:
            # Create thread pool
            with ThreadPoolExecutor(max_workers=min(128, len(prompts))) as executor:
                # Submit all tasks and store futures in order
                futures = []
                for prompt, answer, gt_answer, image_file in zip(prompt_lst, answer_lst, gt_answer_lst, image_file_lst):  # noqa: B905
                    future = executor.submit(
                        self._process_single_item,
                        prompt,
                        answer,
                        gt_answer,
                        image_file,
                        verifier,
                        debug=debug,
                    )
                    futures.append(future)
                for future in futures:
                    reward, extracted_ans, extracted_gt = future.result()
                    all_rewards.append(reward)
                    all_extracted_ans.append(extracted_ans)
                    all_extracted_gt.append(extracted_gt)

        # Check if there are any -inf rewards
        # make -inf rewards to min reward
        # if all -inf, make all rewards to 0
        if any(reward == float("-inf") for reward in all_rewards):
            non_inf_rewards = [r for r in all_rewards if r != float("-inf")]
            if len(non_inf_rewards) > 0:
                min_non_inf = min(non_inf_rewards)
                all_rewards = [min_non_inf if r == float("-inf") else r for r in all_rewards]
            else:
                all_rewards = [0.0 if r == float("-inf") else r for r in all_rewards]

        if log_reward_judge:
            if not (
                len(prompt_lst)
                == len(image_file_lst)
                == len(answer_lst)
                == len(gt_answer_lst)
                == len(datasource_lst)
                == len(all_rewards)
                == len(answer_length_lst)
                == len(uuid_lst)
            ):
                err_msg = (
                    "The length of prompts, image_files, answers, gt_answers, datasources, all_rewards, "
                    "answer_lengths, and uuids should be the same."
                )
                raise ValueError(err_msg)

            save_pobj = mkdir(log_save_dir)
            datasource_dir = save_pobj / datasource
            _ = mkdir(datasource_dir)

            reward_status = "pass@k" if any(reward > 0.75 for reward in all_rewards) else "not_pass@k"
            # Log each reward data pair
            for prompt, image_file, answer, gt_answer, reward, answer_length, uuid in zip(
                prompt_lst,
                image_file_lst,
                answer_lst,
                gt_answer_lst,
                all_rewards,
                answer_length_lst,
                uuid_lst,
                strict=True,
            ):
                # Setup save directory and path
                rollout_save_pobj = datasource_dir / f"rollout_reward_{reward_status}.jsonl"

                # Write reward data to file
                reward_data = {
                    "current_iteration": current_iteration,
                    "prompt": prompt,
                    "image_file": image_file,
                    "answer": answer,
                    "gt_answer": gt_answer,
                    "reward": reward,
                    "answer_token_length": answer_length,
                    "reward_sum_of_this_prompt": sum(all_rewards),
                    "uuid": uuid,
                }
                with open(rollout_save_pobj, "a") as f:
                    f.write(json.dumps(reward_data, ensure_ascii=False) + "\n")

            for prompt, image_file, answer, gt_answer, reward, answer_length, uuid in zip(
                prompt_lst,
                image_file_lst,
                answer_lst,
                gt_answer_lst,
                all_rewards,
                answer_length_lst,
                uuid_lst,
                strict=True,
            ):
                reward_status = "correct" if reward > 0 else "incorrect"
                rollout_save_pobj = datasource_dir / f"rollout_reward_{reward_status}.jsonl"
                with open(rollout_save_pobj, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "current_iteration": current_iteration,
                                "prompt": prompt,
                                "image_file": image_file,
                                "answer": answer,
                                "answer_token_length": answer_length,
                                "gt_answer": gt_answer,
                                "reward": reward,
                                "reward_sum_of_this_prompt": sum(all_rewards),
                                "uuid": uuid,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        if return_extracted_answers:
            return all_rewards, all_extracted_ans, all_extracted_gt

        return all_rewards

    def extract_answer_from_response(
        self, answers: Union[Sequence[str], str], datasources: Union[Sequence[str], str]
    ) -> list:
        # Ensure all inputs are lists
        answer_lst: list[str] = ensure_list(answers)
        datasource_lst: list[str] = ensure_list(datasources)

        all_extracted_ans = []

        for answer, datasource in zip(answer_lst, datasource_lst, strict=True):
            reward_config = self.get_reward_config_from_datasource(datasource)
            verifier = get_verifier_from_config(reward_config, datasource)
            extracted_ans = verifier.extract_answer(answer)
            all_extracted_ans.append(extracted_ans)

        return all_extracted_ans
