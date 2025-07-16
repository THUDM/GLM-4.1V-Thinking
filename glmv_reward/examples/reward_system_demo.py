#!/usr/bin/env python3
"""
VLM Reward System Demo Script

This demo shows how to use the VLM reward system for RL training.
It simulates a rollout scenario where we have:
- Question (from dataset)
- Ground truth answer
- Model response (from policy model)
- Reward evaluation (our system)

Prerequisites:
1. Set ZHIPUAI_API_KEY environment variable.
2. Install the glmv-reward package.
3. Copy the template configuration file and fill in your api key.
Usage:
    python examples/reward_system_demo.py
"""

import os
from pathlib import Path

from glmv_reward import RewardSystem
from glmv_reward.utils.llm import post_query_llm


def main() -> None:
    """Main demo function."""
    print("=" * 60)
    print("VLM Reward System Demo - Simulating RL Rollout Scenario")
    print("=" * 60)

    api_key = os.getenv("ZHIPUAI_API_KEY")
    # Check API key
    if api_key is None:
        print("Error: ZHIPUAI_API_KEY environment variable is not set!")
        print("Please set your Zhipu AI API key:")
        print("export ZHIPUAI_API_KEY='your_api_key_here'")
        return

    # ============================================================================
    # SETUP: Initialize the reward system with configuration
    # ============================================================================
    config_file = Path(__file__).parent / "configs" / "example.yaml"
    print(f"Loading configuration from: {config_file}")

    try:
        reward_system = RewardSystem(config_file)

    except Exception as e:
        print(f"Error initializing reward system: {e}")
        return

    # ============================================================================
    # INPUT DATA: Prepare the three key inputs for reward evaluation
    # In real RL training, these would come from your dataset and policy model
    # ============================================================================

    # 1. Question (from your dataset)
    question = "What is 15 + 27?"

    # 2. Ground truth answer (from your dataset)
    ground_truth = (
        "<think>\nI need to add 15 and 27.\n15 + 27 = 42\n</think>\n"
        "<answer>\n<|begin_of_box|>42<|end_of_box|>\n</answer>"
    )

    # 3. Model response (normally from your policy model, here we simulate using Zhipu AI)
    print(f"\nQuestion: {question}")
    print("Ground Truth: 42")
    print("Generating model response to simulate rollout...")

    # Simulate model response generation (this represents your policy model's output)
    prompt = f"""
        Please solve the following math problem step by step.
        Provide your reasoning in <think> tags and your final answer in <answer> tags with the result inside <|begin_of_box|> and <|end_of_box|> tags.

        Example format:
        <think>
        Let me work through this step by step...
        </think>
        <answer>
        <|begin_of_box|>42<|end_of_box|>
        </answer>

        Question: {question}
""".strip()  # noqa: E501

    try:
        model_response = post_query_llm(
            prompt,
            api_key,
            url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            model="glm-4-flash",
            max_tokens=500,
            temperature=0.1,
        )

        if len(model_response) == 0:
            print("Failed to get response from Zhipu AI")
            return

        print(f"Model Response:\n{model_response}")

    except Exception as e:
        print(f"Error calling Zhipu AI: {e}")
        return

    # ============================================================================
    # REWARD EVALUATION: This is the core of our system
    # Input: question + ground_truth + model_response
    # Output: reward score for RL training
    # ============================================================================

    print("\n" + "=" * 40)
    print("REWARD EVALUATION")
    print("=" * 40)

    try:
        # Call our reward system with the three inputs
        rewards = reward_system.get_reward(
            prompts=[question],  # Question from dataset
            answers=[model_response],  # Response from policy model
            gt_answers=[ground_truth],  # Ground truth from dataset
            datasources=["math"],  # Use math verifier
            return_extracted_answers=True,  # Get detailed results
        )

        # Process results
        if isinstance(rewards, tuple):
            reward_scores, extracted_answers, extracted_ground_truths = rewards
            reward_score = reward_scores[0]
            extracted_answer = extracted_answers[0]
            extracted_gt = extracted_ground_truths[0]
        else:
            reward_score = rewards[0]
            extracted_answer = None
            extracted_gt = None

        # Display evaluation results
        print(f"Extracted Model Answer: {extracted_answer}")
        print(f"Extracted Ground Truth: {extracted_gt}")
        print(f"Reward Score: {reward_score}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # ============================================================================
    # SUMMARY: How to use this in your RL training
    # ============================================================================

    print("\n" + "=" * 40)
    print("INTEGRATION SUMMARY")
    print("=" * 40)
    print("In your RL training loop:")
    print("1. Get question + ground_truth from your dataset")
    print("2. Generate model_response from your policy model")
    print("3. Use reward_system.get_reward() to get reward score")
    print("4. Use reward score to update your policy model")
    print(f"Reward score: {reward_score} (ready for RL training)")


if __name__ == "__main__":
    main()
