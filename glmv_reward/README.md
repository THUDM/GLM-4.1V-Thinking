# VLM Reward System

<div align="center">
<img src="https://raw.githubusercontent.com/MetaGLM/glm-4/main/resources/logo.svg" width="40%"/>
</div>
<p align="center">
    👋 Join our <a href="https://discord.com/invite/8cnQKdAprg" target="_blank">Discord</a> or <a href="https://github.com/THUDM/GLM-4/issues" target="_blank">GitHub Issues</a>
    <br>
    📖 View the <a href="https://open.bigmodel.cn/" target="_blank">Zhipu AI Platform</a> for more details.
    <br>
    💡 Try the demo: <code>python examples/reward_system_demo.py</code>
</p>

## What is VLM Reward System?

VLM Reward System is a powerful evaluation framework designed for **reinforcement learning training** of vision-language models. It automatically evaluates model responses against ground truth answers, providing reward scores that guide your RL training process.

**Key Features:**
- **Easy Integration**: Works with any RL training pipeline
- **Multiple Verifiers**: Math, general reasoning, chart analysis, and more
- **LLM Judge Fallback**: Uses Zhipu AI models for complex evaluations
- **Flexible Configuration**: YAML-based setup for different use cases

## Quick Start

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Set your API key**:
   ```bash
   export ZHIPUAI_API_KEY='your_api_key_here'
   ```

3. **Run the demo**:
   ```bash
   python examples/reward_system_demo.py
   ```

## How It Works

The reward system takes three inputs and outputs a reward score:

```
Input:  Question + Ground Truth + Model Response
        ↓
Output: Reward Score (0.0 - 1.0)
```

**Example Usage in RL Training:**

```python
from vlm_reward_system import RewardSystem

# Initialize the reward system
reward_system = RewardSystem("examples/configs/example.yaml")

# Evaluate model responses
rewards = reward_system.get_reward(
    prompts=["What is 15 + 27?"],
    answers=["<think>15 + 27 = 42</think><answer><|begin_of_box|>42<|end_of_box|></answer>"],
    gt_answers=["<think>15 + 27 = 42</think><answer><|begin_of_box|>42<|end_of_box|></answer>"],
    datasources=["math"]
)

# Use reward in your RL training
print(f"Reward: {rewards[0]}")  # Output: 1.0 (correct answer)
```

## Supported Verifiers

- **Math Verifier**: Evaluates mathematical correctness using symbolic computation
- **General Verifier**: Handles general reasoning tasks
- **Chart Verifier**: Analyzes chart and visualization responses
- **LLM Judge**: Uses Zhipu AI models as fallback for complex evaluations

## Configuration

The system uses YAML configuration files. Example:

```yaml
reward_configs:
  math_verifier_config:
    verifier_type: "math"
    enable_llm_judge_fallback: true
    llm_judge_url:
      - "https://open.bigmodel.cn/api/paas/v4/chat/completions"
```

## Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{vlm-reward-system,
  title={VLM Reward System: A Framework for Reinforcement Learning Training of Vision-Language Models},
  author={VLM Team},
  year={2025},
  url={https://github.com/your-org/vlm-reward-system}
}
```