# VLM Reward System
[中文阅读](./README_zh.md)
<div align="center">
<img src="resources/logo.svg" width="40%"/>
</div>

<p align="center">
    <strong>A Powerful Reward System for Vision-Language Model Reinforcement Learning</strong>
</p>

<p align="center">
    👋 Join our <a href="https://discord.com/invite/8cnQKdAprg" target="_blank">Discord</a> or <a href="https://github.com/THUDM/GLM-4.1V-Thinking/issues" target="_blank">GitHub Issues</a>
    <br>
    📍 Experience our models at <a href="https://chatglm.cn/" target="_blank">ChatGLM</a> and <a href="https://open.bigmodel.cn/" target="_blank">Zhipu AI Platform</a>
    <br>
    🚀 Related Project: <a href="https://github.com/THUDM/GLM-4.1V-Thinking" target="_blank">GLM-4.1V-Thinking</a>
    <br>
    💡 Try the demo: <code>python examples/reward_system_demo.py</code>
</p>

## About

**VLM Reward System** is developed by the **CogVLM Team** at **Zhipu AI (智谱AI)**. As a core component of our vision-language model training infrastructure, this system powers the reinforcement learning training of our GLM-4.1V series models.

**Key Features:**
- **Production-Ready**: Battle-tested in GLM-4.1V-Thinking training
- **Easy Integration**: Works with any RL training pipeline
- **Multiple Verifiers**: Math, general, chart and more
- **Hybrid Verification**: Combines rule-based verifiers with LLM-as-a-judge
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
3. **Configure the reward system**:
   ```bash
   cp examples/configs/example.yaml.template examples/configs/example.yaml
   ```
   Edit the `example.yaml` file to configure the reward system.

4. **Run the demo**:
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
from glmv_reward import RewardSystem

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

## Supported Verifiers

Our reward system includes multiple specialized verifiers, each optimized for different types of reasoning:

### Core Verifiers
- **Math Verifier**: Evaluates mathematical correctness using symbolic computation
- **Biology Verifier**: Specialized for biological and life science questions
- **Chemistry Verifier**: Handles chemistry problems and molecular reasoning
- **Physics Verifier**: Evaluates physics problems and scientific reasoning
- **Geography Verifier**: Specialized for geographical knowledge and spatial reasoning
- **Liberal Arts Verifier**: Handles literature, history, and humanities questions

### Multimodal Verifiers
- **Chart Verifier**: Analyzes chart and visualization responses
- **OCR Verifier**: Evaluates optical character recognition tasks
- **Multi-Image Verifier**: Handles multi-image understanding tasks
- **VQA Verifier**: Specialized for visual question answering
- **Counting Verifier**: Evaluates counting and numerical reasoning in images

### Specialized Task Verifiers
- **General Verifier**: Handles general reasoning tasks with LLM judge fallback
- **Language Mix Verifier**: Detects inappropriate language mixing patterns
- **MMSI Verifier**: Specialized for MMSI task
- **GeoQuest Verifier**: Handles geography-related question answering tasks

## Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{glmvteam2025glm41vthinkingversatilemultimodalreasoning,
      title={GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning},
      author={GLM-V Team and Wenyi Hong and Wenmeng Yu and Xiaotao Gu and Guo Wang and Guobing Gan and Haomiao Tang and Jiale Cheng and Ji Qi and Junhui Ji and Lihang Pan and Shuaiqi Duan and Weihan Wang and Yan Wang and Yean Cheng and Zehai He and Zhe Su and Zhen Yang and Ziyang Pan and Aohan Zeng and Baoxu Wang and Boyan Shi and Changyu Pang and Chenhui Zhang and Da Yin and Fan Yang and Guoqing Chen and Jiazheng Xu and Jiali Chen and Jing Chen and Jinhao Chen and Jinghao Lin and Jinjiang Wang and Junjie Chen and Leqi Lei and Letian Gong and Leyi Pan and Mingzhi Zhang and Qinkai Zheng and Sheng Yang and Shi Zhong and Shiyu Huang and Shuyuan Zhao and Siyan Xue and Shangqin Tu and Shengbiao Meng and Tianshu Zhang and Tianwei Luo and Tianxiang Hao and Wenkai Li and Wei Jia and Xin Lyu and Xuancheng Huang and Yanling Wang and Yadong Xue and Yanfeng Wang and Yifan An and Yifan Du and Yiming Shi and Yiheng Huang and Yilin Niu and Yuan Wang and Yuanchang Yue and Yuchen Li and Yutao Zhang and Yuxuan Zhang and Zhanxiao Du and Zhenyu Hou and Zhao Xue and Zhengxiao Du and Zihan Wang and Peng Zhang and Debing Liu and Bin Xu and Juanzi Li and Minlie Huang and Yuxiao Dong and Jie Tang},
      year={2025},
      eprint={2507.01006},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.01006},
}
```