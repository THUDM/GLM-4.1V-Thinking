# VLM Reward System
<div align="center">
<img src="resources/logo.svg" width="40%"/>
</div>

<p align="center">
    <strong>强大的VLM模型强化学习的奖励系统</strong>
</p>

<p align="center">
    👋 加入我们的 <a href="https://discord.com/invite/8cnQKdAprg" target="_blank">Discord</a> 或 <a href="https://github.com/THUDM/GLM-4.1V-Thinking/issues" target="_blank">GitHub Issues</a>
    <br>
    📍 在 <a href="https://chatglm.cn/" target="_blank">ChatGLM</a> 和 <a href="https://open.bigmodel.cn/" target="_blank">智谱AI平台</a> 体验我们的模型
    <br>
    🚀 相关项目：<a href="https://github.com/THUDM/GLM-4.1V-Thinking" target="_blank">GLM-4.1V-Thinking</a>
    <br>
    💡 试用演示：<code>python examples/reward_system_demo.py</code>
</p>

## 关于

**VLM 奖励系统** 由 **智谱AI (智谱AI)** 的 **CogVLM 团队** 开发。作为我们VLM模型强化训练基础设施的核心组件，该系统为我们的 GLM-4.1V 系列模型的强化学习训练提供支持。

**主要特性：**
- **生产级**：在 GLM-4.1V-Thinking 训练中经过实战检验
- **易于集成**：与任何强化学习训练管道兼容
- **多种验证器**：数学、通用、图表等多种类型
- **混合验证**：结合基于规则的验证器和LLM判断
- **灵活配置**：基于 YAML 的配置，适用于不同使用场景

## 快速开始

1. **安装包**：
   ```bash
   pip install -e .
   ```

2. **设置您的 API 密钥**：
   ```bash
   export ZHIPUAI_API_KEY='your_api_key_here'
   ```
3. **配置奖励系统**：
   ```bash
   cp examples/configs/example.yaml.template examples/configs/example.yaml
   ```
   编辑 `example.yaml` 文件来配置奖励系统。

4. **运行demo**：
   ```bash
   python examples/reward_system_demo.py
   ```

## 工作原理

奖励系统接收三个输入并输出奖励分数：

```
输入：问题 + 标准答案 + 模型响应
        ↓
输出：奖励分数 (0.0 - 1.0)
```

**在强化学习训练中的使用示例：**

```python
from glmv_reward import RewardSystem

# 初始化奖励系统
reward_system = RewardSystem("examples/configs/example.yaml")

# 评估模型响应
rewards = reward_system.get_reward(
    prompts=["15 + 27 等于多少？"],
    answers=["<think>15 + 27 = 42</think><answer><|begin_of_box|>42<|end_of_box|></answer>"],
    gt_answers=["<think>15 + 27 = 42</think><answer><|begin_of_box|>42<|end_of_box|></answer>"],
    datasources=["math"]
)

# 在强化学习训练中使用奖励
print(f"奖励: {rewards[0]}")  # 输出: 1.0 (正确答案)
```

## 配置

系统使用 YAML 配置文件。示例：

```yaml
reward_configs:
  math_verifier_config:
    verifier_type: "math"
    enable_llm_judge_fallback: true
    llm_judge_url:
      - "https://open.bigmodel.cn/api/paas/v4/chat/completions"
```

## 支持的验证器

我们的奖励系统包含多个专门的验证器，每个都针对不同类型的推理进行了优化：

### 核心验证器
- **数学验证器**：使用符号计算评估数学正确性
- **生物验证器**：专门处理生物和生命科学问题
- **化学验证器**：处理化学问题和分子推理
- **物理验证器**：评估物理问题和科学推理
- **地理验证器**：专门处理地理知识和空间推理
- **文科验证器**：处理文学、历史和人文学科问题

### 多模态验证器
- **图表验证器**：分析图表和可视化响应
- **OCR 验证器**：评估光学字符识别任务
- **多图像验证器**：处理多图像理解任务
- **VQA 验证器**：专门用于视觉问答
- **计数验证器**：评估图像中的计数和数值推理

### 专门任务验证器
- **通用验证器**：处理具有大模型判断回退的通用推理任务
- **语言混合验证器**：检测不当的语言混合模式
- **MMSI 验证器**：专门用于 MMSI 任务
- **GeoQuest 验证器**：处理地理相关的问答任务

## 引用

如果您觉得我们的工作有帮助，请考虑引用：

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