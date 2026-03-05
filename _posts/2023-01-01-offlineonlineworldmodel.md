---
layout: post
title: offlineonlineworldmodel
date: 2023-01-01
categories: [Understandings]
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# offline VS online worldmodel

在具身智能和强化学习的语境下，理解“在线（Online）”与“离线（Offline）”世界模型的本质差异，是我们深入研究动力学建模（Dynamics Modeling）和模型预测控制（MPC）的基石。

---

### 一、 在线世界模型 (Online World Models)

#### 1. 定义与机制
在线世界模型的运作范式可以概括为**“交互-学习-规划”的闭环（Interleaved Data Collection and Training）**。
在这个设定下，智能体（Agent）被直接放置在环境中。它利用当前不完美的策略或世界模型在环境中执行动作，收集新的状态转移数据 $(s_t, a_t, r_t, s_{t+1})$ 存入经验回放池（Replay Buffer）。随后，模型从缓冲池中采样数据来更新自己的动力学预测网络和奖励预测网络。伴随着世界模型的升级，策略（Policy）也在模型生成的“想象空间（Latent Imagination）”中同步优化，进而在下一轮采集中获取更高质量的数据。

#### 2. 核心优势与局限
*   **优势（自我纠错与覆盖）**：在线模型具有极强的“在分布（On-distribution）”纠错能力。如果模型在某个状态空间区域预测错误，导致策略在这个区域失败，智能体在真实环境的试错中会收集到该区域的真实物理数据。下一次更新时，模型就会修正这个盲区。
*   **局限（安全与效率惩罚）**：在真实的机器人控制中，“在线探索”是极其危险且昂贵的。随机或试探性的动作极易导致机械臂损坏或环境破坏（安全约束问题）。此外，因为它的表征学习往往依赖当前的策略覆盖范围和特定任务的奖励信号，一旦任务（Reward Function）改变，整个隐空间表征往往会失效，需要从头再来。

#### 3. 代表性工作
*   **Dreamer 系列 (DreamerV1 / V2 / V3)**：由 Danijar Hafner 提出，是在线世界模型的绝对标杆。DreamerV3 使用了循环状态空间模型（RSSM），将确定性隐状态（RNN/GRU）和随机隐状态结合，将图像编码为离散的分类分布（Categorical Representations）。它完全在隐空间中“想象”未来轨迹来训练 Actor-Critic 网络。
*   **IRIS (Transformers are Sample-Efficient World Models)**：由 Micheli 等人提出，将离散自编码器与类似于 GPT 的 Transformer 结合。它在在线收集数据的基础上，利用自回归模型生成未来 Token，在极低的数据采样率下（Sample-efficient）达到了与无模型（Model-free）算法相媲美的效果。

---

### 二、 离线世界模型 (Offline World Models)

#### 1. 定义与机制
离线世界模型的运作范式是**“先见后算（Learn from Logged Data）”**。
在这种设定下，智能体**绝对不被允许**在训练阶段与环境发生任何交互。我们直接喂给模型一个巨大的、预先收集好的静态轨迹数据集 $\mathcal{D} = \{ (o_0, a_0, r_0, o_1), \dots \}$。这个数据集可以是由人类操作员演示录制的（Teleoperation），也可以是随机策略或其他算法生成的混合数据。模型必须仅凭这个静态数据集，抽象出环境的全局物理动力学。

#### 2. 核心优势与局限
*   **优势（安全、可扩展性与任务无关性）**：这是迈向“机器人基础模型（Robotics Foundation Models）”的必经之路。因为无需交互，我们可以安全地利用互联网上海量的异构视频、开源机器人数据集（如 Open X-Embodiment）进行大规模预训练。像我们刚才讲的 **DINO-WM**，就是希望在离线阶段剥离任务特定的奖励，学习一个通用的物理推演引擎。
*   **局限（分布偏移与模型利用 OOD Exploitation）**：这是离线世界模型的致命弱点。当在测试阶段使用规划算法（如交叉熵方法 CEM）在模型中搜索最优轨迹时，优化器会疯狂地去寻找“在模型看来代价极低，但实际上是因为模型在该区域没见过数据而产生的幻觉（Hallucination）”的动作序列。这种现象在学术界被称为**分布外状态利用（Out-of-Distribution Exploitation）**。因为模型无法在线去环境中验证这个“捷径”是否真实，这会导致规划彻底失效。

#### 3. 代表性工作
*   **DINO-WM **：利用预训练视觉特征（DINOv2）在纯离线数据集上进行自回归隐空间动力学建模，实现了无需奖励、零样本的下游视觉规划。
*   **MOPO (Model-Penalized Offline RL) / COMBO**：严格来说这是基于模型的离线强化学习算法。为了对抗前面提到的 OOD 幻觉问题，MOPO 引入了**认知不确定性惩罚（Epistemic Uncertainty Penalization）**。它训练一个世界模型的集成（Ensemble），当多个模型对未来的预测分歧很大时，说明来到了离线数据集未覆盖的区域，算法会人为给予惩罚，迫使规划器“保守”地待在数据分布内。
*   **Genie / 动作条件扩散模型 (Action-conditioned Video Diffusion, 如 AVDC)**：近年来涌现的生成式工作。它们在海量离线游戏视频或机器人视频上训练。输入当前画面和离散/连续动作，直接生成未来的视频帧。虽然它们通常被称为生成式 AI，但其本质正是拟合了环境动力学 $p(o_{t+1}|o_t, a_t)$ 的离线世界模型。

---

### 三、 深度比较

为了让你在研究思路上更清晰，我将两者的本质区别归纳为以下三个维度的对抗：

1. **认知边界的突破方式（Exploration vs. Extrapolation）**：
   在线模型遇到知识盲区时，它的解决手段是**“去试试看”**（Exploration）；离线模型遇到盲区时，它没有试错权，只能依赖神经网络自身的泛化能力进行**“外推猜测”**（Extrapolation）。这也是为什么 DINO-WM 要引入 DINOv2 作为视觉先验——强大的预训练特征为离线外推提供了坚实的物理几何基础。

2. **表征坍塌与任务泛化（Task-Overfitting vs. Task-Agnosticism）**：
   在线模型为了尽快获得高额奖励，其表征网络会主动忽略那些对当前任务无关的视觉细节（比如背景墙壁的颜色、无关物体的运动）。这导致它在单一任务上极强，但换个任务就抓瞎。离线世界模型（特别是无奖励驱动的自监督离线模型，如 DINO-WM）被迫去拟合场景中**所有**正在发生的变化，因此保留了更完备的物理状态，具备了零样本解决新任务（Zero-shot Planning）的潜力。

3. **未来范式（The Future Paradigm）**：
   当前学术界正在走向两者的融合——**Offline Pre-training with Online Fine-tuning**。即：先用海量无奖励的离线数据（视频、人类演示）训练一个庞大且通用的离线世界模型（构建基础物理常识），然后将其部署到特定机器人上，进行少量的在线强化学习微调，以消除具体的分布偏移。
