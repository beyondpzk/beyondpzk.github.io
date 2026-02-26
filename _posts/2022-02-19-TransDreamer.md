---
layout: post
title: TransDreamer
date: 2022-02-19
categories: [WorldModels]
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# TransDreamer

[paper link](https://arxiv.org/abs/2202.09481)


---

# TransDreamer —— 基于 Transformer 世界模型的强化学习

---

## 第一部分：背景与动机 (Background & Motivation)

### 1.1 从 Model-Free 到 Model-Based RL

在强化学习的图谱中，我们长期面临着**样本效率 (Sample Efficiency)** 的挑战。Model-Free 方法（如 DQN, PPO）通常需要与环境进行数百万次的交互才能收敛。

**基于模型的强化学习 (MBRL)** 通过学习一个环境的动态模型（即世界模型，World Model），允许智能体在“想象”中进行训练。
*   **核心优势**：
    1.  **样本效率**：通过在模型中进行规划或策略优化，减少对真实环境的依赖。<alphaxiv-paper-citation paper="2202.09481v2" title="Introduction" page="1" first="Its imagination-based training" last="world model" />
    2.  **知识复用**：世界模型捕捉了环境的物理规律，这些知识是任务无关的 (Task-agnostic)。
    3.  **安全规划**：在执行动作前可以在脑海中预演结果。

### 1.2 Dreamer 范式的回顾

要理解 TransDreamer，首先必须理解其前身 —— **Dreamer (Hafner et al., 2019, 2020)**。

Dreamer 使用 **RSSM (Recurrent State-Space Model)** 作为其世界模型。RSSM 是一个基于 RNN 的状态空间模型，它将状态分解为两部分：
1.  **确定性状态 (Deterministic State, $h_t$)**：由 RNN (如 GRU) 更新，负责记忆历史信息。
2.  **随机状态 (Stochastic State, $z_t$)**：由后验分布或先验分布采样得到，负责捕捉环境的不确定性。

$$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$$
$$z_t \sim P(z_t | h_t)$$

**RNN 的局限性**：
尽管 RSSM 取得了巨大成功，但 RNN 固有的**梯度消失**和**记忆瓶颈**问题，限制了其处理**长程依赖 (Long-term Dependency)** 和**复杂记忆推理 (Memory-based Reasoning)** 的能力。<alphaxiv-paper-citation paper="2202.09481v2" title="Limitations" page="1" first="However, Transformers" last="many domains" />

### 1.3 核心动机：Why Transformer?

Transformer 架构在 NLP 和 CV 领域已经证明了其处理长序列和直接访问历史记忆的优越性。本论文的核心问题是：**我们能否用 Transformer 替换 RNN 来构建一个更强大的世界模型？**

这并非简单的“即插即用”，面临两个挑战：
1.  **架构设计**：如何设计一个支持随机动作条件转换 (Stochastic Action-Conditioned Transitions) 的 Transformer 世界模型？
2.  **训练稳定性**：Transformer 在强化学习中的训练通常比 RNN 更不稳定。<alphaxiv-paper-citation paper="2202.09481v2" title="Challenge" page="1" first="training complex policy" last="is difficult" />

---

## 第二部分：TransDreamer 架构详解 (Architecture Deep Dive)

TransDreamer 是首个完全基于 Transformer 的 MBRL 智能体。其核心创新在于提出了 **TSSM (Transformer State-Space Model)**。

### 2.1 Transformer State-Space Model (TSSM)

TSSM 旨在替代 RSSM 中的 RNN 组件。在 RSSM 中，RNN 的隐状态 $h_t$ 充当了历史信息的压缩摘要。而在 TSSM 中，我们通过**注意力机制 (Attention Mechanism)** 直接访问历史轨迹。

#### 2.1.1 状态定义与输入表示

TSSM 不再维护一个递归的隐状态 $h_t$。相反，它在每个时间步 $t$，将之前的随机状态 $z$ 和动作 $a$ 的序列作为输入。

*   **输入序列**：在时间步 $t$，模型的输入是历史序列 $\{(\hat{z}_1, a_1), (\hat{z}_2, a_2), \dots, (\hat{z}_{t-1}, a_{t-1})\}$。
*   **位置编码**：为了引入时序信息，必须加上位置嵌入 (Positional Embeddings)。

#### 2.1.2 动力学预测 (Dynamics Prediction)

TSSM 利用 Transformer 预测下一时刻的随机状态 $z_t$。这对应于 Dreamer 中的**先验网络 (Prior Network)**。

$$ \hat{H}_t = \text{Transformer}(\{(\hat{z}_i, a_i)\}_{i=1}^{t-1}) $$

这里，$\hat{H}_t$ 是 Transformer 在时间步 $t$ 的输出表示（类似于 RSSM 中的 $h_t$），它聚合了整个历史上下文。

*   **先验分布 (Transition Model)**：
    基于历史预测当前状态的分布：
    $$ z_t \sim P_\phi(z_t | \hat{H}_t) $$

*   **后验分布 (Representation Model)**：
    结合当前观测图像 $x_t$ 的嵌入 $e_t$ 来修正状态估计：
    $$ z_t \sim Q_\phi(z_t | \hat{H}_t, e_t) $$
    注意：观测编码器 (Encoder) 通常是卷积神经网络 (CNN)。

#### 2.1.3 并行训练 (Parallel Training)

与 RNN 必须按时间步顺序展开不同，TSSM 利用了 Transformer 的并行性。在训练阶段，给定一个完整的轨迹，我们可以利用 **Masked Self-Attention (因果掩码)** 一次性计算出所有时间步的先验和后验状态。<alphaxiv-paper-citation paper="2202.09481v2" title="Efficiency" page="2" first="parallel trainability of" last="computational efficiency." />

这意味着训练速度在长序列上可能比 RNN 更具优势，但也带来了显存消耗的增加。

### 2.2 完整的训练目标 (Training Objectives)

TransDreamer 的训练目标与 Dreamer 类似，都是最大化**证据下界 (ELBO)**。

$$ \mathcal{L} = \mathbb{E} \left[ \sum_{t} \underbrace{\ln p(x_t | z_t, \hat{H}_t)}_{\text{Image Recon.}} + \underbrace{\ln p(r_t | z_t, \hat{H}_t)}_{\text{Reward Pred.}} - \beta \underbrace{\text{KL}[Q(z_t | \cdot) \parallel P(z_t | \cdot)]}_{\text{Dynamics Consistency}} \right] $$

1.  **图像重构**：解码器从 $z_t$ 和 $\hat{H}_t$ 重构原始图像 $x_t$。
2.  **奖励预测**：预测当前步的奖励 $r_t$。
3.  **KL 散度**：拉近先验（预测的未来）和后验（实际看到的未来）的距离，这是学习动力学的关键。

### 2.3 策略学习 (Policy Learning)

这部分是 MBRL 的核心：**在想象中学习 (Learning in Imagination)**。

1.  **想象展开 (Rollout)**：从重播缓冲区 (Replay Buffer) 中采样起始状态。
2.  **模拟未来**：使用 TSSM 的先验网络 $P(z_t | \hat{H}_t)$ 和当前的策略网络 $\pi(a_t | z_t, \hat{H}_t)$ 逐步生成未来的轨迹 $\{\hat{z}_\tau, \hat{a}_\tau\}_{\tau=t}^{t+H}$。
3.  **价值评估**：计算想象轨迹上的回报，并使用 Actor-Critic 算法更新策略。

**注意**：在想象阶段，Transformer 必须像 RNN 一样自回归地 (Auto-regressively) 生成步骤，这时无法并行化，推理成本高于 RNN。

---

## 第三部分：实现细节与工程挑战 (Implementation & Challenges)

在将 Transformer 应用于 RL 时，细节决定成败。

### 3.1 记忆机制与滑动窗口

由于显存限制，我们无法让 Transformer 关注无限长的历史。TransDreamer 采用了类似 Transformer-XL 的机制或简单的滑动窗口。
*   **训练时**：从 Replay Buffer 采样固定长度的片段（例如 50-100 步）。
*   **想象时**：需要维护一个 KV-Cache 或历史 buffer，以便 Transformer 能处理超出训练长度的上下文。

### 3.2 训练稳定性

论文指出，直接训练 Transformer 策略网络非常困难。但在 MBRL 框架下，由于有重构损失和 KL 损失作为辅助任务，TSSM 的训练相对稳定。<alphaxiv-paper-citation paper="2202.09481v2" title="Stability" page="1" first="learning a transformer-based" last="facilitate learning." />

### 3.3 计算开销对比

*   **RSSM (RNN)**：
    *   训练：$O(T)$ (时间), $O(T)$ (内存)。
    *   推理/想象：$O(1)$ per step。
*   **TSSM (Transformer)**：
    *   训练：$O(1)$ (时间 - 并行), $O(T^2)$ (内存 - 注意力矩阵)。
    *   推理/想象：$O(T)$ or $O(1)$ with cache per step (但常数项很大)。

**讲师点评**：这是一个关键的 Trade-off。TSSM 换取了更强的记忆能力，但牺牲了推理速度和显存效率。因此，TransDreamer 减少了想象轨迹的数量 (Number of Imagination Trajectories) 以平衡计算量。<alphaxiv-paper-citation paper="2202.09481v2" title="Memory" page="5" first="Due to the" last="than Dreamer." />

---

## 第四部分：实验与分析 (Experiments & Analysis)

论文设计了专门的实验来验证 Transformer 在长程记忆上的优势。

### 4.1 核心实验：Hidden Order Discovery (隐藏顺序发现)

这是一个专门设计的任务，用于测试“长程记忆”和“逻辑推理”。

*   **任务描述**：
    *   场景中有多个不同颜色的球（如 4-6 个）。
    *   每局游戏有一个固定的“正确顺序”。
    *   智能体必须按正确顺序收集球。如果收集错误，所有球重置，但智能体位置不变，且**顺序不变**。
    *   **关键点**：为了高效完成任务，智能体必须**记住**之前的尝试中哪些顺序是错误的，从而进行排除法推理。

*   **2D 与 3D 版本**：
    *   2D Grid：上帝视角，相对简单。
    *   3D Room：第一人称视角 (Unity)，存在严重的**部分可观测性 (Partial Observability)**。

### 4.2 实验结果分析

1.  **胜率对比**：
    在 3D 4-Ball 任务中，TransDreamer 达到了 **18%** 的成功率，而 Dreamer 仅为 **10%**。在更难的 5-Ball 任务中，Dreamer 几乎无法成功 (0%)，而 TransDreamer 仍有表现。<alphaxiv-paper-citation paper="2202.09481v2" title="Success Rate" page="16" first="TransDreamer performs better" last="nearly fails." />

2.  **为什么 TransDreamer 赢了？**
    这验证了 Transformer 的注意力机制能够有效地从遥远的过去提取信息（例如：“我 50 步之前试过先拿红球，结果失败了，所以现在不能拿红球”）。RNN 很难在长序列中保持这种离散的、精确的逻辑信息。

### 4.3 图像生成与世界模型质量

论文定性地展示了“想象”的轨迹。
*   **TransDreamer** 的想象更加清晰，且能准确预测长时后的物体颜色和奖励。
*   **Dreamer** 在长时预测后，物体颜色开始混乱，甚至消失，导致奖励预测失败。<alphaxiv-paper-citation paper="2202.09481v2" title="Quality" page="9" first="Dreamer, on the" last="incorrect." />

### 4.4 标准基准测试 (DMC & Atari)

在 DeepMind Control Suite (DMC) 和 Atari 上：
*   TransDreamer 的表现与 Dreamer **相当 (Comparable)**。
*   **观察**：在这些不需要复杂记忆的任务上，TransDreamer 收敛速度往往比 Dreamer 慢。
*   **解释**：RNN 具有很强的**时序归纳偏置 (Sequential Inductive Bias)**，适合处理马尔可夫性较强或仅需短时记忆的物理控制任务。Transformer 需要更多数据来学习这种时序结构。

---

## 第五部分：总结与讨论 (Conclusion & Discussion)

### 5.1 核心结论

1.  **可行性**：TransDreamer 证明了在 MBRL 框架下，Transformer 完全可以替代 RNN 作为世界模型，且训练稳定。
2.  **长程优势**：在需要记忆推理 (Memory-based Reasoning) 的部分可观测任务中，TransDreamer 显著优于基于 RNN 的 Dreamer。
3.  **通用性代价**：在简单任务上，Transformer 的优势不明显，且计算成本更高。

### 5.2 开放性问题 (课堂讨论)

*   **计算效率问题**：在实际机器人应用中，推理延迟至关重要。Transformer 的 $O(T^2)$ 或 $O(T)$ 复杂度是否可以通过 Linear Attention 或 State Space Models (如 S4, Mamba) 来优化？
*   **世界模型的本质**：世界模型究竟应该记住所有的历史细节（Transformer 方式），还是应该学习一个紧凑的状态压缩（RNN 方式）？
*   **多模态扩展**：Transformer 架构天然适合多模态（文本 + 图像）。TransDreamer 是否是通向通才智能体 (Generalist Agent) 的一步？

### 5.3 课后思考

请同学们思考：如果我们将 TSSM 中的 Transformer 换成最近流行的 **Mamba (State Space Model)** 架构，预期会有什么变化？（提示：Mamba 结合了 RNN 的推理速度和 Transformer 的训练并行性）。

---

**结束语**：TransDreamer 是 MBRL 领域的一个重要里程碑，它打破了 RNN 在世界模型中的统治地位，为利用更强大的序列模型架构打开了大门。希望大家通过这篇论文，能深刻理解模型架构对强化学习智能体能力的根本性影响。


## RSSM 与 TSSM的区别


我们将公式分为三个关键部分来解析：**确定性路径 (Deterministic Path)**、**后验表示 (Posterior / Representation)** 和 **先验预测 (Prior / Transition)**。

---

### 预备知识：符号定义

在深入公式前，我们要对齐符号（Notation）：
*   $x_t$：当前时刻的观测图像（Image Observation）。
*   $e_t$：图像经过 CNN 编码后的特征向量（Embedding），即 $e_t = \text{Encoder}(x_t)$。
*   $a_{t-1}$：上一时刻采取的动作（Action）。
*   $z_t$：我们要学习的**随机隐状态**（Stochastic Latent State），它服从某种分布（通常是高斯分布或 Categorical 分布）。
*   $h_t$ / $\hat{H}_t$：**确定性上下文**（Deterministic Context），这是最重要的区别所在。

---

### 1. 确定性路径 (Deterministic Path)：如何聚合历史？

这是 Table 1 中最根本的区别，决定了模型如何“记忆”过去。

*   **RSSM (Dreamer) - 递归更新**
    $$ h_t = f_{\text{GRU}}(h_{t-1}, z_{t-1}, a_{t-1}) $$
    *   **解读**：
        这是一个标准的 RNN 更新公式。
        1.  **输入**：上一时刻的记忆 $h_{t-1}$，上一时刻的随机状态 $z_{t-1}$，上一时刻的动作 $a_{t-1}$。
        2.  **操作**：通过 GRU 单元（$f_{\text{GRU}}$）融合这些信息。
        3.  **局限**：必须依赖 $h_{t-1}$。如果 $h_{t-1}$ 丢失了信息（例如 100 步之前的关键线索），$h_t$ 也无法挽回。这就是所谓的“马尔可夫属性”的强加——我们假设 $h_{t-1}$ 包含了所有必要的历史。

*   **TSSM (TransDreamer) - 全局注意力**
    $$ \hat{H}_t = f_{\text{Txfm}}(\{(\hat{z}_1, a_1), \dots, (\hat{z}_{t-1}, a_{t-1})\}) $$
    *   **解读**：
        这是 Transformer 的并行处理公式。
        1.  **输入**：**集合** (Set) 或 **序列** (Sequence)。它不依赖单一的 $h_{t-1}$，而是直接读取从 $t=1$ 到 $t-1$ 的所有 $(\hat{z}, a)$ 对。
        2.  **操作**：通过 Self-Attention ($f_{\text{Txfm}}$) 计算。模型可以根据当前的需要，动态地分配权重（Attention Weight）给历史中的任意时刻。
        3.  **优势**：$\hat{H}_t$ 是一个包含了整个历史上下文的丰富表示。它打破了递归的依赖链。
    *   **引用**：TSSM 使用 Transformer 预测先验状态，利用了对历史的并行访问。<alphaxiv-paper-citation paper="2202.09481v2" title="TSSM Dynamics" page="3" first="We propose the" last="temporal dynamics." />

---

### 2. 后验表示 (Posterior / Representation Model)：这是什么？

这个公式描述了：**“当我看到了当前的图像 $x_t$，我认为我现在处于什么状态 $z_t$？”** 这通常用于训练阶段，为模型提供“真值”监督。

*   **RSSM (Dreamer)**
    $$ z_t \sim q_\phi(z_t | h_t, e_t) $$
*   **TSSM (TransDreamer)**
    $$ z_t \sim q_\phi(z_t | \hat{H}_t, e_t) $$

    注意，这两个公式的形式非常相似！
    *   它们都利用了当前的图像编码 $e_t$。
    *   区别仅在于**上下文信息的来源**：RSSM 使用递归压缩的 $h_t$，而 TSSM 使用注意力聚合的 $\hat{H}_t$。
    *   这意味着，即使是处理当前的图像，TransDreamer 也能利用更长远的历史背景来辅助理解（例如：看到一扇关着的门，RNN 可能只知道“门关着”，但 Transformer 可能记得“我 5 分钟前锁了它”）。

---

### 3. 先验预测 (Prior / Transition Model)：接下来会发生什么？

这个公式描述了：**“闭上眼睛（没有图像 $x_t$），仅凭记忆，我认为我现在处于什么状态 $z_t$？”** 这是模型进行**想象 (Imagination)** 和 **规划 (Planning)** 的基础。

*   **RSSM (Dreamer)**
    $$ \hat{z}_t \sim p_\phi(\hat{z}_t | h_t) $$
*   **TSSM (TransDreamer)**
    $$ \hat{z}_t \sim p_\phi(\hat{z}_t | \hat{H}_t) $$

    *   这是 MBRL 的核心。如果 $p_\phi$ 预测得准确，我们就可以在脑海中演练。
    *   在 TransDreamer 中，$\hat{H}_t$ 的质量直接决定了预测的准确性。由于 $\hat{H}_t$ 包含了更丰富的长程依赖信息，TSSM 在需要逻辑推理的任务（如 Hidden Order Discovery）中，能做出比 RSSM 准确得多的未来预测。
    *   **KL 散度损失**：训练的目标之一就是最小化 后验 $q$ 和 先验 $p$ 之间的 KL 散度（$\text{KL}[q || p]$）。这强迫“想象”（先验）尽可能接近“现实”（后验）。

---

### 总结 Table 1 的核心逻辑

| 组件 | 数学本质 (RSSM) | 数学本质 (TSSM) | 核心差异 |
| :--- | :--- | :--- | :--- |
| **上下文** | $h_t = \text{GRU}(h_{t-1}, \dots)$ | $\hat{H}_t = \text{Transformer}(\text{History}_{1:t-1})$ | **递归 vs. 检索** |
| **感知 (Posterior)** | $z \sim q(z \| h_t, \text{image})$ | $z \sim q(z \| \hat{H}_t, \text{image})$ | 上下文来源不同 |
| **想象 (Prior)** | $\hat{z} \sim p(\hat{z} \| h_t)$ | $\hat{z} \sim p(\hat{z} \| \hat{H}_t)$ | 长程预测能力不同 |

**一句话总结**：
Table 1 告诉我们要做的仅仅是将 Dreamer 公式中的 $h_t$（由 RNN 产生的有损压缩）替换为 $\hat{H}_t$（由 Transformer 产生的无损检索），但这简单的替换背后，是**时间复杂度**与**空间复杂度**的权衡，以及**记忆能力**的质变。

