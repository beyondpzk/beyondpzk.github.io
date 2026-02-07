---
layout: post
title: DiffusionPolicy
date: 2023-03-07
categories: [VLA]
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

[paper link](https://arxiv.org/abs/2303.04137) 

今天要深入探讨的是一篇在机器人学习（Robot Learning）领域具有里程碑意义的论文：**《Diffusion Policy: Visuomotor Policy Learning via Action Diffusion》**。这篇论文由哥伦比亚大学、MIT和丰田研究院（TRI）的研究人员共同完成。

在过去的一两年里，生成式模型（Generative Models）特别是扩散模型（Diffusion Models）彻底改变了图像生成领域（如DALL-E 2, Stable Diffusion）。而这篇论文，则是将这股浪潮成功引入机器人操作（Manipulation）领域的代表作。它不仅在15个基准任务上取得了平均46.9%的性能提升，更重要的是，它提出了一种全新的视角来看待“策略（Policy）”的表示问题。

我将分为四个核心模块：
1.  **背景与动机**：为什么传统的行为克隆（Behavior Cloning）方法不够好？
2.  **核心理论**：Diffusion Policy的数学形式化与推断机制。
3.  **关键技术实现**：从网络架构到预测视界（Receding Horizon）的设计。
4.  **实验分析与讨论**：从仿真到真机实验的洞察。

---

# 第一部分：背景与动机——多模态分布的挑战

## 1.1 什么是视觉运动策略（Visuomotor Policy）？
在机器人学习中，特别是模仿学习（Imitation Learning）的范畴下，我们的目标是学习一个函数（策略）$\pi$，它接收当前的视觉观测 $O_t$（Observations），并输出动作 $A_t$（Actions）。
$$ A_t = \pi(O_t) $$
这看起来像是一个标准的监督回归（Supervised Regression）问题。然而，机器人操作数据的独特性使得这不是一个简单的回归任务。

## 1.2 核心挑战：多模态分布（Multimodal Distributions）

想象一个机器人需要绕过桌子上的一个障碍物去抓取物体。它既可以从左边绕过去，也可以从右边绕过去。这两种轨迹在演示数据（Demonstrations）中都存在。
*   如果我们使用标准的均方误差（MSE）回归（即显式策略 Explicit Policy），网络倾向于输出这两种模式的**平均值**。
*   “左”和“右”的平均值是什么？是直接撞向中间的障碍物。

这就是**多模态分布**问题。

## 1.3 现有方法的局限性
为了解决这个问题，学术界之前尝试了多种方案：

1.  **混合高斯模型（GMMs, e.g., LSTM-GMM）**：
    *   *原理*：预测多个高斯分布的加权和。
    *   *局限*：训练不稳定，对参数敏感，且高斯核的数量限制了表达能力的上限。在许多高精度任务中，它难以拟合尖峰分布。

2.  **分类/离散化（Categorical）**：
    *   *原理*：将连续动作空间划分为网格（Bins）。
    *   *局限*：维数灾难。对于高维动作空间（如7自由度机械臂），网格数量呈指数级增长，导致计算不可行或精度极其低下。

3.  **隐式策略（Implicit Policies, e.g., IBC - Implicit Behavior Cloning）**：
    *   *原理*：学习一个能量函数 $E(a, o)$，通过优化 $\text{argmin}_a E(a, o)$ 来找动作。
    *   *局限*：训练极其困难。通常需要负采样（Negative Sampling）来估计配分函数（Partition Function），这导致了著名的训练不稳定性。论文中提到，IBC在许多复杂任务上的表现并不理想（如Lift, Can等任务）。

**Diffusion Policy 的出现，正是为了解决上述所有痛点：它既能完美表达多模态分布，又能保持训练的极度稳定性，同时支持高维动作空间。** <alphaxiv-paper-citation title="Introduction" page="1" first="This formulation allows" last="significantly improving performance." />

---

# 第二部分：Diffusion Policy 理论形式化

## 2.1 从DDPM到机器人策略
去噪扩散概率模型（DDPM）通常我们用它来生成图像，即从高斯噪声中恢复出图像。

在这篇论文中，作者做了一个巧妙的转换：**将“动作序列”视为一张“图像”来进行生成。**

Diffusion Policy 将策略建模为一个条件去噪过程。给定观测 $O_t$，我们通过 $K$ 次迭代去噪，生成动作 $A_t$。

### 2.2 数学表达
标准的DDPM逆向过程（生成过程）如下：
$$ x^{k-1} = \alpha (x^k - \gamma \epsilon_\theta(x^k, k) + \mathcal{N}(0, \sigma^2 I)) $$
其中 $x^k$ 是第 $k$ 步的带噪样本，$\epsilon_\theta$ 是噪声预测网络。

**对于 Diffusion Policy，我们需要做两点关键修改：**
1.  **输出对象**：$x$ 变成了机器人的动作序列 $A_t$。
2.  **条件生成**：去噪过程必须以当前的观测 $O_t$ 为条件。

修改后的公式为（论文 Eq 4）：
$$ A_t^{k-1} = \alpha (A_t^k - \gamma \epsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)) $$

这里非常关键的一点是：**$\epsilon_\theta(O_t, A_t^k, k)$ 实际上是在预测分数函数的梯度 $\nabla \log p(A_t|O_t)$。** (根据Score Matching的理论.)

## 2.3 为什么这比 EBM（能量模型） 更好？
隐式策略（Implicit Policy/EBM）试图直接学习能量函数 $E(x)$。这需要计算配分函数（积分），非常难。
Diffusion Policy 学习的是能量函数的**梯度** $\nabla E(x)$。
*   学习梯度不需要计算归一化常数（因为常数的导数为0）。
*   这就是为什么 Diffusion Policy 的训练比 IBC 稳定得多的根本数学原因。

---

# 第三部分：关键技术实现与工程设计

理论很美，但要让它在物理机器人上工作，需要一系列精妙的工程设计。这也是这篇论文不仅是“Idea paper”更是“System paper”的原因。

## 3.1 闭环动作序列预测（Closed-loop Action Sequences）
这是一个极其重要的设计细节。

传统的策略通常是 $O_t \to a_t$（单步预测）。但 Diffusion Policy 预测的是一个**动作序列** $A_t$。(Trunk)
*   **输入**：过去 $T_o$ 步的观测 $O_t$。
*   **输出**：未来 $T_p$ 步的动作序列。

**为什么要预测序列？**
1.  **时间一致性（Temporal Consistency）**：单步策略容易出现抖动，预测序列能保证动作的连贯和平滑。
2.  **避免短视（Avoiding Myopic Planning）**：模型被迫考虑未来的轨迹，这就隐式地包含了规划（Planning）的能力。

**但是，我们如何执行这个序列？**
这就引入了**后退视界控制（Receding Horizon Control, RHC）**的概念。
假设我们在时刻 $t$ 预测了未来16步的动作。我们**不会**把这16步全部执行完再重新预测。相反，我们只执行前 $T_a$ 步（比如前8步），然后立刻在时刻 $t+T_a$ 重新进行预测。
这种机制既保证了动作的长程平滑性，又赋予了机器人对环境干扰的快速响应能力。 <alphaxiv-paper-citation title="Action Sequences" page="2" first="This design allows" last="and responsiveness." />

## 3.2 视觉条件注入（Visual Conditioning）
网络 $\epsilon_\theta$ 需要同时处理高维的图像输入和低维的动作输入。如何融合？

论文提出了两种架构变体：

1.  **CNN-based (1D Temporal CNN)**:
    *   基于 Janner et al. 的架构。
    *   **融合方式**：FiLM (Feature-wise Linear Modulation)。简单来说，就是用图像特征去仿射变换（缩放和平移）动作处理网络的特征图。
    *   *特点*：擅长低频、平滑的控制任务。

2.  **Transformer-based (Time-series Diffusion Transformer)**:
    *   基于 MinGPT。
    *   **融合方式**：Cross-Attention。将动作嵌入作为 Query，图像嵌入作为 Key 和 Value。
    *   *特点*：对于动作变化剧烈、高频震荡的任务（如 Push-T 任务中的快速调整），Transformer 表现更好，因为它减少了 CNN 带来的过度平滑（Over-smoothing）效应。 <alphaxiv-paper-citation title="Architecture" page="2" first="We propose a" last="velocity control." />

---

# 第四部分：实验分析与讨论

## 4.1 仿真实验结果
论文在4个不同的基准测试（RoboMimic, Kitchen, etc.）共15个任务上进行了评估。
结果是压倒性的：**平均性能提升 46.9%**。

*   在复杂的 **Transport** 任务中，LSTM-GMM 的成功率是 62%，IBC 是 0%，而 Diffusion Policy 达到了 94% 以上。
*   注意 **IBC 的崩溃**。在很多任务中 IBC 成功率为 0。这验证了我们之前的理论分析：EBM 极难训练，经常陷入局部极小值或模式坍塌。

## 4.2 真机实验（Real-world Experiments）
这部分的实验设计非常精彩，展示了 Diffusion Policy 的鲁棒性。

1.  **Push-T 任务**：推一个T型的木块。这是一个典型的多模态任务（可以推T的左边、右边或顶端）。Diffusion Policy 展现了极其精确的接触控制。
2.  **Mug Flip（翻转马克杯）**：这是一个6自由度的高难度任务。机器人需要拿起任意摆放的杯子，把它翻转并挂在架子上。这涉及复杂的重抓取（Regrasp）策略。
    *   *观察*：LSTM-GMM 完全失败。Diffusion Policy 能够处理多种抓取姿势（正手、反手）。 <alphaxiv-paper-citation title="Mug Flip" page="10" first="Although never demonstrated" last="when necessary." />
3.  **Sauce Pouring（倒酱汁）**：处理流体和非刚性物体。
4.  **双臂协调（Bimanual Tasks）**：如叠衣服（Shirt Folding）。

## 4.3 关键消融实验（Ablation Studies）
我们需要关注几个工程参数的影响：

*   **预测视界 ($T_p$)**：预测太短，动作不平滑；预测太长，计算量大且容易累积误差。
*   **执行视界 ($T_a$)**：这是 Receding Horizon 的核心。$T_a < T_p$ 至关重要。如果 $T_a = T_p$（开环执行），由于误差累积，成功率会大幅下降。这就证明了**闭环控制**的必要性。

---

# 第五部分：总结与思考

为什么 **Diffusion Policy** 会成为当前的主流范式。

1.  **数学上的优雅与稳定**：通过学习 Score Function 的梯度，避开了 EBM 训练中的配分函数难题，实现了稳定的训练。
2.  **对多模态的天然适应**：扩散过程本质上是从分布中采样，这使得它能够自然地处理多解问题，而不需要像 GMM 那样预设模式数量，也不像回归那样取平均。
3.  **序列预测与时空一致性**：将动作视为序列（图片），利用了扩散模型在图像生成中被验证过的强大的结构化生成能力。

**遗留问题与未来方向**：
*   **推断速度**：扩散模型需要多次迭代（如 100 步或 16 步），这导致推理速度较慢（尽管论文中优化到了 10-20Hz）。如何进一步加速（如 Consistency Models, Distillation）是当前的研究热点。
*   **数据依赖**：虽然比 IBC 好，但本质上还是 BC，对数据质量有要求。如何结合强化学习（RL）进行微调？

**思考**：
如果我们将 Diffusion Policy 用于导航任务（Navigation），观测空间和动作空间会有什么变化？Receding Horizon 的策略是否需要调整？

# 后面与RL的结合


在 Diffusion Policy 提出（2023年）之后，学术界立刻意识到了它的一个核心局限：**它本质上还是行为克隆（BC）**。也就是说，它只能“模仿”示教者。如果示教者做得不够好，或者环境发生了未见过的变化，它无法像强化学习（RL）那样去“探索”出更优的解。

因此，**Diffusion + RL** 成为了过去两年（2023-2025）最火热的研究方向之一。主要结合方式可以归纳为以下三类流派：

### 1. 离线强化学习：Diffusion 作为“演员”（Policy Head）
这是最早期的结合方式。传统的 RL（如 SAC, PPO）通常假设策略是高斯分布（单模态）。但正如我们课上讲的，机器人动作往往是多模态的。

*   **核心思想**：我们用 Diffusion Model 来替代传统 RL 中的高斯网络（Gaussian Policy）作为 Actor，用来拟合复杂的动作分布。
*   **代表论文**：
    *   **IDQL (Implicit Diffusion Q-Learning)** [arXiv:2304.10573](https://arxiv.org/abs/2304.10573)：这篇论文非常经典。它结合了 IQL（Implicit Q-Learning）和 Diffusion。Diffusion Model 负责从离线数据中生成“候选动作”（拟合数据分布），然后训练一个 Q-function（Critic）来评估这些动作的好坏，最后在推理时通过拒绝采样（Rejection Sampling）或者梯度引导选出 Q 值最高的动作。
    *   **优势**：既能处理多模态数据，又能通过 Q 值找到比示教数据更好的动作。

### 2. 在线微调：先模仿，后强化（RL Fine-tuning）
这是目前最直接的思路。先用 Diffusion Policy 进行模仿学习（预训练），得到一个还不错的策略，然后把它放到环境里，用 RL 接着训练，让它根据奖励（Reward）自我进化。

*   **核心挑战**：Diffusion 的生成过程是一个多步的去噪链（比如100步）。要计算 RL 的梯度（Policy Gradient）并反向传播穿过这一百步，显存消耗巨大且梯度极不稳定。
*   **代表论文**：
    *   **DPPO (Diffusion Policy Policy Optimization)** [arXiv:2409.00588](https://arxiv.org/abs/2409.00588)：这是2024年的一篇重磅工作，专门解决上述问题。它提出了一套完整的框架，使得我们可以用类似 PPO 的算法来微调 Diffusion Policy。实验表明，经过微调后，机器人不仅动作更流畅，而且完成任务的成功率远超原始的示教者。
    *   **方法**：它不需要反向传播穿过整个去噪链，而是巧妙地利用了 Score Function 的性质来估计梯度，使得微调变得可行且高效。

### 3. 奖励引导生成（Reward-Guided Generation）
这种方法不改变 Diffusion Policy 的权重，而是在推理（Inference）阶段“外挂”一个导航员。

*   **核心思想**：训练一个独立的价值函数 $V(s, a)$ 或分类器。在 Diffusion 去噪的每一步，我们计算 $\nabla_a V(s, a)$，用这个梯度去“推”生成的动作，让它向高价值区域偏移。
*   **类比**：就像你在画画（Diffusion 生成），旁边站着一个老师（RL Critic）。你每画一笔，老师就提醒你“往左一点更好”，最后画出来的结果就会既像原来的风格，又符合老师的要求。
*   **相关工作**：这种思想最早见于 **Decision Diffuser**，虽然它更多用于规划，但其核心逻辑被广泛用于机器人控制中。

### 总结
如果用一句话概括 Diffusion Policy 和 RL 的关系：
*   **Diffusion Policy** 提供了 **“像人”**的先验（平滑、拟人、多模态）。
*   **RL** 提供了 **“成功”**的导向（最大化奖励、适应新环境）。

未来的趋势一定是二者的深度融合：**用 Diffusion 保证动作不发生灾难性的变形，用 RL 提升任务的极限性能。**
