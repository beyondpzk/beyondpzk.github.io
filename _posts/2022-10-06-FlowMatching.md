---
layout: post
title: FlowMatching
date: 2022-10-06
categories: [AIGC]
tags: [AIGC]
---

[TOC]

# FlowMatching

- [paper地址](https://arxiv.org/abs/2210.02747)


# Flow Matching for Generative Modeling

**论文**：*Flow Matching for Generative Modeling* (ICLR 2023)

---

## 第一部分：生成模型的演进与 CNF 的困境 (Introduction & Context)

### 1.1 全景图
这篇论文之前，我们需要先理清两条并行的技术路线：

1.  **Diffusion Models (扩散模型)**：大家都很熟悉，Stable Diffusion, DALL-E 2 背后的技术。它们基于随机微分方程 (SDE)，通过去噪来生成数据。优点是训练极其稳定（这就是为什么它们火了），但缺点是采样极其低效，路径弯曲复杂。
2.  **Continuous Normalizing Flows (CNFs, 连续归一化流)**：这是我们今天的重点。CNF 基于常微分方程 (ODE)。

**核心问题**：在 Flow Matching 出现之前，CNF 虽然数学形式优美（确定性、可逆），但几乎无法在大规模数据上训练。为什么？

### 1.2 CNF 的数学定义回顾
让我们回顾一下 CNF。CNF 也是将简单的噪声分布 $p_0$ (如高斯) 变换为复杂的数据分布 $p_1$ (如 ImageNet)。
这个变换是通过一个随时间变化的向量场 $v_t(x)$ 定义的 ODE 来实现的：
$$ \frac{d\phi_t(x)}{dt} = v_t(\phi_t(x)) $$
其中 $\phi_t(x)$ 是流映射 (Flow map)。

**难点在于似然计算**。根据 *Instantaneous Change of Variables Theorem*，对数密度的变化率是：
$$ \frac{\partial \log p_t(x)}{\partial t} = -\text{Tr}\left( \frac{\partial v_t}{\partial x} \right) = -\text{div}(v_t(x)) $$
要训练这个模型，最大化似然需要计算雅可比矩阵的迹（Trace of Jacobian）。对于高维图像，这个计算复杂度是 $O(d^2)$ 甚至 $O(d^3)$。虽然有 Hutchinson Trace Estimator 可以近似，但在训练中需要求解整个 ODE 积分，这不仅慢，而且数值不稳定。

### 1.3 本文的突破口
这篇文章的作者 Lipman 等人提出：**我们要放弃极大似然训练 (Maximum Likelihood Training)**。
不要去解那个昂贵的 ODE 积分来计算似然。相反，我们要用一种 **“无需模拟 (Simulation-Free)”** 的回归方法。我们直接告诉模型：“在这个时刻 $t$，你应该往哪个方向流”，这就是 **Flow Matching**。

---

## 第二部分：Flow Matching 核心理论 (The Theory of Flow Matching)

### 2.1 目标：回归向量场
假设存在一个理想的概率路径 $p_t(x)$，它从噪声 $p_0$ 平滑过渡到数据 $p_1$。既然有概率路径，这就意味着必然存在一个**生成这一路径的向量场** $u_t(x)$。
这由连续性方程 (Continuity Equation) 保证。 (这里需要补充一些东西.)

我们要训练神经网络 $v_t(x; \theta)$ 去逼近这个理想向量场 $u_t(x)$。
目标函数非常直观：
$$ \mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim U[0,1], x \sim p_t(x)} \| v_t(x) - u_t(x) \|^2 $$
<alphaxiv-paper-citation title="FM Objective" page="3" first="The Flow Matching" last="u_t(x)‖^2" />

这看起来很简单，但有一个巨大的陷阱：**我们根本不知道 $u_t(x)$ 是什么！**
我们只有数据 $x_1$ (来自目标数据分布 $q(x)$) 和噪声 $x_0$ (来自高斯 $p(x)$)。我们不知道中间的 $p_t$ 长什么样，更不知道生成它的宏观向量场 $u_t$ 是什么。对于复杂的数据集（如 ImageNet），这个 $u_t$ 是 intractable（不可计算）的。

### 2.2 破局：Conditional Flow Matching (CFM)
这是整篇论文最天才的一步。
既然宏观的 $p_t$ 搞不定，我们把问题分解到**微观**层面。

我们定义**条件概率路径** $p_t(x|x_1)$。这表示：给定**某一张**特定的目标图片 $x_1$，噪声 $x_0$ 是如何变成这张 $x_1$ 的？这个微观路径可以设计得非常简单（比如高斯分布）。
如果路径简单，那么生成这个微观路径的**条件向量场** $u_t(x|x_1)$ 也就很简单，是可以直接写出公式的。

**关键定理 (Theorem 2)**：
我们不需要回归宏观向量场 $u_t(x)$，我们只需要回归微观向量场 $u_t(x|x_1)$。
作者证明了：
$$ \nabla_\theta \mathcal{L}_{FM}(\theta) = \nabla_\theta \mathcal{L}_{CFM}(\theta) $$
其中 CFM 的损失函数为：
$$ \mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)} \| v_t(x) - u_t(x|x_1) \|^2 $$
<alphaxiv-paper-citation title="CFM Theorem" page="4" first="has the same" last="L_CFM(θ)." />

为什么这两个不一样的 Loss 会有相同的梯度？既然梯度一样,那优化这两个就是等价的.
展开平方项：
$$ \| v - u \|^2 = \|v\|^2 - 2\langle v, u \rangle + \|u\|^2 $$
$$ \| v - u(\cdot|x_1) \|^2 = \|v\|^2 - 2\langle v, u(\cdot|x_1) \rangle + \|u(\cdot|x_1)\|^2 $$
这里 $\|v\|^2$ 项是一样的。常数项对梯度无影响。关键在于交叉项 $\langle v, u \rangle$。
数学上可以证明，宏观向量场 $u_t(x)$ 其实就是微观向量场 $u_t(x|x_1)$ 的加权平均（边缘化）：
$$ u_t(x) = \mathbb{E}_{x_1 \sim p(x_1|x)} [u_t(x|x_1)] $$
(我觉得上面这个公式是MeanFlows的出发点.)
所以，只要神经网络 $v_t$ 学会了拟合每一个样本的微观方向，在统计期望上，它自然就学会了正确的宏观方向。(我的理解: 训练时虽然有多种多样的路径对, 但是训完模型后,模型就知道哪些路是最容易的,这也对应Rectified flow中的那个图,为什么训完后会调头,不会交叉;)

> **比喻**： (我觉得这个比喻不好)
> 想象你要预测下班高峰期的人流（宏观向量场）。
> 方法一（FM）：你试图直接测量整个城市每个路口的人流速度矢量。这太难了。
> 方法二（CFM）：你随机抽取一百万个人，问每个人：“你家在哪？公司在哪？你打算怎么走？”如果你能预测每一个个体的简单直线运动，把这些预测聚合起来，你其实就完美预测了整个城市的宏观人流。

---

## 第三部分：如何设计路径？ (Instantiations of Probability Paths)

现在我们有了 CFM 框架，剩下的问题就是：我们选什么样的微观路径 $p_t(x|x_1)$？

### 3.1 扩散路径 (Diffusion Paths) —— 连接过去
论文首先展示了 Flow Matching 可以完全兼容并包含扩散模型。
对于扩散模型，条件分布通常是：
$$ p_t(x|x_1) = \mathcal{N}(x | \alpha_t x_1, \beta_t^2 I) $$
这是一个随时间收缩方差的高斯分布。
它对应的条件向量场是： (Why?)
$$ u_t(x|x_1) = \frac{\sigma'_t(x_1) (x - \mu_t(x_1))}{\sigma_t(x_1)} + \mu'_t(x_1) $$
结论是：如果我们用 FM 框架去训练扩散路径，得到的模型效果比传统的 Score Matching 还要好，训练更稳定。但这并没有解决根本问题——**路径依然是弯曲的**。

### 3.2 最优传输路径 (Optimal Transport Paths) —— 通向未来
这是本文最想推销的方案。既然我们可以自定义路径，为什么不定义一个最简单的？
两点之间，直线最短。

**OT 条件路径定义**：
我们定义均值 $\mu_t$ 随时间线性变化，从 $0$ 变到 $x_1$。
$$ p_t(x|x_1) = \mathcal{N}(x | t x_1, (1 - (1 - \sigma_{min})t)^2 I) $$
这里 $\sigma_{min}$ 是一个小量，防止 $t=1$ 时方差坍缩为 0 导致数值问题。
<alphaxiv-paper-citation title="OT Path" page="5" first="Example II: Optimal" last="σ_min)t." />

**OT 条件向量场推导 (Detailed derivation)**：
对于这种线性插值路径 $\psi_t(x_0) = (1-t)x_0 + t x_1$（假设 $\sigma_{min} \approx 0$ 简化理解）。
这个粒子的速度是什么？对时间求导：
$$ \frac{d}{dt}\psi_t(x_0) = x_1 - x_0 $$
速度是恒定的！
但是向量场 $u_t(x)$ 需要是位置 $x$ 的函数，而不是初始点 $x_0$ 的函数。
我们要把 $x_0$ 换掉。因为 $x = (1-t)x_0 + t x_1$，所以 $x_0 = \frac{x - t x_1}{1-t}$。
代入速度公式，经过整理得到论文中的公式 21：
$$ u_t(x|x_1) = \frac{x_1 - (1-\sigma_{min})x}{1 - (1-\sigma_{min})t} $$
<alphaxiv-paper-citation title="OT Vector Field" page="5" first="u_t(x|x_1) =" last="σ_min)t" />

**这一步的物理意义**：
这个向量场引导样本**走直线**。这意味着：
1.  **Transport Cost 最小**：这是 Optimal Transport 的本质。
2.  **ODE Solver 极度友好**：如果你在解一个 $\frac{dx}{dt} = \text{常数}$ 的方程，哪怕用最简单的 Euler 方法，一步就能算准。虽然后期变成了复杂的神经网络，但这种“由于设计而产生的直线倾向”被保留了下来。

---

## 第四部分：实验与结果分析 (Experiments & Analysis)

### 4.1 训练效率
看 Figure 2 和 Figure 3。
*   **Diffusion Vector Field**：你看那个场，是随时间剧烈变化的，中间还要绕弯。
*   **OT Vector Field**：方向几乎不变，只是模长在变。
这导致神经网络更容易学习 OT 场。论文提到 FM 的收敛速度显著快于 Diffusion。 <alphaxiv-paper-citation title="Consistent Direction" page="6" first="OT VF has" last="constant direction" />

### 4.2 ImageNet 上的 SOTA
在 ImageNet 64x64 和 128x128 上，Flow Matching (FM-OT) 在 NLL (Negative Log Likelihood) 和 FID (Fréchet Inception Distance) 上都击败了当时的顶尖扩散模型（如 ADM）。
这一点很重要，因为这是第一次证明 CNF 这种基于 ODE 的方法，在生成质量上可以和基于 SDE 的扩散模型硬碰硬。 <alphaxiv-paper-citation title="Performance" page="1" first="consistently better performance" last="diffusion-based methods" />

### 4.3 采样速度 (NFE Analysis)
这是工业界最看重的。看 **Figure 7**。
*   横轴是 NFE (Number of Function Evaluations)，即调用神经网络的次数。
*   纵轴是 FID (越低越好)。
*   看那条红线 (FM-OT)：在 NFE=10 到 20 的时候，FID 已经非常低了。
*   对比蓝线 (Diffusion)：在低 NFE 时效果很差，必须要在 NFE > 100 时才能追上 FM。
这就是**直线路径**带来的巨大红利。我们可以用 `dopri5` 这种自适应步长求解器，它会自动发现路径很直，然后迈大步子，极大地节省计算量。 <alphaxiv-paper-citation title="Sampling Speed" page="9" first="FM with OT" last="diffusion baselines." />

---

## 第五部分：总结与展望 (Conclusion)

### 5.1 Takeaways
1.  **Simulation-Free Training**：FM 让我们不再需要解 ODE 就能训练 ODE 模型。
2.  **Unified View**：FM 把 Diffusion 降级为一种特殊的、非最优的路径选择。
3.  **Optimal Transport**：直线路径是生成模型的未来（这一点已经被后续的 Stable Diffusion 3, Flux 等模型证实，它们都转向了 Rectified Flow / Flow Matching 架构）。

### 5.2 潜在的局限性
*   **训练时的随机性**：虽然路径是直的，但我们需要对 $t$ 进行采样。如果 $t$ 接近 1，且 $\sigma_{min}$ 很小，向量场的数值可能会很大（除以接近0的数），这在工程实现上需要注意（通常设置 $\sigma_{min}=1e-5$）。
*   **模型容量**：虽然任务简单了，但要在大分辨率上拟合高频细节，依然需要巨大的神经网络参数量。


# 为什么公式(21)意味着让粒子走直线


理解公式 (21) 为什么代表直线运动，我们可以从**物理直观**和**数学推导**两个层面来理解。

为了方便，令 $\sigma_{min} = 0$（即假设噪声完全消除），这样公式会变得非常干净，物理意义一目了然。

---

### 1. 物理直观：剩余距离 / 剩余时间

当 $\sigma_{min} = 0$ 时，公式 (21) 简化为：
$$ u_t(x|x_1) = \frac{x_1 - x}{1 - t} $$

这时候，我们可以这样解读公式中的每一项：

*   **分子 $(x_1 - x)$**：这是一个向量，指向**目标** $x_1$ 减去 **当前位置** $x$。
    *   这决定了**方向**：永远指向终点 $x_1$。既然方向永远指向终点，且终点固定，那么轨迹必然是一条直线。
*   **分母 $(1 - t)$**：这是**剩余时间**（总时间是1，当前时间是 $t$）。
*   **整体含义**：
    $$ \text{速度} = \frac{\text{剩余距离}}{\text{剩余时间}} $$

**想象**：
> “想象你要在 1 小时内从学校走到市中心（直线距离）。
> 过了 0.5 小时，你还剩一半路程。你的速度应该是多少？
> 速度 = 剩余路程 / 剩余时间。
> 只要你时刻保持这个速度公式，你就是在做**匀速直线运动**。”

这就是公式 (21) 最本质的物理含义：它描述了一个**匀速直线**奔向目标的运动过程。

---

### 2. 数学推导：从结果反推原因

在论文中，作者的逻辑是反过来的：**不是因为有了这个向量场才走出了直线，而是因为我们定义了直线路径，才推导出了这个向量场。**


**Step 1: 定义直线路径 (Flow Map)**
我们强制规定粒子 $x$ 从 $x_0$（噪声）到 $x_1$（数据）走的是线性插值（Linear Interpolation）：
$$ \psi_t(x_0) = (1 - t)x_0 + t x_1 $$
*(注：这里为了简化展示，暂时忽略 $\sigma_{min}$，其逻辑完全一致)*

**Step 2: 计算速度 (Velocity)**
对时间 $t$ 求导，得到速度：
$$ v_t = \frac{d}{dt}\psi_t(x_0) = x_1 - x_0 $$
注意，这里的速度 $x_1 - x_0$ 是一个常数向量！这意味着粒子在做**匀速**运动。

**Step 3: 坐标变换 (Coordinate Change)**
这步最关键。Vector Field $u_t(x)$ 必须是**当前位置** $x$ 的函数，不能包含初始位置 $x_0$（因为在推理阶段我们不知道 $x_0$）。我们需要把 $x_0$ 消掉。

由 Step 1 的公式：
$$ x = (1 - t)x_0 + t x_1 $$
反解出 $x_0$：
$$ (1-t)x_0 = x - t x_1 \implies x_0 = \frac{x - t x_1}{1 - t} $$

**Step 4: 代入速度公式**
将 $x_0$ 代回 Step 2 的速度公式：
$$ v_t = x_1 - \left( \frac{x - t x_1}{1 - t} \right) $$
通分整理：
$$ v_t = \frac{x_1(1-t) - (x - t x_1)}{1 - t} $$
$$ v_t = \frac{x_1 - t x_1 - x + t x_1}{1 - t} $$
$$ v_t = \frac{x_1 - x}{1 - t} $$

**结论**：
这就是公式 (21) 的由来。
所以，公式 (21) 仅仅是“匀速直线运动”这个物理过程在欧拉视角（Eulerian viewpoint，即向量场视角）下的数学表达。

<alphaxiv-paper-citation title="OT Vector Field" page="5" first="u_t(x|x_1) =" last="σ_min)t" />

### 3. 稍微复杂一点的情况 ($\sigma_{min} > 0$)

如果加上 $\sigma_{min}$（论文原版公式），逻辑是一样的，只是“直线”没变，“匀速”变成了“变速直线”。

原公式：
$$ x_t = (1 - (1 - \sigma_{min})t)x_0 + t x_1 $$
这里 $x_0$ 的系数不再是 $(1-t)$，但这依然是一个关于 $t$ 的**一次函数**。
$$ x_t = A \cdot t + B $$
只要位置 $x$ 是时间 $t$ 的一次函数，轨迹在空间中就一定是一条直线。

### 总结

1.  **几何上**：分子 $x_1 - (1-\sigma_{min})x$ 主要是 $x_1 - x$ 的变体，保证了向量场方向始终指向目标与当前位置的连线方向。
2.  **动力学上**：这个公式就是通过强行定义 $x(t)$ 为线性函数，$u_t = \dot{x}(t)$ 反推出来的结果。它保证了 Optimal Transport 的核心属性——路径最短（直线）且运输成本最低。

