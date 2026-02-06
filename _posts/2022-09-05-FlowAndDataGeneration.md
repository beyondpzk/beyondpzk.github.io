---
layout: post
title: FlowAndDataGeneration
date: 2022-09-05
categories: [AIGC]
tags: [AIGC]
---

[TOC]

# FlowAndDataGeneration


这次的主体是：**“如何对抗雅可比行列式 (Jacobian Determinant)”**。
流模型的发展史，就是一部与计算复杂度 $O(d^3)$ 的行列式进行斗争的历史。从 NICE 的“完全避开”，到 RealNVP 的“对角化”，再到 CNF 的“转化为迹(Trace)”，最后到 Flow Matching 的“完全抛弃似然计算”。


---

# 从 Normalizing Flows 到 Flow Matching

**目标**：理清流模型 (Flow-based Models) 的发展脉络，理解离散流与连续流的区别，最终引出 Flow Matching 解决的历史痛点。

---

## 第一部分：基础与离散流模型 (The Era of Discrete Flows)

### 1.1 引入：生成模型的三大流派 (10 分钟)
*   **GAN (对抗生成网络)**：
    *   优点：生成质量高，速度快。
    *   缺点：训练不稳定 (Min-Max game)，没有显式的概率密度 $p(x)$。
*   **VAE (变分自编码器)**：
    *   优点：理论基础好，训练较稳。
    *   缺点：优化的是下界 (ELBO)，生成的图像通常模糊。
*   **Flow-based Models (流模型)**：
    *   **核心卖点**：精确的对数似然估计 (Exact Log-likelihood)，可逆 (Invertible)，潜在变量与数据一一对应。

### 1.2 数学基础：变量变换定理 (Change of Variables)
这是所有流模型的基石。

设 $z \sim p_z(z)$ 是简单分布（如标准高斯），$x = f(z)$ 是生成的数据。如果 $f$ 是可逆函数，那么：
$$ p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right| $$
或者写成对数形式（更常用）：
$$ \log p_x(x) = \log p_z(z) - \log \left| \det \frac{\partial f(z)}{\partial z} \right| $$

> **📝 重点**：
> 画一个“挤压气球”的图示。
> *   $z$ 空间是松散的气体。
> *   $f$ 变换把气体挤压到 $x$ 空间的流形上。
> *   Jacobian Determinant 就是**体积变化的倍率**。为了保证概率总和为1，体积变小了，密度就要变大。

*   **核心痛点**：计算 $N \times N$ 矩阵的行列式复杂度是 $O(N^3)$。对于 $64 \times 64$ 的图像，$N=12288$，直接算是完全不可能的。
*   **解决思路**：设计特殊的神经网络结构，使得 Jacobian 矩阵是**三角阵**或**对角阵**，这样行列式就等于对角线元素之积（复杂度 $O(N)$）。

### 1.3 NICE: Non-linear Independent Components Estimation
*(Dinh et al., 2014)* —— **流模型的鼻祖**

通过仔细的设计神经网络，让Jacobian好计算. 

*   **核心设计：加性耦合层 (Additive Coupling Layer)**
    把输入 $x$ 拆分成两半 $(x_{1:d}, x_{d+1:D})$。
    $$
    \begin{cases}
    y_{1:d} = x_{1:d} \\
    y_{d+1:D} = x_{d+1:D} + m(x_{1:d})
    \end{cases}
    $$
    其中 $m$ 可以是任意复杂的神经网络（不需要可逆）。

*   **Jacobian 矩阵长什么样？**
    $$
    J = \begin{pmatrix} I & 0 \\ \frac{\partial m}{\partial x_1} & I \end{pmatrix}
    $$
    这是一个下三角矩阵，对角线全是 1。
*   **结论**：$\det(J) = 1$。它是**保体积 (Volume Preserving)** 的变换。
*   **缺点**：因为保体积，它很难极度压缩或扩展空间密度，表达能力受限。

### 1.4 RealNVP: Real Non-Volume Preserving
*(Dinh et al., 2016)* —— **真正的工业级基石**

*   **核心改进：仿射耦合层 (Affine Coupling Layer)**
    NICE 只有加法，RealNVP 引入了乘法（缩放）。
    $$
    \begin{cases}
    y_{1:d} = x_{1:d} \\
    y_{d+1:D} = x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})
    \end{cases}
    $$
    这里 $s$ (scale) 和 $t$ (translation) 是神经网络。

*   **Jacobian**：
    $$
    J = \begin{pmatrix} I & 0 \\ \frac{\partial \dots}{\partial x_1} & \text{diag}(\exp(s)) \end{pmatrix}
    $$
*   **行列式**：$\det(J) = \prod \exp(s_i) = \exp(\sum s_i)$。
    计算极其简单，且不再强制保体积。
*   **工程细节**：介绍了 Checkerboard Masking（棋盘格掩码）和 Channel Squeezing（通道压缩），这对处理图像至关重要。

### 1.5 GLOW: Generative Flow with 1x1 Convolutions 
*(Kingma & Dhariwal, 2018)* —— **流模型的高光时刻**

*   **背景**：RealNVP 固定了通道的切分方式，导致通道间信息交流不够充分。
*   **创新点：可逆 1x1 卷积**。
    在耦合层之前，先乘一个 $1 \times 1$ 的可学习矩阵 $W$（相当于对通道进行全排列融合）。
    $$ y = W x $$
    $W$ 的行列式计算复杂度是 $O(c^3)$，其中 $c$ 是通道数（通常很小，如 3, 64, 128），所以可以接受。
*   **地位**：Glow 是第一个生成出高质量人脸（如 CelebA-HQ）的流模型，证明了流模型的潜力。

---

## 第二部分：从离散走向连续 (The Era of Continuous Flows)

### 2.1 思考：层数的极限
*   **ResNet 的视角**：
    残差连接 $x_{t+1} = x_t + f(x_t)$ 可以看作是欧拉离散化 (Euler discretization)。
    $$ x_{t+1} = x_t + \Delta t \cdot f(x_t) $$
    当 $\Delta t \to 0$，这就变成了常微分方程 (ODE)。
    $$ \frac{dx(t)}{dt} = f(x(t), t) $$

上面的 $f$ 可以看成是神经网络的每一层。

### 2.2 Neural ODE & FFJORD 
*(Chen et al., NeurIPS 2018 - Best Paper)*

*   **Continuous Normalizing Flows (CNFs)**：
    不再是一层一层的离散变换，而是定义一个随时间变化的向量场 $v_t(x)$。
    $$ x(t_1) = x(t_0) + \int_{t_0}^{t_1} v_t(x(t)) dt $$
    我们可以用任何 ODE Solver (如 Runge-Kutta) 来求解。

*   **FFJORD 的创新**：
    **瞬时变量变换公式 (Instantaneous Change of Variables)**。
    我们不再计算 Jacobian determinant，而是计算 Jacobian 的**迹 (Trace)**。
    $$ \frac{\partial \log p_t(x(t))}{\partial t} = -\text{Tr}\left( \frac{\partial v_t}{\partial x} \right) $$
*   **Hutchinson's Trace Estimator**：
    计算 Trace 不需要算出整个 Jacobian 矩阵，可以用随机向量投影来估算，把复杂度从 $O(d^3)$ 降到 $O(d^2)$ 甚至更低。

### 2.3 CNF 的优缺点
*   **优点 (Pros)**：
    1.  **自由形式 (Free-form Jacobian)**：因为我们要的是 Trace 而不是 Determinant，所以向量场 $v_t$ 不需要像 RealNVP 那样搞特殊的三角结构。可以是任意神经网络！这大大释放了模型能力。
    2.  **参数效率**：参数共享（同一个网络用于所有时间步），模型更小。
*   **缺点 (Cons) —— *这是引出 Flow Matching 的关键***：
    1.  **训练极慢**：为了计算梯度，需要解 ODE 积分。前向传播解一次，反向传播（Adjoint Method）还要解一次。
    2.  **数值不稳**：训练过程中，向量场可能变得很震荡 (stiff)，导致 ODE Solver 需要走几千步才能求出结果，训练时间爆炸。

> **💡 教学互动**：
> 问学生：“如果我们能在不解 ODE 的情况下训练 CNF，那会怎样？”
> 答：“那我们就拥有了 CNF 的强大表达能力，同时拥有普通神经网络的训练速度。”
> -> **引出 Flow Matching。**

---

## 第三部分：Flow Matching 的救赎 (The Flow Matching Solution)

### 3.1 视角的转变：从 MLE 到 Regression (15 分钟)
*   **旧范式 (CNF/FFJORD)**：
    *   方法：最大似然估计 (MLE)。
    *   过程：随机猜一个向量场 $\to$ 拼命积分算 $p(x)$ $\to$ 发现概率不对 $\to$ 根据梯度调整向量场。
    *   比喻：这就好比你要造一个滑梯。你造了一段，扔个球下去，看它滚到哪，如果不准，就微调滑梯，再扔球。**非常低效，因为“扔球（模拟）”很贵。**

*   **新范式 (Flow Matching)**：
    *   方法：直接回归 (Direct Regression)。
    *   过程：我直接在空间中画出我想要的路径（比如直线）。我强制向量场在任何位置都指向这条路径的切线方向。
    *   比喻：你在滑梯上画好了一条线，直接命令每一段滑梯：“你就照着这个斜率造”。**不需要扔球模拟。**

### 3.2 串联历史

1.  **NICE/RealNVP/Glow**：为了能训练，不得不阉割模型结构（必须用 Coupling Layer），导致表达能力受限或需要极深的网络。
2.  **Neural ODE/FFJORD**：解放了模型结构（可以用任意网络），但掉进了计算陷阱（ODE 积分太慢）。
3.  **Flow Matching**：
    *   保留了 CNF 的**任意模型结构**（Free-form Jacobian）。
    *   去掉了 CNF 的**积分训练过程**（Simulation-free）。
    *   引入了 **Optimal Transport**（比 Diffusion 更好的路径）。

### 3.3 深入理解 Conditional Flow Matching

*   **Why it works?**
    以前我们不敢做 Regression，因为不知道目标向量场 $u_t$ 是什么。
    Flow Matching 的天才之处在于发现：**拟合个体的微观向量场，等价于拟合群体的宏观向量场。**
    这使得我们可以绕过 intractable 的边缘分布，直接利用条件分布（高斯）进行监督训练。

---

### 建议阅读材料 (Reading List for Students)
1.  *NICE: Non-linear Independent Components Estimation* (2014)
2.  *Density estimation using Real NVP* (2016)
3.  *Glow: Generative Flow with Invertible 1x1 Convolutions* (2018)
4.  *Neural Ordinary Differential Equations* (2018) - **必读经典**
5.  *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models* (2018)
6.  *Flow Matching for Generative Modeling* (2023) 
