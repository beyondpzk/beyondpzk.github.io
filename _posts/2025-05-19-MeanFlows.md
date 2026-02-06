---
layout: post
title: MeanFlows 
date: 2025-05-19 19:52
tags: []
---

[TOC]

# meanflows

- [paper地址](https://arxiv.org/abs/2505.13447)

我的理解: 我觉得meansflows是针对 rectified flow的弱点来的,尤其是训练时的弱点.

基于论文内容，MeanFlow 的训练目标、Loss 函数以及 Ground Truth (GT) 的构建方式非常独特。它不像传统的监督学习那样直接有一个固定的“标签”，而是通过一个**微分恒等式**构造了一个**自洽（Self-consistent）的回归目标**。

以下是详细的数学原理和步骤解析：

### 1. 核心理论基础：MeanFlow Identity

理解 Loss 的前提是理解论文推导出的核心公式——**MeanFlow Identity（平均流恒等式）**。

*   **定义：**
    *   $v(z_t, t)$：瞬时速度（Instantaneous Velocity），即 Flow Matching 中的速度场。
    *   $u(z_t, r, t)$：平均速度（Average Velocity），即从时间 $r$ 到 $t$ 的位移除以时间间隔。
*   **恒等式：**
    论文推导出 $u$ 和 $v$ 必须满足以下微分关系（论文公式 6）：
    $$ \frac{d}{dt} u(z_t, r, t) = \frac{v(z_t, t) - u(z_t, r, t)}{t - r} $$
    
    **移项后，我们可以得到 $u$ 的表达式：**
    $$ u(z_t, r, t) = v(z_t, t) - (t - r) \underbrace{\frac{d}{dt} u(z_t, r, t)}_{\text{全导数}} $$

    **这个移项后的公式，就是训练目标的来源。**

---

### 2. 训练目标 (Regression Target) 与 GT 构建

MeanFlow 的训练本质上是训练神经网络 $u_\theta$ 去拟合上述恒等式的右边。

#### Ground Truth (GT) 的构成：$u_{\text{tgt}}$
训练时的“目标值” $u_{\text{tgt}}$ 并不是预先计算好的固定值，而是由**已知物理量**和**网络当前的导数预测**动态组合而成的。

根据论文公式 (10) 和 (11)，目标值 $u_{\text{tgt}}$ 定义为：

$$ u_{\text{tgt}} = \underbrace{v_t}_{\text{数据决定的瞬时速度}} - (t - r) \times \underbrace{\left( v_t \cdot \nabla_z u_\theta + \partial_t u_\theta \right)}_{\text{网络预测的全导数 (JVP)}} $$

这里包含两个关键部分：

1.  **$v_t$ (瞬时速度，真正的 Ground Truth 来源)：**
    *   这是唯一来自数据的外部信号。
    *   在 Flow Matching 框架下，对于一条直线路径（Straight Path），给定数据点 $x$（图片）和噪声 $\epsilon$，在时间 $t$ 的位置是 $z_t = (1-t)x + t\epsilon$。
    *   此时的**条件瞬时速度**是已知的：**$v_t = \epsilon - x$** (或者 $x - \epsilon$，取决于具体定义，论文中 $v_t = \epsilon - x$ 对应 $z_1=\epsilon, z_0=x$)。
    *   **注意：** 这个 $v_t$ 不需要模型预测，是直接算出来的。

2.  **全导数项 (网络自身的性质)：**
    *   $\frac{d}{dt} u$ 被展开为 $\frac{\partial u}{\partial z} \frac{dz}{dt} + \frac{\partial u}{\partial t}$。
    *   其中 $\frac{dz}{dt}$ 就是 $v_t$。
    *   这一项通过 **Jacobian-Vector Product (JVP)** 计算。即计算网络输出 $u_\theta$ 对输入 $(z, r, t)$ 的导数，并沿着向量 $(v_t, 0, 1)$ 方向投影。

#### Stop-Gradient (停止梯度)
为了避免训练不稳定和二阶导数计算（Double Backpropagation），论文对目标值使用了 **Stop-Gradient (sg)** 操作：
$$ \text{Target} = \text{sg}(u_{\text{tgt}}) $$
这意味着在计算 Loss 对网络参数 $\theta$ 的梯度时，**不**对 $u_{\text{tgt}}$ 里的 $u_\theta$ 求导。目标值被视为一个常数。

---

### 3. Loss 函数

论文使用的 Loss 函数形式如下（公式 9）：

$$ \mathcal{L}(\theta) = \mathbb{E}_{t, r, x, \epsilon} \left[ \| u_\theta(z_t, r, t) - \text{sg}(u_{\text{tgt}}) \|^2 \right] $$

**详细展开后：**

$$ \mathcal{L}(\theta) = \| u_\theta(z_t, r, t) - \text{sg}\left( v_t - (t-r)(\underbrace{v_t \cdot \nabla_z u_\theta + \partial_t u_\theta}_{\text{JVP}}) \right) \|^2 $$

**Loss 的直观解释：**
*   网络预测的平均速度 $u_\theta$，应该等于“瞬时速度 $v_t$”减去“因时间变化导致的修正项”。
*   如果 $t=r$，则 $t-r=0$，Loss 变为 $\| u_\theta - v_t \|^2$，这退化为标准的 Flow Matching（平均速度等于瞬时速度）。
*   如果 $t \neq r$，模型就被迫学习如何根据当前的瞬时速度和变化率，去推断跨越时间段的平均速度。

**加权 Loss (Adaptive Weighting):**
在实际训练中（Section 4.3），为了平衡不同时间步的学习难度，作者使用了一个自适应权重 $w$：
$$ \mathcal{L}_{\text{final}} = w \cdot \| \Delta \|^2 $$
其中 $w = \frac{1}{\| \Delta \|^2 + c}$（$c$ 是小常数），这使得 Loss 表现得像 Pseudo-Huber Loss，能提高训练稳定性。

---

### 4. 训练流程总结 (Step-by-Step)

根据论文 Algorithm 1，一步训练的具体操作如下：

1.  **采样时间：** 随机采样两个时间点 $t$ 和 $r$（通常 $t, r \in [0, 1]$，且 $t > r$）。
2.  **采样数据：** 采样一张真实图片 $x$ 和高斯噪声 $\epsilon$。
3.  **构造输入 $z_t$：** 使用线性插值构造当前时刻的噪声图：$z_t = (1-t)x + t\epsilon$。
4.  **计算瞬时速度 $v_t$：** 直接计算 $v_t = \epsilon - x$。这是物理真值。
5.  **前向传播与 JVP：**
    *   将 $(z_t, r, t)$ 输入网络 $u_\theta$。
    *   同时利用自动微分框架（如 PyTorch 的 `torch.func.jvp`）计算全导数 $dudt = \text{jvp}(u_\theta, (z_t, r, t), (v_t, 0, 1))$。
6.  **构建目标 $u_{\text{tgt}}$：** 计算 $u_{\text{tgt}} = v_t - (t - r) \times dudt$。
7.  **计算 Loss：** 计算预测值 $u$ 和目标值 $\text{sg}(u_{\text{tgt}})$ 之间的加权平方误差。
8.  **反向传播：** 更新网络参数 $\theta$。

### 5. 特殊情况：带 CFG 的 GT 构建
如果使用 Classifier-Free Guidance (CFG)，GT 会发生变化（Section 4.2）：
*   **目标场变化：** 目标不再是单纯的 $v_t$，而是混合了 CFG 权重的速度场。
*   **公式：** $u_{\text{tgt}}$ 中的 $v_t$ 被替换为 $\tilde{v}_t = \omega v_t + (1-\omega) u_{\theta}(z_t, t, t)$。
*   这意味着在训练阶段，模型不仅要拟合物理速度，还要拟合“被引导后”的速度场，从而使得推理时只需 1 步即可完成 CFG 生成。

下面是详细解释.

# CFG 的GT构建

关于 **带 CFG（Classifier-Free Guidance）的 Ground Truth (GT) 构建**，这是 MeanFlow 论文中非常精彩的一部分，因为它巧妙地解决了传统 CFG 推理速度慢的问题。

以下是关于带 CFG 的 GT 构建的详细完整解读：

---

### 1. 核心思想：把 CFG “内化”到训练目标中

*   **传统 CFG 的痛点：**
    在采样（推理）阶段，传统方法需要计算公式：$\text{Output} = \text{Uncond} + \omega (\text{Cond} - \text{Uncond})$。这意味着每生成一步，都要跑两次网络（一次有条件，一次无条件），导致计算量翻倍（2-NFE）。

*   **MeanFlow 的解决方案：**
    作者定义了一个新的**混合物理场** $v_{\text{cfg}}$。既然我们知道推理时想要的是混合后的结果，不如直接训练网络去预测这个**已经混合好的平均速度** $u_{\text{cfg}}$。这样推理时只需要跑一次网络（1-NFE）。

---

### 2. 构造新的“混合瞬时速度” ($\tilde{v}_t$)

在普通训练中，瞬时速度的 GT 是 $v_t = \epsilon - x$。
在 CFG 训练中，我们需要构造一个**混合了引导尺度的瞬时速度** $\tilde{v}_t$ 作为基础。

根据论文公式 (13) 和 (19)，新的瞬时速度定义为：

$$ \tilde{v}_t = \omega \cdot v_t + (1 - \omega) \cdot u_\theta(z_t, t, t) $$

这里包含三部分：
1.  **$\omega$ (Guidance Scale)：** 引导强度（例如 2.0 或 7.5），这是训练时的超参数。
2.  **$v_t$ (Conditional Velocity)：** 数据决定的物理真值（即 $\epsilon - x$）。这代表了“有条件”的理想方向。
3.  **$u_\theta(z_t, t, t)$ (Unconditional Velocity)：** **这是关键点。**
    *   当 $r=t$ 时，平均速度等于瞬时速度。
    *   这里使用**模型自己预测的**、在 $t$ 时刻的瞬时速度，来近似“无条件”的速度场。
    *   注意：这一项通常是把类别条件置空（Drop Condition）后得到的输出。

**直观理解：** 训练目标不再是纯粹的物理真实速度，而是“物理真实速度”和“模型自己认为的无条件速度”的一个线性组合。

---

### 3. 构造带 CFG 的训练目标 ($u_{\text{tgt}}$)

有了上面的 $\tilde{v}_t$，我们再次利用 **MeanFlow Identity** 来构造最终的回归目标 $u_{\text{tgt}}$。

公式 (18) 如下：

$$ u_{\text{tgt}} = \tilde{v}_t - (t - r) \times \underbrace{\left( \tilde{v}_t \cdot \nabla_z u^{\text{cfg}}_\theta + \partial_t u^{\text{cfg}}_\theta \right)}_{\text{基于 } \tilde{v}_t \text{ 计算的 JVP}} $$

**具体步骤变化：**
1.  **计算 JVP 时：** 投影向量不再是 $(v_t, 0, 1)$，而是变成 $(\tilde{v}_t, 0, 1)$。这意味着我们计算的是沿着 CFG 混合轨迹的导数。
2.  **计算目标值时：** 基准速度变成了 $\tilde{v}_t$。

---

### 4. 进阶技巧：Improved CFG (Appendix B.1)

论文在附录中提出了一个改进版（Improved CFG），引入了一个混合参数 $\kappa$，进一步提升了效果。

**问题：** 基础版公式只利用了“无条件”的模型输出。
**改进：** 作者认为应该同时混合“有条件”和“无条件”的模型输出到目标中。

**改进后的混合瞬时速度公式 (Eq. 21)：**

$$ \tilde{v}_t = \omega (\epsilon - x) + \kappa \cdot u_\theta(z_t, t, t | c) + (1 - \omega - \kappa) \cdot u_\theta(z_t, t, t | \emptyset) $$

*   **$u_\theta(z_t, t, t | c)$：** 模型预测的**有条件**瞬时速度。
*   **$u_\theta(z_t, t, t | \emptyset)$：** 模型预测的**无条件**瞬时速度。
*   **$\kappa$：** 一个新的超参数，用于调节混合比例。

**为什么这样做？**
这相当于一种**自蒸馏 (Self-Distillation)**。模型在训练过程中，不仅在学习拟合数据（$\epsilon - x$），还在学习拟合“自己之前的预测混合”。这使得模型预测的平均速度场 $u_{\text{cfg}}$ 更加平滑、一致，从而在单步生成时画质更好。

---

### 5. 总结：带 CFG 的训练与推理

#### 训练阶段 (Training)
1.  随机采样 $t, r, x, \epsilon$。
2.  计算物理速度 $v_t = \epsilon - x$。
3.  让模型预测当前的瞬时速度 $u_{\text{inst}} = u_\theta(z_t, t, t)$（可能包含有条件和无条件两次前向）。
4.  **合成目标速度场：** $\tilde{v}_t = \text{Mix}(v_t, u_{\text{inst}}, \omega)$。
5.  **计算 JVP：** 基于 $\tilde{v}_t$ 计算全导数。
6.  **计算 Loss：** $\| u_\theta(z_t, r, t) - \text{sg}(u_{\text{tgt}}) \|^2$。

#### 推理阶段 (Inference / Sampling)
**极度简单：**
因为模型 $u_\theta$ 已经学会了预测“混合后的平均速度”，所以推理时**不需要**做任何 CFG 公式计算，也不需要跑两次模型。

$$ z_0 = z_1 - (1 - 0) \cdot u_\theta(z_1, 0, 1 | c) $$

**只需 1 次 NFE，就能得到带 Guidance 效果的高质量图像。**
