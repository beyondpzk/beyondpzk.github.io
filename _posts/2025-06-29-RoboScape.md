---
layout: post
title: RoboScape
date: 2025-06-29
categories: [WorldModels]
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# RoboScape: Physics-informed Embodied World Model


[paper link](https://arxiv.org/abs/2506.23135) 

**一句话总结：** RoboScape通过引入**时序深度预测（Temporal Depth Prediction）**和**关键点动力学学习（Keypoint Dynamics Learning）**两个辅助任务，在统一的Transformer架构中“强迫”模型理解物理规律，从而生成既逼真又符合物理逻辑的机器人操作视频。

根据论文中的Figure 2, 我整体理解一下这个工作, 首先, 核心还是视频生成, 即输入历史的RGB-sequence,生成未来的RGB Frame, 但是仅有pixel的生成可能不符合物理规律, 因此增加了未来深度图的监督, 以及关键点的对齐, 这样相当于不仅在pixel层面进行约束,而且也在物理几何方面有约束.

RoboScape 是一个以动作条件（Action-Conditioned）为核心的视频生成模型。为了克服纯视觉生成的物理幻觉，它通过特征注入引入了显式的几何先验（Depth），并通过关键点轨迹一致性引入了隐式的动力学先验（Dynamics），从而在无需传统物理引擎的情况下，实现了符合物理规律的具身世界模拟。



# RoboScape——物理感知的具身世界模型

**参考文献：** Shang et al., "RoboScape: Physics-informed Embodied World Model", arXiv:2506.23135v1, 2025.

---

## 第一部分：课程导入与背景 (Introduction & Context)

### 1.1 从生成式AI到具身智能的鸿沟
在过去的一两年里，我们见证了Sora、Genie等视频生成模型的爆发。这些模型生成的视频在视觉上极具欺骗性，但在物理逻辑上往往经不起推敲。比如，一个人走路时脚陷入了地面，或者一个杯子被拿起时突然发生了形变。

对于娱乐视频生成，这些瑕疵是可以容忍的。但当我们谈论**具身智能（Embodied Intelligence）**和**机器人学习（Robotic Learning）**时，这种“物理幻觉”是致命的。

*   **核心问题：** 机器人需要基于对世界的预测来决策。如果世界模型（World Model）预测“推一下杯子，杯子会穿过桌子掉下去”，机器人学到的策略就是错误的。
*   **现状：** 现有的具身世界模型（如IRASim, iVideoGPT）主要关注RGB像素的优化。它们是“视觉拟合者”，而非“物理理解者”。
*   **挑战：** 如何在不引入昂贵的传统物理引擎（Physics Engine）模拟的前提下，让神经网络学到3D几何一致性和物体动力学（如刚体、柔性体特性）？

### 1.2 本文切入点：RoboScape
《RoboScape: Physics-informed Embodied World Model》。这篇论文由清华大学和Manifold AI联合发表。

**一句话总结：** RoboScape通过引入**时序深度预测（Temporal Depth Prediction）**和**关键点动力学学习（Keypoint Dynamics Learning）**两个辅助任务，在统一的Transformer架构中“强迫”模型理解物理规律，从而生成既逼真又符合物理逻辑的机器人操作视频。

---

## 第二部分：数据工程——构建物理感知的具身数据集

在深度学习中，Data is the fuel。RoboScape之所以能生效，很大程度上通过巧妙的数据管线（Pipeline）引入了物理先验（Physical Priors）。

### 2.1 数据处理管线 (Data Processing Pipeline)
作者基于AGIBOT-World数据集构建了训练数据。普通的视频数据是RGB序列，但我们要训练物理感知，必须有“物理标签”。作者采用了**Model-based Annotation**（基于模型的标注）策略：

1.  **物理属性标注 (Physical Property Annotating)：**
    *   **深度信息：** 使用 **Video Depth Anything** 模型。这是一个强大的单目深度估计大模型，它为每一帧RGB视频生成对应的深度图序列。这是为了教模型理解“几何”。
    *   **关键点追踪：** 使用 **SpatialTracker** 模型。它可以追踪视频中的像素点运动轨迹。这是为了教模型理解“运动和形变”。

2.  **视频切片与清洗 (Slicing & Filtering)：**
    *   使用 **TransNetV2** 检测镜头边界，防止把不同场景剪在一起。
    *   使用 **Intern-VL**（视觉语言模型）来提取动作语义（如“pick the bottle”），这对于后续的文本/动作条件控制至关重要。
    *   使用 **FlowNet** 过滤掉那些运动模糊或几乎静止的低质量片段。

### 2.2 思考
**Q:** 为什么要用预训练模型（如Depth Anything）生成的伪标签（Pseudo-labels）来训练，而不是直接用仿真器（Simulator）生成的Ground Truth？
**A:** 虽然仿真器数据完美，但Sim-to-Real Gap（虚实迁移鸿沟）很难跨越。RoboScape的方法允许我们在**真实世界视频**上进行训练（只要能跑Depth Anything），这极大地扩展了数据的多样性。这是一种“知识蒸馏”——将专用大模型（深度、追踪）的知识压缩进我们的世界模型中。

---

## 第三部分：核心方法论——RoboScape模型架构 (Methodology)

参考论文 **Figure 2**。

### 3.1 总体架构：多任务自回归Transformer
RoboScape本质上是一个预测下一帧（Next-token Prediction）的自回归模型。

*   **输入：** 历史观测 $o_{1:t}$，历史动作 $a_{1:t}$。
*   **输出：** 下一帧观测 $o_{t+1}$。

#### 3.1.1 视觉Tokenizer (Visual Tokenization)
模型不直接在像素空间操作，而是使用 **MAGVIT-2** 作为Tokenizer。
*   **压缩：** 将 $T \times H \times W \times 3$ 的RGB视频压缩为离散的Latent Tokens $s_{1:T} \in \mathbb{R}^{T \times H' \times W' \times D}$。
*   **深度图Tokenization：** 同样的，将深度图 $d_{1:T}$ 也Tokenize为深度Tokens $z_{1:T}$。

### 3.2 核心创新：双分支协同自回归 Transformer (DCT)
这是模型结构中最精彩的部分。为了让模型同时学好“画图”（RGB）和“建模结构”（Depth），作者设计了一个双分支结构。

#### 结构拆解：
1.  **RGB分支 ($F_{RGB}$)：** 负责预测下一时刻的RGB Token。
2.  **Depth分支 ($F_{Depth}$)：** 负责预测下一时刻的Depth Token。

**公式表达：**
$$ \hat{s}_t = F_{RGB}(s_{1:t-1} \oplus c_{1:t-1} \oplus e_{1:t-1}) $$
$$ \hat{z}_t = F_{Depth}(z_{1:t-1} \oplus c_{1:t-1} \oplus e_{1:t-1}) $$
其中，$c$ 是动作embedding，$e$ 是位置embedding。

#### 3.3 深度注入机制 (Depth Injection)
如果两个分支各跑各的，那就是两个独立的模型。RoboScape引入了**跨分支交互（Cross-branch Interaction）**。

在每一个时空Transformer块（ST-Transformer Block）中，深度的特征被注入到RGB分支中：
$$ h_{RGB}^l = h_{RGB}^l + W^l(h_{depth}^l) $$
*   **物理含义：** 这一步至关重要。$h_{depth}$ 包含了场景的3D几何信息。通过将其投影并加到RGB特征上，模型在生成RGB像素时，实际上是“看着”深度信息在画图。这保证了生成的物体不会出现空间错位。

### 3.4 隐式材质理解：关键点动力学学习 (Keypoint Dynamics Learning)
仅仅有深度还不够。深度图无法完全描述物体的物理材质（如布料是软的，金属是硬的）。作者提出通过**关键点运动**来隐式编码这些属性。

#### 3.4.1 自适应采样 (Adaptive Sampler)
论文没有追踪所有点，而是追踪**动得最厉害的点**。 (变化最大的点才是最应该关注的.)
*   计算运动幅度：$M_i = \sum ||p_{i}^{t+1} - p_{i}^{t}||^2$。
*   选择Top-K个关键点。
*   **直觉：** 在机器人操作中，运动剧烈的区域通常是机器人末端执行器（End-effector）和被操作物体。这正是物理交互发生的地方。

#### 3.4.2 损失函数设计
为了让模型学会物理规律，作者设计了两个特殊的Loss：

1.  **关键点一致性 Loss ($L_{Keypoint}$):**
    $$ L_{Keypoint} = \frac{1}{(T-1)K} \sum_{i=1}^{K} \sum_{t=2}^{T} \| \hat{s}_t(p_i^t) - \hat{s}_1(p_i^1) \|_2^2 $$
    *   **解释：** 这个公式在强迫模型做一件事——同一个物体上的同一个点，在不同时间步的Feature representation应该是一致的（或者可预测的）。这实际上是在约束光流（Optical Flow）和物体的一致性，防止物体在运动中“融化”或改变纹理。

2.  **注意力引导 Loss ($L_{Attention}$):**
    $$ L_{Attention} = - \sum_{t=1}^{T} A_t \odot s_t \log p(\hat{s}_t) $$
    *   作者定义了一个Mask $A_t$，在关键点轨迹附近的区域给予更高的权重 $\gamma$。
    *   **作用：** 强迫模型重点关注那些正在发生物理交互的区域（如抓取点），而不是背景墙壁。

### 3.5 总体优化目标
$$ L = L_{RGB} + \lambda_1 L_{Depth} + \lambda_2 L_{Keypoint} + \lambda_3 L_{Attention} $$
这就构成了一个统一的、端到端的物理感知训练框架。

---

## 第四部分：实验结果与分析 (Experiments)

### 4.1 视频生成质量
 **Table 1**。作者对比了IRASim, iVideoGPT, Genie, CogVideoX。
*   **SOTA表现：** RoboScape在LPIPS（感知质量）、PSNR（像素信噪比）和AbsRel（几何一致性）上全面领先。
*   **定性分析：** 看 **Figure 3**。注意那个“拖动布料”的例子。由于引入了关键点动力学，RoboScape生成的布料褶皱变化符合物理规律，而没有物理约束的模型生成的布料可能会像液体一样流走。

### 4.2 下游任务：机器人策略学习 (Policy Learning)
这是验证World Model是否有用的“金标准”。能不能用生成的假数据训练真机器人？

*   **实验设置：** 使用Diffusion Policy和$\pi_0$模型。
*   **结果 (Table 3):**
    *   在Robomimic任务中，仅使用合成数据训练，达到了91%的成功率（真实数据是92%）。这非常惊人，意味着Sim-to-Real几乎无损。
    *   在LIBERO这一复杂的长程任务中，混合合成数据能显著提升成功率。
    *   **结论：** RoboScape生成的视频不仅仅是“看着像”，其中的物理动力学是“对的”，所以策略网络能从中学习到正确的操作逻辑。

### 4.3 策略评估 (Policy Evaluation)
*   **场景：** 在不部署机器人的情况下，用世界模型来测试一个策略好不好。
*   **相关性：** RoboScape预测的成功率与真实模拟器(Ground Truth Simulator)的成功率相关系数高达 **0.953** (Pearson)，而IRASim是负相关。
*   **意义：** 这意味着我们可以用RoboScape作为一个低成本的“离线模拟器”来筛选策略。

---

## 第五部分：总结与讨论 (Conclusion & Future Work)

### 5.1 核心贡献回顾
1.  **Unified Framework:** 不再是“视频生成 + 物理外挂”，而是将物理知识（深度、关键点）内化为训练任务。
2.  **Physics as Supervision:** 提出了通过预测深度和追踪关键点来作为物理监督信号，这是一个非常通用的思路。
3.  **Data Pipeline:** 证明了利用现有大模型生成伪标签来提升特定领域模型性能的可行性。

### 5.2 开放讨论 (Open Discussion)
*   **局限性：** 模型依赖于Depth Anything和SpatialTracker的准确性。如果伪标签错了，模型也会学错。这被称为“Error Propagation”。如何解决？
*   **未来方向：** 目前还是基于2D视频生成。是否可以直接生成3D Gaussian Splatting或NeRF来实现更好的视角一致性？
*   **Scaling Law:** 论文提到了模型参数量从34M增加到544M性能提升明显。对于World Model，它的上限在哪里？

---

## 思考
1.  论文中提到将Depth特征注入RGB分支，为什么不反过来，将RGB特征注入Depth分支？或者采用双向注入？请从特征语义层级角度分析。
2.  $L_{Keypoint}$ 损失函数假设了特征的一致性。在物体发生剧烈形变（如海绵被压扁）或遮挡（Occlusion）时，这个假设还成立吗？如果不成立，会对模型产生什么影响？

## QA1. 在机器人任务上如何验证

在下游机器人任务上如何验证？(Diffusion Policy & $\pi_0$)

验证的核心逻辑是：**如果世界模型生成的“假数据”足够逼真（物理规律正确），那么用这些“假数据”训练出来的策略（Policy），应该能在“真环境”里通过考试。**

具体验证流程分为三步走：

#### 第一步：数据合成 (Data Synthesis)
*   **原料：** 给定初始帧 $o_0$ 和一串动作序列 $a_{1:T}$（这些动作可以是从数据集中采样的，也可以是随机生成的）。
*   **生成：** 让 RoboScape 预测出一整段视频 $o_{1:T}$。
*   **产物：** 我们得到了一组成对的数据：`{视频(Observation), 动作(Action)}`。但这视频是模型生成的“梦境”。

#### 第二步：策略训练 (Policy Training)
作者使用了两种目前最先进的策略网络架构（Policy Architectures）作为“学生”，在上述“梦境数据”上进行训练：

1.  **Diffusion Policy (扩散策略):**
    *   这是目前机器人模仿学习（Imitation Learning）的SOTA方法。它把预测动作建模为一个去噪过程。
    *   **训练目标：** 输入当前的观测图像，输出应该执行的动作分布。
    *   **关键点：** 如果 RoboScape 生成的视频里，杯子被推倒的方式不对（比如穿模了），Diffusion Policy 就会学到错误的视觉-动作映射。

2.  **$\pi_0$ (Pi-Zero):**
    *   这是一个基于 Flow Matching 的大型基础策略模型（类似于机器人领域的 GPT）。
    *   **作用：** 作者引入 $\pi_0$ 是为了证明 RoboScape 生成的数据质量极高，甚至能用来微调（Fine-tune）这种大规模策略模型。

#### 第三步：真机/真仿真器测试 (Evaluation in Ground Truth)
这是“期末考试”。
*   **考场：** 真实的物理仿真器（Ground Truth Simulator），如 **LIBERO** 或 **Robomimic** 环境。注意，这里虽然也是仿真器，但它是基于牛顿力学引擎（MuJoCo等）算出来的，被视为“真实世界”。
*   **考试过程：**
    1.  把刚才用“假数据”训练好的 Diffusion Policy / $\pi_0$ 放入这个“真环境”。
    2.  让机器人执行任务（比如“把红色方块放到绿色盘子里”）。
    3.  **统计成功率 (Success Rate)。**

#### 实验结果解读 (Table 3)
*   **RoboScape 的成绩：** 在 Robomimic 任务上，仅用 RoboScape 生成的数据训练，Diffusion Policy 达到了 **91.2%** 的成功率。
*   **对比：** 用真实数据训练的上限是 **92.6%**。
*   **结论：** 这意味着 RoboScape 生成的数据在物理真实性上已经**无限接近**真实数据（Sim-to-Real Gap 极小）。这就是为什么说它是一个“Physics-informed”的世界模型——它没骗过策略网络，它教给了策略网络真正的物理规律。

---

**总结一下：**
RoboScape 的 Transformer 是**从零训练**的（但在预训练Tokenizer的特征空间上）。验证方法是**“以假乱真”**——看用假视频训练出来的机器人，能不能在真环境里干活。如果能干活，说明这个“假视频”里的物理规律是真的。


## Q2.整个网络是 train from scratch吗

答案是：**核心动力学部分是 Train from Scratch，但感知部分利用了预训练模型。** 这是一个典型的“站在巨人肩膀上”的混合策略。

我们可以把整个管线拆分为三个部分来看：

1.  **视觉感知与压缩 (Visual Tokenizer) —— 预训练 (Pre-trained)**
    *   **组件：** **MAGVIT-2**。
    *   **状态：** 这是一个已经在海量通用视频数据上预训练好的Tokenizer。
    *   **作用：** 它的任务是将 $256 \times 256$ 的 RGB 图像压缩成 $32 \times 32$ 的离散 Token。
    *   **原因：** 如果从零开始训练Tokenizer，收敛极慢且需要极其庞大的数据。复用MAGVIT-2相当于让模型天生就“认识”边缘、纹理和颜色，不需要重新学“怎么看世界”。

2.  **物理监督信号生成 (Physics Annotators) —— 预训练且冻结 (Frozen)**
    *   **组件：** **Video Depth Anything** (深度估计) 和 **SpatialTracker** (关键点追踪)。
    *   **状态：** 这两个是大模型，直接拿来用（Inference only），不参与训练更新。
    *   **作用：** 它们是“老师”，负责给训练数据打上深度和运动的标签（Pseudo-labels）。

3.  **核心世界模型 (Dynamics Transformer) —— 从零训练 (Train from Scratch)**
    *   **组件：** 论文核心的 **DCT (Dual-branch Co-autoregressive Transformer)**。
    *   **状态：** 这部分参数是随机初始化的，完全在 **AgiBotWorld-Beta 数据集**（50k 视频片段，约 650万个训练样本）上从头训练。
    *   **原因：** 虽然它利用了预训练的 Tokenizer，但**“机器人动作 $a_t$ 如何影响环境 $s_{t+1}$”** 这一物理规律是领域特定的（Domain-specific）。通用视频模型（如Sora）不懂机器人的运动学，所以必须依靠专门的机器人数据从零学习动力学。

