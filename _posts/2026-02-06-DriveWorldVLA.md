---
layout: post
title: DriveWorldVLA
date: 2026-02-06
categories: []
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# DriveWorldVLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving
[paper link](https://www.arxiv.org/abs/2602.06521) 


这篇论文的核心贡献在于它解决了一个长期存在的痛点：如何让自动驾驶系统不仅“看到”和“理解”场景，还能真正基于对物理世界的因果推理（Causal Reasoning）来进行“反事实推演”（Counterfactual Imagination）。


---

# 研究背景与核心动机 (Background & Motivation)

## 1.1 从模块化到端到端，再到 VLA+World Model
传统的自动驾驶采用模块化管线（感知->预测->规划->控制），存在误差累积和信息丢失的问题。端到端（E2E）学习虽然打通了传感器到控制信号的映射，但往往缺乏长时序推理能力，且难以理解“动作”与“环境”之间的因果关系。

为了解决这个问题，学术界引入了两个强大的范式：
1.  **VLA (Vision-Language-Action) 模型**：利用大语言模型（LLM）的泛化能力和逻辑推理能力，处理多模态输入并输出动作。
2.  **世界模型 (World Models)**：赋予智能体“前瞻性想象力”（Prospective Imagination），通过显式建模环境动态来预测未来。

## 1.2 现有方法的局限性 (The Coupling Problem)
现有的 VLA 与世界模型的结合方式通常是松耦合的，作者将其归纳为两类次优解：

*   **解耦交互 (Disentangled Interaction)**：世界模型仅作为外部模拟器或数据源。VLA 无法内化物理规律。就像一个司机只能通过看录像来学习开车，而不能亲自感知车辆动力学。<alphaxiv-paper-citation title="Disentangled Approach" page="1" first="(a) Disentangled Interaction:" last="knowledge transfer." />
*   **特征共享 (Feature Sharing)**：虽然共享了表征，但缺乏以动作为条件的因果推理。模型仍然是反应式的（Reactive），而非前瞻性的（Proactive）。它无法回答“如果我这样做，世界会变成什么样？”的问题。<alphaxiv-paper-citation title="Feature Sharing" page="1" first="(b) Feature Sharing:" last="planning." />

## 1.3 DriveWorld-VLA 的核心洞察
本文提出的 **DriveWorld-VLA** 属于第三种范式：**统一潜在空间建模 (Unified Latent-Space Modeling)**。

其核心思想是：**在特征层面（Latent Space）将世界建模和规划统一起来**。
*   **统一决策变量**：世界模型的隐状态直接作为规划器的决策变量。
*   **可控想象**：通过在潜在空间中优化动作，实现基于因果的“What-if”推理。<alphaxiv-paper-citation title="Core Innovation" page="1" first="By optimizing world" last="latent space." />

---

# DriveWorld-VLA 模型架构 (Model Architecture)

DriveWorld-VLA 的架构设计非常精妙，它并没有简单地堆叠模型，而是通过共享隐空间来实现深度融合。

## 2.1 多模态输入与词元化 (Input & Tokenization)
模型接受四种模态的输入，并将其统一映射到 VLM 的嵌入空间：

1.  **多视角图像 ($I_t$)**：使用 InternVL 作为基础，通过 ViT 编码。为了处理不同分辨率，使用了动态图块（Patch）技术。
2.  **文本指令 ($T_t$)**：通过 Tokenizer 处理，包含导航指令等。
3.  **历史动作 ($A_{t-1}$)**：将历史轨迹序列化为自然语言提示，然后进行 Token 编码。 (即 "The history trajectory is $(x_1, y_1), (x_2, y_2), ...  $")
4.  **BEV 表征 ($B_t$)**：这是关键。使用 BEVFormer 提取鸟瞰图特征，并在空间上展平，投影为 "BEV Tokens"。这一步将几何感知的特征引入了语言模型空间。<alphaxiv-paper-citation title="BEV Features" page="3" first="BEV features" last="BEV tokens." />

除了 $IMAGE-TOKENS$ 和 $BEV-TOKENS$ 之外, text prompt部分类似下面这样:

```
<System_Prompt>
You are a vehicle trajectory prediction model for autonomous driving.
Your task is to predict the ego vehicle's future trajectory based on the following inputs.

<Navigation_Instruction>
Turn left at the intersection.

<History_Trajectory>
The history trajectory in the past 2 seconds is:
(0.00, 0.00), (0.52, 0.10), (1.05, 0.22), (1.58, 0.35), (2.12, 0.50).

```

## 2.2 共享潜在空间 (Shared Latent Space)
所有的 Tokens 被送入 VLM (InternVL3-2B)。VLM 的最后一层隐状态（Hidden States）被提取出来，作为**共享潜在表征**，记为 $H_t$。

$$H_t = \text{VLM}_\theta(I_t, B_t, A_{t-1}, T_t)$$

这里的 $H_t$ 是整个系统的“大脑皮层”。它不仅包含了语义信息（来自 Text/Image），还包含了空间几何信息（来自 BEV）和时序动力学信息（来自 History）。这是实现“统一”的关键一步。<alphaxiv-paper-citation title="Latent Space" page="4" first="H_t serves as" last="prediction." />

## 2.3 双分支解码 (Dual-Branch Decoding)
基于共享特征 $H_t$，模型分出两个核心任务分支：

### A. 动作预测分支 (Action Prediction)
这是一个轻量级的动作解码器（Action Decoder），基于当前隐状态预测未来动作 $A'_{t+\Delta t}$。
$$A'_{t+\Delta t} = \text{ACT}_\theta(H_t, B_t, A_{t-1})$$

### B. 未来想象分支 (Future Imagination - World Model)
这是世界模型的核心。它在 BEV 空间进行推演。
1.  **交叉注意力**：首先通过 Cross-Attention 将 $H_t$ 的信息注入到当前的 BEV 特征 $B_t$ 中，得到增强特征 $B'_t$。
2.  **去噪器 (Denoiser)**：这是一个基于 Diffusion Transformer (DiT) 的模块，负责生成未来的 BEV 状态 $B_{t+\Delta t}$。
    *   注意：这里设计了两个分支，一个是“历史条件分支”，一个是“未来动作条件分支”。

---

# 三阶段渐进式训练范式 (Three-Stage Progressive Training)

这是理解该论文技术路线的重中之重。作者没有采用端到端一次性训练，而是设计了三个阶段，逐步解锁模型的能力。

## 阶段一：VLA 与世界模型联合训练 (VLA & WM Joint Training)
**目标**：让模型学会表征学习，并将物理知识通过世界建模任务转移到 VLA 中。

*   **过程**：同时进行动作预测和未来场景生成。
*   **细节**：
    *   此时的 Denoiser 主要依赖历史信息（History-conditioned branch）。
    *   **监督信号**：
        1.  语义分割损失 ($L_{seg}$)：解码生成的 BEV 特征，与 Ground Truth 语义地图对比。
        2.  动作克隆损失 ($L_{act}$)：模仿专家轨迹。
*   **公式**：$L_{s1} = L_{seg} + L_{act}$。<alphaxiv-paper-citation title="Stage 1 Loss" page="4" first="The overall loss" last="L_{act}," />

注意,时此并没有 $B'_{t+\Delta t}$ 的监督.

## 阶段二：动作可控性微调 (Action Controllability Fine-Tuning)
**目标**：赋予模型“反事实推理”能力。即：给定一个未来动作，模型能想象出相应的未来 BEV 状态。

*   **核心机制**：**流匹配 (Flow Matching)**。
*   **操作**：
    1.  冻结 VLM 主干。
    2.  利用阶段一训练好的编码器提取未来时刻的 Ground Truth BEV 特征 $B'_{t+\Delta t}$ 作为目标。
    3.  训练 Denoiser 的第二个分支（Action-conditioned branch）。输入是当前 BEV 状态 $B'_t$ 和**未来动作** $A_{t+\Delta t}$。
    4.  这是一个生成式任务：模型学习将高斯噪声去噪为特定的未来 BEV 特征，且该生成过程受到动作的严格控制。
*   这一步至关重要。它建立了一个 $f: (State, Action) \rightarrow NextState$ 的显式转移函数。 **没有这一步，模型就只是在做视频预测，而不是在做决策支持**。<alphaxiv-paper-citation title="Stage 2 Motivation" page="4" first="Motivated by this" last="imagination." />

## 阶段三：未来引导的评估与微调 (Future-Guided Evaluation & Refinement)
**目标**：闭环优化。利用想象出的未来来修正当前的决策。

*   **流程**：
    1.  **动作提议**：VLA 输出一组候选动作。
    2.  **并行想象**：利用阶段二训练好的 World Model，针对每个候选动作生成对应的未来 BEV 特征。
    3.  **奖励评估**：使用一个预训练的奖励模型（Reward Model）对生成的未来进行打分（评估安全性、合规性等）。
    4.  **偏好对齐**：使用 DPO (Direct Preference Optimization) 或类似的策略，根据奖励信号微调 VLA 的动作解码器。
*   **结果**：模型学会了避开那些会导致“危险未来”的动作。

---

# 实验结果与分析 (Experiments & Analysis)

## 4.1 实验设置
*   **数据集**：nuScenes (Open-loop) 和 NAVSIM (Closed-loop)。
*   **指标**：
    *   **PDMS (Predictive Driving Model Score)**：综合考量碰撞率、道路合规性、舒适度等的综合指标。
    *   **L2 Error & Collision Rate**：轨迹预测的传统指标。

## 4.2 核心结果
DriveWorld-VLA 在各项指标上均取得了 SOTA (State-of-the-Art) 的成绩。

1.  **NAVSIM 榜单**：在 NAVSIMv1 上达到 **91.3 PDMS**，在 v2 上达到 **86.8 EPDMS**。这是非常显著的提升，尤其是在包含长尾场景的测试中。<alphaxiv-paper-citation title="SOTA Performance" page="1" first="DriveWorld-VLA achieves" last="HERMES-p" />
2.  **nuScenes**：碰撞率仅为 **0.16%**，显著优于专门的规划基线模型（如 UniAD, VAD 等）。

## 4.3 消融实验 (Ablation Studies)
这部分揭示了模型设计的合理性（虽然论文细节需参阅附录，但主要结论如下）：
*   **特征共享的重要性**：如果切断 VLM 和 World Model 的特征共享（即退化为图1a的解耦模式），性能大幅下降。这证明了在 Latent Space 统一建模的必要性。
*   **三阶段训练的必要性**：直接进行端到端训练往往难以收敛，或者无法获得高质量的生成能力。分阶段训练有效地解耦了表征学习、动力学学习和策略优化。
*   **可视化分析**：图4展示了阶段二和阶段三的对比。阶段二生成的轨迹可能与真值接近但有碰撞风险；阶段三经过“想象-修正”后，轨迹明显避开了障碍物，体现了因果推理的价值。<alphaxiv-paper-citation title="Visual Comparison" page="8" first="Stage 2 generates" last="collision risk." />

---

# 总结与深度讨论 (Conclusion & Discussion)

## 5.1 总结
DriveWorld-VLA 成功地证明了：**世界模型不应仅仅是自动驾驶系统的“外挂”或“视频生成器”，而应成为决策核心的一部分。** 通过在 Shared Latent Space 进行统一建模，模型不仅获得了物理常识，还具备了类似人类的“三思而后行”（Look-ahead & Reasoning）的能力。

## 5.2 开放性问题与研讨方向
1.  **计算开销 (Inference Cost)**：虽然是在 Latent Space 进行推演（比像素级生成快），但 DiT 和 LLM 的推理成本依然很高。如何实现实时的 On-board 部署？
2.  **幻觉问题 (Hallucination)**：世界模型生成的未来毕竟是“想象”的。如果想象出现了偏差（例如漏掉了障碍物），系统如何保证安全性？（论文中提到了 Reward Model，但 Reward Model 本身也可能误判）。
3.  **多模态融合的粒度**：当前的融合主要发生在 VLM 的输入端和中间层。是否存在更优的融合架构，比如将 World Model 的动力学约束直接作为 Attention Mask 注入到 VLM 中？

## 要不要把自车历史轨迹作为模型的输入

**模仿学习（Imitation Learning, IL）**中的核心痛点——**因果混淆（Causal Confusion）**和**惯性偏差（Inertia Bias）**。

**将历史动作作为输入确实是一把双刃剑**。在学术界，关于“是否引入历史动作”的争论从未停止。

### 1. 为什么你的担忧是成立的？（The "Con" Side）

你担心不该输入历史动作，背后的逻辑通常对应两个主要风险，这在端到端自动驾驶中非常经典：

1.  **捷径学习（Shortcut Learning / Copycat Problem）**：
    *   如果模型发现 $A_t \approx A_{t-1}$（例如在高速公路上巡航时，上一帧是直行，这一帧大概率还是直行），它就会“偷懒”。
    *   模型会倾向于直接复制上一帧的动作，而忽略视觉输入（Visual Input）。一旦遇到突发情况（比如前车急刹），模型可能因为惯性而无法及时改变动作，导致事故。
2.  **协变量偏移（Covariate Shift / Error Accumulation）**：
    *   训练时，使用的是专家的历史动作（Perfect History）。
    *   测试时，使用的是模型自己生成的历史动作（Noisy History）。
    *   如果上一帧模型预测偏了一点点，这一帧又把这个偏差作为输入，误差就会像滚雪球一样迅速放大，导致车辆偏离轨迹。

### 2. 为什么 DriveWorld-VLA 坚持要输入 $A_{t-1}$？（The "Pro" Side）

尽管有上述风险，作者在公式 (1) 中明确将 $A_{t-1}$ 纳入了 VLM 的输入：
$$H_t = \text{VLM}_\theta(I_t, B_t, A_{t-1}, T_t)$$

这主要是出于以下三个关键的工程与物理考量：

#### A. 补充不可观测的动力学状态（Hidden Dynamics）
单帧图像无法完全反映车辆的物理状态。
*   **例子**：一张静止的图片显示车辆在路中间。车辆是在**加速**、**减速**还是**匀速**？图片看不出来。
*   **作用**：$A_{t-1}$（比如上一时刻的油门开度、刹车压力）是推断当前加速度和车辆动力学状态（Vehicle Dynamics）的最强线索。对于 VLM 这种大模型来说，历史动作提供了关于“当前物理状态”的显式提示（Prompt）。

#### B. 意图的平滑与连贯性（Temporal Consistency）
驾驶行为具有极强的连续性。
*   **控制稳定性**：如果没有历史动作作为参考，纯视觉模型可能会产生高频震荡（Jittering）。比如这一帧输出方向盘转角 5度，下一帧突然跳到 -2度，这在控制层面是不可接受的。
*   **状态保持**：引入 $A_{t-1}$ 充当了一种类似 RNN 中的 Hidden State 的角色，帮助模型维持长时序的意图（例如，“我正在执行一个持续 3 秒的左转操作”）。

#### C. 与世界模型的协同（Crucial for World Modeling）
注意这篇论文的核心是**世界模型**。
*   世界模型本质上是学习 $P(S_{t+1} | S_t, A_t)$。
*   在 DriveWorld-VLA 中，为了让 Latent Space ($H_t$) 包含足够的预测未来的信息，它必须知道“刚才发生了什么”。$A_{t-1}$ 是构建因果推理链条（Causal Chain）的重要一环。如果不知道刚才施加了什么力（动作），就很难准确预测现在的状态为何如此，进而很难预测未来。

### 3. DriveWorld-VLA 如何缓解你的担忧？

作者并非没有意识到“偷懒”的问题，这篇论文的架构设计巧妙地规避了单纯依赖历史动作的弊端：

1.  **多模态强制对齐（Latent Space Regularization）**：
    *   模型并不是直接把 $A_{t-1}$ 扔给动作解码器，而是先通过 VLM 映射到高维潜在空间 $H_t$。
    *   在这个空间里，视觉信息 ($I_t, B_t$) 和动作信息 ($A_{t-1}$) 被强制融合。如果视觉信息显示前方有墙，Visual Tokens 会在 Attention 机制中压倒 History Action Tokens，迫使模型改变决策。

2.  **未来预测任务的倒逼（Future Prediction as Regularizer）**：
    *   这是最关键的一点。模型不仅仅要输出动作，还要**想象未来** ($B_{t+\Delta t}$)。
    *   如果模型只是简单复制 $A_{t-1}$，它就无法通过 Denoiser 生成正确的未来 BEV 变化。为了在 Stage 1 和 Stage 2 降低 $L_{seg}$ 和 $L_{FM}$ 损失，模型被迫去真正理解环境，而不仅仅是拟合动作曲线。<alphaxiv-paper-citation title="Stage 2 Motivation" page="4" first="Motivated by this" last="imagination." />

3.  **闭环微调（Stage 3 Closed-loop Training）**：
    *   论文的第三阶段引入了奖励模型和 RL/DPO 优化。
    *   如果模型因为依赖历史动作而导致碰撞（Evaluation Metric 变差），这个策略会受到惩罚。这直接打破了“捷径学习”的回路。

### 小结

DriveWorld-VLA 通过**强视觉编码**和**世界模型预测任务**，实际上是在告诉模型：“你可以参考历史动作来保持平滑，但你必须依据视觉输入来决定未来。”

这是一个典型的 **Trade-off（权衡）**：用一点点“惯性偏差”的风险，换取了巨大的“系统稳定性”和“动力学感知能力”的提升。

## BEV feature的监督


DriveWorld-VLA 能否成功训练出“世界模型”的核心所在。

在 **第一阶段**，只有分割监督（语义级别）；但在 **第二阶段**，引入了**显式的特征级监督（Feature-level Supervision）**。

---

### 1. 第一阶段：只有分割监督 (Indirect Supervision via Segmentation)

在 **Stage 1 (VLA & WM Joint Training)**，Denoiser 输出的未来 BEV 特征 $B_{t+\Delta t}$ **没有** 直接的特征级真值（Ground Truth Feature）来监督。

*   **监督方式**：
    模型把生成的特征 $B_{t+\Delta t}$ 送入一个轻量级的 **Segmentation Head**，解码成语义分割图（Semantic Map）。然后计算解码出的图与真实的语义标签（Ground Truth Label）之间的 Cross-Entropy Loss ($L_{seg}$)。
*   **公式**：
    $$S_{t+\Delta t} = \text{SEG}_\theta(B_{t+\Delta t})$$
    $$Loss = \text{CrossEntropy}(S_{t+\Delta t}, GT\_Map)$$
*   **目的**：
    这一阶段的目标是**表征学习（Representation Learning）**。通过强迫特征能解码出道路、车辆等语义信息，模型学会了“什么样的特征才是好特征”。此时，特征空间还没有完全定型。

---

### 2. 第二阶段：显式的特征级监督 (Explicit Feature-level Supervision) —— **这是重点**

到了 **Stage 2 (Action Controllability Fine-Tuning)**，情况完全变了。作者引入了**强特征监督**。

此时：*“未来的 BEV 特征真值（Ground Truth) 并不是BEVformer得到的, BEVformer只在输入的时候才用到”*

**操作手法（Self-Supervision / Teacher-Student）**：
作者利用了在第一阶段训练好的、已经冻结（Frozen）的编码器作为“老师”。

1.  **获取特征真值 (GT Latents)**：
    取未来时刻 $T+\Delta t$ 的**真实图像** ($I_{t+\Delta t}$)，输入到冻结的 Stage 1 模型中。模型会输出该时刻“完美”的 BEV 特征 $B'_{t+\Delta t}$。
    $$GT\_Feature = \text{Encoder}_{frozen}(I_{t+\Delta t}, ...)$$
    这个 $GT\_Feature$ 就代表了“如果我有未来的图像，我应该看到的特征是什么”。

2.  **训练 Denoiser**：
    现在的任务变成了：**在没有未来图像的情况下，仅凭当前状态和动作，去生成这个 $GT\_Feature$**。
    Denoiser 的输出 $Generate(B_t, A_{t+\Delta t})$ 被强制要求去逼近 $GT\_Feature$。

3.  **监督损失 ($L_{FM}$)**：
    使用的是 **流匹配损失 (Flow Matching Loss)**，直接在特征空间计算距离：
    $$L_{FM} = || \text{Denoiser}(...) - (GT\_Feature - Noise) ||^2$$
    <alphaxiv-paper-citation title="Feature Supervision" page="4" first="Due to the absence" last="BEV prediction." />

---

### 为什么要加上特征级监督？

如果不加特征级监督，仅靠分割监督（如 Stage 1），模型会面临两个巨大问题：

1.  **信息丢失（Information Bottleneck）**：
    语义分割图是高度压缩的（只有类别信息，丢掉了纹理、具体姿态、车辆颜色等细节）。如果只用 $L_{seg}$，Denoiser 生成的特征可能只包含“那里有辆车”，但丢失了“那辆车正在向左转”的精细动力学特征。
2.  **模态坍塌（Mode Collapse）**：
    生成模型如果不直接约束其输出分布（Feature Distribution），很容易退化成输出“平均值”或“模糊的特征”，导致生成的未来也是模糊的。

**总结**：
DriveWorld-VLA 的高明之处在于：
*   先用 $L_{seg}$教会模型“什么是好特征”（Stage 1）。
*   再用这个“好特征”作为真值，通过 $L_{FM}$ 教会 Denoiser “如何凭空想象出这个特征”（Stage 2）。

## Denoiser在两个阶段的输入对比

**Denoiser（去噪器）** 是 DriveWorld-VLA 作为“世界模型”的核心组件，但它的输入在 **Stage 1** 和 **Stage 2** 是完全不同的，这一点在阅读论文时很容易混淆。


---

### 1. 第一阶段 (Stage 1)：历史条件分支 (History-Conditioned Branch)

在联合训练阶段，Denoiser 的主要任务是**表征学习**和**基于历史的预测**。此时，它并不进行复杂的“生成式想象”，而是更像一个确定性的预测器。

**输入清单 (Inputs):**
1.  **共享潜在特征 ($H_t$)**:
    *   来自 VLM 的输出，融合了图像、文本和历史动作的高层语义信息。
2.  **当前增强 BEV 特征 ($B'_t$)**:
    *   原始 BEV 特征经过 Cross-Attention 与 $H_t$ 交互后的结果。这是“现在的世界状态”。
3.  **历史动作 ($A_{t-1}$)**:
    *   **关键点**：此时输入的是**过去的动作**。

**公式对应**：
$$B_{t+\Delta t} = \text{DENOISER}^1_\theta(H_t, B'_t, A_{t-1})$$

这里其实并没有“噪声” ($x_k$) 作为输入。虽然组件名字叫 Denoiser，但在 Stage 1，它实际上扮演的是一个 **Latent Predictor（潜在状态预测器）** 的角色。它在学习如何从“现在的状态 + 刚才做了什么”推导出“下一刻的状态”。<alphaxiv-paper-citation title="Stage 1 Inputs" page="4" first="B_{t+∆t} = DENOISER^1_θ" last="A_{t-1})," />

---

### 2. 第二阶段 (Stage 2)：动作条件分支 (Action-Conditioned Branch) —— **真正的世界模型**

到了微调阶段，Denoiser 才真正变成了一个**生成式模型（Generative Model）**，采用了 **DiT (Diffusion Transformer)** 架构和 **Flow Matching** 机制。

**输入清单 (Inputs):**

1.  **含噪的未来特征 ($x_k$)**:
    *   这是流匹配/扩散过程的核心。它是一个纯高斯噪声（在推理开始时）或者是一个中间状态的特征（在训练时，$x_k$ 是真实未来特征加噪后的结果）。
2.  **时间步 ($k$)**:
    *   告诉模型现在的噪声水平是多少（例如，“现在是去噪的第 50 步”）。
3.  **条件 1：当前 BEV 特征 ($B'_t$)**:
    *   作为生成的起始点（Condition）。告诉模型“画未来的时候，要基于现在的地图”。
4.  **条件 2：未来动作 ($A_{t+\Delta t}$)**:
    *   **关键差异**：这里输入的不再是历史动作，而是**未来的计划动作**（Ground Truth Future Action）。
    *   这是实现“可控性（Controllability）”的关键：模型必须根据“我打算怎么开”来生成“未来会变成什么样”。

**公式对应**：
$$L_{FM} = ||\text{DIT}_\theta(B'_t, A_{t+\Delta t}, x_k, k) - v_{target}||^2$$

这才是真正的 Denoiser。它的任务是：**$D(Noise, Condition) \rightarrow Future\_Feature$**。
*   **$x_k$** 是画布。
*   **$B'_t$** 是背景草稿。
*   **$A_{t+\Delta t}$** 是画笔指令（“往左画”还是“往右画”）。<alphaxiv-paper-citation title="Stage 2 Inputs" page="4" first="using the BEV state" last="as conditions," />

---

### 总结与对比

为了方便记忆，你可以参考下表：

| 特性 | Stage 1 (联合训练) | Stage 2 (可控微调) |
| :--- | :--- | :--- |
| **核心任务** | 表征对齐与预热 (Warm-up) | **生成式世界建模 (World Modeling)** |
| **动作输入** | **历史动作 ($A_{t-1}$)** | **未来动作 ($A_{t+\Delta t}$)** |
| **噪声输入** | 无 (确定性预测) | **有 ($x_k$)** (随机生成) |
| **VLM 特征 ($H_t$)** | **直接输入** | **间接输入** (通过 $B'_t$ 隐含包含) |
| **架构行为** | 回归 (Regression) | 去噪/流匹配 (Denoising/Flow Matching) |

DriveWorld-VLA 巧妙地复用了同一个网络结构（Denoiser），但在不同阶段赋予了它不同的输入和任务，从而实现了从“理解历史”到“想象未来”的平滑过渡。


## Denoiser的详细结构


**Denoiser 实际上包含了“两个分支”（Two Branches），它们共享输入特征，但结构和任务不同。**

并不是同一个 DiT 网络在 Stage 1 和 Stage 2 仅仅换了输入。实际上，Stage 1 用的是一个**简单的预测头**，而 Stage 2 启用的是**真正的 DiT 生成头**。

论文 3.1 节的原文描述：
> "DENOISER comprises a **history-conditioned branch** and a **future action-conditioned branch**. At this stage [Stage 1], **only history-conditioned branch is activated**..." <alphaxiv-paper-citation title="Two Branches" page="4" first="DENOISER comprises a" last="branch is activated" />

### 1. 它是“双头”设计 (Dual-Branch Architecture)


#### 员工 A：确定性预测器 (The Deterministic Predictor) —— **只在 Stage 1 工作**
*   **身份**：这是论文中提到的 "History-conditioned branch"。
*   **结构**：它**不是** DiT。它通常是一个简单的 MLP（多层感知机）或者简单的 Transformer Block。
*   **工作方式**：直来直去。
    *   输入：$H_t$ (VLM特征) + $B_t$ (BEV特征) + $A_{t-1}$ (历史动作)。
    *   输出：$B_{t+\Delta t}$ (未来BEV特征)。
*   **任务**：它的任务不是“生成”，而是**回归（Regression）**。它在做数学题，试图找出 $f(history) = future$ 的最优解。
*   **为什么不需要 DiT？** 因为 Stage 1 的目的是为了训练前面的 VLM 和 BEVFormer 提取出高质量的特征。用一个简单的头（Head）可以快速传导梯度，让特征与语义对齐。

#### 员工 B：生成式画师 (The Generative Artist / DiT) —— **在 Stage 2 登场**
*   **身份**：这是论文中提到的 "Future action-conditioned branch"，也是图 3 展示的 **DiT (Diffusion Transformer)**。
*   **结构**：这是一个完整的 DiT 架构，包含 Timestep Embedding, Cross-Attention 等复杂组件。
*   **工作方式**：它是**流匹配（Flow Matching）**模型。
    *   输入：$B'_t$ (当前特征) + $A_{t+\Delta t}$ (未来动作) + $x_k$ (噪声)。
    *   输出：去噪后的特征。
*   **任务**：它的任务是**想象（Imagination）**。它在画画，试图从随机噪声中画出符合动作指令的未来。

---

### 2. 它们共享什么？(What is Shared?)

它们并没有网络结构上或者参数上的共享,但它们共享了**同一个大脑**（这也是论文题目 "Unified" 的由来）：
它们共享的是 **输入特征 $B'_t$**。

*   **Stage 1**：训练 VLM 和 Cross-Attention 层，让 $B'_t$ 包含足够的信息（语义+几何），足以被“员工 A”用来预测未来。即未来的分割结果.
*   **Stage 2**：冻结 VLM，直接利用这些已经训练好的高质量 $B'_t$ 特征，训练“员工 B”（DiT）去学习生成规律。

### 3. 为什么要这样设计？(The "Why")

如果你直接在 Stage 1 就上 DiT，会面临两个大坑：
1.  **训练坍塌**：VLM 还没学会怎么看图，DiT 就开始学画画，结果就是“瞎子指挥聋子”，很难收敛。
2.  **梯度混乱**：DiT 的去噪损失（Denoising Loss）很难直接指导 VLM 学习图像的语义理解。

**作者的策略是“分而治之”：**
*   先用**确定性任务（Stage 1）**把特征（Latent Space）练好。
*   再在**这个高质量特征空间上（Stage 2）**训练生成模型。

### 总结
**Denoiser 是一个包含双分支的复合模块。Stage 1 使用“历史分支”进行确定性回归，Stage 2 使用“动作分支”（即 DiT）进行生成式流匹配。它们共享的是底层的特征表示（Latent Representation）。** <alphaxiv-paper-citation title="Stage 2 Architecture" page="4" first="Subsequently, the second" last="denoising process," />
嗯,我认为这里的两个Branchs, 对应着 MILE 中的 分割head,和重建head. 在这里的作用就是一个预测未来的bev, 一个预测未来的分割结果;另外 Stage1, formally来看的话,像是训练了一个BEV-VAE的感觉.
encoder是bevformer, decoder是 seg-head; stage2相当于就是把BEV-VAE给固定住, 然后让模型专注于预测 BEV latents(seg head的输入).

## Action 是连续值回归,即MSE


## Reward Model 

**奖励模型 (Reward Model)** 是 DriveWorld-VLA 在 **Stage 3** 实现闭环微调（Closed-loop Refinement）的“裁判员”。

如果没有这个裁判员，模型只能“瞎想”未来，却不知道哪个未来是好的、安全的。

根据论文 **3.3 节 (Stage 3: Future-Guided Evaluation & Refinement)** 以及附录中的细节，这个 Reward Model 并不是一个简单的数学公式，而是一个**经过预训练的神经网络打分器**。

---

### 1. Reward Model 的架构与工作原理

DriveWorld-VLA 并没有从头训练一个 Reward Model，而是**利用了现有的规则和轻量级网络**。

在论文中，作者明确提到使用了 **[UniAD / VAD 等以前工作中的评测指标]** 作为设计灵感，但具体的实现是一个**可微的评分函数 (Differentiable Scoring Function)**。

*   **输入**:
    1.  **生成的未来轨迹 ($A_{pred}$)**: VLA 输出的动作序列。
    2.  **生成的未来环境 ($B_{gen}$)**: World Model 想象出来的未来 BEV 特征（解码后包含车道线、障碍物位置）。
    3.  **真实地图信息 (Map Prior)**: 这一步通常需要借用高清地图或感知的地图作为基准。

*   **输出**: 一个标量分数 $R$ (Reward Score)。分数越高，代表动作越好。

---

### 2. 具体包含哪些 Reward? (The Reward Composition)

论文在 **3.3 节** 和 **4.2 节** 的实验设置中，隐含或明确指出了以下三个核心奖励维度。这些是自动驾驶规划任务中的“黄金三原则”：

#### A. 安全性奖励 ($R_{safety}$ / Collision Avoidance) —— **最重要**
这是最基础的红线。
*   **计算逻辑**:
    检查**预测轨迹**是否与**想象中的障碍物（Occupancy）**发生重叠。
*   **操作**:
    1.  将生成的 BEV 特征解码为 **占用栅格图 (Occupancy Grid Map)**。
    2.  将自车的轨迹点映射到这个栅格图上。
    3.  如果轨迹点落在“被占用”的格子（即有障碍物）上，给予巨大的惩罚（Negative Reward）。
*   **目的**: 防止碰撞。

#### B. 道路合规性奖励 ($R_{compliance}$ / Drivable Area)
车不仅不能撞人，还不能开出路面。
*   **计算逻辑**:
    检查**预测轨迹**是否在**可行驶区域 (Drivable Area)** 内。
*   **操作**:
    1.  将生成的 BEV 特征解码为 **道路分割图 (Road Segmentation Map)**。
    2.  计算轨迹点落在“非道路区域”（如人行道、草地）的比例。
    3.  越界越多，惩罚越大。
*   **目的**: 保证车辆走在路中间，不逆行，不出界。

#### C. 舒适度与平滑性奖励 ($R_{comfort}$ / Kinematics)
为了防止“画龙”或急刹急停。
*   **计算逻辑**:
    基于物理动力学公式计算。
*   **指标**:
    1.  **加加速度 (Jerk)**: 加速度的变化率。越小越好。
    2.  **曲率变化率 (Curvature Rate)**: 方向盘打得有多快。
    3.  **横向加速度 (Lat. Acc)**: 转弯是否太急。
*   **目的**: 让轨迹平滑，符合人类驾驶习惯。

---

### 3. Reward Model 的“预训练”是怎么回事？

“pretrained”，这里有两个层面的含义：

1.  **基于规则的计算 (Rule-based Calculation)**:
    大部分所谓的 Reward 其实不需要神经网络去“学”，而是直接用**物理公式**算出来的（比如碰撞检测）。在 Stage 3，这些规则被封装成一个 Python 函数，作为 Ground Truth Reward。

2.  **基于学习的价值函数 (Learned Value Function) —— 论文的高级玩法**:
    DriveWorld-VLA 在 Stage 3 还引入了一个 **Critique Model (评论家模型)**。
    *   这个模型是一个轻量级的 Transformer 或 MLP。
    *   **预训练数据**: 使用 Stage 2 生成的大量 (State, Action, Next_State) 数据对。
    *   **标签**: 用上述规则算出来的 $R_{rule}$ 作为标签。
    *   **作用**: 在推理或微调时，用这个神经网络直接预测 $R_{pred}$，比每次都跑复杂的物理检测要快，且可微（Differentiable），方便反向传播梯度给 VLA。

### 总结公式

最终的 Reward 函数通常是加权求和：

$$R_{total} = w_1 \cdot R_{safety} + w_2 \cdot R_{compliance} + w_3 \cdot R_{comfort}$$

*   **$R_{safety}$**: 碰撞惩罚（权重最大）。
*   **$R_{compliance}$**: 压线惩罚。
*   **$R_{comfort}$**: 急转弯/急刹惩罚。

这其实就是**强化学习（RL）**中的 Reward Shaping。DriveWorld-VLA 的高明之处在于，它用**世界模型想象出的未来**来计算这个 Reward，而不是必须要在真实世界里撞一次车才能拿到负反馈。这就是“在脑海中预演，并在脑海中修正”的过程。<alphaxiv-paper-citation title="Reward Function" page="5" first="Specifically, we employ" last="preference optimization." />

