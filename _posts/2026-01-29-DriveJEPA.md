---
layout: post
title: DriveJEPA
date: 2026-01-29
categories: [WorldModel]
toc: 
    sidebar: left
---

[TOC]
# DriveJEPA

[论文链接](https://arxiv.org/abs/2601.22032)


# Drive-JEPA —— 视频联合嵌入预测架构与多模态轨迹蒸馏在端到端驾驶中的应用

**论文题目：** Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving

---

## 第一部分：背景与动机

### 1.1 端到端自动驾驶的现状与瓶颈

在深入 Drive-JEPA 之前，我们需要回顾端到端自动驾驶（End-to-End Autonomous Driving）的核心理念及其当前面临的两大主要挑战。

*   **从模块化到端到端：**
    传统自动驾驶采用模块化管道（感知 $\to$ 预测 $\to$ 规划），虽然可解释性强，但存在累积误差和信息丢失。端到端方法试图通过统一的神经网络，直接将原始传感器数据映射到驾驶行为，旨在减少信息丢失并利用大规模数据。 <alphaxiv-paper-citation title="Introduction" page="1" first="End-to-end autonomous driving" last="neural model." />
*   **挑战一：表征学习的局限性（Representation Bottleneck）**
    目前端到端模型通常依赖视频预训练来理解场景。主流的“世界模型”（World Models）方法分为两类：
    1.  **视频生成式（Video-generative）：** 试图重建或生成像素级视频。这计算量巨大，且过于关注视觉细节（如树叶的纹理），而这些细节对驾驶决策往往无关紧要。
    2.  **潜空间动力学（Latent World Models）：** 预测特征的演变。但这通常只作为辅助目标，并未展示出随着预训练规模扩大而带来的显著性能提升。 <alphaxiv-paper-citation title="World Models" page="1" first="However, pretraining video" last="limited improvements." />
*   **挑战二：多模态行为的监管缺失（Supervision Bottleneck）**
    驾驶本质上是多模态的（Multimodal）。在一个路口，左转、直行或右转可能都是合法的。然而，人类驾驶数据集通常每一种场景只提供**一条**轨迹（Ground Truth）。如果我们只用这一条轨迹做监督，模型会丢失其他可行解的多样性，导致在未见过的场景中泛化能力差。 <alphaxiv-paper-citation title="Ambiguity" page="1" first="This limitation is" last="multimodal behaviors." />

### 1.2 核心解决方案概览

Drive-JEPA 针对上述两个痛点提出了针对性的解决方案：

1.  **针对表征学习：** 引入 **V-JEPA (Video Joint-Embedding Predictive Architecture)** 进行预训练。它不重建像素，而是在特征空间预测未来，从而高效地学习对规划有用的语义特征。
2.  **针对多模态监管：** 提出 **多模态轨迹蒸馏 (Multimodal Trajectory Distillation)**。利用仿真器（Simulator）生成多条安全轨迹作为“伪教师”，补充单一的人类数据。

---

## 第二部分：Drive-JEPA 方法论详解

### 2.1 架构总览 (Figure 2 解析)

Drive-JEPA 的框架包含三个核心组件：
1.  **驾驶视频预训练 (Driving Video Pretraining)**：使用 V-JEPA 学习视觉编码器。
2.  **基于锚点的提案生成 (Waypoint-anchored Proposal Generation)**：生成候选轨迹。
3.  **多模态轨迹蒸馏与选择 (Distillation & Selection)**：利用仿真器数据优化轨迹分布，并选择最佳路径。 <alphaxiv-paper-citation title="Framework" page="2" first="Specifically, our framework" last="Trajectory Selection." />

### 2.2 核心组件一：V-JEPA 驾驶视频预训练

这是该论文的一大亮点，它将 LeCun 提出的 JEPA 架构成功适配到了驾驶领域。

*   **原理：** V-JEPA 不同于生成式模型（如 MAE 或 VideoMAE），它不预测被遮挡的像素，而是预测被遮挡区域的**潜在特征（Latent Representation）**。
*   **流程：**
    1.  **输入：** 连续的驾驶视频帧。
    2.  **掩码策略：** 随机遮挡视频的时空块。
    3.  **目标：** 编码器提取可见部分的特征，预测器（Predictor）根据这些特征预测被遮挡部分的特征表示。
*   **优势：** 这种方法避免了像素级重建的高昂计算成本，专注于学习场景的高层语义（如物体运动、道路拓扑），这与规划任务（Planning）更加对齐。
*   **成果：** 作者在 208 小时的视频数据上进行了预训练，相比之前的像素重建方法，计算效率更高。 <alphaxiv-paper-citation title="V-JEPA" page="2" first="In the first" last="collapse prevention." />

### 2.3 核心组件二：基于锚点的提案生成

有了强大的特征提取器后，如何生成规划轨迹？Drive-JEPA 采用了一种“提案-选择”（Proposal-Selection）的范式，但这与传统的固定词表不同。

*   **动态提案：** 模型不是从固定的轨迹库中选，而是动态生成提案。
*   **机制：**
    1.  **查询初始化：** 使用自车状态（Ego Status）初始化一组可学习的查询向量（Queries）。
    2.  **迭代优化：** 使用 Deformable Attention 机制，这些查询向量在 BEV（鸟瞰图）特征图上聚合信息，并迭代地修正轨迹锚点（Anchors）。
    3.  **输出：** 输出 $N$ 条候选轨迹，每条轨迹由一系列时空路点 $(x, y, \text{heading})$ 组成。 <alphaxiv-paper-citation title="Proposal Gen" page="2" first="In the second" last="refine proposals iteratively." />

### 2.4 核心组件三：多模态轨迹蒸馏 (核心创新)

这是解决“单一人类轨迹监管”问题的关键。

*   **问题：** 如果只用人类的一条轨迹做 Loss（如 L2 距离），模型会倾向于坍缩到单一模态，或者输出多模态的平均值（这是不安全的）。
*   **解决方案：** 引入仿真器作为“老师”。
    1.  **仿真生成：** 在训练阶段，利用仿真器（Simulator）基于当前场景生成大量随机轨迹。
    2.  **筛选：** 筛选出那些符合动力学约束且无碰撞的“高质量”轨迹。
    3.  **蒸馏：** 将这些轨迹作为额外的监督信号。模型生成的 $N$ 条提案不仅要逼近人类轨迹，还要覆盖仿真器生成的这些合法的多模态轨迹。
*   **意义：** 这极大地丰富了训练信号，教会模型“除了人类这样做，那样做也是安全的”。 <alphaxiv-paper-citation title="Distillation" page="2" first="proposals are supervised" last="from the simulator." />

### 2.5 动量感知轨迹选择 (Momentum-aware Selection)

生成了多条轨迹后，如何选择最终执行的那一条？

*   **评分器：** 模型预测每条轨迹的安全性（碰撞风险）、舒适度和交通规则符合度。
*   **动量机制：** 为了防止控制信号在帧与帧之间剧烈跳变（造成“画龙”现象），引入了动量感知惩罚。当前选择的轨迹应与上一帧规划的轨迹保持一定的一致性。
*   **公式逻辑：** 最终得分 = 预测质量得分 - 轨迹形变惩罚。 <alphaxiv-paper-citation title="Selection" page="2" first="incorporates a momentum-aware" last="trajectory distortion." />

---

## 第三部分：实验结果与讨论

### 3.1 实验设置

*   **数据集：** NAVSIM v1 和 NAVSIM v2（基于 nuPlan 的大规模闭环仿真评测基准），以及 Bench2Drive。
*   **评估指标：** PDMS (Predictive Driving Model Score)。这是一个综合指标，不仅仅看轨迹与人类的重合度（L2 error），更看重闭环仿真中的安全性、舒适度和进度。 <alphaxiv-paper-citation title="Evaluation" page="2" first="We validate Drive-JEPA" last="Jia et al., 2024)." />

### 3.2 核心结果分析

1.  **State-of-the-Art (SOTA) 表现：**
    *   在 NAVSIM v1 上，Drive-JEPA 达到了 **93.3 PDMS**。
    *   在 NAVSIM v2 上，达到了 **87.8 EPDMS**。
    *   **结论：** 这刷新了目前的最佳成绩，证明了该框架的有效性。 <alphaxiv-paper-citation title="Results" page="1" first="The complete Drive-JEPA" last="state-of-the-art." />

2.  **V-JEPA 的有效性（Perception-Free Setting）：**
    *   为了验证 V-JEPA 预训练是否真的有用，作者在一个纯感知无关（Perception-Free）的设置下进行了测试（即不使用检测框等中间任务，纯端到端）。
    *   **结果：** 仅使用 V-JEPA 预训练的 ViT 编码器配合简单的 Transformer 解码器，就比之前的方法高出 **3 PDMS**。
    *   **解读：** 这有力地证明了 V-JEPA 学到了对规划极其关键的特征，而不仅仅是视觉特征。 <alphaxiv-paper-citation title="Perception-Free" page="1" first="outperforms prior methods" last="perception-free setting." />

3.  **多模态蒸馏的贡献：**
    *   消融实验显示，引入多模态轨迹蒸馏显著提升了驾驶质量。尤其是在 Bench2Drive 这种复杂场景较多的测试中，模型能够处理更多样化的路况，避免了单一模仿人类可能导致的死板或危险行为。 <alphaxiv-paper-citation title="Ablation" page="2" first="Multimodal Trajectory Distillation" last="multimodal trajectories." />

### 3.3 总结与讨论

**总结：**
Drive-JEPA 成功地将非生成式视频预训练（V-JEPA）与基于仿真器的知识蒸馏结合起来。它不仅提升了特征的鲁棒性，还解决了端到端学习中数据分布稀疏的问题。

**思考题：**
1.  **V-JEPA vs. 生成式模型：** 既然生成式模型（如 Sora）展现了强大的物理世界理解能力，为什么 Drive-JEPA 认为像素级生成对驾驶是不必要的？未来随着算力提升，这一观点会改变吗？
2.  **仿真器的依赖：** 该方法依赖仿真器生成“伪教师”轨迹。如果仿真器本身的物理模型不准确（Sim-to-Real Gap），会对实车部署造成什么影响？如何缓解？
3.  **安全性保障：** 虽然使用了多模态蒸馏，但端到端模型仍然是个“黑盒”。在实际部署中，我们如何为这种基于神经网络的规划器加上确定性的安全围栏（Safety Guardrails）？



# 思考题解答


好的，这是为您准备的针对那三个讨论题的详细回答。这些回答结合了 **Drive-JEPA** 论文的具体内容以及更广泛的自动驾驶领域知识，旨在帮助学生深入理解背后的工程权衡和理论基础。

---

### 讨论题 1：V-JEPA vs. 生成式模型（如 Sora/VideoMAE）

**问题回顾：** 既然生成式模型展现了强大的物理世界理解能力，为什么 Drive-JEPA 认为像素级生成对驾驶是不必要的？未来随着算力提升，这一观点会改变吗？

**详细解答：**

1.  **信息密度与相关性的权衡（Signal-to-Noise Ratio）：**
    *   **核心观点：** Drive-JEPA 的核心论点是“驾驶决策不需要像素级的完美”。在驾驶场景中，像素空间包含了大量与规划无关的高频信息（例如：树叶随风摆动的纹理、路边广告牌的具体内容、云层的形状）。生成式模型（Generative Models）强迫网络去学习并重建这些细节，这不仅浪费了巨大的计算资源，还可能导致模型“过拟合”到视觉细节上，而忽略了物体间的相对运动、遮挡关系等对驾驶至关重要的**语义信息**。
    *   **论文佐证：** 作者明确指出，像素级目标（Pixel-level objective）会带来沉重的计算负担，并且可能过分强调与决策无关的视觉细节。 <alphaxiv-paper-citation title="Generative Limitations" page="1" first="pixel-level objective incurs" last="to decision making." />
    *   **V-JEPA 的优势：** V-JEPA 在 **特征空间（Latent Space)**进行预测。它实际上是在学习一种“抽象”，即只保留那些在时间上具有预测性的信息（通常是物体的位置、类别、运动状态），而丢弃不可预测的噪音（光照的随机闪烁、纹理细节）。这种抽象恰恰是规划模块（Planner）最需要的。(我觉得这是做和规划相关的worldmodel要一直铭记的初衷,即到底什么样的信息是planner模块需要的。)

2.  **计算效率与实时性（Efficiency）：**
    *   端到端自动驾驶模型需要在车端芯片上实时运行。生成式模型的解码器（Decoder）通常非常庞大，推理延迟高。而 V-JEPA 的预测头（Predictor）是轻量级的，且在推理时甚至不需要运行预测头（只用 Encoder），这使得它在部署时极其高效。

3.  **未来的展望（Future Perspective）：**
    *   **算力不是唯一瓶颈：** 即使未来算力无限，**Yann LeCun**（JEPA 的提出者）的理论认为，**在不确定性极高的世界中进行像素级预测在数学上是不适定（Ill-posed）的**。例如，预测一辆车“可能会左转”是容易的，但预测这辆车左转时每一个像素的 RGB 值是非常难且没必要的。
    *   **可能的融合：** 未来更有可能出现的趋势是**混合架构**。即底层使用 V-JEPA 学习物理常识和动力学，上层使用轻量级的生成模块用于“可解释性可视化”（例如生成未来场景给人类安全员看，而不是给控制算法看）。只要驾驶的本质是“做决策”而非“画图”，特征空间学习（Latent Learning）大概率仍是主流。

---

### 讨论题 2：仿真器的依赖与 Sim-to-Real Gap

**问题回顾：** 该方法依赖仿真器生成“伪教师”轨迹。如果仿真器本身的物理模型不准确，会对实车部署造成什么影响？如何缓解？

**详细解答：**

1.  **Drive-JEPA 中仿真器的角色：**
    *   首先需要明确，Drive-JEPA 使用仿真器并不是为了生成**图像（Sensor Data）**，而是为了生成**轨迹（Future Trajectories）**。这是一个关键的区别。
    *   论文提到，他们使用仿真器来生成“多样化的轨迹”，并根据动力学约束筛选出无碰撞的路径作为额外的监督信号。 <alphaxiv-paper-citation title="Sim Distillation" page="1" first="distills diverse" last="human trajectories." />

2.  **Sim-to-Real Gap 的具体表现：**
    *   **动力学差异（Dynamics Gap）：** 仿真器通常使用简化的车辆动力学模型（如单车模型 Bicycle Model）。然而，真实车辆在高速转弯、湿滑路面或轮胎磨损情况下的响应是高度非线性的。如果模型在训练时认为“以 80km/h 速度急转弯是安全的（因为仿真器没报错）”，在实车上可能会导致侧滑或失控。
    *   **行为逻辑差异（Behavioral Gap）：** 仿真器中的其他车辆（NPC）通常遵循规则（Rule-based），行为比较死板。这可能导致模型学不到如何处理真实世界中人类驾驶员的博弈、犹豫或违规行为。

3.  **缓解策略：**
    *   **保守性筛选（Conservative Filtering）：** 在生成蒸馏数据时，必须设置比真实物理极限更严格的阈值。例如，如果物理极限侧向加速度是 $0.8g$，仿真筛选时可能限制在 $0.5g$。
    *   **闭环微调（Closed-loop Finetuning）：** 在仿真训练后，必须在真实数据上进行少量的微调，或者使用**逆强化学习（Inverse Reinforcement Learning）**，让模型重新对齐人类的驾驶风格。
    *   **混合监督（Hybrid Supervision）：** Drive-JEPA 并没有完全抛弃人类数据，而是将“仿真轨迹”与“人类轨迹”结合。人类轨迹保证了风格像人（拟人化），仿真轨迹提供了多样性和安全性边界（鲁棒性）。这种组合本身就是一种缓解 Sim-to-Real Gap 的手段。 <alphaxiv-paper-citation title="Selection Mechanism" page="1" first="momentum-aware selection" last="safe behavior." />

---

### 讨论题 3：安全性保障与安全围栏（Safety Guardrails）

**问题回顾：** 虽然使用了多模态蒸馏，但端到端模型仍然是个“黑盒”。在实际部署中，我们如何为这种基于神经网络的规划器加上确定性的安全围栏？

**详细解答：**

1.  **神经网络的概率本质 vs. 驾驶的确定性要求：**
    *   Drive-JEPA 输出的是轨迹的概率分布或评分。神经网络本质上是基于统计的，它无法保证 100% 不犯错（例如在长尾分布的场景下）。

2.  **Drive-JEPA 内部的软约束（Soft Guardrails）：**
    *   论文中提到的 **“动量感知选择机制”（Momentum-aware selection mechanism）** 就是一种内置的软约束。它强迫模型输出的轨迹在时间上是连续的，防止出现“上一帧左转，下一帧突然右转”的危险抖动。 <alphaxiv-paper-citation title="Momentum Selection" page="1" first="incorporates a momentum-aware" last="trajectory distortion." />

3.  **外部的硬约束（Hard Guardrails）—— 工业界标准做法：**
    在实际部署中，通常采用 **“规划-校验-回退”（Plan-Check-Fallback）** 架构：
    *   **第一层：模型规划（Planner）：** Drive-JEPA 生成 Top-K 条候选轨迹。
    *   **第二层：规则校验（Rule-based Checker）：** 这是一个轻量级、确定性的代码模块。它利用高精地图和感知边界框，对候选轨迹进行物理校验。
        *   *碰撞检测：* 轨迹是否与障碍物重叠？
        *   *动力学检测：* 曲率是否过大？加速度是否超标？
        *   *交通规则：* 是否闯红灯或逆行？
    *   **第三层：轨迹截断或回退（Fallback）：**
        *   如果 Drive-JEPA 的最优轨迹未通过校验，则顺次检查第二、第三优轨迹。
        *   如果所有轨迹都失败，系统将触发**最小风险策略（Minimum Risk Maneuver, MRM）**，通常是紧急制动或缓慢靠边停车。

4.  **总结：**
    Drive-JEPA 负责“聪明地驾驶”（舒适、高效、拟人），而外部的规则系统负责“不撞车”。

# 其他

1. 在做Drive-JEPA的pre-training的时候, 是在独立的视频片段上面进行的; 但是在下游规划任务中, 模型的输入是 "Multi-View" 的图像序列.
2. Vision Encoder 用V-JEPA训完之后, 会进行微调, 即只是作为初始化的weights. 预训练的目的是为了让编码器先"看懂"视频中的物理规律(即获得更好的Representation).
论文中的 Table 7 对比了不同的vision pretrain model 的效果.
