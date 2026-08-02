---
title: SGImagineNav
date: 2025-08-09
categories: [VLN]
---

# SGImagineNav：基于场景图想象世界模型的具身导航

> **论文**：*Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation*  
> **作者**：Junzhe Wu, Ruihan Xu, Hang Liu, Avery Xi, Henry X. Liu, Ram Vasudevan, Maani Ghaffari  
> **发布时间**：2025-08-09  
> **arXiv**：https://arxiv.org/abs/2508.06990

---

## 摘要

SGImagineNav 提出了一种基于**符号世界模型（Symbolic World Modeling）**的想象式导航框架。与仅依赖历史观测的现有方法不同，SGImagineNav 维护一个**层次化场景图（Hierarchical Scene Graph）**，并利用大语言模型（LLM）预测和探索环境的未见部分，从而在未见环境中主动估计目标位置。在 NavVerse 基准的零样本评估中，SGImagineNav 作为模块化方法的代表，展现出**最强的安全性能**（最低碰撞率），但成功率低于端到端 VLA 方法。

---

## 一、研究背景与动机

### 1.1 问题定义

语义导航要求智能体在**未见环境**中导航到指定目标。传统方法主要分为两类：

- **端到端方法**：直接学习"观测→动作"的映射，缺乏对环境的显式建模；
- **模块化方法**：先建图再规划，但通常只利用已观测区域，无法主动推断未探索区域。

### 1.2 核心洞察

SGImagineNav 的核心论点是：

> **人类导航时会主动想象未见区域的可能布局，从而更高效地搜索目标。** 例如，要找厨房时，人类会推断"厨房通常在卧室附近"，而不是盲目随机探索。

因此，智能体应该能够：
1. 基于已有观测构建全局环境表示；
2. 利用常识和语义知识**想象**未见区域的内容；
3. 根据想象结果**主动**规划探索路径。

---

## 二、方法架构

### 2.1 整体框架

```
当前观测 (RGB / 深度 / 语义)
        ↓
[场景图构建模块]  →  层次化场景图 (Hierarchical Scene Graph)
        ↓
[LLM 想象模块]    →  预测未见区域的语义内容
        ↓
[自适应导航策略]  →  利用语义捷径 / 探索未知区域
        ↓
        动作输出
```

### 2.2 层次化场景图

SGImagineNav 维护一个**持续演化的层次化场景图**，包含多个抽象层级：

| 层级 | 内容 | 作用 |
|---|---|---|
| 物体层 | 检测到的物体及其属性 | 局部语义 grounding |
| 房间层 | 房间类型与布局 | 中层空间推理 |
| 建筑层 | 建筑结构与功能分区 | 全局拓扑理解 |

场景图不是静态地图，而是**随着探索动态更新**的世界模型。

### 2.3 LLM 驱动的想象模块

SGImagineNav 利用 LLM 的常识推理能力，对**未见区域**进行预测：

- **语义补全**：根据已有场景推断未探索区域可能包含的物体（如"卧室通常有床和衣柜"）；
- **目标定位**：基于目标物体的语义属性，推断其最可能出现的位置；
- **路径规划**：结合想象的环境布局，生成更高效的探索路径。

### 2.4 自适应导航策略

SGImagineNav 采用**自适应策略**，在两种模式之间切换：

1. **利用语义捷径（Exploit Semantic Shortcuts）**：当 LLM 预测目标位置置信度高时，直接导航到预测位置；
2. **探索未知区域（Explore Unknown Regions）**：当预测不确定时，优先探索可能包含目标的新区域。

这种策略平衡了"利用已有知识"和"探索新信息"的 trade-off。

---

## 三、实验评估

### 3.1 在 NavVerse 上的零样本表现

NavVerse 基准将 SGImagineNav 作为**模块化方法**的代表进行评估。关键结果：

#### ObjNav

| 指标 | SGImagineNav | 对比 |
|---|---|---|
| All Scenes SR | 6.42% | UniNaVid 11.62% |
| Collision Rate | **0.21** | UniNaVid 0.31 |
| Avg. Distance to Obstacles | **0.98** | UniNaVid 0.81 |
| Navigable Surface Ratio | **0.16** | UniNaVid 0.12 |

#### PlaceNav

| 指标 | SGImagineNav | 对比 |
|---|---|---|
| All Scenes SR | 4.88% | UniNaVid 11.38% |
| Collision Rate | **0.05** | UniNaVid 0.16 |
| Avg. Distance to Obstacles | **1.91** | UniNaVid 1.98 |

### 3.2 关键发现

1. **安全性最强**：SGImagineNav 在所有任务上的碰撞率都显著低于 RL/VLA 方法；
2. **成功率偏低**：模块化方法在零样本设置下成功率明显低于端到端 VLA；
3. **过度保守**：低碰撞率可能源于过于保守的探索策略，导致 timeout 和 wrong-goal 比例较高。

---

## 四、优势与局限

### 4.1 优势

- **显式世界建模**：场景图提供可解释的环境表示，便于调试和分析；
- **安全性高**：保守的导航策略显著降低碰撞风险；
- **无需训练**：零样本即可工作，不依赖大量演示数据。

### 4.2 局限

- **成功率不足**：过度依赖 LLM 想象可能导致错误的目标定位；
- **计算开销**：LLM 推理和场景图维护带来较高延迟；
- **零样本泛化有限**：在复杂 indoor-to-outdoor 场景中，想象模块的准确性下降。

---

## 五、与相关工作的关系

| 工作 | 与 SGImagineNav 的关系 |
|---|---|
| **传统 SLAM + 规划** | SGImagineNav 用 LLM 想象替代了部分几何推理 |
| **ImagineNav / ImagineNav++** | 同样使用"想象"思想，但 SGImagineNav 基于场景图而非图像生成 |
| **模块化 VLA** | SGImagineNav 是模块化代表，与端到端 VLA 形成对比 |
| **NavVerse** | SGImagineNav 是 NavVerse 评估的基线之一，展现了模块化方法的安全性优势 |

---

## 六、结论

SGImagineNav 证明了**符号世界模型 + LLM 想象**在语义导航中的价值。它通过显式构建层次化场景图和利用 LLM 预测未见区域，实现了高安全性的零样本导航。然而，其成功率受限于想象准确性和保守策略，表明**纯模块化方法在复杂跨上下文导航中仍面临挑战**。

对于实际应用，SGImagineNav 的启示是：**安全性和可解释性很重要，但需要与更强的语义 grounding 能力结合**，才能在不牺牲成功率的前提下保持安全。

---

*参考：Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation (arXiv:2508.06990, Aug 9, 2025)*
