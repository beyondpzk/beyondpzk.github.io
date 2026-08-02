---
title: PoliFormer
date: 2024-06-28
categories: [VLN]
---

# PoliFormer：基于大规模 On-Policy RL 的 Transformer 导航器

> **论文**：*PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators*  
> **作者**：Kuo-Hao Zeng, Zichen Zhang, Kiana Ehsani, Rose Hendrix, Jordi Salvador, Alvaro Herrasti, Ross Girshick, Aniruddha Kembhavi, Luca Weihs  
> **机构**：PRIOR @ Allen Institute for AI  
> **发布时间**：2024-06-28  
> **arXiv**：https://arxiv.org/abs/2406.20083  
> **项目主页**：https://poliformer.allen.ai/

---

## 摘要

PoliFormer 是一个仅使用 RGB 观测的室内导航智能体，通过**大规模端到端强化学习**训练，能够在真实世界中零样本泛化，尽管完全在仿真环境中训练。它采用**视觉 Transformer 编码器 + 因果 Transformer 解码器**的架构，支持长期记忆和推理。PoliFormer 在 CHORES-S 基准上取得 **85.5% 的目标导航成功率**，比此前最佳方法提升 **28.5%**，并可在无需微调的情况下扩展到目标跟踪、多目标导航和开放词汇导航等下游任务。

---

## 一、研究背景与动机

### 1.1 室内导航的瓶颈

传统室内导航方法面临两个主要瓶颈：

1. **仿真到真实的鸿沟**：在仿真中训练的模型往往无法直接部署到真实机器人；
2. **性能平台期**：现有方法在复杂基准上的成功率长期停滞在 50%~60%。

### 1.2 核心洞察

PoliFormer 的核心论点是：

> **大规模 on-policy RL + Transformer 架构，可以突破室内导航的性能平台期，实现真正的零样本仿真到真实迁移。**

关键设计选择：
- **RGB-only**：不依赖深度或里程计，更贴近真实部署条件；
- **On-policy RL**：直接从环境交互中学习，避免离线数据的分布偏移；
- **Transformer**：利用自注意力机制建模长期依赖和历史记忆。

---

## 二、模型架构

### 2.1 整体结构

```
RGB 观测序列
    ↓
[Vision Transformer Encoder]  →  视觉特征
    ↓
[Causal Transformer Decoder]  →  长期记忆与推理
    ↓
[Actor-Critic Head]           →  动作分布 + 价值估计
```

### 2.2 Vision Transformer Encoder

- 将输入图像分割为 patch，通过 ViT 提取视觉特征；
- 支持多帧历史观测的时序建模；
- 仅使用 RGB，不依赖深度图或语义分割。

### 2.3 Causal Transformer Decoder

- 采用因果掩码（causal mask），确保当前动作只依赖历史观测；
- 支持**长期记忆**：通过自注意力机制回顾整个 episode 的历史；
- 使模型能够推理"我之前去过哪里"和"目标可能在哪里"。

### 2.4 大规模训练

| 配置 | 数值 |
|---|---|
| 训练交互次数 | 数亿次（hundreds of millions） |
| 并行环境 | 多机器并行 rollout |
| 训练策略 | On-policy RL（PPO 类算法） |
| 优化器 | AdamW |
| 学习率 | 余弦衰减 |

---

## 三、实验评估

### 3.1 主要基准结果

| 基准 | 成功率 | 相比此前最佳 | 备注 |
|---|---|---|---|
| **CHORES-S** | **85.5%** | +28.5% | 目标导航，最大提升 |
| ObjectNav (Habitat) | SOTA | - | 标准物体导航 |
| Multi-Object Nav | SOTA | - | 多目标导航 |
| Open-Vocabulary Nav | 强 | - | 零样本开放词汇 |

### 3.2 真实世界泛化

PoliFormer 在两种真实机器人上验证：

| 机器人 | 任务 | 表现 |
|---|---|---|
| **LoCoBot** | 室内目标导航 | 无需微调，直接部署 |
| **Stretch RE-1** | 室内目标导航 | 无需微调，直接部署 |

**关键结论**：PoliFormer 是**首个在纯仿真训练后，无需任何适应即可在真实机器人上达到高性能的 RGB-only 导航模型**。

### 3.3 在 NavVerse 上的零样本表现

NavVerse 将 PoliFormer 作为 **RL 方法**的代表进行评估：

| 指标 | PoliFormer | 说明 |
|---|---|---|
| ObjNav All Scenes SR | 5.81% | 远低于其在 CHORES-S 上的表现 |
| ObjNav Indoor SR | 12.67% | 室内场景尚可 |
| ObjNav Outdoor SR | **0.00%** | 完全无法处理室外 |
| PlaceNav All Scenes SR | 1.63% | 无法处理语义地点目标 |
| Collision Rate | 0.24~0.36 | 碰撞率较高 |

**分析**：PoliFormer 在室内 ObjectNav 上表现尚可，但完全无法泛化到室外或 PlaceNav，说明其训练分布高度特化于室内物体导航。

---

## 四、关键技术创新

### 4.1 大规模 On-Policy RL

PoliFormer 证明：**只要训练规模足够大，on-policy RL 可以学到通用导航策略**，不需要复杂的课程学习或离线预训练。

### 4.2 Transformer 长期记忆

Causal Transformer Decoder 使模型能够：
- 记住已经探索过的区域；
- 避免重复搜索；
- 推断未探索区域的可能性。

### 4.3 零样本下游扩展

PoliFormer 可以无需微调直接扩展到：
- **目标跟踪**：跟踪动态目标；
- **多目标导航**：依次访问多个目标；
- **开放词汇导航**：导航到任意文本描述的物体。

---

## 五、优势与局限

### 5.1 优势

- **真正的零样本仿真到真实**：无需领域适应、微调或真实数据；
- **性能突破**：CHORES-S 上 85.5% 的成功率大幅超越此前工作；
- **RGB-only**：部署简单，不依赖额外传感器；
- **可扩展性**：易于扩展到多种下游导航任务。

### 5.2 局限

- **训练分布特化**：在室内物体导航上表现极佳，但完全无法处理室外或语义地点导航；
- **高计算成本**：数亿次交互的训练需要大量 GPU 资源；
- **碰撞率较高**：在 NavVerse 评估中碰撞率高于模块化方法；
- **无显式世界模型**：无法像 SGImagineNav 那样主动想象未见区域。

---

## 六、与相关工作的关系

| 工作 | 与 PoliFormer 的关系 |
|---|---|
| **传统 SLAM + 规划** | PoliFormer 是端到端学习，不依赖显式建图 |
| **Habitat ObjectNav 基线** | PoliFormer 大幅超越这些基线 |
| **VLA 方法（如 UniNaVid）** | PoliFormer 是纯 RL，不依赖语言模型；UniNaVid 在 NavVerse 上成功率更高 |
| **NavVerse** | PoliFormer 是 NavVerse 评估的 RL 基线，展现室内特化的局限性 |

---

## 七、结论

PoliFormer 是室内导航领域的里程碑工作。它证明了**大规模 on-policy RL + Transformer 架构**可以突破长期存在的性能平台期，实现真正的零样本仿真到真实迁移。CHORES-S 上 85.5% 的成功率为 RGB-only 室内导航树立了新标杆。

然而，NavVerse 的评估也暴露了 PoliFormer 的**高度特化性**：它在室内物体导航上表现极佳，但完全无法泛化到室外或语义地点导航。这提示我们：**单一任务的 SOTA 不等于通用导航能力**，跨上下文泛化仍是未解决的挑战。

---

*参考：PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators (arXiv:2406.20083, Jun 28, 2024)*
