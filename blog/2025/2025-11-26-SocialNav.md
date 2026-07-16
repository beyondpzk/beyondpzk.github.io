---
title: SocialNav
date: 2025-11-26
categories: [VLN]
---

# SocialNav：面向社交感知具身导航的人本启发式基础模型

> **论文**：*SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation*
> **作者**：Ziyi Chen, Yingnan Guo, Zedong Chu, Minghua Luo, Yanfen Shen, Mingchao Sun, Junjun Hu, Shichao Xie, Kuan Yang, Pei Shi, Zhining Gu, Lu Liu, Honglin Han, Xiaolong Wu, Mu Xu, Yu Zhang, Ning Guo
> **单位**：1Amap, Alibaba Group, China；2Zhejiang University, China
> **发布时间**：2025-11-26（arXiv），CVPR 2026
> **arXiv**：[https://arxiv.org/abs/2511.21135](https://arxiv.org/abs/2511.21135)
> **项目主页**：[https://amap-eai.github.io/SocialNav/](https://amap-eai.github.io/SocialNav/)

## 摘要

在社交场景中，机器人不仅要抵达目标，还必须遵守人类社会规范，例如走人行道、避开草坪、不阻挡行人等。现有方法多聚焦于最短路径与几何避障，难以处理高层社会规范与低层轨迹生成之间的耦合。为此，本文提出 **SocialNav**，一种面向社交感知具身导航（socially-aware embodied navigation）的分层基础模型，采用 "Brain-Action" 架构：

- **Brain 模块**：基于视觉-语言模型（VLM）进行高层语义推理，输出社会可通行区域、思维链（CoT）解释与视觉问答（VQA）。
- **Action Expert**：基于条件流匹配（conditional flow matching）生成低层、社会合规的轨迹。

为支撑训练，作者构建了 **SocNav Dataset**（700 万样本），包含专家轨迹金字塔（ETP）与认知激活数据集（CAD）；并提出首个面向具身导航的流式强化学习框架 **SAFE-GRPO**（Socially-Aware Flow Exploration GRPO），通过规范感知奖励显式鼓励社会合规行为。实验表明，SocialNav 在 CityWalker 开环基准、SocNav 闭环基准及真实世界 Unitree Go2 部署中均显著超越现有方法。

## 一、研究背景与动机

### 1.1 社交导航的现实需求

随着具身智能体逐步进入人类社会，导航任务不再只是几何意义上的点到点运动。一个能在商场、校园、街道等人流密集场景中运行的机器人，需要理解并遵守隐性的社会规范：

- 沿人行道、斑马线行走；
- 避免穿越草坪、花坛、私人区域；
- 与行人保持安全距离，不切断他人路径；
- 在十字路口遵守通行规则。

这些规则往往无法仅通过碰撞检测或最短路径规划来捕捉，而需要结合语义理解、常识推理与对人类行为的预判。

### 1.2 现有方法的局限

现有视觉导航研究大致可分为三类，但均存在明显不足：

| 方向 | 代表工作 | 主要局限 |
|------|----------|----------|
| 经典 SLAM / 路径规划 | ORB-SLAM、A*、Dijkstra | 缺乏语义与社会规范理解 |
| 端到端模仿学习 | GNM、ViNT、NoMaD、CityWalker | 侧重于行为克隆，难以学习社会规则背后的因果结构 |
| VLM 推理增强 | NavCoT、SayNav、Discuss-before-Moving | 高层推理与低层动作生成脱节，缺乏闭环控制能力 |

特别是，纯模仿学习（Imitation Learning, IL）虽然能从专家轨迹中学到运动先验，但无法真正"理解"社会规范：它只能模仿表面行为，遇到分布外场景时容易失效。因此，作者认为需要一种能够同时理解社会规范并生成合规轨迹的统一框架。

## 二、核心贡献

SocialNav 的核心贡献可从模型、数据、训练范式与评测平台四个维度概括：

### 2.1 模型层面：Brain-Action 分层架构

首次将 VLM 的高层社会语义推理与条件流匹配的低层轨迹生成紧耦合：

- Brain 输出可解释的社会可通行区域与 CoT 推理；
- Action Expert 以 Brain 的潜在特征为条件，生成机器人可执行的轨迹序列。

### 2.2 数据层面：SocNav Dataset

构建了一个 700 万样本的大规模异质数据集：

- **Expert Trajectories Pyramid（ETP）**：涵盖互联网视频、高保真仿真场景、真实机器人数据；
- **Cognitive Activation Dataset（CAD）**：提供社会可通行区域识别、导航 CoT、通用 VQA 等认知监督信号。

### 2.3 训练层面：SAFE-GRPO

提出首个面向具身导航的流式强化学习框架，将确定性 ODE 流策略扩展为随机 SDE，并通过规范感知奖励显式优化社会合规性。

### 2.4 评测层面：SocNav Benchmark

结合 Isaac Sim 物理仿真与 3D Gaussian Splatting（3DGS）真实感渲染，在 9 个新采集的大规模社交场景（共 73K m²）上构建闭环评测平台。

## 三、方法详解

### 3.1 问题定义

SocialNav 将任务建模为基于视觉历史条件的点到点导航问题。在每个时刻 $t$，智能体接收最近 $n=5$ 帧 RGB 观测 $\bm{O}_{t-n:t}$ 及对应 2D 位置 $\bm{P}_{t-n:t}$，给定目标点 $\bm{g} \in \mathbb{R}^2$，策略 $\pi_\theta$ 预测未来 $m=5$ 步的动作序列：

$$
\bm{A}_{t+1:t+m} = \pi_\theta(\bm{O}_{t-n:t}, \bm{P}_{t-n:t}, \bm{g})
$$

动作空间为低层轨迹（waypoints），最终由机器人底层运动策略跟踪执行。

### 3.2 整体架构：Brain-Action 分层设计

SocialNav 的整体架构如图 3 所示，包含两个核心模块：

| 模块 | 功能 | 基础模型 |
|------|------|----------|
| **Brain Module** | 高层语义推理：输出社会可通行区域、CoT、VQA | Qwen2.5-VL (3B) |
| **Action Expert** | 低层轨迹生成：将语义先验转化为机器人动作 | 条件流匹配（Conditional Flow Matching） |

两个模块通过 VLM 最后一层隐特征 $\bm{Z}_{\text{VLM}}$ 进行条件连接，实现"高层理解指导低层动作"的闭环。

### 3.3 Brain 模块：基于 VLM 的社会语义推理

Brain 模块以 Qwen2.5-VL 为骨干，执行三类生成式任务：

1. **社会可通行区域预测**：以多边形形式标注 sidewalks、crosswalks、stairs 等可通行区域；
2. **导航思维链（CoT）**：生成逐步文本推理，解释当前导航决策；
3. **通用视觉问答（VQA）**：回答自由形式问题以增强场景理解。

这些任务通过 CAD 数据集中对应的 120 万可通行区域样本、82.5 万 CoT 样本、100 万 VQA 样本进行监督训练。

### 3.4 Action Expert：条件流匹配生成轨迹

Action Expert 采用条件流匹配建模动作分布。给定 VLM 提供的条件特征 $\bm{Z}_{\text{VLM}}$，轨迹生成过程为：

$$
\bm{Z}_{\text{VLM}} = \pi_{\text{VLM}}(\bm{O}_{t-n:t}, \bm{P}_{t-n:t}, \bm{g})
$$

$$
\bm{A}_{t+1:t+m} = \pi_{\text{flow}}(\bm{x}_t, t; \bm{Z}_{\text{VLM}})
$$

其中 $\pi_{\text{flow}}$ 为 Diffusion Transformer（12 层、12 头、隐藏维度 1536），推理时进行 $K=5$ 步去噪。条件流匹配的优势在于能够建模多模态动作分布，并自然支持基于强化学习的进一步探索。

### 3.5 多阶段训练流程

SocialNav 的训练分为三个阶段，逐步注入导航能力与社交规范：

#### 3.5.1 阶段 1：预训练（Pre-training）

在 ETP 的互联网视频数据 $\mathcal{D}_{\text{video}}$、仿真数据 $\mathcal{D}_{\text{sim}}$ 以及 CAD 的认知数据 $\mathcal{D}_{\text{cog}}$ 上进行端到端预训练：

- $\mathcal{D}_{\text{video}}$（200 万伪轨迹）：通过互联网城市漫游视频，经 $\pi^3$ 三维重建、MoGe 尺度对齐后采样得到，提供丰富的真实世界运动先验；
- $\mathcal{D}_{\text{sim}}$（170 万轨迹）：基于 4,490 个高保真 3D 场景及 3.37 km² 动态城市 SocCity 生成，包含标准路径与恢复轨迹；
- $\mathcal{D}_{\text{cog}}$：提供社会语义推理监督。

训练使用 AdamW，学习率 $5 \times 10^{-5}$，批量大小 192，在 96 张 H20 GPU 上训练 3 个 epoch。

#### 3.5.2 阶段 2：真实世界数据微调（Fine-tuning）

冻结 VLM，仅在真实机器人数据 $\mathcal{D}_{\text{real}}$（34 万轨迹，来自 SCAND、Huron、Recon、CityWalker 遥操作数据等）上微调 Action Expert，以缩小 sim-to-real 差距。批量大小 256，学习率 $1 \times 10^{-5}$，使用 32 张 H20 GPU。

#### 3.5.3 阶段 3：SAFE-GRPO 强化学习

在模仿学习基础上，进一步通过 SAFE-GRPO 对齐社会规范。该阶段使用 SocCity 仿真数据，提供精确的道路标注以支持奖励计算。

### 3.6 SAFE-GRPO：社会感知的流式强化学习

SAFE-GRPO 的核心思想是将确定性流策略的 ODE 转化为随机 SDE，以支持受控探索：

$$
d\bm{x}_t = \bm{v}_{\text{flow}}(\bm{x}_t, t; \bm{Z}_{\text{VLM}}) dt + \sigma_t d\bm{w}_t
$$

其中 $\sigma_t$ 控制探索幅度。关键在于，随机性仅引入在流积分过程中，而来自 VLM Brain 的语义条件 $\bm{Z}_{\text{VLM}}$ 保持固定，从而保证探索始终受高层社会语义引导。

总体奖励函数综合社会合规、专家一致性、运动平滑与导航效率：

$$
\mathcal{R} = \mathcal{R}_{\text{social}} + \lambda_{\text{expert}} \mathcal{R}_{\text{expert}} + \lambda_{\text{smooth}} \mathcal{R}_{\text{smooth}} + \lambda_{\text{eff}} \mathcal{R}_{\text{eff}}
$$

- $\mathcal{R}_{\text{social}}$：基于语义占用图，鼓励与社会可通行区域保持一致；
- $\mathcal{R}_{\text{expert}}$：鼓励与专家轨迹一致；
- $\mathcal{R}_{\text{smooth}}$：鼓励运动连续性；
- $\mathcal{R}_{\text{eff}}$：鼓励向目标高效前进。

SAFE-GRPO 在 16 张 H20 GPU 上训练，rollout 批量大小 128，学习率 $5 \times 10^{-7}$。

## 四、实验

### 4.1 实验设置与评测指标

作者在三种场景下评估 SocialNav：

1. **CityWalker 开环基准**：使用最大平均方向误差 MAOE（Maximum Average Orientation Error），越低越好；
2. **SocNav Benchmark 闭环基准**：使用成功率 SR、路径完成率 RC、成功加权路径长度 SPL、社会合规率 DCR/TCR；
3. **真实世界部署**：在 Unitree Go2 四足机器人上测试。

社会合规指标定义如下：

$$
\text{DCR} = \begin{cases} \dfrac{d_{\text{compliant}}}{d_{\text{actual}}}, & \text{if } s=1 \\[8pt] 0, & \text{otherwise} \end{cases}
$$

其中 $s$ 为任务成功指示变量，$d_{\text{compliant}}$ 为在社会合规区域内行驶的距离，$d_{\text{actual}}$ 为实际总行驶距离。TCR 定义类似。

### 4.2 CityWalker 开环评测结果

表 1 展示了在 CityWalker 基准上的 MAOE 对比。SocialNav 在所有场景与总体指标上均取得最低误差。

**表 1：CityWalker 开环评测（MAOE，越低越好）**

| 方法 | 场景均值 | 转弯 | 过街 | 绕行 | 近距 | 人群 | 全部样本 |
|------|----------|------|------|------|------|------|----------|
| 数据占比 | — | 8% | 12% | 12% | 6% | 7% | 55% |
| GNM | 16.2 | 31.1 | 14.8 | 12.5 | 14.7 | 12.8 | 11.0 |
| ViNT | 16.5 | 31.1 | 15.4 | 12.9 | 14.8 | 13.3 | 11.6 |
| NoMaD | 19.1 | 35.1 | 18.5 | 15.6 | 18.1 | 14.3 | 12.8 |
| CityWalker | 15.2 | 26.6 | 14.1 | 13.9 | 14.3 | 12.0 | 10.4 |
| **SocialNav (Full)** | **10.2** | **20.1** | **8.8** | **8.4** | **8.9** | **7.6** | **7.2** |

尤其在人群（Crowd）与过街（Crossing）等社会敏感场景中，SocialNav 的 MAOE 显著低于 CityWalker，说明其轨迹更符合人类步行规范。

### 4.3 SocNav Benchmark 闭环评测结果

表 2 展示了在 SocNav Benchmark 上的闭环性能。SocialNav (Full) 在导航性能与社会合规性上均大幅领先基线。

**表 2：SocNav Benchmark 闭环评测**

| 方法 | SR↑ | RC↑ | SPL↑ | DCR↑ | TCR↑ |
|------|-----|-----|------|------|------|
| GNM* | 43.3 | 62.4 | 37.0 | 26.5 | 28.7 |
| ViNT* | 45.6 | 66.2 | 39.5 | 31.4 | 33.8 |
| NoMaD* | 41.1 | 60.5 | 35.4 | 29.5 | 31.6 |
| CityWalker | 47.8 | 64.7 | 44.7 | 36.1 | 36.6 |
| SocialNav* | 65.0 | 78.4 | 62.3 | 58.0 | 56.7 |
| **SocialNav (Full)** | **86.1** | **91.2** | **77.4** | **82.5** | **82.9** |

注：带 * 的模型表示仅在真实数据 $\mathcal{D}_{\text{real}}$ 上训练。

关键发现：

- 与 CityWalker 相比，SocialNav (Full) 的 SR 从 47.8 提升至 86.1，RC 从 64.7 提升至 91.2，SPL 从 44.7 提升至 77.4；
- 社会合规指标提升更为显著：DCR 从 36.1 提升至 82.5，TCR 从 36.6 提升至 82.9；
- 即使在相同真实数据上训练，SocialNav* 的 SR 65.0、DCR 58.0 也显著优于 GNM*、ViNT*、NoMaD*，说明模型架构本身具有更强的泛化能力。

### 4.4 真实世界部署结果

表 3 展示了在 Unitree Go2 机器人上的真实世界测试结果。SocialNav 在街道过街、办公园区、商场三种场景中均取得最高成功率。

**表 3：真实世界部署成功率**

| 方法 | 街道过街 | 办公园区 | 商场 | 平均 SR |
|------|----------|----------|------|---------|
| GNM* | 9/20 | 10/20 | 8/20 | 45.0 |
| ViNT* | 7/20 | 12/20 | 8/20 | 45.0 |
| NoMaD* | 9/20 | 11/20 | 10/20 | 50.0 |
| CityWalker | 12/20 | 13/20 | 12/20 | 62.5 |
| **SocialNav (Full)** | **18/20** | **16/20** | **17/20** | **85.0** |

SocialNav 在真实环境中运行频率超过 5 Hz，能够满足实时导航需求。

### 4.5 消融实验

表 4 展示了数据组成与训练阶段对性能的影响。

**表 4：数据组成与训练阶段消融**

| 编号 | 配置 | $\mathcal{D}_{\text{real}}$ | $\mathcal{D}_{\text{video}}$ | $\mathcal{D}_{\text{sim}}$ | $\mathcal{D}_{\text{cog}}$ | IL | RL | SR | RC | SPL | DCR | TCR |
|------|------|--------|--------|--------|--------|----|----|----|----|----|----|----|
| 1 | SocialNav* | ✓ | — | — | — | ✓ | — | 65.0 | 78.4 | 62.3 | 58.0 | 56.7 |
| 2 | + $\mathcal{D}_{\text{video}}$ | ✓ | ✓ | — | — | ✓ | — | 76.7 | 84.8 | 70.1 | 62.9 | 64.6 |
| 3 | + $\mathcal{D}_{\text{sim}}$ | ✓ | ✓ | ✓ | — | ✓ | — | 82.2 | 86.0 | 77.8 | 69.8 | 68.2 |
| 4 | + $\mathcal{D}_{\text{cog}}$ | ✓ | ✓ | ✓ | ✓ | ✓ | — | 84.4 | 88.1 | 79.4 | 78.2 | 78.4 |
| 5 | + RL（无 $\mathcal{D}_{\text{cog}}$） | ✓ | ✓ | ✓ | — | ✓ | ✓ | 80.0 | 89.1 | 78.9 | 68.1 | 66.9 |
| 6 | + RL（完整数据） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **86.1** | **91.2** | 77.4 | **82.5** | **82.9** |

消融实验揭示了几个关键结论：

1. **互联网视频数据**带来显著性能增益（SR +11.7），说明大规模真实世界运动先验的重要性；
2. **仿真数据**增强了恢复能力与鲁棒性（SR 从 76.7 提升至 82.2）；
3. **认知数据**对社会合规至关重要：DCR 从 69.8 提升至 78.2，TCR 从 68.2 提升至 78.4；
4. **SAFE-GRPO 需要认知先验**：没有 CAD 时，RL 反而降低社会合规指标（DCR 68.1 vs 69.8），说明高层社会理解是 RL 对齐的基础；
5. **RL 带来轻微路径效率下降**：SPL 从 79.4 降至 77.4，反映了社会合规路径通常更长但更安全的本质权衡。

## 五、优势与局限

### 5.1 主要优势

- **统一的分层架构**：首次将 VLM 高层推理与流匹配低层控制紧耦合，实现可解释的社会感知导航；
- **大规模异质数据**：700 万样本覆盖视频、仿真、真实机器人三类数据源，兼具规模与现实性；
- **认知激活数据**：通过 CoT 与可通行区域预测，使模型真正"理解"社会规则，而非简单模仿；
- **首个流式导航 RL 框架**：SAFE-GRPO 将流匹配与 GRPO 结合，显式优化社会合规行为；
- **高保真评测平台**：SocNav Benchmark 结合 Isaac Sim 物理仿真与 3DGS 渲染，提供 73K m² 的 9 个新场景；
- **真实世界验证**：在 Unitree Go2 上实现 85% 成功率与 5 Hz 以上实时运行。

### 5.2 局限与未来方向

- **奖励函数依赖手工设计**：当前 $\mathcal{R}_{\text{social}}$ 基于语义占用图，难以覆盖所有情境化的人类习俗；未来可引入 VLM 提供更丰富、自适应的奖励信号。
- **社会规范范围有限**：当前主要聚焦于可通行区域与基本空间礼仪，尚未涉及更复杂的文化规范、动态行人交互等。
- **SPL 的轻微下降**：社会合规路径往往更长，如何在合规与效率之间取得更好平衡仍需研究。
- **计算资源需求高**：完整训练需要 96 张 H20 GPU 进行预训练，对学术界复现构成一定门槛。

## 六、历史意义与后续影响

### 6.1 历史意义

SocialNav 是首批将"社会规范理解"作为核心能力、并以基础模型方式系统解决的具身导航工作之一。它标志着视觉导航研究从"几何可达"向"社会可接受"的重要转变，将高层语义推理、低层动作生成与强化学习对齐整合在同一框架中。

### 6.2 后续影响

- **数据-模型-评测三位一体**：SocNav Dataset 与 Benchmark 为后续社会导航研究提供了标准化训练与评测基础；
- **Brain-Action 架构启示**：该分层设计范式可扩展至其他需要高层推理指导低层控制的具身任务，如社交操作、人机协作等；
- **SAFE-GRPO 的泛化价值**：将流匹配与 GRPO 结合的思路，可推广至其他需要细粒度奖励对齐的连续控制任务；
- **真实世界部署验证**：证明了大规模预训练 VLA 模型在真实四足机器人上的可行性，推动了从仿真到现实的落地进程。

## 七、总结

SocialNav 通过 Brain-Action 分层架构、SocNav 大规模数据集、SAFE-GRPO 强化学习框架以及高保真 SocNav Benchmark，系统性地解决了社交感知具身导航问题。实验表明，该模型在导航成功率与社会合规性上均显著超越现有方法，并在真实 Unitree Go2 机器人上验证了其鲁棒性与实时性。尽管在社会规范的广度、奖励函数的自适应性以及计算成本方面仍有提升空间，SocialNav 无疑为构建真正具备社会意识的具身智能体迈出了关键一步。
