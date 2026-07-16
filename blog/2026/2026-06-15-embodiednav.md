---
title: EmbodiedNav
date: 2026-06-15
categories: [VLA]
---

# EmbodiedNav：具身导航统一基础模型演进综述

> 本文按发表时间梳理了四篇代表性工作——Uni-NaVid、NavFoM、ABot-N0、Qwen-RobotNav，从“单一模型统一多任务”到“可外部配置的 Agent 导航原语”的演进脉络，并在文末给出对比表格。

---

## 一、引言

具身导航（Embodied Navigation）长期被任务专用架构割裂：VLN、ObjectNav、目标跟踪、自动驾驶等任务各自拥有独立的输入输出格式与模型结构。近年来，随着视觉-语言模型（VLM）与大语言模型（LLM）的发展，研究者开始尝试用**统一的 Vision-Language-Action（VLA）架构**覆盖多种导航任务。本文按时间顺序梳理四个里程碑工作，展示该领域如何从“统一任务格式”走向“统一且可配置的导航基础模型”。

---

## 二、Uni-NaVid（2024.12）—— 视频 VLA 统一导航的先行者

**论文**：*Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks*  
**机构**：北京大学、Galbot、北京智源人工智能研究院  
**核心目标**：用单一视频 VLA 模型同时处理 VLN、ObjectNav、具身问答（EQA）和人体跟随四种任务。

### 2.1 核心思想

Uni-NaVid 将四类导航任务统一为：**egocentric RGB 视频流 + 自然语言指令 → 未来 4 步离散低级动作**。其关键假设是：不同导航任务本质上都依赖“在线视频历史与语言指令的对齐”，统一格式能促进任务间的技能迁移。

### 2.2 模型架构

- **视觉编码器**：EVA-CLIP，每帧输出 256 个 patch token。
- **在线视觉 token 合并**：借鉴 Atkinson-Shiffrin 记忆模型，将历史帧分为：
  - **当前帧**（64 tokens，保留细粒度空间信息）
  - **短期记忆**（最近 64 帧，每帧 4 tokens）
  - **长期记忆**（更早历史，每帧 1 token，并按余弦相似度增量合并）
- **语言模型**：Vicuna-7B，接收合并后的视觉 token、`<NAV>` 任务指示 token 与语言指令，自回归输出动作 token。
- **动作空间**：离散动作 `{FORWARD, TURN-LEFT, TURN-RIGHT, STOP}`。
- **前瞻预测**：一次输出未来 4 步动作，支持非阻塞执行，推理速度约 **5 Hz**。

### 2.3 数据与训练

| 数据类型 | 规模 | 说明 |
|---|---|---|
| 多任务导航数据 | 3.6M | VLN、ObjectNav、EQA、Human Following |
| 开放世界视频 VQA / 描述 | 2.3M | LLaMA-VID、Panda-70M 等 |
| **合计** | **5.9M** | 覆盖 861 个 Habitat 场景 |

训练分两步：先对齐投影器，再端到端联合微调 LLM 与投影器，使用 40×H800 训练约 35 小时（约 1400 GPU 小时）。

### 2.4 关键指标

| 基准 | 关键结果 |
|---|---|---|
| VLN-CE R2R Val-Unseen | SR 47.0% / SPL 42.7% |
| VLN-CE RxR Val-Unseen | SR 48.7% / SPL 40.9% |
| HM3D ObjectNav | SR 73.7% / SPL 37.1%（仅 RGB） |
| HM3D-OVON zero-shot | SR 39.5% / SPL 19.8% |
| 真实世界 VLN | 简单 92% / 复杂 84% |

### 2.5 主要贡献

1. 提出首个基于视频的 VLA 统一导航模型，验证多任务联合训练的协同效应。
2. 在线视觉 token 合并机制解决长历史视频输入的推理效率问题。
3. 仅依赖 RGB 视频与语言指令实现真实机器人 zero-shot 部署。

---

## 三、NavFoM（2025.09）—— 跨任务、跨具身的导航基础模型

**论文**：*Embodied Navigation Foundation Model (NavFoM)*  
**机构**：北京大学、Galbot、USTC、BAAI、阿德莱德大学、浙江大学等  
**核心目标**：打破任务与具身（embodiment）的双重割裂，构建跨四足、轮式、无人机、车辆平台的统一导航基础模型。

### 3.1 核心思想

NavFoM 进一步将导航抽象为：**多视角 ego-centric 视频 + 语言指令 → 未来 8 个连续航点轨迹**。它不仅统一任务，还统一不同机器人平台的相机配置与运动学差异。

### 3.2 模型架构

- **视觉编码器**：DINOv2 + SigLIP 串联，每帧每视角 576 patch tokens。
- **Grid Pooling**：当前帧细粒度 64 tokens，历史帧粗粒度 4 tokens。
- **时序-视角指示 token（TVI tokens）**：显式编码每个视觉 token 所属的相机方位角与时间步，使模型能区分 1/2/4/6/8 目相机、不同历史长度、image/video/navigation 三类样本。
- **预算感知的时序采样（BATS）**：基于遗忘曲线，在固定 token 预算下对近期帧高概率采样、历史帧保留非零下界，实现推理速度与长程记忆的平衡。
- **语言模型**：Qwen2-7B。
- **轨迹头**：3 层 MLP，预测归一化的 8 个航点 `(x, y, z, θ)`，地面平台使用 `(x, y, θ)`。

### 3.3 数据与训练

| 数据类型 | 规模 | 说明 |
|---|---|---|
| 导航样本 | 8.02M | 四足、轮式、无人机、车辆 |
| Image QA | 3.15M | 开放世界图像问答 |
| Video QA | 1.61M | 开放世界视频问答 |
| **合计** | **12.7M** | 含 Sekai 网络视频伪导航数据 |

导航任务分布：VLN 3.37M、ObjectNav 1.02M、Active Tracking 0.897M、自动驾驶 0.681M、Web-Video 2.03M。使用 56×H100 训练约 72 小时（约 4032 GPU 小时）。

### 3.4 关键指标

| 基准 | 关键结果 |
|---|---|---|
| VLN-CE R2R Val-Unseen（四目） | SR 61.7% / SPL 55.3% |
| VLN-CE RxR Val-Unseen（四目） | SR 64.4% / SPL 56.2% |
| HM3D-OVON Val-Unseen（zero-shot） | SR 45.2% / SPL 31.9% |
| EVT-Bench 单目标（四目） | SR 88.4% / TR 80.7% |
| NAVSIM navtest（八目） | PDMS 84.3 |

### 3.5 主要贡献

1. 首次同时在**跨任务、跨具身、多相机配置**三个维度上实现统一训练。
2. 提出 TVI tokens 与 BATS，显式建模视觉 token 的时空身份与上下文预算。
3. 通过多任务共调，搜索与跟踪任务获得数量级的性能提升。

---

## 四、ABot-N0（2026.02）—— 五大任务的“Grand Unification”与真实世界部署

**论文**：*ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation*  
**机构**：阿里巴巴 AMAP CV Lab  
**核心目标**：在单一架构内统一 Point-Goal、Object-Goal、Instruction-Following、POI-Goal、Person-Following 五种核心导航任务，并实现真实世界 Agentic 部署。

### 4.1 核心思想

ABot-N0 采用层次化 **Brain-Action** 架构：
- **Brain（认知大脑）**：基于 Qwen3-4B，负责语义理解、空间推理与任务分解；
- **Action（动作专家）**：基于 Flow Matching，负责生成连续、多模态、社会合规的轨迹分布。

所有任务统一输出局部 BEV 坐标系下的 **5 个航点** `(x, y, θ)`。

### 4.2 模型架构

- **Universal Multi-Modal Encoder**：统一编码 RGB 观测、视觉历史、文本/坐标目标、推理任务描述。
- **Cognitive Brain**：Qwen3-4B，设独立 Reasoning Head 与 Action Head，以任务 token 条件化切换。
- **Action Expert**：Flow Matching 生成连续航点分布，支持多模态轨迹采样，避免确定性回归的平均化问题。
- **Agentic Navigation System**：集成 Agentic Planner、层次化 Topo-Memory、Self-Reflector 与 Neural Controller，实现长程任务执行与真实世界闭环控制。

### 4.3 数据与训练

ABot-N0 Data Engine 是当前规模最大的具身导航数据引擎之一：

| 数据类型 | 规模 | 说明 |
|---|---|---|
| 高保真 3D 场景 | 7,802 个 | 覆盖 10.7 km² |
| 专家轨迹 | 16.9M | Point-Goal 4.0M、Instruction 2.8M、Object-Goal 3.6M、POI-Goal 2.5M、Person-Following 4.0M |
| 认知推理样本 | 5.0M | 可通行区域、社会导航 CoT、VLN 推理、OVON 推理、POI Grounding、通用 VQA |
| **合计** | **21.9M** | |

训练采用三阶段课程学习：
1. **Cognitive Warm-up**：用推理数据微调 LLM；
2. **Unified Sensorimotor SFT**：联合训练轨迹与推理数据（20% 推理回放）；
3. **SAFE-GRPO**：冻结 Brain，仅微调 Action Expert，将社会合规奖励注入动作分布。

### 4.4 关键指标

| 任务/基准 | 关键结果 |
|---|---|---|
| Point-Goal / CityWalker（开环） | MAOE 11.2 |
| Point-Goal / SocNav（闭环） | SR 88.3% / SPL 79.2% / DCR 85.1% |
| VLN-CE R2R Val-Unseen | SR 66.4% / SPL 63.9% |
| VLN-CE RxR Val-Unseen | SR 69.3% / SPL 60.0% |
| HM3D-OVON Val-Unseen | SR 54.0% / SPL 30.5% |
| BridgeNav POI-Goal | SR 88.68% @ 0.3m |
| EVT-Bench（AT） | SR 67.3% / TR 79.5% |

真实世界部署在 Unitree Go2 上实现 **2Hz VLA 推理 + 10Hz Neural Controller 闭环控制**。

### 4.5 主要贡献

1. 首次在单一 VLA 模型中实现五种核心导航任务的“Grand Unification”。
2. 构建最大规模具身导航数据引擎（16.9M 轨迹 + 5.0M 推理）。
3. 通过 Agentic Navigation System 将基础模型能力扩展到真实世界长程任务。

---

## 五、Qwen-RobotNav（2026.06）—— 可外部配置的 Agent 导航原语

**论文**：*Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System*  
**机构**：阿里巴巴通义千问团队  
**核心目标**：将多任务导航的核心挑战重新定义为“观察上下文建模”，使同一模型可被上层 Agent 在推理时动态调用。

### 5.1 核心思想

Qwen-RobotNav 认为，不同导航任务共享同一感知-规划骨干，但对“如何消费视觉流”有根本不同的需求：
- VLN 需要长程全局记忆以反复对照路标；
- 目标跟踪只需要最近几帧高分辨率输入；
- 目标搜索需要在探索（全局记忆）与趋近（局部聚焦）之间切换。

因此，关键不是增加任务头，而是让**观察上下文成为外部可控的一等公民**。

### 5.2 模型架构

- **骨干**：直接继承 Qwen3-VL（SigLIP-2 ViT + Qwen3 LLM），零结构改动。
- **动作头**：4 层 MLP，输出 8 个航点 `(x, y, θ)`。
- **参数化观察接口 `Φ = (B, γ, wc, m, b_min, b_max)`**：
  - `B`：token 预算；
  - `γ`：时间衰减（控制历史帧权重）；
  - `wc`：每相机权重；
  - `m`：帧采样模式（random / latest）；
  - `b_min / b_max`：每帧每相机 token 上下界。
- **自然语言标签**：用 “Front View / Time step 0” 等普通词汇表达相机身份与时间顺序，无需新增位置编码。
- **具身提示前缀**：通过 system prompt 区分室内机器人与自动驾驶汽车。

### 5.3 数据与训练

| 数据类型 | 规模 | 说明 |
|---|---|---|
| 导航轨迹规划数据 | 13.3M（85%） | VLN、PointNav、ObjNav、Tracking、Driving |
| 自动生成视频数据 | 40K | T2V 生成 + VLM 过滤 + 位姿恢复 |
| 视觉-语言推理数据 | 2.37M（15%） | 通用 VQA、视觉定位、图像描述、导航专用推理、离散多轮 VLN |
| **合计** | **15.6M** | |

训练时对所有观察参数进行每样本随机采样，使模型在推理时能 zero-shot 适应任意配置。8B 模型总计约 **2816 H100 GPU 小时**。

### 5.4 关键指标

| 基准 | 关键结果 |
|---|---|---|
| VLN-CE R2R Val-Unseen（全景，8B） | SR 72.1% / SPL 66.6% |
| VLN-CE RxR Val-Unseen（全景，8B） | SR 76.5% / nDTW 72.5% |
| HM3D v2 ObjectNav（4B，单目 RGB） | SR 75.6% |
| HM3D-OVON Unseen（4B） | SR 53.1% |
| EVT-Bench STT（4B） | TR 90.0% |
| NAVSIM navtest（4B） | PDMS 91.4 |
| HM-EQA / MT-EQA / EXPRESS | 76.7 / 54.4 / 79.27 |

### 5.5 主要贡献

1. 提出**参数化观察接口**，使导航模型可被上层 LLM Planner 动态配置。
2. 通过训练时随机化实现推理时任意观察策略的 zero-shot 泛化。
3. 将导航模型封装为 Agent 可调用的导航原语，支撑长程记忆与上下文压缩。

---

## 六、演进趋势总结

1. **从离散动作到连续航点**：Uni-NaVid 输出离散低级动作，后续工作统一输出连续航点轨迹，更利于真实机器人控制。
2. **从任务统一到跨具身统一**：NavFoM 引入 TVI tokens 与 BATS，将统一范围从任务扩展到机器人平台。
3. **从模仿学习到社会价值对齐**：ABot-N0 通过 SAFE-GRPO 将社会合规显式注入动作分布，关注真实世界安全。
4. **从固定观察到可配置观察**：Qwen-RobotNav 将观察上下文作为外部可控变量，使导航模型成为 Agent 的模块化组件。
5. **数据规模持续增长**：从 5.9M → 12.7M → 21.9M → 15.6M，显示大规模异构数据是统一导航基础模型的关键。

---

## 七、四篇工作对比

| 维度 | Uni-NaVid | NavFoM | ABot-N0 | Qwen-RobotNav |
|---|---|---|---|---|
| **发表时间** | 2024.12 | 2025.09 | 2026.02 | 2026.06 |
| **机构** | 北大、Galbot、BAAI | 北大、Galbot、USTC、BAAI 等 | 阿里 AMAP CV Lab | 阿里通义千问团队 |
| **视觉编码器** | EVA-CLIP | DINOv2 + SigLIP | ViT（SigLIP-B/16） | SigLIP-2 ViT |
| **语言模型** | Vicuna-7B | Qwen2-7B | Qwen3-4B | Qwen3-VL |
| **核心架构创新** | 在线视觉 token 合并（当前/短期/长期） | TVI tokens + BATS 采样 | Brain-Action 分层 + Flow Matching Action Expert | 参数化观察接口 + 训练时随机化 |
| **统一任务** | VLN、ObjectNav、EQA、Human Following | VLN、ObjectNav、Tracking、Driving、UAV | Point-Goal、Object-Goal、Instruction、POI-Goal、Person-Following | VLN、PointNav、ObjNav、Tracking、Driving |
| **输出动作形式** | 4 步离散动作 {前/左/右/停} | 8 个连续航点 `(x,y,z,θ)` | 5 个 BEV 航点 `(x,y,θ)` | 8 个航点 `(x,y,θ)` |
| **训练数据总量** | **5.9M**（3.6M 导航 + 2.3M VQA/视频描述） | **12.7M**（8.02M 导航 + 4.76M QA） | **21.9M**（16.9M 轨迹 + 5.0M 推理） | **15.6M**（13.3M 轨迹 + 2.37M VL 推理 + 40K 生成视频） |
| **训练计算** | ~1400 H800 GPU 小时 | ~4032 H100 GPU 小时 | 未公开 | ~2816 H100 GPU 小时 |
| **真实世界部署** | Unitree GO2，非阻塞 5Hz | 四足、人形、无人机、轮式机器人 | Unitree Go2，2Hz VLA + 10Hz Neural Controller | Unitree Go2 / Jetson Thor，~5Hz |
| **关键局限** | 动作离散、任务范围有限 | 自动驾驶未利用结构化信息、UM 拆分落后 | 消融实验与训练成本披露不足 | 跟踪 SR 偏保守、AlpaSim zero-shot 有差距 |

### 任务能力说明

| 工作 | 能做的典型任务 |
|---|---|
| **Uni-NaVid** | 自然语言路径跟随、物体搜索、具身问答、人体跟随 |
| **NavFoM** | 室内 VLN、开放词汇目标搜索、主动视觉跟踪、无人机户外导航、自动驾驶规划 |
| **ABot-N0** | 点目标导航、物体目标导航、指令跟随、POI 精确定位、行人跟随，及室内外长程复合任务 |
| **Qwen-RobotNav** | 指令跟随、点目标导航、物体搜索、目标跟踪、自动驾驶，并作为上层 Agent 的导航原语 |

---

## 八、结语

从 Uni-NaVid 到 Qwen-RobotNav，具身导航领域正在经历从“任务专用”到“统一基础模型”再到“Agent 可调原语”的范式转变。未来的关键方向包括：更细粒度的消融与可解释性、真实世界大规模量化评估、在线适应与强化学习闭环、三维场景记忆与语义地图融合，以及更轻量高效的边缘部署。

---

## 附录：训练数据集来源与存储空间预估

> 本节汇总四篇工作中明确提及的训练数据来源，并基于公开信息或合理假设对原始存储占用进行粗略估算。实际占用与数据预处理、压缩策略、是否缓存特征等因素强相关，以下数字仅供参考。

### A.1 数据集来源汇总

#### Uni-NaVid

| 数据集/来源 | 类型 | 样本数 | 说明 |
|---|---:|---|---|
| VLN-CE R2R / RxR | 仿真导航 | 2.4M | Habitat 连续环境渲染的 RGB-指令-动作序列 |
| HM3D ObjectNav | 仿真导航 | 483K | L3MVN 成功轨迹 |
| MP3D-EQA | 仿真导航 + 问答 | 250K | 240K 视频-动作 + 10K 视频-答案 |
| 自建 Habitat 3.0 Human Following | 仿真导航 | 544K | 8 个 avatar + 干扰行人 |
| LLaMA-VID 视频 QA | 开放世界 VQA | 含于 2.3M | 防止灾难性遗忘 |
| Panda-70M 视频描述 | 开放世界 Caption | 含于 2.3M | 保持开放世界视频理解 |

#### NavFoM

| 数据集/来源 | 类型 | 样本数 | 说明 |
|---|---:|---|---|
| VLN-CE R2R / RxR | 仿真导航 | 3.37M | 室内轮式 VLN |
| OpenUAV | 仿真导航 | 含于 3.37M | 户外无人机 VLN |
| HM3D ObjectNav | 仿真导航 | 1.02M | 基于 L3MVN 成功片段 |
| EVT-Bench | 仿真跟踪 | 897K | 主动视觉跟踪 |
| nuScenes | 真实驾驶 | 681K 中 27K 片段 | 自动驾驶开环数据 |
| OpenScene | 仿真/真实驾驶 | 654K | 自动驾驶场景 |
| Sekai Web-Video | 网络视频伪导航 | 2.03M | 18.2 万 YouTube 视频 + VLM 指令 + SLAM 轨迹 |
| 开放世界 Image QA | 开放世界 VQA | 3.15M | 与导航数据共调 |
| 开放世界 Video QA | 开放世界 VQA | 1.61M | 与导航数据共调 |

#### ABot-N0

| 数据集/来源 | 类型 | 样本数 | 说明 |
|---|---:|---|---|
| 互联网视频伪轨迹（π3 + MoGe） | 真实视频伪导航 | 2.0M | 单目视频三维重建 + 尺度对齐 |
| 高保真 3D 场景合成 | 仿真导航 | 1.7M | 导航图采样 + 最优路径 + 恢复轨迹 |
| SCAND / HuRoN / Recon / CityWalker | 真实机器人遥操作 | 340K | 真实物理动力学与传感器噪声 |
| VLN-CE R2R / RxR | 仿真导航 | ~1.5M | 教师强制展开 + 筛选 |
| Door-Traversal | 仿真导航 | 300K | 狭窄通道与门口穿越 |
| Language-Guided Person Search | 仿真导航 | 200K | 语言查询定位目标人物 |
| Short-Horizon | 仿真导航 | 800K | 旋转/平移/复合原子动作 |
| HM3D + OVON | 仿真导航 | 1.8M | 6 类封闭类别 + 145 场景开放词汇 |
| OVON-sub | 仿真导航 | 200K | 从目标首次可见处截断的短程子集 |
| InteriorGS | 仿真导航 | 1.6M | 1,000 场景、700+ 物体类别 |
| BridgeNav 合成 | 仿真/合成视频 | 2.5M | 分割+深度+A*+Wan2.1-I2V 合成 POI 视频 |
| 合成跟踪序列 | 仿真跟踪 | 4.0M | 3 种距离 × 3 类挑战 |
| Navigable Areas Analysis | 推理数据 | 1.2M | 户外可通行区域多边形标注 |
| Social Navigation CoT | 推理数据 | 800K | Qwen-VL-Max 教师生成的结构化推理 |
| Instruction-Following Reasoning | 推理数据 | 1.3M | 长程指令分解为里程碑节点 |
| Object-Goal Reasoning | 推理数据 | 100K | 四步结构化推理链 |
| POI Grounding | 推理数据 | 500K | POI 名称与入口像素坐标 VQA |
| General VQA | 推理数据 | 1.1M | Blip3、COCO、MAmmoTH-VL、RefCOCO、Objects365、R2R-EnvDrop、ScanQA 等 |

#### Qwen-RobotNav

| 数据集/来源 | 类型 | 样本数 | 说明 |
|---|---:|---|---|
| VLN-CE R2R | 仿真导航 | 1.49M | 教师强制展开，单/多相机 |
| VLN-CE RxR | 仿真导航 | 4.14M | 更长路径、多语言、密集路标 |
| Habitat (MP3D / HM3D) PointNav | 仿真导航 | 984K | 直接接近 / 短程 / 长程 / 命令式 |
| MP3D + HM3D-OVON | 仿真导航 | 2.0M | 骨架图探索轨迹 + VLM 开放式目标标注 |
| EVT-Bench | 仿真跟踪 | 1.49M | 单目标跟踪，拥挤室内场景 |
| nuScenes + OpenScene | 真实/仿真驾驶 | 3.2M | 多视图 + 可选自车状态/历史轨迹 |
| 自动生成视频数据 | 生成视频 | 40K | T2V 生成 + VLM 过滤 + 位姿恢复 |
| 通用 VQA | 开放世界 VQA | ~669K | 维持开放世界视觉理解 |
| 视觉定位 | 开放世界定位 | ~178K | RefCOCO / COCO / Objects365 |
| 图像描述 | 开放世界 Caption | ~6K | 基础视觉-语言对齐 |
| 多图像推理/比较 | 开放世界推理 | ~38K | 跨帧空间推理 |
| 导航专用推理 | 导航推理 | 873K | 自由式 QA + 结构化多视角推理 |
| 离散多轮 VLN | 导航推理 | 362K | CVDN / SOON / REVERIE / SRDF 等 |

---

### A.2 存储空间预估方法

VLA 导航训练数据的存储占用通常由以下部分组成：

1. **原始 RGB 图像/视频**：占绝大部分空间。单张仿真 RGB（640×480）压缩后约 50–150 KB；真实世界图像或视频更高。
2. **3D 场景资产/扫描**：HM3D、MP3D、高保真场景等 mesh/纹理/语义标注，单个场景通常 10–200 MB。
3. **元数据（指令、航点、标注）**：通常仅几 KB 每样本，可忽略。
4. **预提取视觉特征**：若采用 NavFoM 式的特征缓存，空间可压缩 1–2 个数量级。

下面按“**原始图像/视频 + 场景资产**”和“**压缩/特征缓存后**”两个口径分别估算。

---

### A.3 各工作存储占用预估

#### 假设条件

- 仿真 RGB 图像：平均每帧 100 KB；
- 每条训练样本平均包含 4 帧历史/当前图像（部分任务如 Driving、Tracking 帧数更多，VQA 更少，取平均）；
- 视频/网络视频按平均每分钟 5 MB 估算；
- 3D 场景资产平均每个场景 50 MB；
- 预提取特征后，每条样本约占原始图像的 5%–10%。

| 工作 | 数据总量 | 原始图像/视频 + 场景资产 预估 | 预提取特征/压缩后 预估 |
|---|---:|---:|---:|
| **Uni-NaVid** | 5.9M | **~1.5 – 3 TB** | **~80 – 200 GB** |
| **NavFoM** | 12.7M | **~4 – 8 TB** | **~200 – 500 GB** |
| **ABot-N0** | 21.9M | **~8 – 15 TB** | **~400 GB – 1 TB** |
| **Qwen-RobotNav** | 15.6M | **~5 – 10 TB** | **~250 – 700 GB** |

**说明**：
- ABot-N0 的 7,802 个高保真 3D 场景（10.7 km²）本身可能占据 **数百 GB 到 1 TB+** 的场景资产；此外 POI-Goal 使用 Wan2.1-I2V 生成视频、Person-Following 使用 4M 合成跟踪序列，都会显著推高原始视频/图像体积。
- NavFoM 包含 Sekai 的 18.2 万 YouTube 视频（2.03M 伪导航样本），若按平均 2–5 分钟、5 MB/分钟计算，仅原始视频就可能 **1.8 – 4.5 TB**。
- Qwen-RobotNav 明确提到 40K T2V 生成视频；真实场景视频数据（nuScenes 完整约 300GB，OpenScene 数十 GB）也贡献较大比例。

---

### A.4 公开数据集参考大小

| 数据集 | 典型大小 | 说明 |
|---|---|---|
| nuScenes（完整） | ~300 GB | 包含 6 相机、LiDAR、Radar、标注 |
| HM3D（场景资产） | ~50 – 100 GB | 约 1,000 个高保真室内场景 |
| MP3D（场景资产） | ~10 – 20 GB | 90 个场景 |
| Panda-70M | 数 TB 级 | 7000 万视频片段 |
| Sekai | ~1.8 – 4.5 TB | 18.2 万 YouTube 视频（估算） |
| VLN-CE R2R / RxR | < 10 GB | 文本指令 + 导航图 + 小量渲染图 |
| EVT-Bench | < 100 GB | 仿真跟踪数据 |

---

### A.5 小结

- 若按**原始多媒体资产**口径，四个工作的训练数据合计约 **18 – 36 TB**；
- 若按**可训练样本（图像/特征缓存）**口径，合计约 **1 – 3 TB**；
- 实际训练时通常会使用**预提取视觉特征缓存**（如 NavFoM 的粗粒度 token 数据库），可显著降低显存与磁盘 I/O 压力，但原始资产仍需保留用于复现或新特征提取。

---

*报告整理时间：2026-06-23*  
*参考论文：Uni-NaVid (arXiv:2412.06224)、NavFoM (arXiv:2509.12129)、ABot-N0 (arXiv:2602.11598)、Qwen-RobotNav (2026-06-17)*
