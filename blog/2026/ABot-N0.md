---
title: ABot-N0
date: 2026-02-12
categories: [VLA]
---

# ABot-N0：面向通用具身导航的 VLA 基础模型

> **论文**：*ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation*  
> **作者**：AMAP CV Lab, Alibaba Group（通讯作者：Zedong Chu、Shichao Xie；项目负责人：Mu Xu、Xiaolong Wu）  
> **发布时间**：2026-02-12  
> **arXiv**：https://arxiv.org/abs/2602.11598 (arXiv:2602.11598v1 [cs.RO])  
> **项目主页**：https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/

---

## 摘要

ABot-N0 是阿里巴巴 AMAP CV Lab 提出的**统一视觉-语言-动作（Vision-Language-Action, VLA）具身导航基础模型**，首次在单一架构内实现了 Point-Goal、Object-Goal、Instruction-Following、POI-Goal 与 Person-Following 五种核心导航任务的“Grand Unification”，并在 CityWalker、SocNav、VLN-CE (R2R/RxR)、HM3D-OVON、BridgeNav、EVT-Bench 等 **7 个权威基准**上取得 SOTA。

它的核心设计可概括为三点：
1. **Brain-Action 层次架构**：用预训练 LLM（Qwen3-4B）做高层语义推理，用基于 Flow Matching 的 Action Expert 做连续轨迹生成；
2. **规模最大的导航数据引擎**：在 7,802 个高保真 3D 场景（覆盖 10.7 km²）中合成 **16.9M** 条专家轨迹与 **5.0M** 条认知推理样本；
3. **可部署的 Agentic 导航系统**：通过 Agentic Planner、层次化 Topo-Memory 与 Neural Controller，已在 **Unitree Go2** 四足机器人（NVIDIA Jetson Orin NX）上实现 **2Hz VLA 推理 + 10Hz 闭环控制**的真实世界部署。

---

## 一、核心指标与贡献

### 1.1 SOTA 亮点

| 任务/基准 | 关键指标 | ABot-N0 | 对比最佳基线 |
|---|---|---|---|
| Point-Goal 开环 / CityWalker | MAOE ↓ | **11.2** | CityWalker 15.2（↓26.3%） |
| Point-Goal 闭环 / SocNav | SR ↑ / SPL ↑ / DCR ↑ | **88.3% / 79.2% / 85.1%** | CityWalker 47.8% / 44.7% / 36.1% |
| Instruction-Following / R2R-CE Val-Unseen | SR ↑ / SPL ↑ | **66.4% / 63.9%** | NavFoM 61.7% / 55.3% |
| Instruction-Following / RxR-CE Val-Unseen | SR ↑ / SPL ↑ | **69.3% / 60.0%** | NavFoM 64.4% / 56.2% |
| Object-Goal / HM3D-OVON Val-Unseen | SR ↑ / SPL ↑ | **54.0% / 30.5%** | MTU3D 40.8% / 12.1% |
| POI-Goal / BridgeNav SR(0.1m/0.2m/0.3m) | SR ↑ | **32.14% / 71.50% / 88.68%** | OmniNav 18.78% / 46.99% / 72.39% |
| Person-Following / EVT-Bench AT | SR ↑ / TR ↑ | **67.3% / 79.5%** | TrackVLA++ 51.2% / 63.4% |

### 1.2 三大核心贡献

1. **统一的具身导航基础模型与全面 SOTA**：ABot-N0 首次在单一架构内统一五种核心导航任务，在 7 个基准上刷新 SOTA。
2. **规模最大的具身导航数据引擎**：ABot-N0 Data Engine 涵盖 7,802 个高保真 3D 场景（覆盖面积 10.7 km²），合成了约 **16.9M** 条专家轨迹和 **5.0M** 条认知推理样本。
3. **可部署的 Agentic 导航系统**：提出 Agentic Navigation System，集成 Agentic Planner、层次化 Topo-Memory 与 Neural Controller，已在 **Unitree Go2** 四足机器人（NVIDIA Jetson Orin NX）上实现 **2Hz VLA 推理 + 10Hz 闭环控制**的真实世界部署。

---

## 二、研究背景与问题定义

### 2.1 为什么需要统一导航模型？

具身导航是连接高层认知推理与低层连续运动控制的关键桥梁，但现有研究长期陷入“碎片化范式（fragmented paradigm）”。不同任务往往依赖独立的专用架构：
- **Point-Goal** 强调度量坐标下的路径规划与避障；
- **Object-Goal** 需要主动搜索与语义推理；
- **Instruction-Following** 要求长程语言-视觉对齐；
- **POI-Goal** 关注室内外过渡的“最后几米”精确定位；
- **Person-Following** 涉及动态目标跟踪与社会交互。

这种任务隔离限制了跨任务泛化，也阻碍了智能体从大规模异构数据中学习统一的物理先验。

### 2.2 关键洞察与动机

ABot-N0 的核心论点是：

> 尽管导航任务在目标表示上差异巨大（坐标、物体类别、自然语言、POI 名称、动态人物），但它们共享同一套“感知-推理-运动”循环。通过统一的多模态编码、共享的 LLM 认知大脑与分布式的连续动作生成，可以在单一模型中实现多任务统一。

具体而言，论文采用层次化 **“Brain-Action”** 设计：
- **Brain（认知大脑）**：基于预训练 LLM，负责语义理解、空间推理与任务分解；
- **Action（动作专家）**：基于 Flow Matching，负责生成连续、多模态、社会合规的轨迹分布。

### 2.3 统一任务形式化

在 ABot-N0 框架中，所有导航任务被统一为：

**输入**：当前观测 `O_t`、历史视觉记忆 `M^S`、任务指令/目标 `G`、推理任务描述 `R`。

**输出**：局部 BEV 坐标系下的短期航点序列：

```
W = {(x_1, y_1, θ_1), (x_2, y_2, θ_2), ..., (x_5, y_5, θ_5)}
```

其中 `(x_i, y_i)` 表示第 `i` 个航点的位置，`θ_i` 表示局部朝向角。该统一动作表示使得五种任务可在同一动作空间内训练与推理。

### 2.4 五种任务的具体定义

- **Point-Goal Navigation（点目标导航）**：智能体需要到达局部坐标系中精确定义的度量坐标 `(x, y)`。这是最基础的移动原语，强调鲁棒的移动与避障能力。ABot-N0 在训练时通过 4.0M 条轨迹覆盖了从理想最优路径到带噪声真实机器人演示的完整分布。

- **Object-Goal Navigation（物体目标导航）**：智能体在未见环境中主动搜索并导航到特定物体类别（如“冰箱”“沙发”）。该任务严重依赖语义推理与多模态信息融合，且室内布局异质性与环境遮挡显著增加了难度。ABot-N0 不仅支持封闭类别（HM3D ObjectNav），还支持开放词汇查询（OVON）。

- **Instruction-Following Navigation（指令跟随导航）**：智能体执行长程、复杂的自然语言路径指令，如“穿过客厅，经过厨房，在红色沙发旁停下”。该任务强调语言输入与序列动作执行的严格对齐，要求模型在每一步判断指令进度与下一个里程碑。

- **POI-Goal Navigation（兴趣点目标导航）**：该任务由 BridgeNav 提出，要求智能体识别特定 Points of Interest（POI，如“瑞幸咖啡”“金辉大厦”）并精确导航到其物理入口。它桥接了室外粗粒度场景与室内细粒度空间，解决“最后几米”导航挑战。

- **Person-Following Navigation（行人跟随导航）**：智能体实时跟踪动态人体目标，涉及目标检测、运动预测、遮挡处理与社会距离保持。这是人机交互与社会导航中的关键能力。

这五种任务在目标表示、时间尺度、社会约束上差异巨大，但 ABot-N0 通过统一的多模态编码与动作输出将它们纳入同一训练与推理框架。

### 2.5 与相关工作的关系

在 ABot-N0 之前，已有若干工作尝试统一导航任务：
- **Uni-NaVid** 提出基于视频的 VLA 模型统一多种具身导航任务，但在长程语言对齐与社会合规方面仍有局限；
- **NavFoM** 采用 Flow Matching 进行连续环境导航，但在任务覆盖面上不及 ABot-N0；
- **TrackVLA / TrackVLA++** 专注于 Person-Following，缺乏对 POI-Goal、Object-Goal 等任务的统一支持；
- **CityWalker** 从网络视频学习城市导航，但主要针对 Point-Goal；
- **SocialNav** 与本工作密切相关，提出了社会感知导航的基础模型与 SAFE-GRPO 训练框架，ABot-N0 在此基础上扩展到五种任务的统一架构与更大规模数据引擎。

ABot-N0 的差异化在于：它不仅统一了任务接口，还构建了最大规模的数据引擎，并通过 Brain-Action 架构将 LLM 推理能力与 Flow Matching 的连续动作生成能力深度结合，同时通过 Agentic Navigation System 实现了真实世界长程任务部署。

---

## 三、模型架构与方法

### 3.1 整体结构

ABot-N0 采用三层架构：

```
异构输入（RGB 观测 + 视觉历史 + 目标描述 + 推理任务）
            ↓
    Universal Multi-Modal Encoder
            ↓
      Cognitive Brain (Qwen3-4B)
      ├─ Reasoning Head（推理头）
      └─ Action Head（动作头）
            ↓
      Action Expert (Flow Matching)
            ↓
    5 个 BEV 航点 W = {(x, y, θ)} × 5
```

### 3.2 Universal Multi-Modal Encoder（通用多模态编码器）

为了实现五种任务的“Grand Unification”，ABot-N0 设计了灵活的 token 化编码方案。

#### 3.2.1 Flexible Vision Interface（灵活视觉接口）

- **Current Observation `O_t`**：使用 Vision Transformer（ViT）编码 RGB 观测。
  - **全景模式（Panoramic Mode）**：分别输入左、前、右三个视角，并通过视角专属的特殊 token 区分，避免图像拼接带来的几何畸变。
  - **前视模式（Front-View Mode）**：仅处理单个前向图像。
- **Episodic Visual Memory `M^S`**：维护显式的视觉历史缓冲区，将相关历史帧编码后拼接到上下文窗口中，以处理部分可观测马尔可夫决策过程（POMDP）。

#### 3.2.2 Heterogeneous Navigation Target Encoder（异构导航目标编码器）

ABot-N0 通过统一接口解释不同类型的目标：

- **语义目标（文本型）**：Instruction-Following、Object-Goal、POI-Goal、Person-Following 的目标均使用 LLM 的文本 tokenizer 直接嵌入，包括自然语言指令、物体类别、POI 名称或人物描述。
- **几何目标（坐标型）**：Point-Goal 任务中的目标坐标 `(x, y)` 定义在局部 BEV 坐标系下，通过 MLP 投影到共享嵌入维度，作为伪 token（pseudo-tokens）输入 LLM。

#### 3.2.3 Reasoning Task Encoder（推理任务编码器）

该模块注入特定的任务描述，激活 LLM 内的相关推理回路（例如“Where is Luckin Coffee?”或“Identify the zebra crossing area”）。这些任务在训练时作为辅助目标，帮助 Brain 构建对环境鲁棒的表征。

### 3.3 The Cognitive Brain（认知大脑）

ABot-N0 的骨干网络是预训练的 **Qwen3-4B**。它接收推理指令、导航目标、视觉历史与当前观测作为输入。

与近期 LLM 中常用的顺序 CoT（Chain-of-Thought）不同，ABot-N0 采用**任务条件化设计**：
- **Reasoning Head（推理头）** 与 **Action Head（动作头）** 是两个独立分支，基于任务 token 形成“If-Else”关系，而非严格的串行流水线。
- **Cognitive Activation**：训练时显式监督模型执行推理任务，如分析场景可通行性、识别社会规范、grounding 不同 POI 等，以将高层语义表征与物理世界约束对齐。
- **Navigation Decision Support**：对于导航任务，Brain 利用推理任务培养出的丰富物理 grounded 隐状态上下文，直接条件化 Action Expert。

### 3.4 The Action Expert（动作专家）

为了精确运动控制，ABot-N0 采用基于 **Flow Matching** 的动作头，其输入为 Brain 提供的上下文。Action Expert 预测局部 BEV 坐标系下的短期轨迹计划，由 5 个航点组成，每个航点包含 2D 坐标与偏航角：

```
W = {(x_1, y_1, θ_1), (x_2, y_2, θ_2), ..., (x_5, y_5, θ_5)}    (1)
```

其中 `(x_i, y_i)` 表示位置，`θ_i` 表示第 `i` 步的局部朝向角。

论文选择 Flow Matching 而非确定性回归，主要基于两点原因：

1. **连续精度**：Flow Matching 能够生成高精度的连续平移与旋转值，这对平滑机器人控制与稳定朝向调整至关重要。
2. **多模态分布建模**：在大规模导航数据集中，相似的输入条件往往对应多种合理的专家行为（例如从障碍物左侧或右侧绕行）。确定性回归倾向于对这些不同模式取平均，导致无效或碰撞路径；而 Flow Matching 能有效建模这种复杂分布，使规划器采样出反映专家演示多样性的有效轨迹。

#### 3.4.1 Flow Matching 的基本思想

Flow Matching 是一种生成模型方法，它通过定义一条从简单先验分布（如标准高斯噪声）到目标数据分布的连续变换路径，来学习一个向量场 `v_t`。给定条件上下文 `c`（由 Brain 提供），Action Expert 学习条件向量场：

```
dW_t / dt = v_t(W_t | c)
```

其中 `W_t` 表示时刻 `t` 的航点状态。训练时，模型通过 Conditional Flow Matching（CFM）损失监督：

```
L_CFM = E_{t, W_t, c} [ || v_θ(W_t, t | c) − v_t^*(W_t | c) ||^2 ]
```

这里 `v_t^*` 是目标向量场，通常由专家轨迹与高斯噪声之间的线性插值构造。相比扩散模型，Flow Matching 不需要复杂的随机微分方程求解，训练与推理更直接；相比确定性回归，它又保留了生成多模态连续输出的能力。

#### 3.4.2 在 ABot-N0 中的具体实现

在 ABot-N0 中，Action Expert 以 Brain 输出的上下文为条件，预测 5 个航点的连续分布。每个航点包含位置 `(x, y)` 与朝向 `θ`，因此输出空间为 15 维。通过 Flow Matching：
- **推理时**：从先验噪声出发，沿学习到的向量场进行常微分方程（ODE）积分，得到一条确定性的高质量轨迹；也可通过多次采样获得多模态候选轨迹，供后续选择或评估。
- **条件化**：上下文 `c` 不仅包含当前视觉观测与历史，还包含任务 token、目标描述以及 Reasoning Head 产生的语义信息，从而实现“认知引导的动作生成”。

### 3.5 从航点到机器人速度：Neural Controller

在 Agentic Navigation System 中，ABot-N0 生成的 2Hz 航点不足以在密集动态环境中保证稳定避障。因此系统引入基于 **CE-Nav** 的 Neural Controller：
- 利用预训练的 **VelFlow** 专家（条件归一化流模型）作为引导先验；
- 在 Isaac Sim 中通过强化学习微调一个动力学感知的 refiner，使其适配具体机器人硬件与底层运动策略；
- 推理时，Neural Controller 以最新航点与 LiDAR 实时占用图作为输入，输出精确且动力学可行的机体速度指令 `(v_x, v_y, v_yaw)`，运行频率超过 **10Hz**。

---

## 四、数据与训练

### 4.1 ABot-N0 Data Engine 总体规模

ABot-N0 Data Engine 由三层组成：
1. **High-Fidelity 3D Scene Ecosystem**：提供照片级真实、带语义标注的环境；
2. **ABot-N0 Trajectories Dataset**：聚合五种导航范式的专家演示；
3. **ABot-N0 Reasoning Dataset**：为决策提供显式的空间-社会逻辑。

| 数据类别 | 规模 | 说明 |
|---|---|---|
| 高保真 3D 场景 | 7,802 个 | 覆盖 10.7 km²（室内 6.25 km² + 室外 4.42 km²） |
| 导航图总长度 | 384,754 米 | 用于无碰撞轨迹合成 |
| 专家轨迹 | 16.9M | 跨五种导航范式 |
| 认知推理样本 | 5.0M | 激活 VLA Brain 的高层推理能力 |

### 4.2 高保真 3D 场景生态

#### 4.2.1 室内环境：从居家到公共场所

- **居家场景（Residential / Home）**：整合 HM3D 与 InteriorGS，共 **2,318** 个住宅单元，适用于 Object-Goal 与 Instruction-Following。
- **办公场景（Office）**：狭窄走廊、隔间与会议室，考验细粒度运动控制。
- **商场（Shopping Mall）**：总面积 **3.02 km²**，具有复杂拓扑、多层楼面与玻璃门面等反光表面。
- **交通枢纽（Transit Station）**：通道化人流、闸机瓶颈、大型候车厅与狭窄换乘走廊。

#### 4.2.2 室外环境：静态扫描与动态城市

- **真实世界扫描（Real-World Scans）**：利用高精度 LiDAR 与摄影测量重建 **37** 个室外场景。
  - **路口（Intersection）**：学习斑马线、车道、人行道几何；道路网标注严格禁止进入车行道。
  - **公园（Park）**：8 个非结构化公园环境，包含狭窄步道与开放广场。
- **动态虚拟城市 SocCity**：大规模（**3.37 km²**）虚拟城市场景，支持车辆与行人的动态仿真，并提供层次化占用图以区分人行道、斑马线、车行道与动态障碍物，用于训练 **SAFE-GRPO**。

### 4.3 ABot-N0 Trajectories Dataset（16.9M）

| 任务 | 子集/来源 | 轨迹样本数 | 关键说明 |
|---|---|---|---|
| **Point-Goal** | 互联网视频伪轨迹 | 2.0M | π3 稠密三维重建 + MoGe 尺度对齐 |
| | 高保真 3D 场景 | 1.7M | 在导航图上采样 (s,g) 并计算最优路径，含恢复轨迹 |
| | 真实机器人演示 | 340K | SCAND、HuRoN、Recon、CityWalker 遥操作数据 |
| | **小计** | **~4.0M** | |
| **Instruction-Following** | VLN-CE R2R | ~200K | 教师强制展开，从 600K 经动作分布平衡筛选 |
| | VLN-CE RxR | ~1.3M | 更长路径、更复杂拓扑、更密集路标描述；从 1.8M 筛选 |
| | Door-Traversal | 0.3M | 6,000 个 clips，针对狭窄通道与门口 |
| | Language-Guided Person Search | 0.2M | 基于语言查询在 2–5 个 avatar 中定位目标 |
| | Short-Horizon | 0.8M | 12,000 个 clips，旋转/平移/复合原子动作 |
| | **小计** | **~2.8M** | |
| **Object-Goal** | HM3D + OVON | 1.8M | 6 类目标 × 80 场景（HM3D）+ 145 场景开放词汇（OVON） |
| | OVON-sub | 0.2M | 从目标首次可见处截断的短程子集 |
| | InteriorGS | 1.6M | 1,000 场景、700+ 物体类别，基于可见性估计生成 |
| | **小计** | **~3.6M** | |
| **POI-Goal** | BridgeNav 合成 | 2.5M | 分割+深度→局部占用图→A* 规划→Wan2.1-I2V 合成第一人称视频 |
| | **小计** | **2.5M** | |
| **Person-Following** | 合成跟踪序列 | 4.0M | 3 种距离（2.0m/1.5m/1.2m）× 3 类挑战（STT/DT/AT）× 400K + 400K 目标缺失 |
| | **小计** | **4.0M** | |
| **合计** | | **16.9M** | |

#### 4.3.1 Point-Goal 数据工程细节

Point-Goal 数据由三条互补流水线汇聚而成：

1. **互联网视频伪轨迹（2.0M）**：
   - **Structure Recovery**：使用 **π3** 从单目视频进行稠密三维重建；
   - **Metric Alignment**：使用 **MoGe** 解决尺度歧义，将重建路径对齐到真实度量单位；
   - **Episode Synthesis**：沿相机运动流形采样多样化点目标对，生成运动学一致的导航样本。
2. **高保真 3D 场景合成（1.7M）**：
   - 在导航图上采样可达坐标对 `(s, g)`；
   - 使用路径规划算法计算最优路径；
   - 特别加入**恢复轨迹（recovery trajectories）**：将智能体初始化在路径外或近碰撞状态，强制模型学习错误修正行为。
3. **真实机器人演示（340K）**：
   - 来源包括 SCAND、HuRoN、Recon、CityWalker 遥操作数据；
   - 提供真实的物理动力学与传感器噪声特征；
   - 统一转换为标准 point-goal 格式，使模型内化非完整运动约束与传感器-执行延迟。

#### 4.3.2 Object-Goal 在 InteriorGS 上的可见性驱动生成

InteriorGS 包含 1,000 个高保真室内场景与 700+ 物体类别，但缺乏预定义的可见位置标注。为此，论文提出基于**目标可见性估计**的自定义轨迹生成管线：
- 选择最近可见位置作为 clip 终点；
- 避免使用最近自由空间作为终点的朴素策略带来的噪声；
- 最终生成 40K 个 clips，对应 1.6M 训练轨迹。

#### 4.3.3 Instruction-Following 数据工程细节

VLN-CE R2R 与 RxR 是 Instruction-Following 的核心来源。R2R 包含约 10K 个连续 clips，每个 clip 对应一条最短路径导航任务与一段自然语言指令。通过教师强制（teacher forcing）协议将轨迹展开为逐步样本，并采用动作分布平衡策略，将初始 600K 样本筛选为 200K 训练样本。

RxR 在规模与复杂度上显著超过 R2R：路径更长、拓扑更复杂、指令语言密度更高，频繁涉及地标与复杂空间关系。RxR 子集包含约 20K 个连续 clips，初始池 1.8M 样本经平衡后筛选为 1.3M 训练样本。两者合计 1.5M。

此外，论文还专门构造了三种增强数据：
- **Door-Traversal（0.3M）**：在 InteriorGS 门口附近采样初始位姿，强制多样化穿越后朝向与行进距离，并为每条轨迹标注多条不同指令，以提升几何与语言多样性。
- **Language-Guided Person Search（0.2M）**：在场景中随机放置 2–5 个数字人 avatar，根据语言描述指定目标人物，计算到目标的地测最短路径。
- **Short-Horizon（0.8M）**：聚焦短程原子动作，如“左转 60 度”“前进 1 米”“后退 3 米”，用于增强细粒度执行能力并稳定训练。

#### 4.3.4 POI-Goal 数据合成流程

POI-Goal 数据通过以下四步合成：
1. 给定单张输入图像，使用图像分割与深度估计生成局部占用栅格地图；
2. 使用 A* 算法规划从起点到 POI 入口的轨迹作为真值路径；
3. 将输入图像与生成的轨迹输入 **Wan2.1-I2V Diffusion Transformer** 视频生成模型，合成第一人称导航观测视频；
4. 从生成的视频中采样 2.5M 条训练轨迹。

该方法的关键优势在于：无需昂贵的大规模真实 POI 导航采集，即可生成覆盖多样城市场景与入口形态的训练数据。

#### 4.3.5 Person-Following 数据构造

Person-Following 数据参考 TrackVLA 的构造方法，但使用开源的人体运动轨迹自行生成。具体而言：
- 设置三种目标跟随距离：2.0m、1.5m、1.2m；
- 每种距离下生成三类跟踪挑战：Single-Target Tracking（STT，单目标跟踪）、Distracted Tracking（DT，干扰跟踪）、Ambiguity Tracking（AT，遮挡模糊跟踪）；
- 每类距离-挑战组合包含 400K 样本，基础样本共 3.6M（3 距离 × 3 类别 × 400K）；
- 额外补充 400K 目标缺失样本，最终得到 4.0M 训练样本。

每个样本包含当前帧图像、历史帧序列、未来轨迹与对应导航指令。

### 4.4 ABot-N0 Reasoning Dataset（5.0M）

推理数据集旨在为 Brain 提供显式因果逻辑，结构如下：

| 子集 | 规模 | 内容 |
|---|---|---|
| Navigable Areas Analysis | 1.2M | 户外视频流中标注社会合规可通行区域（人行道、斑马线）多边形 |
| Social Navigation CoT | 0.8M | 使用 Qwen-VL-Max 作为教师生成结构化 CoT 推理（如“红灯，必须等待”） |
| Instruction-Following Reasoning | 1.3M | 将长程指令分解为子指令，标注里程碑节点作为 Visual CoT |
| Object-Goal Reasoning | 0.1M | 基于 OVON 生成目标可见性、空间关系、路径规划、原子动作四步推理链 |
| POI Grounding | 0.5M | 街景图像中 POI 名称与入口像素坐标 `(u, v)` 的 VQA 对 |
| General VQA | 1.1M | Blip3、COCO、MAmmoTH-VL、RefCOCO、Objects365、R2R-EnvDrop、ScanQA 等 |
| **合计** | **5.0M** | |

#### 4.4.1 Navigable Areas Analysis（1.2M）

该子集从多样户外视频流中采集 1.2M 张 egocentric 图像，人工标注社会合规可通行区域（如人行道、斑马线）的多边形，并严格排除非可通行区域（车行道、草坪）。模型被训练为根据视觉观测输出这些区域的语义标签与多边形坐标。该任务确保下游任务生成的航点始终位于安全边界内。

#### 4.4.2 Social Navigation CoT（0.8M）

利用 SocialNav 的自动化流水线，以 Qwen-VL-Max 作为教师生成结构化 Chain-of-Thought 推理。每个样本包含决策背后的显式逻辑，例如“交通灯是红色，因此必须等待”。这为社会导航中的复杂交互（避让行人、遵守交通灯）提供了可解释的推理监督。

#### 4.4.3 Instruction-Following Reasoning（1.3M）

将长程指令分解为原子子指令，并为每个子指令标注时空终止点“milestone node”，形成 Visual CoT。通过 Gemini-3 Pro 与 Qwen3-VL 自动处理 VLN-CE R2R/RxR 的 30,000 个 clips，生成 1.3M 训练样本。每个样本包含当前/历史观测、已完成指令、未来指令与预测动作。

#### 4.4.4 Object-Goal Reasoning（0.1M）

基于 OVON 数据集，使用 MLLM 根据全景观测与目标信息生成四步结构化推理链：
1. 目标可见性评估；
2. 空间关系分析；
3. 路径规划；
4. 对应原子动作序列。

通过一致性检查过滤幻觉：仅保留与真值轨迹对齐的样本。最终得到 0.1M 高质量样本。

#### 4.4.5 POI Grounding（0.5M）

从 BridgeNav 数据集中筛选街景图像，利用 Qwen3-VL-Plus 自动生成结构化标注 `⟨POI Name, Entrance Coordinates (u, v)⟩`，共 0.5M VQA 对。该数据集将语义 POI 目标精确 ground 到物理入口像素坐标。

#### 4.4.6 General VQA（1.1M）

整合 Blip3、COCO2014/2017、MAmmoTH-VL、RefCOCO 系列、Objects365、R2R-EnvDrop、ScanQA 等数据集，共 1.1M 样本，用于保持通用视觉-语言表征与增强未知环境泛化能力。

### 4.5 训练策略

ABot-N0 采用三阶段课程学习（curriculum learning）训练流程，以平衡高层语义推理与低层航点控制。

#### 4.5.1 Phase 1: Cognitive Warm-up（认知预热）

- 在“学习如何移动”之前，先学习“看到了什么”和“如何推理”。
- 使用 ABot-N0 Reasoning Dataset 对 LLM 骨干进行微调。
- **冻结 Vision Encoder 与文本 tokenizer**，仅微调 LLM 核心。
- 损失函数为 Next Token Prediction（NTP）交叉熵损失。
- **Action Expert 保持冻结**，确保梯度全部用于优化视觉-语言表征与认知基础。

#### 4.5.2 Phase 2: Unified Sensorimotor SFT（统一感觉运动监督微调）

- 引入 ABot-N0 Trajectory Dataset，将五种导航任务统一为单一多任务训练体制。
- 为防止灾难性遗忘，采用**混合训练策略**：在轨迹数据中加入约 **20%** 的推理数据回放缓冲区。
- **双头优化（Dual-Head Optimization）**：联合优化 AR Head（自回归推理）与 Action Expert（Flow Matching）。
- 联合损失函数为：

```
L_Phase2 = λ_txt L_NTP(θ_brain) + λ_flow L_CFM(θ_action | θ_brain)    (2)
```

其中 `L_NTP` 为文本生成的交叉熵损失，`L_CFM` 为条件 Flow Matching 损失。该方式将高层语义计划 ground 到精确的连续物理动作中。

#### 4.5.3 Phase 3: Post-Training Value Alignment via SAFE-GRPO

- 第二阶段通过模仿学习（IL）获得通用导航能力，但 IL 只能捕捉专家行为的表层统计，难以掌握复杂社会环境中的因果结构。
- 因此引入 **SAFE-GRPO（Socially-Aware Flow Exploration GRPO）**，一种基于流的强化学习框架，显式强制社会合规。
- **冻结 Brain**，仅微调 Action Expert。
- 在 SocCity 的专家轨迹上训练，利用其丰富标注计算精确奖励。
- 复合奖励函数为：

```
R = w_soc R_social + w_exp R_expert + w_sm R_smooth + w_eff R_eff    (3)
```

各项含义：
- **R_social（社会合规奖励）**：基于 SocCity 提供的真值语义占用图。若预测轨迹穿越不可通行或社会限制区域（草坪、限制车道、行人），给予重罚，使模型内化“可通行几何 ≠ 社会可接受路径”。
- **R_expert（专家相似奖励）**：防止奖励作弊，鼓励策略保持在专家演示的合理分布内。
- **R_smooth / R_eff（平滑与效率奖励）**：惩罚抖动运动，鼓励向目标前进。

通过该价值对齐后训练，ABot-N0 在执行所有导航任务时都能严格遵守社会规范。

---

## 五、实验评估

### 5.1 评估设置概述

论文在 7 个权威基准上对 ABot-N0 进行了全面评估：
- **Point-Goal**：CityWalker（开环）、SocNav（闭环）
- **Instruction-Following**：VLN-CE R2R-CE / RxR-CE Val-Unseen
- **Object-Goal**：HM3D-OVON
- **POI-Goal**：BridgeNav
- **Person-Following**：EVT-Bench

#### 5.1.1 主要评估指标说明

| 指标 | 全称 | 含义 | 适用任务 |
|---|---|---|---|
| MAOE ↓ | Maximum Average Orientation Error | 未来 horizon 内预测动作与真值动作的最大角度偏差 | Point-Goal 开环 |
| SR ↑ | Success Rate | 智能体在阈值内到达目标的成功率 | 通用 |
| SPL ↑ | Success weighted by Path Length | 成功率按路径长度加权，惩罚冗余路径 | 通用 |
| RC ↑ | Route Completion | 实际完成路径占最优路径的比例 | Point-Goal 闭环 |
| DCR ↑ | Distance Compliance Rate | 行驶距离中位于社会可通行区域的比例 | SocNav |
| TCR ↑ | Time Compliance Rate | 行驶时间中位于社会可通行区域的比例 | SocNav |
| NE ↓ | Navigation Error | 终点到目标的地测距离 | VLN-CE |
| OS ↑ | Oracle Success | 在任意时刻进入目标阈值内的比例 | VLN-CE |
| TR ↑ | Tracking Rate | 目标保持在有效距离与视野内的时间比例 | Person-Following |
| CR ↓ | Collision Rate | 碰撞发生率 | Person-Following |

这些指标共同衡量了导航的准确性、效率、安全性与社会合规性。

### 5.2 Point-Goal 任务

#### 5.2.1 CityWalker 开环评估

| Method | Mean ↓ | Turn ↓ | Crossing ↓ | Detour ↓ | Proximity ↓ | Crowd ↓ | Other ↓ | All ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GNM | 16.2 | 31.1 | 14.8 | 12.5 | 14.7 | 12.8 | 11.0 | 12.1 |
| ViNT | 16.5 | 31.1 | 15.4 | 12.9 | 14.8 | 13.3 | 11.6 | 12.6 |
| NoMaD | 19.1 | 35.1 | 18.5 | 15.6 | 18.1 | 14.3 | 12.8 | 12.1 |
| CityWalker | 15.2 | 26.6 | 14.1 | 13.9 | 14.3 | 12.0 | 10.4 | 11.5 |
| **ABot-N0** | **11.2** | **21.3** | **9.8** | **12.8** | **8.1** | **8.8** | **6.3** | **7.6** |

MAOE（Maximum Average Orientation Error）衡量未来 horizon 内预测动作与真值动作的最大角度偏差，越低越接近人类驾驶行为。ABot-N0 平均 MAOE 从 CityWalker 的 15.2 降至 11.2。

#### 5.2.2 SocNav 闭环评估

| Method | SR ↑ | RC ↑ | SPL ↑ | DCR ↑ | TCR ↑ |
|---|---:|---:|---:|---:|---:|
| GNM* | 43.3 | 62.4 | 37.0 | 26.5 | 28.7 |
| ViNT* | 45.6 | 66.2 | 39.5 | 31.4 | 33.8 |
| NoMaD* | 41.1 | 60.5 | 35.4 | 29.5 | 31.6 |
| CityWalker | 47.8 | 64.7 | 44.7 | 36.1 | 36.6 |
| **ABot-N0** | **88.3** | **92.1** | **79.2** | **85.1** | **85.4** |

ABot-N0 在 SR 上接近翻倍（88.3% vs 47.8%），更重要的是 DCR（距离合规率）与 TCR（时间合规率）分别达到 85.1% 与 85.4%，证明模型在到达目标的同时主动尊重社会规范。

### 5.3 Instruction-Following 任务

在 VLN-CE R2R-CE 与 RxR-CE Val-Unseen 上的结果：

| Method | Obs. | R2R-CE Val-Unseen | | | RxR-CE Val-Unseen | | |
|---|---|---|---:|---:|---:|---:|---:|
| | RGB | Pano | NE ↓ | OS ↑ | SR ↑ | SPL ↑ | NE ↓ | SR ↑ | SPL ↑ |
| StreamVLN | ✓ | | 4.98 | 64.2 | 56.9 | 51.9 | 6.22 | 52.9 | 46.0 |
| InternVLA-N1 (S1+S2) | ✓ | ✓ | 4.83 | 63.3 | 58.2 | 54.0 | 5.91 | 53.5 | 46.1 |
| NavFoM (Four views) | ✓ | | 4.61 | 72.1 | 61.7 | 55.3 | 4.74 | 64.4 | 56.2 |
| **ABot-N0** | ✓ | | **3.78** | **70.8** | **66.4** | **63.9** | **3.83** | **69.3** | **60.0** |

注：表中仅列出代表性方法，完整表格见原论文 Table 3。

- R2R-CE 上 SR 66.4%，超越 NavFoM 4.7%，SPL 提升 8.6%；
- RxR-CE 上 SR 与 SPL 分别达到 69.3% 与 60.0%。

### 5.4 Object-Goal 任务

在 HM3D-OVON 上的结果：

| Method | Obs. | Val-Seen | | Val-Seen-Synonyms | | Val-Unseen | |
|---|---|---|---:|---:|---:|---:|---:|
| | RGB | Pano | Depth | Odo | SR ↑ | SPL ↑ | SR ↑ | SPL ↑ | SR ↑ | SPL ↑ |
| MTU3D | ✓ | ✓ | ✓ | ✓ | 55.0 | 23.6 | 45.0 | 14.7 | 40.8 | 12.1 |
| NavFoM (Four views) | ✓ | | | | 40.1 | 27.1 | 45.4 | 32.6 | 45.2 | 31.9 |
| **ABot-N0** | ✓ | | | | **55.3** | **32.1** | **55.4** | **33.2** | **54.0** | **30.5** |

- ABot-N0 仅用 RGB 输入即超越需要深度/里程计的 MTU3D；
- 在 Val-Unseen 上超越 MTU3D 13.2% SR；
- MTU3D 从 Val-Seen 到 Val-Unseen 下降 14.2%，而 ABot-N0 仅下降 1.3%，显示极强的泛化能力。

### 5.5 POI-Goal 任务

在 BridgeNav 上的结果：

| Method | SR(0.1m) ↑ | SR(0.2m) ↑ | SR(0.3m) ↑ | TR(mean) ↓ | TR(best) ↓ | TR(worst) ↓ |
|---|---:|---:|---:|---:|---:|---:|
| NoMaD | 4.13 | 15.07 | 29.20 | 31.35 | 5.45 | 85.91 |
| Citywalker | 13.79 | 41.02 | 65.96 | 15.58 | 0.76 | 56.47 |
| OmniNav | 18.78 | 46.99 | 72.39 | 14.16 | 0.99 | 53.79 |
| **ABot-N0** | **32.14** | **71.50** | **88.68** | **9.84** | **0.44** | **51.38** |

- 在最严格的 0.1m 阈值下，ABot-N0 取得 32.14% SR，相对 OmniNav 提升 70.1%；
- 平均轨迹偏差 TR(mean) 降低 30.5%，最佳场景提升 55.6%。

POI-Goal 任务要求智能体从室外场景识别并精确进入特定 POI 入口，入口宽度在现实场景中差异很大。ABot-N0 在 0.1m 阈值下的显著优势说明其具备卓越的细粒度控制能力，能够成功通过基线经常失败的狭窄入口。路径效率指标的提升则表明其生成的轨迹不仅到达目标，而且更接近几何最优路径。

### 5.6 Person-Following 任务

在 EVT-Bench 单目设置上的结果：

| Method | Single-Target Tracking (STT) | | | Distracted Tracking (DT) | | | Ambiguity Tracking (AT) | | |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | SR ↑ | TR ↑ | CR ↓ | SR ↑ | TR ↑ | CR ↓ | SR ↑ | TR ↑ | CR ↓ |
| TrackVLA++ | 86.0 | 81.0 | 2.10 | 66.5 | 68.8 | 4.71 | 51.2 | 63.4 | 15.9 |
| **ABot-N0** | **86.9** | **87.6** | 8.54 | **66.7** | **75.4** | 11.6 | **67.3** | **79.5** | **7.05** |

- 简单 STT 任务 SR 86.9%，超越 TrackVLA++ 0.9%；
- 最具挑战的 AT 任务中，SR 与 TR 分别提升 16.1%，展示了强大的目标预测与重识别能力。

值得注意的是，ABot-N0 在 AT 任务中的提升最为显著。AT 场景包含频繁严重遮挡，需要目标重识别与运动预测能力，而这正是 Brain-Action 架构与视觉历史记忆的优势所在。相比之下，反应式基线在面对遮挡时更容易丢失目标。

### 5.7 Agentic Navigation System 架构

为了将 ABot-N0 基础模型能力扩展到复杂真实世界任务，论文提出了 **Agentic Navigation System**。该系统包含四个核心模块：Agentic Planner、Actor、短期 Episodic Memory 与长期 Topo-Memory。整个任务执行过程被建模为 POMDP。

#### 5.7.1 系统组成

```
用户指令 I
    ↓
Agentic Planner (VLM, 云端 RTX 4090)
    ├── 解析模糊意图
    ├── 查询 Topo-Memory M^L
    ├── CoT 分解为子任务 G = {g_1, g_2, ..., g_n}
    └── 失败时触发 Self-Reflector 重新规划
            ↓
Actor (ABot-N0 + Neural Controller, 边缘 Jetson Orin NX)
    ├── ABot-N0: 2Hz 航点生成
    └── Neural Controller: 10Hz 速度控制
            ↓
观测/轨迹反馈 → Episodic Memory M^S / Topo-Memory M^L
```

#### 5.7.2 Map as Memory：Topo-Memory

Topo-Memory 是一个层次化拓扑记忆模块，将地图视为持续更新的外部记忆，而非静态背景。它包含四层：

| 层级 | 抽象粒度 | 作用 |
|---|---|---|
| Block Layer | 房间/街区 | 粗粒度跨区域定位与长程任务分解 |
| Road Layer | 路口/门口 | 刻画物理连通性，提供刚性可达性约束 |
| Function Layer | 休息区/厨房/电梯厅 | 将抽象语言意图翻译为功能可达目标 |
| Object/POI Layer | 具体物体/商铺 | 作为“最后几米”Object/POI-Goal 的视觉-语义锚点 |

此外，系统采用 **“One Map”** 策略统一室内外空间表示，整合 AMAP 全局路由与实时视觉决策，可执行“离开家→出小区→过街→到达 3 公里外商场内某餐厅”这类多阶段任务。

#### 5.7.3 Agentic Planner

Planner `P` 接收用户指令 `I`、当前观测 `O_t`、Topo-Memory `M^L` 与 Episodic Memory `M^S`，通过 CoT 推理将指令分解为可执行子任务序列：

```
G = P(I, M^L, M^S, O_t)
```

Planner 提供三项关键能力：
- **模糊性消解**：将自由自然语言解析为结构化子任务序列；
- **记忆感知规划**：先检索 `M^L`，例如已知厨房在 `(x, y)`，则优先生成 Point-Goal 任务快速接近，再切换局部 Object-Goal 策略；
- **粗到细分解（Coarse-to-Fine）**：利用 Point-Goal 完成长程 Approaching，利用 Object/POI-Goal 完成局部 Reaching，利用 Instruction-Following/Person-Following 完成 Interaction。

数学上，整个规划-执行过程可分解为：

```
P(W | I, M^L, M^S) = P(G | I, M^L, M^S) · ∏_{j=1}^{N} P(W^j | g_j, M^L, M^S)
        └─ Agentic Planning ─┘        └──────── ABot-N0 ────────┘
```

#### 5.7.4 Self-Reflector 与重规划

每个子任务 `g_i` 执行完毕后，VLM-based Self-Reflector `S` 评估完成状态：

```
(r, f) = S(M^L, M^S, g_i)
```

若失败（`r = False`），系统利用反馈 `f` 触发重规划：

```
G' = P(I, M^L, M^S, O_t, f)
```

论文中的典型案例是“我要可乐”：Planner 首先规划去零食架（Snack Rack），Self-Reflector 发现没有可乐后，反馈“Coke not found on the snack rack”，系统重规划前往自动售货机（Vending Machine）并成功找到可乐。

#### 5.7.5 Neural Controller 与闭环控制

ABot-N0 生成航点的频率为 2Hz，不足以保证密集动态环境中的实时避障。Neural Controller 基于 CE-Nav 框架：
- 以预训练 VelFlow 专家为条件归一化流先验；
- 在 Isaac Sim 中通过 RL 微调出适配 Unitree Go2 运动学的 refiner；
- 推理时接收最新航点与 LiDAR 实时占用图，输出机体速度 `(v_x, v_y, v_yaw)`，频率超过 10Hz。

#### 5.7.6 真实世界部署硬件与性能

| 组件 | 配置 |
|---|---|
| 机器人平台 | Unitree Go2 X（12 个驱动自由度） |
| 视觉感知 | 3 个单目 RGB 相机，合计水平 FOV ≈ 270° |
| 几何感知 | Unitree 4D LiDAR L2（全向点云） |
| 全局定位 | RTK-GNSS |
| 边缘计算 | NVIDIA Jetson Orin NX（157 TOPS，16GB RAM） |

- **混合云-边缘架构**：Planner 在云端 RTX 4090 运行；ABot-N0 与 Neural Controller 在 Jetson Orin NX 本地运行，保证低延迟与断网安全。
- **边缘优化**：采用 93M SigLIP-B/16 视觉骨干 + token merging（merge size=4），在仅损失 3% 性能的前提下实现 2Hz VLA 推理。
- **验证场景**：覆盖 Point-Goal、Object-Goal、POI-Goal、Instruction-Following、Person-Following 等单任务，以及室内外长程复合任务（如“带我去奶茶店买伯牙绝弦并占座”“带我去公园放松一下”）。

---

## 六、消融研究

原技术报告并未设立独立的消融实验章节，而是在方法描述中隐含了若干关键设计选择。下面根据论文内容总结可被视作消融洞察的要点：

### 6.1 Flow Matching vs 确定性回归

论文明确指出，采用 Flow Matching 而非确定性回归的原因在于：
- 大规模导航数据中相似输入对应多种有效专家行为；
- 确定性回归会平均化不同模式，产生无效或碰撞路径；
- Flow Matching 能够采样多模态轨迹，保持专家行为多样性。

这可以视为对动作生成头选择的关键消融。

### 6.2 混合训练策略（20% 推理回放）

在 Phase 2 中，论文通过在轨迹数据中加入约 20% 的推理数据回放缓冲区，防止了灾难性遗忘。这一比例是维持 Brain 推理能力与 Action Expert 运动能力平衡的重要超参。

### 6.3 SAFE-GRPO 对社会合规的影响

SocNav 闭环评估中 DCR/TCR 的大幅提升（85.1%/85.4% vs 基线 36.1%/36.6%）可视为 SAFE-GRPO 价值对齐有效性的强间接证据。该后训练阶段冻结 Brain、仅微调 Action Expert，说明社会合规性主要体现在低层动作分布上。

### 6.4 边缘端 token merging 的精度-效率权衡

真实部署中，通过 token merging（merge size=4）将边缘端 VLA 推理提升到 2Hz，性能仅下降 3%。这验证了视觉 token 压缩策略在边缘部署中的有效性。

---

## 七、关键洞察与局限

### 7.1 关键洞察

1. **Brain-Action 层次化解耦是关键**：将高层语义推理（LLM Brain）与低层连续动作生成（Flow Matching Action Expert）解耦，使模型既能利用 LLM 的常识推理，又能生成精确、多模态的轨迹。这种解耦避免了“所有计算都压在 LLM 内”导致的连续动作精度不足，也避免了“纯反应式策略”缺乏语义理解的问题。

2. **统一动作表示实现 Grand Unification**：所有任务共享同一组 BEV 航点输出，使异构任务能在单一训练目标下联合学习。这种“异构输入、同构输出”的设计是 ABot-N0 能够统一五种任务的核心机制。

3. **数据规模与社会合规并重**：16.9M 轨迹提供强大泛化基础，而 SAFE-GRPO 将社会规范显式注入动作分布，避免“可通行几何 ≠ 社会可接受路径”的隐患。SocNav 闭环评估中 DCR/TCR 的大幅提升是社会价值对齐成功的直接证据。

4. **Agentic 系统弥补基础模型局限**：通过 Agentic Planner、Topo-Memory 与 Self-Reflector，将短程 VLA 能力扩展为长程、可容错的真实世界任务执行系统。基础模型负责“当下这一步怎么走”，Agentic 系统负责“全局任务怎么拆”。

5. **视觉历史与目标编码的灵活性**：全景/前视模式、文本/坐标目标、任务条件化双头设计，使同一模型适配不同机器人形态与任务输入。这种灵活性对于真实世界部署至关重要。

6. **推理数据作为正则化与对齐手段**：5.0M 推理样本不仅提升了 Brain 的语义理解能力，还在 Phase 2 中作为 20% 回放缓冲区防止灾难性遗忘，使模型在学会运动的同时保持推理能力。

7. **从仿真到真实的工程路径**：ABot-N0 展示了“大规模仿真/合成数据训练 → 安全对齐 → Agentic 系统封装 → 边缘部署”的完整工程路径，为具身智能落地提供了可复用的范式。

### 7.2 局限

1. **缺乏显式消融实验**：论文未提供详尽的模块消融表（如去掉推理数据、去掉 SAFE-GRPO、不同 waypoint 数量、不同 Flow Matching 采样步数等），难以量化每个设计选择的边际贡献。对于希望复现或改进该工作的研究者而言，这是一个信息缺口。

2. **训练计算成本未披露**：16.9M 轨迹 + 5.0M 推理样本 + 三阶段训练 + Flow Matching + SAFE-GRPO 的具体 GPU 小时数、batch size、学习率、训练周期等细节未在论文中完整给出。这使得外部研究者难以准确估计复现成本与模型规模效应。

3. **真实世界量化评估有限**：虽然论文展示了大量可视化案例，但真实场景中的统计性成功率、失败模式分析、与仿真指标的 gap、动态行人密度对性能的影响等定量结果较少。大部分真实世界验证仍停留在定性演示阶段。

4. **边缘部署性能损失**：token merging 带来 3% 性能下降，在更严苛的精度场景（如 0.1m POI 导航）下可能更为明显。如何在资源受限设备上进一步缩小与云端模型的差距仍是一个开放问题。

5. **任务条件化设计的可解释性**：Reasoning Head 与 Action Head 的“If-Else”分支机制虽然高效，但其内部如何共享/隔离知识、推理任务对导航性能的具体贡献比例、以及双头之间是否存在信息瓶颈，尚缺乏深入分析。

6. **对 SocCity 仿真环境的依赖**：SAFE-GRPO 依赖 SocCity 提供的真值语义占用图计算社会合规奖励。在向真实世界迁移时，如何获得同等精度的语义占用图、以及如何处理标注缺失或错误的场景，是实际部署中需要解决的问题。

7. **开放词汇导航的边界**：虽然 OVON 与 POI-Goal 支持开放词汇查询，但模型对极端罕见物体、模糊语言描述或文化特定 POI 名称的处理能力仍有待验证。

### 7.3 未来方向

1. **更细粒度的消融与可解释性分析**：未来工作应系统地消融推理数据比例、Flow Matching 采样步数、waypoint 数量、视觉历史长度等超参，以量化每个组件的贡献。

2. **真实世界大规模量化评估**：需要建立标准化的真实世界导航基准，评估长程任务成功率、社会合规率、人机交互舒适度等指标。

3. **在线学习与终身导航**：Topo-Memory 已具备动态更新能力，未来可结合在线学习与持续学习，使机器人在长期运行中不断提升效率与适应性。

4. **更轻量的边缘部署**：探索模型蒸馏、量化、神经架构搜索等方法，在保持性能的同时进一步降低边缘延迟与功耗。

5. **多机器人/多智能体协同**：将 ABot-N0 扩展到多机器人协同导航与人群密集场景中的社会交互规划。

---

## 八、结论

ABot-N0 代表了具身导航领域向**统一基础模型**迈进的重要一步。它通过 **Brain-Action** 层次架构、**Flow Matching** 动作生成、**16.9M 专家轨迹 + 5.0M 推理样本**的大规模数据引擎，以及 **SAFE-GRPO** 社会价值对齐，首次在单一 VLA 模型中实现了五种核心导航任务的 Grand Unification，并在 7 个权威基准上取得 SOTA。

更进一步，ABot-N0 并未止步于仿真指标。论文提出的 **Agentic Navigation System** 通过 Agentic Planner、层次化 **Topo-Memory** 与高速 **Neural Controller**，将基础模型能力成功部署到 **Unitree Go2** 四足机器人上，在真实室内外动态环境中实现了 2Hz VLA 推理与 10Hz 闭环控制，验证了从仿真到真实世界的可迁移性。

对于希望构建通用具身 Agent 的研究者与工程师而言，ABot-N0 提供了一个重要的范式参考：**以统一的动作表示连接异构任务，以层次化架构桥接认知与运动，以大规模数据与社会对齐保证泛化与安全**。尽管报告在消融实验、训练成本与真实世界量化评估方面仍有完善空间，但其在统一导航基础模型、数据引擎规模与真实部署闭环方面的贡献，使其成为具身导航领域的重要里程碑。

---

*参考：ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation (arXiv:2602.11598v1, Feb 12, 2026)*
