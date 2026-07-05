---
title: NavFoM：跨具身、跨任务的导航基础模型
date: 2025-09-16
categories: [VLA]
---

# NavFoM：跨具身、跨任务的导航基础模型

> **论文**: *Embodied Navigation Foundation Model* (NavFoM)  
> **作者**: Jiazhao Zhang*, Anqi Li*, Yunpeng Qi*, Minghan Li*, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang†, He Wang†  
> **机构**: 北京大学、Galbot、USTC、BAAI、阿德莱德大学、浙江大学、Differential Robotics  
> **arXiv**: [arXiv:2509.12129v2](https://arxiv.org/abs/2509.12129)  
> **项目主页**: [https://pku-epic.github.io/NavFoM-Web/](https://pku-epic.github.io/NavFoM-Web/)

---

## 导语：导航也需要自己的“Foundation Model”

在具身智能（Embodied AI）的版图中，导航是最基础、也最具挑战的能力之一。无论是室内轮式机器人跟随语言指令到达目标，无人机在陌生社区中搜索地标，还是自动驾驶车辆规划未来轨迹，它们的本质都是同一种智能：**根据 ego-centric 视觉观测和语言/目标指令，在三维空间中做出移动决策**。

然而，长期以来，这些任务被割裂为不同的研究社区：VLN（Vision-and-Language Navigation）、ObjNav / OVON（Object Goal Navigation）、主动视觉跟踪（EVT）、无人机导航（OpenUAV）、自动驾驶（nuScenes / NAVSIM）。每个社区有自己的数据集、指标、模型结构，甚至不同的相机配置与动作空间。这种割裂导致模型往往只能“专精一门”，难以像 LLM 那样从海量异构数据中学习到通用能力。

2025 年 9 月，来自北京大学、Galbot 等机构的研究者发布了 **NavFoM（Embodied Navigation Foundation Model）**，试图为导航领域构建一个统一的“基础模型”。NavFoM 将多种导航任务统一为“多视角视频 + 语言指令 → 未来航点轨迹”的生成问题，通过 **TVI tokens** 与 **BATS** 等关键设计，在 800 万导航样本与 476 万开放世界 QA 样本上联合训练，最终在 7 个公开基准上取得 SOTA 或极具竞争力的性能，并成功部署到四足、人形、无人机、轮式机器人等真实平台。

---

## 一、核心贡献速览

### 1.1 一句话总结

NavFoM 是一个面向 **cross-embodiment（跨具身）** 与 **cross-task（跨任务）** 的导航基础模型，把 VLN、目标搜索、主动视觉跟踪、自动驾驶等任务统一为同一生成框架，并通过显式时空建模与预算感知采样，在多个基准上实现 SOTA，同时展现跨真实机器人平台的部署能力。

### 1.2 SOTA 亮点

| 任务 / 基准 | 设置 | 关键结果 | 对比基线 / 说明 |
|---|---|---|---|
| VLN-CE R2R Val-Unseen | 单目前向 RGB | **56.2% SR / 51.2% SPL** | 超越 StreamVLN-RGB-only (55.7% SR / 50.9% SPL) |
| VLN-CE R2R Val-Unseen | 四目环视 RGB | **61.7% SR / 55.3% SPL** | 超越此前 RGB-D+里程计 SOTA HNR (61.0% SR / 51.0% SPL) |
| VLN-CE RxR Val-Unseen | 单目前向 RGB | **57.4% SR / 49.4% SPL / 60.2 nDTW** | 超越 StreamVLN (51.8% SR / 45.0% SPL) |
| VLN-CE RxR Val-Unseen | 四目环视 RGB | **64.4% SR / 56.2% SPL / 65.8 nDTW** | 超越 HNR (56.3% SR / 46.7% SPL / 63.5 nDTW) |
| OpenUAV (L1, UO Full) | 四目无人机 | **29.83% SR / 27.20% SPL** | 超越 TravelUAV (22.42% SR / 20.51% SPL) |
| HM3D-OVON Val-Unseen | 单目 / 四目 zero-shot | **43.6% SR / 31.3% SPL** → **45.2% SR / 31.9% SPL** | 超越此前 SOTA MTU3D (40.8% SR / 12.1% SPL) 与 Uni-NaVid* (39.5% SR / 19.8% SPL) |
| EVT-Bench 单目标 | 单目 / 四目 | **85.0% SR / 80.5% TR** → **88.4% SR / 80.7% TR** | 超越 TrackVLA (85.1% SR / 78.6% TR) |
| EVT-Bench 干扰目标 | 单目 / 四目 | **61.4% SR / 68.2% TR** → **62.0% SR / 67.9% TR** | 超越 TrackVLA (57.6% SR / 63.2% TR) |
| NAVSIM navtest | 八目环视 | PDMS **84.3** | 与专用端到端驾驶模型相当 |
| nuScenes 开环 | 六目环视 | Avg L2 **0.42 m** / Avg Collision **0.12%** | 未显式建模车道/他车 |

> 注：除特别标注外，所有结果均 **无需针对特定任务或相机配置做 fine-tuning**，直接同一模型推理。

### 1.3 三大核心创新

1. **时序-视角指示 token（TVI tokens）**：显式编码每个视觉 token 的相机方位角与时间步，使模型能区分不同相机配置和历史长度，支持 image QA、video QA、navigation 三种样本在统一框架下共调。
2. **预算感知的时序采样策略（BATS）**：受遗忘曲线启发，在固定 token 预算下动态平衡近期帧与历史帧，保证推理速度与显存可控。
3. **跨任务、跨具身的统一训练框架**：构建 802 万导航样本 + 476 万开放世界 QA 样本，提出离线特征缓存、原子动作轨迹化等工程优化，将训练速度提升 2.9 倍、显存降低 1.8 倍。

---

## 二、背景：为什么导航需要统一模型？

### 2.1 任务割裂与具身割裂

具身导航的研究长期被以下问题割裂：

- **任务割裂**：VLN 关注长语言路径指令，ObjNav / OVON 关注目标类别/描述，主动跟踪关注动态目标，自动驾驶关注车辆轨迹。它们的输入输出形式、评价指标、模型结构各不相同。
- **具身割裂**：轮式机器人、四足、无人机、车辆的相机布局、运动学、观察空间差异巨大，多数方法只能服务于特定平台。
- **数据割裂**：每个任务/平台各自收集数据，难以形成规模效应，模型也无法从异构导航经验中迁移通用知识。

近年来，Vision-Language Models（VLMs）在开放世界理解上展现出强大泛化能力，但它们在导航领域的应用仍多局限于单一任务或单一具身。例如：

- **NaVid** 专注于视频式 VLN；
- **Uni-NaVid** 尝试统一多个室内导航任务；
- **NaVILA** 面向四足机器人；
- **StreamVLN** 强调流式上下文建模；
- **ViNT / GNM** 探索视觉导航的表征学习。

这些工作虽然各自取得了显著进展，但尚未有一个模型能够同时在 **多种任务** 与 **多种具身** 上实现统一训练与零样本部署。NavFoM 的目标正是打破这种割裂。

### 2.2 NavFoM 的关键洞察

NavFoM 的核心洞察来自两点观察：

1. **人类主要依赖视觉完成导航**。这与 GNM、ViNT、NaVid 等“纯视觉”导航方法的成功一致。因此，可以把通用导航任务统一为：输入 ego-centric RGB 视频（单目或多目）和语言指令，输出可执行轨迹。

2. **观察序列的“时空身份”必须显式编码**。当模型同时处理不同相机数量（1、2、3、4、6、8 目）、不同历史长度（室内几十步、户外几百米、驾驶连续帧）以及 image/video/navigation 三种不同样本时，普通的位置编码或简单拼接会导致 LLM 无法区分 token 来自哪个视角、哪个时刻、哪种任务。因此需要 TVI tokens 作为“结构化提示”。

---

## 三、统一任务形式化

NavFoM 将所有导航任务统一为如下形式：

给定语言指令 $L$ 和多相机 RGB 观测序列 $I_{1:T}^{1:N}$，模型 $\pi$ 输出未来航点轨迹：

$$
\tau = \{a_1, a_2, \dots \}, \quad a \in \mathbb{R}^4 = (x, y, z, \theta)
$$

其中 $(x,y,z)$ 表示位置，$\theta$ 表示偏航角。$z$ 维度仅在无人机中使用；地面机器人/车辆使用 $(x, y, \theta)$。

统一映射写作：

$$
\pi\bigl(L, I_{1:T}^{1:N}\bigr) \mapsto \tau_T
$$

这一形式化兼容现有大多数导航任务：

- **VLN**：提供长语言路径指令；
- **OVON / ObjNav**：提供目标类别或描述；
- **主动视觉跟踪**：提供目标外观描述；
- **自动驾驶**：提供高层驾驶指令或车辆状态。

---

## 四、模型架构与方法

### 4.1 整体流程

NavFoM 基于现成的视频型 VLM（Qwen2-7B）扩展而来，整体流程如下：

```
多视角 RGB 图像序列 I_{1:T}^{1:N}
    ↓
[DINOv2 + SigLIP] 视觉编码 → 视觉特征 V
    ↓
Grid Average Pooling（细粒度 64 tokens / 粗粒度 4 tokens）
    ↓
Cross-Modality Projector P(·)（2 层 MLP）→ 视觉 token E^V
    ↓
TVI tokens 组织 + BATS 采样
    ↓
与语言 token E^L 拼接 → Qwen2-7B LLM
    ↓
Action token 隐藏状态 E^A_T
    ↓
Planning Head A_θ（3 层 MLP）→ 归一化未来 8 个航点
    ↓
乘以任务级缩放因子 α_task → 绝对坐标轨迹 τ_T
```

核心公式：

$$
E_T^A = \text{LLM}\bigl(E_{1:T}^{1:N}, E^L\bigr), \qquad \tau_T = A_\theta\bigl(E_T^A\bigr)
$$

对于 QA 任务，模型使用常规 LM Head 进行自回归 next-token 预测，与导航分支共享 LLM backbone。

### 4.2 观测编码与 Grid Pooling

视觉编码器采用 **DINOv2** 与 **SigLIP** 的串联方案。对于每帧每个视角，先得到 $P = 576$ 个 patch 特征：

$$
V_{1:T}^{1:N} \in \mathbb{R}^{P \times C}
$$

为降低长历史视频带来的 token 数量爆炸，使用 **Grid Average Pooling** 在 patch 维度进行下采样：

$$
V^{\text{fine/coarse}} = \text{GridPool}\bigl(V, \tfrac{64}{P} \text{ 或 } \tfrac{4}{P}\bigr)
$$

- **细粒度** $V^{\text{fine}} \in \mathbb{R}^{64 \times C}$：用于当前最新帧与 image QA；
- **粗粒度** $V^{\text{coarse}} \in \mathbb{R}^{4 \times C}$：用于导航历史帧与 video QA。

随后通过 2 层 MLP 的跨模态投影器 $P(\cdot)$ 将视觉特征映射到 LLM 的 latent 空间。

### 4.3 时序-视角指示 Token（TVI tokens）

这是 NavFoM 最核心的架构创新之一。视觉 token 本身不包含“来自哪个相机”或“属于哪一帧”的信息。NavFoM 为每个视觉 token 前置一个 **TVI token**，其嵌入由三部分组成：

- **基础嵌入** $E_{\text{Base}}$：标识这是视觉 token；
- **时间嵌入** $\text{TimePE}(t)$：基于 sinusoidal position encoding；
- **视角嵌入** $\text{AnglePE}(\phi)$：对相机方位角 $\phi$ 的编码，保持 $0 \equiv 2\pi$ 的循环连续性。

TVI token 的形式化定义：

$$
E_{\text{TVI}}^T =
\begin{cases}
E_{\text{Base}} + P_{\text{time}}\bigl(\text{TimePE}(t)\bigr) + P_{\text{angle}}\bigl(\text{AnglePE}(\phi)\bigr), & \text{Navigation} \\[6pt]
E_{\text{Base}} + P_{\text{time}}\bigl(\text{TimePE}(t)\bigr), & \text{Video QA} \\[6pt]
E_{\text{Base}}, & \text{Image QA}
\end{cases}
$$

TVI token 满足三个关键属性：

1. **Viewpoint-Awareness**：方位角编码保持圆周连续性，几何上相近的视角在嵌入空间中也相近；
2. **Time-Awareness**：跨所有相机视角唯一标识时间顺序，对非均匀采样具有鲁棒性；
3. **Separability**：通过组合不同嵌入组件即可区分 image QA、video QA、navigation 三种样本。

论文中的 UMAP 可视化显示，TVI tokens 在嵌入空间中按方位角与时间清晰聚类，证明 LLM 学到了有效的时空信息。

### 4.4 预算感知的时序采样（BATS）

长历史视频会产生大量视觉 token，直接全部送入 LLM 会拖慢训练与推理。NavFoM 提出 **Budget-Aware Temporal Sampling（BATS）**，核心思想是：受**遗忘曲线**启发，越新的帧采样概率越高，同时给历史帧保留非零下界。

给定 token 预算 $B_{\text{token}}$、当前最新时间步 $T$，对第 $t$ 帧的采样概率为：

$$
P(t) = (1 - \varepsilon) e^{k(t - T)/T} + \varepsilon, \qquad k > 0
$$

其中 $\varepsilon = 0.1$ 保证最低采样概率，$k$ 通过 Brent 方法离线求解以满足 token 预算约束。

BATS 的优势：

- **性能**：在 RxR 四目设置下，$B=2048$ 时 BATS 的 SR 为 **64.4%**，nDTW 为 **65.8**；等距采样仅为 **62.4% SR / 63.9 nDTW**。
- **效率**：BATS 在整个导航过程中保持稳定的推理时间，而保留全部帧的方法随历史增长显著变慢。

### 4.5 Token 组织与轨迹预测头

NavFoM 针对不同任务采用统一的 token 组织模式：

- **Image QA**：仅使用 $E_{\text{Base}}$ + 细粒度视觉 token（64 tokens / 图）；
- **Video QA**：使用 $E_{\text{Base}} + P_{\text{time}}$ + 粗粒度视觉 token（4 tokens / 帧）；
- **Navigation**：使用完整 TVI token + 粗粒度历史帧 + 细粒度最新帧。

LLM 输出的 action hidden state 经过 3 层 MLP 的 planning head 预测未来 $M = 8$ 个归一化航点：

$$
\tau_T = \{a_1, \dots, a_M\}_T = \alpha_{\text{task}} \cdot A_\theta\bigl(E_T^A\bigr)
$$

不同任务/具身的轨迹尺度差异巨大，因此采用任务级缩放因子 $\alpha_{\text{task}}$ 将真值与预测归一化到 $[-1, 1]$，推理时再反归一化。

### 4.6 训练目标

- **导航损失**：预测航点与真值航点的 MSE，只计算有效动作维度；
- **QA 损失**：标准 next-token cross-entropy 损失；
- **总损失**：$\mathcal{L} = \beta \mathcal{L}_{\text{nav}} + \mathcal{L}_{\text{QA}}$，其中 $\beta = 10$。

---

## 五、数据与训练工程

### 5.1 训练数据规模

NavFoM 共使用 **12.7M** 训练样本：

| 数据类型 | 样本数 | 说明 |
|---|---|---|
| 导航样本 | **8.02 M** | 跨四足、轮式、无人机、车辆 |
| Image QA | 3.15 M | 开放世界图像问答 |
| Video QA | 1.61 M | 开放世界视频问答 |
| **总计** | **12.7 M** | 超过 NaVid / Uni-NaVid 等前作 |

导航样本按任务细分：

| 任务 | 样本数 | 数据来源 / 平台 |
|---|---|---|
| Vision-and-Language Navigation | 3.37 M | VLN-CE R2R / RxR + OpenUAV |
| Object Goal Navigation | 1.02 M | HM3D ObjectNav |
| Active Visual Tracking | 0.897 M | EVT-Bench |
| Autonomous Driving | 0.681 M | nuScenes + OpenScene |
| Web-Video Navigation | 2.03 M | Sekai 数据集 |

### 5.2 关键数据工程

- **多视角与相机随机化**：固定前视相机 + 随机 1–8 个环绕相机，相机高度 0.6–1.5 m 随机，HFoV 75°–120° 随机；
- **离散动作轨迹化**：统一规定前进 12.5 cm 或旋转 15° 为一个原子操作，将离散动作序列累积为连续航点轨迹；
- **网络视频数据**：Sekai 提供约 18.2 万 YouTube 视频，由 VLM 生成指令、SLAM 生成轨迹，引入真实世界视觉域与长程上下文。

### 5.3 训练策略与加速

- **硬件**：56 块 NVIDIA H100 GPU；
- **训练时间**：约 72 小时，合计 **4,032 GPU 小时**；
- **Epoch**：单 epoch；
- **初始化**：DINOv2、SigLIP 预训练权重冻结，Qwen2-7B 预训练权重可训练；
- **离线视觉特征缓存**：仅缓存粗粒度视觉 token（4 tokens / 帧），训练速度提升 **2.9×**，显存降低 **1.8×**。

---

## 六、实验结果

### 6.1 VLN-CE R2R / RxR

| Method | Observation | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|---|---|---|---|---|---|---|
| HNR* | RGB-D + Odo | 61.0 | 51.0 | 56.3 | 46.7 | 63.5 |
| **NavFoM (Four views)** | RGB only | **61.7** | **55.3** | **64.4** | **56.2** | **65.8** |

| Method | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|---|---|---|---|---|---|
| StreamVLN | 55.7 | 50.9 | 51.8 | 45.0 | 62.1 |
| **NavFoM (Single view)** | **56.2** | **51.2** | **57.4** | **49.4** | **60.2** |

NavFoM 四目仅使用 RGB 就超越了依赖 RGB-D + 里程计的 HNR，说明多视角视觉信息可以有效替代深度与里程计。

### 6.2 户外无人机导航（OpenUAV）

| Method | UO Set SR↑ | UO Set SPL↑ |
|---|---|---|
| TravelUAV | 22.42 / 20.51 | — |
| **NavFoM (Four views)** | **29.83** / **27.20** | — |

在未使用俯视相机的情况下取得 SOTA。不过在 **Unseen-Map (UM)** 拆分上仍显著落后，因为该拆分需要平均行进约 300 米并探索大规模未知社区。

### 6.3 开放词汇目标搜索（HM3D-OVON）

| Method | Val-Unseen SR↑ | Val-Unseen SPL↑ |
|---|---|---|
| Uni-NaVid* | 39.5 | 19.8 |
| MTU3D | 40.8 | 12.1 |
| **NavFoM* (Single view)** | **43.6** | **31.3** |
| **NavFoM* (Four views)** | **45.2** | **31.9** |

NavFoM 的目标搜索训练数据为单目，四目为 zero-shot 泛化。SPL 提升尤为明显（31.9 vs. 19.8），说明模型学到了更高效的路径规划。

### 6.4 主动视觉跟踪（EVT-Bench）

| Method | Single Target SR↑ | Single Target TR↑ | Distracted Target SR↑ | Distracted Target TR↑ |
|---|---|---|---|---|
| TrackVLA | 85.1 | 78.6 | 57.6 | 63.2 |
| **NavFoM (Single view)** | 85.0 | **80.5** | **61.4** | **68.2** |
| **NavFoM (Four views)** | **88.4** | **80.7** | **62.0** | 67.9 |

### 6.5 自动驾驶

**NAVSIM navtest 闭环**：

| Method | PDMS↑ |
|---|---|
| DiffusionDrive | **88.1** |
| LAW | 84.6 |
| **NavFoM (Eight views)** | **84.3** |

NavFoM 未显式建模车道、他车、交通信号等驾驶先验，仅通过多视角 RGB 与车辆状态达到与专用驾驶模型相当的水平。

**nuScenes 开环**：Avg L2 **0.42 m**，Avg Collision **0.12%**。

### 6.6 真实世界部署

NavFoM 在远程服务器（NVIDIA RTX 4090）上运行，通过 Internet 与机器人通信：

- 在 **1600 token 预算**下，生成 8 个航点的轨迹耗时 **不超过 0.5 秒**；
- 机器人异步压缩并上传最新观测，同时执行上一帧动作；
- 各机器人使用自身的局部轨迹跟踪器执行预测轨迹。

验证平台包括：Unitree Go2（四足）、Unitree G1 / Galbot G1（人形）、NCS-β 无人机、配备 RealSense 的轮式机器人。论文在 110 个可复现案例上展示了 VLN、搜索、跟踪三种能力，并在复杂室内环境中展示了跨任务/跨具身切换能力。

---

## 七、消融研究：什么真正重要？

### 7.1 多任务协同训练

加入其他任务数据共调带来一致提升。特别是搜索任务从约 10.3% 提升至 45.2%（四目 zero-shot），跟踪任务从约 12.6% 提升至 62.0%。这说明 VLN 与驾驶数据中的多视角、开放集能力可以有效地迁移到目标搜索与跟踪任务。

### 7.2 相机数量的影响

在 VLN-CE RxR 上，从单目到四目 SR 持续上升（四目 64.4%），但扩展到六目后出现轻微下降。作者分析，六目相较四目并未显著增加环境覆盖，但更多视角 token 挤占了历史帧预算，导致历史上下文被稀释。

### 7.3 TVI tokens 与 BATS 的有效性

在 RxR 四目设置下对比不同策略（$B=2048$）：

| Strategy | NE↓ | SR↑ | SPL↑ | nDTW↑ |
|---|---|---|---|---|
| Uniform Sampling | 4.90 | 62.4 | 54.0 | 63.9 |
| Token Merging | 5.01 | 63.2 | 54.9 | 64.4 |
| **BATS** | **4.74** | **64.4** | **56.2** | **65.8** |

TVI token 设计的消融：

| Identity Token Design | NE↓ | SR↑ | SPL↑ | nDTW↑ |
|---|---|---|---|---|
| Viewpoint-history positional embedding | 6.27 | 52.3 | 46.3 | 58.7 |
| Individual learned special tokens | 5.52 | 59.1 | 52.0 | 59.6 |
| Handcraft tokens (w/o MLPs) | 6.06 | 53.6 | 46.1 | 58.0 |
| **TVI tokens (full)** | **4.74** | **64.4** | **56.2** | **65.8** |

结论：显式时空建模（TVI）与遗忘曲线式采样（BATS）都是必要的。

---

## 八、与相关工作的对比

| 工作 | 核心定位 | 跨任务 | 跨具身 | 关键局限 |
|---|---|---|---|---|
| **NaVid** | 视频式 VLN | VLN 为主 | 室内轮式 | 未统一其他导航任务 |
| **Uni-NaVid** | 统一室内导航任务 | VLN / ObjNav / OVON | 室内轮式 | 未覆盖无人机、驾驶 |
| **NaVILA** | 四足机器人 VLA | VLN | 四足 | 单一具身 |
| **StreamVLN** | 流式上下文 VLN | VLN | 室内轮式 | 任务范围窄 |
| **ViNT / GNM** | 视觉导航表征 |  Goal-reaching | 轮式 | 无语言指令，任务简单 |
| **NavFoM** | 导航基础模型 | VLN / OVON / Tracking / Driving | 四足 / 轮式 / 无人机 / 车辆 | 长程探索仍需提升 |

NavFoM 的独特之处在于：它不是设计多个任务 head，而是通过 TVI tokens 让同一 LLM 理解“token 来自哪里、属于何时”，从而在统一框架下处理多样输入。这种“上下文即结构”的思路与 π 系列中通过多样化 prompt 条件来利用异构数据有异曲同工之妙。

---

## 九、关键洞察、局限与未来方向

### 9.1 关键洞察

1. **跨任务/跨具身统一的关键是“观察上下文”的显式建模**：TVI tokens 让 LLM 能够区分不同视角、时间和任务类型，这是统一训练的前提。
2. **多任务共调带来显著迁移**：尤其是训练/评估分布差异大的任务（搜索、跟踪），从多目 VLN 和驾驶数据中获得多视角与开放集能力。
3. **视觉 token 预算必须显式管理**：BATS 证明，在固定预算下，遗忘曲线式采样优于均匀采样与 token merging。
4. **粗-细粒度视觉 token 分工是长视频高效训练的关键**：历史帧用 4 tokens 压缩，最新帧用 64 tokens 保留细节。

### 9.2 局限

1. **Unseen-Map 等长程探索任务仍显著落后**：300 米级大尺度探索对数据与策略要求更高。
2. **六目及以上配置的收益递减**：token 预算被视角 token 挤压，历史上下文受损。
3. **自动驾驶未充分利用结构化驾驶信息**：未显式建模车道线、交通信号、他车意图。
4. **真实世界量化评估规模有限**：主要展示 110 个可控案例，更大规模真实场景量化评测仍是开放问题。
5. **对 web-video 数据的噪声较为宽容**：如何更精细地筛选高质量数据值得研究。

### 9.3 未来方向

- **更大规模、更高质量的跨具身导航数据**：特别是真实世界长程探索数据；
- **在线适应与强化学习闭环**：让模型在部署中持续学习；
- **3D 场景记忆与语义地图融合**：结合神经辐射场、3D Gaussian Splatting 等显式场景表示；
- **更智能的多视角 token 编码**：在视角数量增加时保持效率；
- **与 VLA 操控模型的结合**：构建“导航 + 操作”的统一具身智能体。

---

## 十、结语

NavFoM 是导航领域迈向“基础模型”的重要一步。它通过 **TVI tokens** 显式编码视觉 token 的时空身份，通过 **BATS** 在固定 token 预算下高效组织长历史观测，通过 **12.7M 异构样本** 的联合训练获得通用导航能力。在 7 个基准上，NavFoM 无需任务特定 fine-tuning 即取得 SOTA 或接近 SOTA 的性能，并成功部署到四足、人形、无人机、轮式机器人等真实平台。

更重要的是，NavFoM 展示了一种新的研究范式：不是为每个任务和每个机器人设计专门模型，而是将异构数据统一为“视觉-语言-动作”的序列建模问题，让模型从规模中涌现通用能力。这与 VLA 领域 π 系列的发展脉络高度一致——无论是机器人操作还是机器人导航，通向通用具身智能的道路，或许都需要这样一个统一的“视觉-语言-动作”基础模型。

---

## 参考资料

1. Zhang, J., et al. *Embodied Navigation Foundation Model*. arXiv:2509.12129v2, 2025. [https://arxiv.org/abs/2509.12129](https://arxiv.org/abs/2509.12129)
2. Project Page: [https://pku-epic.github.io/NavFoM-Web/](https://pku-epic.github.io/NavFoM-Web/)
3. NaVid: [https://arxiv.org/abs/2402.15852](https://arxiv.org/abs/2402.15852)
4. Uni-NaVid: [https://arxiv.org/abs/2412.06224](https://arxiv.org/abs/2412.06224)
5. NaVILA: [https://arxiv.org/abs/2412.04453](https://arxiv.org/abs/2412.04453)
6. StreamVLN: [https://arxiv.org/abs/2507.05240](https://arxiv.org/abs/2507.05240)
7. ViNT: [https://arxiv.org/abs/2306.14846](https://arxiv.org/abs/2306.14846)
8. GNM: [https://arxiv.org/abs/2210.03370](https://arxiv.org/abs/2210.03370)
