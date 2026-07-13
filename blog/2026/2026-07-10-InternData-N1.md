---
title: "InternData-N1：面向通用视觉-语言导航的大规模统一数据集"
category: VLN
date: 2026-07-10
authors:
  - InternRobotics (Shanghai AI Lab)
affiliation: Shanghai Artificial Intelligence Laboratory
license: CC BY-NC-SA 4.0
---

# InternData-N1：面向通用视觉-语言导航的大规模统一数据集

> InternData-N1 是上海人工智能实验室 InternRobotics 团队发布的面向视觉-语言导航（VLN）的大规模统一数据集。它将多个主流基准整合到标准化的 LeRobot 格式中，包含 **240,000+ 条轨迹**、覆盖 **3,000+ 场景**，为训练和评估具身导航系统提供多样化、高质量的训练数据。数据集在 Hugging Face 上以 CC BY-NC-SA 4.0 协议开源（需申请授权），是 InternVLA-N1 双系统导航基础模型与 InternNav 开源工具箱的数据基石。

---

## 一、背景与动机

### 1.1 VLN 数据集的无序局面

视觉-语言导航（VLN）领域经过多年发展，形成了多个各具特色的基准数据集：

| 数据集 | 场景数 | 轨迹数 | 特点 | 格式 |
|---|---|---|---|---|
| R2R | 90 | ~21K | 最早、最经典的室内 VLN 基准 | JSON |
| RxR | 90 | ~58K | 多语言、密集路标、更长轨迹 | JSON |
| R2R-CE | 90 | ~21K | 连续动作空间版本 | JSON + MP3D 网格 |
| RxR-CE | 90 | ~58K | 连续动作空间版本 | JSON + MP3D 网格 |
| VLN-PE | 90 | ~21K | 加入物理运动控制器 | JSON + 控制器参数 |

这些数据集格式各异、场景重叠、预处理方式不同，给研究者和工程师带来了显著的复现和迁移成本。过去一个常见的问题是：一个模型在 R2R 上训好了，迁移到 RxR 时发现数据格式、动作空间、评估协议都不一样，需要大量适配工作。

### 1.2 需要统一数据集

DualVLN（InternVLA-N1）论文在提出双系统架构的同时，配套推出了 InternData-N1 数据集，核心目标是以统一的数据格式降低多任务训练的门槛。它不仅整合了现有的 VLN-CE 和 VLN-PE 数据，还新增了 VLN-N1 子集——基于 3D-Front 室内场景合成的扩展数据。

---

## 二、数据集概览

### 2.1 基本信息

| 属性 | 值 |
|---|---|
| **发布机构** | InternRobotics, Shanghai AI Lab |
| **发布日�期** | 2025 年 7 月 26 日 |
| **许可证** | CC BY-NC-SA 4.0（社区授权协议，需申请）|
| **访问方式** | Hugging Face gated dataset（自动审批）|
| **Hugging Face 下载量** | 61,447+ |
| **Hugging Face 收藏量** | 80+ |
| **格式** | LeRobot 统一格式 |
| **任务类型** | 机器人导航、视觉-语言导航 |

### 2.2 核心规模

| 指标 | 数值 |
|---|---|
| **总轨迹数** | 240,000+ |
| **总场景数** | 3,000+ |
| **子集数量** | 3（VLN-CE / VLN-PE / VLN-N1）|
| **文件总数** | 20,829 |
| **语言** | 英语 |

### 2.3 文件结构概览

```
InternData-N1/
├── vln_ce/                      # 连续环境 VLN
│   ├── raw_data/                # 原始标注（JSON）
│   │   ├── r2r/                 # R2R 标注（train/val_seen/val_unseen）
│   │   └── rxr/                 # RxR 标注
│   └── traj_data/               # 场景级轨迹文件
│       ├── r2r/
│       │   ├── 17DRP5sb8fy.tar.gz   # 按场景命名的轨迹包
│       │   └── ... (90 个场景)
│       └── rxr/
│           └── ... (更多场景)
│
├── vln_pe/                      # 物理控制器 VLN
│   ├── raw_data/
│   │   ├── embeddings.json.gz   # 预计算特征嵌入
│   │   └── r2r/                 # R2R 标注
│   └── traj_data/
│       └── r2r/
│           └── ... (90+ 场景)
│
└── vln_n1/                      # InternRobotics 新增子集
    └── traj_data/
        └── 3dfront_d435i/
            └── ... (3D-Front 室内场景轨迹)
```

---

## 三、三大子集详解

### 3.1 VLN-CE（Continuous Environment）

**定位**：传统的 VLN 连续环境基准数据，覆盖 R2R 和 RxR 两个经典数据集。

- **来源**：基于 Matterport3D 的 90 个室内场景
- **内容**：轨迹级数据包含 RGB 观测、深度图、相机位姿、动作序列、语言指令
- **格式**：LeRobot 标准格式，包含每帧观测和动作标签
- **规模**：921 个文件（7 个标注文件 + 914 个场景轨迹包）
- **应用**：训练 VLN-CE 模型（如 StreamVLN、DualVLN）

VLN-CE 子集的价值在于它提供了一个标准化的数据接口。过去研究者需要分别处理 R2R 的离散导航图和 RxR 的多语言标注，InternData-N1 把它们统一为同一套格式，使得在混合数据上训练成为简单的"读目录→加载→训练"流程。

### 3.2 VLN-PE（Physical Environment）

**定位**：加入物理运动控制器的 VLN 数据，弥合仿真与真实之间的鸿沟。

- **来源**：基于 VLN-PE 基准（Wang et al., 2025），覆盖 R2R 轨迹
- **特色**：除了标准的观测和轨迹外，还包含机器人运动学控制参数
- **规模**：16,132 个文件（4 个标注/嵌入文件 + 16,128 个场景轨迹包）
- **应用**：训练能够输出连续、物理可行的轨迹的模型（如 InternVLA-N1）

VLN-PE 的核心差异在于动作空间。VLN-CE 的动作通常是离散的（前进 0.25 米、左转 15°等），而 VLN-PE 包含了连续控制信号（速度命令、轨迹航点），使模型可以直接输出可被真实机器人执行的指令。这也是 DualVLN 论文中零样本迁移到 VLN-PE 并获得 SOTA 能的基础之一。

### 3.3 VLN-N1（New Subset）

**定位**：InternRobotics 新增的扩展数据，覆盖更多样化的场景。

- **来源**：3D-Front 室内场景数据集，使用 Intel D435i 相机配置
- **特色**：
  - 覆盖 3,000+ 场景（相比 MP3D 的 90 个有数量级提升）
  - 大量长走廊、复杂室内拓扑（商场、办公室）
  - 包含多样化的照度条件和布局风格
- **规模**：3,774 个文件（3D-Front 场景轨迹包）
- **应用**：扩展模型的泛化能力，减少对 MP3D 过拟合

VLN-N1 的加入是 InternData-N1 区别于此前所有 VLN 数据集的关键。它大量学习了 ABot-N0 的数据工程哲学——用更多的合成场景来弥补真实扫描数据的不足。DualVLN 在 RxR 上 SR 达到 61.4（远超此前 SOTA 的 52.9），VLN-N1 的数据多样性功不可没。

---

## 四、数据统计与分析

### 4.1 纵向对比：InternData-N1 vs 此前 VLN 数据集

| 维度 | R2R/RxR（原始）| VLN-PE（原始）| **InternData-N1（统一）**|
|---|---|---|---|
| 总场景数 | ~90 | ~90 | **3,000+** |
| 总轨迹数 | ~79K | ~21K | **240,000+** |
| 格式统一性 | ✗ 各数据集格式不同 | ✗ 独立格式 | **✓ LeRobot 统一格式** |
| 动作空间 | 离散 + 连续（CE）| 连续（PE）| **连续（CE+PE+N1）** |
| 跨子集训练 | 需手动适配 | 需手动适配 | **开箱即用** |
| 3D-Front 场景 | ✗ | ✗ | **✓ 3,000+** |
| 授权方式 | 需逐一申请 | 需申请 | **统一授权** |

### 4.2 VLN-N1 子集的规模对比

在场景覆盖上，VLN-N1 子集是一个重要的突破：

- MP3D（R2R/RxR 基础）：90 个室内场景
- 3D-Front（VLN-N1 基础）：**18,000+ 个室内场景**（InternData-N1 使用了 3,000+ 个）
- 场景类型：从纯住宅扩展到办公楼、商场、医院、学校等公共空间

---

## 五、在 InternVLA-N1 / DualVLN 中的角色

InternData-N1 是 InternVLA-N1 双系统导航基础模型的数据基石。

### 5.1 训练流程中的位置

```
InternData-N1
├── VLN-CE → 训练 System 2 的像素目标定位 + 自导向视角调整
├── VLN-PE → 训练 System 1 的轨迹生成 + 物理可行输出
└── VLN-N1 → 扩展泛化能力，缓解 MP3D 过拟合
       ↓
InternNav 工具箱（数据加载 + 预处理 + 模型训练）
       ↓
InternVLA-N1 模型（DualVLN 双系统架构）
```

DualVLN 的两阶段训练对数据集有不同需求：
- Stage 1（训练 System 2 的像素目标定位）：主要使用 VLN-CE 中的 R2R/RxR 数据，通过将 3D 轨迹投影到 2D 图像来构建监督信号。
- Stage 2（冻结 System 2，训练 latent queries + DiT）：主要使用 VLN-PE 数据，需要连续的轨迹标签来训练 Diffusion Transformer。
- VLN-N1 在两个阶段都作为数据增强使用，尤其是在长程场景和多样化布局上的表现提升。

### 5.2 与 InternVLA-N1 模型的关系

| 组件 | 模型 | 数据来源 |
|---|---|---|
| System 2（VLM 规划器）| Qwen-VL-2.5 (7B) | InternData-N1 VLN-CE |
| System 1（DiT 策略）| Diffusion Transformer (12层) | InternData-N1 VLN-PE |
| 完整系统 | InternVLA-N1 | InternData-N1（全部）|

---

## 六、如何使用

### 6.1 申请访问

InternData-N1 采用 gated access 机制：

1. 访问 [Hugging Face 数据集页](https://huggingface.co/datasets/InternRobotics/InternData-N1)
2. 同意社区授权协议（CC BY-NC-SA 4.0）
3. 填写申请表单（姓名、邮箱、所属机构、研究兴趣、职位等）
4. 通常自动审批，立即获得下载权限

```python
# 安装 Hugging Face datasets
# pip install datasets

from datasets import load_dataset

# 登录 Hugging Face（需要先同意授权）
# huggingface-cli login

# 加载数据集（需指定子集和配置）
dataset = load_dataset("InternRobotics/InternData-N1", split="train")
print(dataset[0])
```

### 6.2 使用 InternNav 工具箱

推荐配合 [InternNav](https://github.com/InternRobotics/InternNav) 工具箱使用：

```bash
git clone https://github.com/InternRobotics/InternNav.git
cd InternNav

# 安装依赖
pip install -e .

# InternData-N1 数据可自动下载（需先通过 Hugging Face 授权）
```

InternNav 提供了数据加载、预处理、模型训练和评估的完整流程支持。

### 6.3 相关资源总览

| 资源 | 链接 |
|---|---|
| **数据集** | [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1) |
| **模型** | [InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger) |
| **代码** | [InternNav GitHub](https://github.com/InternRobotics/InternNav) |
| **技术报告** | [InternVLA-N1 Technical Report](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf) |
| **论文** | [DualVLN (arXiv:2512.08186)](https://arxiv.org/abs/2512.08186) |
| **项目主页** | [InternVLA-N1 Website](https://internrobotics.github.io/internvla-n1.github.io/) |
| **文档** | [InternNav Documentation](https://internrobotics.github.io/user_guide/internnav/index.html) |

---

## 七、局限与展望

### 7.1 已知局限

1. **室内为主**：InternData-N1 的 3,000+ 场景全部为室内环境，缺乏室外城市导航数据。这限制了模型向自动驾驶、城市漫游等室外场景的直接迁移。

2. **静态场景假设**：数据采集时场景不包含动态行人或其他移动障碍物。虽然 DualVLN 提出了 Social-VLN 基准来评估动态避障，但训练数据本身仍是静态的。

3. **语言单一**：目前仅支持英语标注，缺乏多语言支持（原始 RxR 含多语言，但在 InternData-N1 中是否保留多语言字段需确认）。

4. **格式迁移成本**：LeRobot 格式虽然统一，但对于习惯原有 JSON 格式的用户，需要一定的学习成本。InternNav 工具箱封装减轻了这一问题。

### 7.2 未来方向

- **室外场景扩展**：加入城市级导航数据（类似 ABot-N0 的 SocCity），支持室内外一体化导航训练。
- **动态场景数据**：包含行人和移动障碍物的轨迹数据，支持动态避障和社交导航的训练。
- **多模态扩展**：除了 RGB-D，加入语音指令、触觉反馈等多模态输入。
- **持续更新**：InternData-N1 的命名（N1 → N2 → ...）暗示了这是一个持续演进的数据集系列。

---

## 八、总结

InternData-N1 是 VLN 领域的一个重要基础设施贡献。它通过标准化的 LeRobot 格式、三项互补的子集和 3,000+ 场景的覆盖范围，大幅降低了多任务训练的工程成本。

它的最大价值在于**打通了"数据"和"模型"之间的工程鸿沟**——过去一个 VLN 研究者需要花大量时间处理数据格式、重写预处理管道、对齐不同数据集的评估协议。InternData-N1 + InternNav 工具箱的组合，使得训练一个多任务导航模型变成：

```
load_dataset → train_model → evaluate
```

对于希望快速入门 VLN 的研究者和工程师，InternData-N1 搭配 InternNav 工具箱是目前最推荐的起点。对于希望复现或改进 DualVLN/InternVLA-N1 的同仁，它更是不可或缺的基础设施。

---

## 参考文献

[1] Meng Wei et al. *Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation*. arXiv:2512.08186, Dec 2025.

[2] InternRobotics. *InternVLA-N1: An Open Dual-System Navigation Foundation Model with Learned Latent Plans*. 2025.

[3] InternNav Contributors. *InternNav: InternRobotics' Open Platform for Building Generalized Navigation Foundation Models*. GitHub, 2025.

[4] Peter Anderson et al. *Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments*. CVPR, 2018.

[5] Alexander Ku et al. *Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding*. EMNLP, 2020.

[6] Liuyi Wang et al. *Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities*. arXiv:2507.13019, 2025.

[7] Jacob Krantz et al. *Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments*. ECCV, 2020.
