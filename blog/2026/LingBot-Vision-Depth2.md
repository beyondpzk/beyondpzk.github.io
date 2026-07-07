---
title: LingBot-Vision 与 LingBot-Depth 2.0：具身智能的空间视觉基座再升级
date: 2026-07-07
categories: [Vision]
---

# LingBot-Vision 与 LingBot-Depth 2.0：具身智能的空间视觉基座再升级

> **论文 / 技术报告**: *Masked Depth Modeling for Spatial Perception*（LingBot-Depth 1.0）
> **作者**: Bin Tan, Changjiang Sun, Xiage Qin, Hanat Adai, Zelin Fu, Tianxiang Zhou, Han Zhang, Yinghao Xu, Xing Zhu, Yujun Shen, Nan Xue
> **机构**: Robbyant（蚂蚁集团灵波科技）
> **arXiv**: [arXiv:2601.17895](https://arxiv.org/abs/2601.17895)
> **GitHub**: [Robbyant/lingbot-depth](https://github.com/Robbyant/lingbot-depth)
> **发布时间**: LingBot-Depth 2.0 / LingBot-Vision 于 2026-07-06/07 发布

---

## 导语：从“看清每个像素有多远”到“看懂整个三维世界”

在具身智能的技术栈中，视觉感知是最靠近物理世界的一层。机器人要抓杯子、导航穿过房间、与人交互，首先需要一个稳定、精确、对真实场景鲁棒的空间感知系统。而深度估计——即判断每个像素到相机的距离——正是空间感知的核心。

2026 年 1 月，蚂蚁集团旗下具身智能公司 **灵波科技（Robbyant）** 开源了 **LingBot-Depth**，提出 **Masked Depth Modeling（MDM）** 方法，将 RGB-D 相机失效区域的深度空洞视为“自然掩码”，用视觉上下文补全深度，在真实场景中把深度误差降低了 **70% 以上**。该工作已被 **ECCV 2026** 接收，并以 Apache 2.0 协议发布了代码、模型权重和 300 万 RGB-D 配对数据。

2026 年 7 月，灵波科技联合 **奥比中光** 再次发布 **LingBot-Depth 2.0**，并同步推出其视觉基座模型 **LingBot-Vision**。新版本基于 **1.5 亿规模数据**训练，强调从“深度补全”向“通用空间视觉基座”跃迁；奥比中光的无本体数据采集设备 **EGO-RGBD** 也将适配 LingBot-Depth，形成“算法 + 硬件 + 数据”的闭环。

本文将系统梳理 LingBot-Depth 1.0 的技术原理、LingBot-Depth 2.0 与 LingBot-Vision 的升级逻辑，以及这次软硬协同对具身智能产业的意义。

---

## 一、为什么深度估计对具身智能如此重要？

### 1.1 RGB-D 相机的理想与现实

理论上，RGB-D 相机能直接输出每个像素的彩色图像与对应深度，是机器人感知三维世界最直接的传感器。然而真实场景中，深度传感器常常失效：

- **高反光表面**（金属、玻璃、釉面）导致 ToF / 结构光无法返回有效信号；
- **低纹理区域**（白墙、桌面）使双目匹配失去依据；
- **透明/半透明物体**（水杯、塑料袋）产生空洞或虚假深度；
- **远距离与边缘区域**深度噪声大、覆盖率下降。

这些失效区域在深度图上表现为“空洞”或“离群值”。传统后处理方法（中值滤波、双边滤波、基于 RGB 引导的补全）往往只能在局部平滑，难以恢复真实几何。

### 1.2 从“修复深度图”到“理解几何歧义”

LingBot-Depth 的核心洞察是：**深度传感器的不准确区域，本质上反映了底层几何的歧义**。与其把空洞当作需要填充的“噪声”，不如把它们当作“掩码信号”——模型需要根据 RGB 视觉上下文推断被掩码部分的深度。

这与自然语言处理中的 Masked Language Modeling（MLM）有异曲同工之妙：BERT 通过掩码词学习语言表示，LingBot-Depth 则通过掩码深度学习空间表示。模型不仅补全深度，还学到了 RGB 与深度模态之间的对齐隐式表示，这一表示可直接迁移到下游任务。

---

## 二、LingBot-Depth 1.0：Masked Depth Modeling

### 2.1 核心思想

LingBot-Depth 将深度补全形式化为一个 **Masked Depth Modeling（MDM）** 问题：

- 输入：RGB 图像 + 原始（含噪声/空洞）深度图；
- 掩码：根据深度置信度或传感器特性，将不可靠像素标记为掩码区域；
- 目标：模型利用 RGB 上下文，预测被掩码像素的真实深度；
- 输出：完整、精确、带有真实尺度的稠密深度图。

关键在于，掩码不是人为随机添加的，而是**由真实传感器失效模式自然产生的**。这让模型学到的补全能力直接对应实际部署中的 failure case。

### 2.2 模型架构

根据官方 GitHub 与 ModelScope 信息，LingBot-Depth 1.0 的架构如下：

| 组件 | 配置 |
|---|---|
| 编码器 | ViT-Large/14（24 层） |
| 输入嵌入 | RGB 与深度分别进行 patch embedding |
| 解码器 | ConvStack decoder，分层上采样 |
| 参数量 | 约 300M |
| 训练目标 | Masked Depth Modeling |
| 许可证 | Apache 2.0 |

**分离式 patch embedding** 设计让模型在 early fusion 之前保留模态特异性：RGB 提供纹理、语义与上下文，深度提供粗略几何线索。ViT 编码器在统一 latent 空间中融合两者，ConvStack 解码器再逐步恢复空间分辨率。

### 2.3 数据工程：200 万真实 + 100 万合成

LingBot-Depth 1.0 开源了 **300 万 RGB-D 配对数据**，其中：

- **200 万真实数据**：使用奥比中光 Gemini 330 系列双目 3D 相机采集，并经后处理校验；
- **100 万合成数据**：来自仿真环境，用于扩充场景多样性、材质分布与相机位姿。

这一数据规模在当时已经相当大，但更关键的是 **数据质量 pipeline**：团队不仅收集了 RGB-D 图像，还对深度进行了验证与清洗，确保模型学习的是“可信深度”而非传感器噪声。

### 2.4 能力与应用

根据官方介绍，LingBot-Depth 1.0 可支撑多种下游任务：

- **深度补全与精修**（Depth Completion & Refinement）：填充空洞、降低噪声、恢复真实尺度；
- **场景重建**（Scene Reconstruction）：作为高保真室内建图的深度先验；
- **4D 点跟踪**（4D Point Tracking）：在度量空间中跟踪动态点；
- **灵巧操作**（Dexterous Manipulation）：为抓取提供精确几何理解。

实验显示，在挑战性材质和复杂场景中，LingBot-Depth 的深度精度超过直接使用高端 RGB-D 相机的原始输出，深度误差降低超过 70%。

---

## 三、LingBot-Depth 2.0：从专用模型到空间视觉基座

### 3.1 发布时间与合作背景

2026 年 7 月 6 日至 7 日，蚂蚁灵波科技正式发布 **LingBot-Depth 2.0**，并宣布与 **奥比中光** 达成深度合作。奥比中光作为国内 3D 视觉感知龙头企业，其 Gemini 系列相机、Femto 系列相机以及最新推出的无本体数据采集设备，将为 LingBot-Depth 提供硬件落地入口。

### 3.2 数据规模：从 300 万到 1.5 亿

LingBot-Depth 2.0 最显著的升级是训练数据规模：

| 版本 | 训练数据规模 | 真实/合成比例 |
|---|---|---|
| LingBot-Depth 1.0 | 300 万 RGB-D 对 | 200 万真实 + 100 万合成 |
| LingBot-Depth 2.0 | **1.5 亿规模** | 未公开具体比例，但强调大规模真实场景采集 |

从 300 万到 1.5 亿，数据量提升了 **50 倍**。这种规模跃迁通常意味着：

- 覆盖更多室内/室外场景、光照条件、物体材质；
- 包含更多极端 case（反光、透明、低纹理、长距离）；
- 能够训练更大容量、更通用的视觉基座模型。

### 3.3 能力升级方向

虽然官方尚未发布完整技术报告，但基于新闻稿与产业合作信息，LingBot-Depth 2.0 的提升方向可能包括：

1. **更强的泛化能力**：在未见过的环境、相机和任务中保持深度估计稳定性；
2. **更高的精度与覆盖率**：对细小物体、远距离区域、极端材质的补全效果更好；
3. **更快的推理速度**：针对边缘设备优化，适配机器人实时控制需求；
4. **与视觉基座模型协同**：LingBot-Vision 提供通用视觉表示，LingBot-Depth 2.0 提供几何先验。

### 3.4 产业认证：奥比中光“深度视觉实验室”

据报道，LingBot-Depth 2.0 已入驻奥比中光专业的 **深度视觉实验室**，接受多项工业级测试。认证结果显示：

- 在精度表现和极端材质对抗中性能亮眼；
- 对细小物体的捕捉能力强；
- 长距离收敛速度快，满足硬件工程师的苛刻指标。

这意味着 LingBot-Depth 2.0 不再只是研究原型，而是开始向可量产、可部署的工业级感知组件演进。

---

## 四、LingBot-Vision：LingBot-Depth 2.0 的视觉基座模型

### 4.1 定位：从深度模型到通用视觉基座

LingBot-Vision 是随 LingBot-Depth 2.0 同步推出的 **视觉基座模型（Vision Foundation Model）**。它的定位可以理解为灵波科技在空间感知领域的“通用视觉大脑”：

- 不仅为深度估计提供特征，还可能服务于语义理解、场景分类、物体检测、视觉导航等任务；
- 作为统一视觉编码器，为 LingBot-VLA、LingBot-Map、LingBot-World 等上层模型提供视觉表示；
- 通过与 LingBot-Depth 2.0 联合训练，实现语义与几何的深度融合。

### 4.2 为什么是“视觉基座”而非“深度模型”？

具身智能的发展趋势表明，单一任务模型很难满足复杂机器人系统的需求。一个通用视觉基座模型可以：

- **减少重复计算**：一次视觉编码，多个任务共享；
- **促进跨任务迁移**：在深度估计中学到的几何知识，可帮助操作与导航；
- **统一数据利用**：将图像-文本、视频-动作、RGB-D 等多种数据纳入同一表示空间。

LingBot-Vision 的推出，标志着灵波科技在空间感知层面的布局从“做深”走向“做宽”。

### 4.3 与 LingBot 技术栈的协同

回顾灵波科技过去半年的技术路线图，可以看到一条清晰的具身智能闭环：

| 模型 | 功能 | 发布时间 |
|---|---|---|
| LingBot-Depth | 深度补全、空间感知 | 2026-01 |
| LingBot-VLA | 视觉-语言-动作模型 | 2026-01 |
| LingBot-World | 世界模型 / 视频生成 | 2026-02 |
| LingBot-Map | 实时 3D 重建与建图 | 2026-04 |
| **LingBot-Vision** | **视觉基座模型** | **2026-07** |
| **LingBot-Depth 2.0** | **升级版空间感知模型** | **2026-07** |

LingBot-Vision 与 LingBot-Depth 2.0 的加入，使这条链路从“看清”到“看懂”再到“决策-模拟-建图”形成了更完整的闭环。

---

## 五、软硬协同：奥比中光 EGO-RGBD 数采设备

### 5.1 数据采集是具身智能的瓶颈

高质量、带标注的机器人数据一直是具身智能的核心瓶颈。传统数据采集依赖：

- 昂贵的人形机器人本体；
- 复杂的动作捕捉与传感器标定；
- 大量人工遥操作。

而“无本体数据采集”成为一种新趋势：使用可穿戴或手持设备采集第一人称视角数据，再迁移到机器人上训练。

### 5.2 EGO-RGBD：让每一帧都带高质量深度标注

奥比中光最新推出的 **无本体数据采集产品矩阵**中，**RGB-D 版本的 EGO 设备**将适配灵波科技专门为数采场景优化的 LingBot-Depth 版本。这意味着：

- 采集的每一帧 RGB 图像都能同步获得高质量深度标注；
- 无需昂贵的本体即可大规模采集带几何信息的数据；
- 采集数据可直接用于训练 LingBot-Depth、LingBot-Vision 以及上层 VLA 模型。

后续，双方还将进一步集成更高级别的商业版本模型，形成从数据采集、模型训练到终端部署的完整方案。

### 5.3 SDK 全面深度集成

除硬件外，灵波科技的模型还将通过 SDK 与奥比中光的相机生态深度集成。下游机器人厂商在使用奥比中光硬件时，可以“一键开启”空间感知能力，降低具身智能系统的集成门槛。

---

## 六、与相关工作的关系

### 6.1 深度估计与补全

| 工作 | 核心方法 | 特点 |
|---|---|---|
| **Depth Anything V2** | 基于大规模数据集的鲁棒单目深度估计 | 无需 RGB-D 输入，但深度尺度可能不一致 |
| **Metric3D** | 绝对尺度单目深度估计 | 强调真实尺度，但依赖相机内参 |
| **ZoeDepth / MiDaS** | 混合数据集训练的单目深度 | 通用性强，但无法补全传感器空洞 |
| **LingBot-Depth** | **Masked Depth Modeling + RGB-D 补全** | 针对 RGB-D 传感器失效区域，恢复真实尺度深度 |

LingBot-Depth 的独特之处在于：它不是从 RGB 中“猜测”深度，而是**以 RGB-D 相机原始输出为条件，利用视觉上下文修正和补全深度**。这种方法在机器人部署中更具实用性，因为它保留了传感器的真实尺度信息。

### 6.2 视觉基座模型

| 工作 | 机构 | 定位 |
|---|---|---|
| **SAM 2** | Meta | 通用图像/视频分割 |
| **DINOv2** | Meta | 自监督视觉特征 |
| **SigLIP / CLIP** | Google / OpenAI | 视觉-语言对齐 |
| **Depth Anything V2 Encoder** | 港中文等 | 深度感知视觉编码 |
| **LingBot-Vision** | Robbyant | 面向具身智能的空间视觉基座 |

LingBot-Vision 的差异在于它**面向具身智能场景**，强调空间几何、多视角一致性、动态场景理解，而不仅是语义或分割能力。

---

## 七、关键洞察与技术启示

### 7.1 传感器失效不是噪声，而是结构化掩码

LingBot-Depth 1.0 最重要的方法论贡献，是把深度传感器的失效区域重新定义为“掩码信号”。这一视角转变使得模型能够：

- 利用 RGB 上下文推断几何；
- 学习模态间的对齐表示；
- 将补全任务与下游感知/操作任务无缝连接。

### 7.2 数据规模是空间感知能力提升的关键

从 300 万到 1.5 亿，LingBot-Depth 2.0 的升级再次验证了一个朴素但重要的规律：**在视觉任务中，数据规模的扩大往往比架构微调带来更显著的能力跃迁**。对于具身智能而言，真实场景数据的采集与清洗能力，正在成为核心竞争力。

### 7.3 软硬协同是具身智能落地的主旋律

LingBot-Depth 2.0 与奥比中光的合作说明，算法公司需要与传感器/硬件公司深度合作，才能：

- 获得高质量、大规模的传感器原生数据；
- 针对特定相机 ISP 和深度引擎优化模型；
- 降低下游客户的集成成本，加速产业落地。

### 7.4 视觉基座是连接感知与决策的枢纽

LingBot-Vision 的推出表明，灵波科技正在构建一个统一的视觉表示层。未来，这一基座可能同时服务：

- LingBot-Depth（几何）
- LingBot-Map（建图）
- LingBot-VLA（决策与操作）
- LingBot-World（模拟与预测）

这种分层架构与 LLM 领域中“基础模型 + 专用适配器”的思路高度一致。

---

## 八、局限与未来方向

### 8.1 当前局限

1. **LingBot-Depth 2.0 与 LingBot-Vision 的技术细节尚未完全公开**：截至发布，完整论文、模型架构细节、训练 recipe 仍未披露，社区复现和深入研究存在一定门槛。
2. **1.5 亿数据的构成与质量**：真实数据与合成数据的比例、采集场景分布、标注精度等信息尚不清楚。
3. **泛化性验证有限**：公开报道主要强调实验室和工业认证结果，在开放世界、大规模真实部署中的表现仍需观察。
4. **开源协议与可获取性**：LingBot-Depth 1.0 已开源，但 2.0 是否完全开源、商业版本如何授权，仍有待明确。

### 8.2 未来方向

- **端到端联合训练**：将 LingBot-Vision、LingBot-Depth 2.0 与 LingBot-VLA 联合训练，实现感知-决策一体化；
- **多模态融合**：引入触觉、力觉、音频等模态，构建更全面的物理世界模型；
- **实时性与边缘部署**：针对机器人端侧算力优化模型，支持低延迟、低功耗运行；
- **开放生态**：进一步开放数据、模型和工具链，吸引更多开发者和机器人厂商共建生态；
- **长程记忆与语义地图**：结合 LingBot-Map 的 3D 重建能力，构建可更新的场景记忆。

---

## 九、结语

LingBot-Depth 1.0 通过 **Masked Depth Modeling** 重新定义了 RGB-D 深度补全问题，将传感器失效区域转化为模型学习的结构化信号，并以 300 万真实与合成数据展示了强大的空间感知能力。2026 年 7 月，**LingBot-Depth 2.0** 与 **LingBot-Vision** 的发布，则将这一能力推向了新的高度：1.5 亿训练数据、视觉基座模型、与奥比中光的软硬协同，标志着灵波科技正在从“做一个好的深度模型”走向“构建具身智能的空间视觉基础设施”。

对于机器人行业而言，这意味着：未来使用奥比中光等 RGB-D 硬件的厂商，或许可以像今天调用云端 API 一样，便捷地获得高质量深度感知与视觉理解能力。而当感知层足够可靠时，上层 VLA、世界模型、长期记忆等能力也将获得更坚实的物理基础。

---

## 参考资料

1. Tan, B., Sun, C., Qin, X., et al. *Masked Depth Modeling for Spatial Perception*. arXiv:2601.17895, 2026. [https://arxiv.org/abs/2601.17895](https://arxiv.org/abs/2601.17895)
2. GitHub: Robbyant/lingbot-depth. [https://github.com/Robbyant/lingbot-depth](https://github.com/Robbyant/lingbot-depth)
3. ModelScope: Robbyant/lingbot-depth. [https://modelscope.cn/models/Robbyant/lingbot-depth](https://modelscope.cn/models/Robbyant/lingbot-depth)
4. Pandaily: Ant Group Open-Sources LingBot-Depth. [https://pandaily.com/ant-group-open-sources-ling-bot-depth-next-generation-spatial-perception-model-based-on-masked-depth-modeling](https://pandaily.com/ant-group-open-sources-ling-bot-depth-next-generation-spatial-perception-model-based-on-masked-depth-modeling)
5. Quantum Zeitgeist: Robbyant’s LingBot-Depth AI Cuts Depth Error By 70% For Robotics. [https://quantumzeitgeist.com/robbyant-lingbot-depth-ai-robotics-depth-sensing/](https://quantumzeitgeist.com/robbyant-lingbot-depth-ai-robotics-depth-sensing/)
6. 新浪财经：蚂蚁灵波发布空间感知模型 LingBot-Depth 2.0，联合奥比中光加速产业落地. [https://finance.sina.com.cn/tob/2026-07-06/doc-inifvzfx8683285.shtml](https://finance.sina.com.cn/tob/2026-07-06/doc-inifvzfx8683285.shtml)
7. 新浪财经：国产基模一战封神，11 亿参数逆袭 70 亿大魔王. [https://k.sina.com.cn/article_5953740931_162dee08306703km78.html](https://k.sina.com.cn/article_5953740931_162dee08306703km78.html)
8. 智源社区：狂跑一万帧丝滑不崩！LingBot-Map 开源. [https://hub.baai.ac.cn/view/54127](https://hub.baai.ac.cn/view/54127)

---

> 本博客以 LingBot-Depth 2.0 / LingBot-Vision 发布日期 2026-07-07 作为时间标记，归类为 Vision。
