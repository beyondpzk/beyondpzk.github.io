---
title: Uni-LaViRA
date: 2026-05-26
categories: [VLN]
---

# Uni-LaViRA：以语言-视觉-机器人动作翻译统一具身导航

> **论文**：*Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation*
> **作者**：Hongyu Ding, Sizhuo Zhang, Ziming Xu, Jinwen Guo, Hongxiu Liu, Xingzhi Cheng, Zixuan Chen, Haifei Qi, Duo Wang, Hao Xu, Jieqi Shi, Yifan Zhang, Jing Huo, Jian Cheng, Yang Gao, Jiebo Luo
> **单位**：南京大学、中国科学院自动化研究所、北京航空航天大学、BMW（南京）信息技术有限公司、美国罗切斯特大学
> **发布时间**：2026-05-26（arXiv）
> **arXiv**：[2605.27582](https://arxiv.org/abs/2605.27582)
> **项目主页**：[https://xetroubadour.github.io/Uni-LaViRA/](https://xetroubadour.github.io/Uni-LaViRA/)

## 摘要

Uni-LaViRA 提出了一种**零训练（training-free）**的统一具身导航智能体架构。作者认为，导航任务的决策结构可以归纳为单一的**Language-Vision-Robot Actions Translation**：

- **Language Action**：输出语义级方向指令（如 front / left / right / back）；
- **Vision Action**：在选定视角的原始像素上输出视觉目标（bounding box）；
- **Robot Action**：通过几何反投影与路径规划，将像素目标转换为机器人可执行的 3-D 航点。

由于“方向语言”与“像素框”均位于预训练多模态大语言模型（MLLM）的**自然输出流形（natural output manifold）**内，导航可以完全由 MLLM 推理完成，而无需在机器人轨迹数据上训练 VLA。为处理长程指令与错误恢复，论文进一步提出 **TODO List Memory（TDM）** 与 **Second Chance Backtrack（SCB）**。在 VLN-CE R2R、RxR、HM3D-v2、HM3D-OVON、MP3D-EQA 与 OpenUAV 六个基准上，Uni-LaViRA 以零训练代价达到了与最新导航基础模型相当甚至更好的表现。

## 一、研究背景与动机

### 1.1 具身导航的多样性

过去十年，具身导航分化出多个任务家族：

| 任务家族 | 代表基准 | 输入 | 目标 |
|---|---|---|---|
| VLN-CE | R2R、RxR | 自然语言路线描述 | 按指令移动到终点 |
| ObjectNav | HM3D-v2、HM3D-OVON | 物体类别或开放词汇 | 找到指定物体实例 |
| EQA | MP3D-EQA | 自然语言问题 | 探索场景并回答问题 |
| Aerial-VLN | OpenUAV | 飞行指令 | 在户外三维空间执行指令 |

这些任务在指令格式、感知模态与机器人形态上差异显著，但共享同一决策本质：**感知视觉场景 → 理解语言输入 → 发出空间动作序列**。

### 1.2 当前主流的“ Scaling Law ”路线

近两年，导航领域的主流范式是不断放大**视觉-语言-动作（VLA）基础模型**：训练数据从不足 1M 轨迹增长到超过 16M，VLN-CE R2R 的成功率从 40% 以下提升至约 70%。这条路线将跨任务、跨形态的泛化能力押注在数据规模上。

Uni-LaViRA 提出**互补观点**：对于导航这类以**无接触（contact-free）**空间运动为主的任务，其动作空间本身落在预训练 MLLM 的自然输出流形内，因此泛化可以通过**结构设计**获得，而不必完全依赖数据规模。

### 1.3 从 LaViRA 到 Uni-LaViRA

LaViRA 是作者前期的会议版本，仅在 VLN-CE 单一任务上验证了 Language-Vision-Robot 三层分解。Uni-LaViRA 在此基础上解决了三个瓶颈：

1. **单任务范围**：将架构扩展到四种任务家族；
2. **长指令注意力漂移**：RxR 平均 120 词，UAV 飞行计划多阶段，LaViRA 容易遗漏中间子目标；
3. **错误恢复盲目**：LaViRA 的倒退回退会丢弃失败轨迹，导致重复犯错。

## 二、核心贡献

Uni-LaViRA 的贡献可概括为以下四点：

1. **统一四层任务的 agentic 架构**：同一套 prompt 接口、动作集合与控制器栈覆盖 VLN-CE、ObjectNav、EQA、Aerial-VLN。
2. **TODO List Memory（TDM）**：显式维护动态更新的子目标清单，将长程计划“朗读”回当前注意力窗口。
3. **Second Chance Backtrack（SCB）**：把失败子轨迹作为推理上下文，使单次导航变为自纠错过程。
4. **跨四种真实机器人部署**：轮式 Agilex Cobot Magic、Unitree G1 人形、Unitree Go1 四足、自研四旋翼 UAV，仅替换底层控制器。

表 1 对比了 LaViRA 与 Uni-LaViRA 的覆盖范围：

| 维度 | LaViRA | Uni-LaViRA |
|---|---|---|
| 任务家族 | 1 | 4 |
| 真实机器人平台 | 2 | 4 |
| 工作记忆 | Prompt 历史 | TDM |
| 错误恢复 | 单步回退 | SCB |
| 失效分析 | 定性 | 定量 + 规模化 |

## 三、方法详解

### 3.1 统一问题定义与接口

在每一步决策时刻 $t$，智能体接收：

- 任务描述 $\mathcal{T}$（自然语言）；
- 第一人称 RGB-D 观测 $\mathcal{O}_t$；
- 当前位姿 $(x_t, y_t, z_t, \theta_t)$；
- 结构化历史 $\mathcal{H}_t$（已访问航点、关键观测、先前决策）。

输出为机器人原生动作空间中的低层动作 $\mathcal{A}_t$。Uni-LaViRA 将策略分解为：

$$
\mathcal{A}_t = \pi_{\mathrm{robot}}\Big(\phi_{\mathrm{vis}}\big(\phi_{\mathrm{lang}}(\mathcal{T}, \mathcal{O}_t, \mathcal{H}_t)\big), D_t, \mathbf{K}, \mathrm{pose}_t\Big)
$$

其中：

- $\phi_{\mathrm{lang}}$：Language Action MLLM；
- $\phi_{\mathrm{vis}}$：Vision Action MLLM；
- $D_t$：深度图；$\mathbf{K}$：相机内参；
- $\pi_{\mathrm{robot}}$：与 embodiment 相关的确定性控制器。

### 3.2 Language Action：高层规划

Language Action 模型采用 **Gemini-3.1-Pro**，在每个决策步被调用。输入包括任务描述、四视角全景图（front/left/right/back）以及结构化历史。它从离散动作集合中发出一个 tool call：

$$
\mathcal{A}_t^{\mathrm{lang}} \in \big\{\texttt{turn}(\textsc{dir}), \texttt{backtrack}(\mathrm{wp}_k), \texttt{go\_stair}(\textsc{up/down}), \texttt{double\_check}(\textsc{stop})\big\}
$$

其中 $\textsc{dir} \in \{\mathrm{front}, \mathrm{left}, \mathrm{right}, \mathrm{back}\}$。每个调用以 JSON 输出，包含 progress_analysis、action_reasoning 以及 TDM 更新字段，便于解析与调试。

### 3.3 Vision Action：中层视觉 grounding

Vision Action 模型采用 **Qwen3.5-27B**，接收任务描述、Language Action 产生的进度描述，以及对应方向的单视角图像。它输出：

$$
\mathcal{A}_t^{\mathrm{vis}} = \big(\texttt{select}(\mathrm{bbox}/\mathrm{point}), \texttt{target\_desc}(\mathrm{text})\big)
$$

- $\texttt{select}$ 在原始像素上返回 2-D bounding box 或点；
- $\texttt{target\_desc}$ 返回简短目标描述（如“通往卧室的门口”）。

直接在像素上 grounding 摆脱了对预训练 waypoint predictor 的依赖，也能指向远处或不可遍历的语义线索（如走廊开口）。prompt 明确抑制过近目标，以避免贪婪热图跟随的局部最优。

### 3.4 Robot Action：低层控制

Robot Action 控制器是确定性几何模块：

1. 在 Vision Action 输出的 box 内选取代表像素 $(u^*, v^*)$；
2. 读取该像素深度 $d_t$，反投影到相机坐标系：

$$
\mathbf{p}_{\mathrm{cam}} = d_t \, \mathbf{K}^{-1} [u^*, v^*, 1]^\top
$$

3. 利用当前位姿转换到世界坐标系 $\mathbf{p}_{\mathrm{world}}$；
4. 在全局累计地图上规划短程轨迹并执行。

地面机器人使用 2-D 占用栅格 + Fast-Marching；UAV 使用 3-D 体素栅格 + 可见性图搜索。控制器是系统中唯一与 embodiment 相关的组件，跨平台部署时无需修改上层 MLLM。

### 3.5 TODO List Memory（TDM）

TDM 维护一个有序列表 $\mathcal{L}_t = [\ell_t^{(1)}, \ldots, \ell_t^{(n_t)}]$，每个条目为三元组：

$$
\ell_t^{(i)} = (\mathrm{content}^{(i)}, \mathrm{status}^{(i)}, \mathrm{result}^{(i)})
$$

其中 $\mathrm{status}^{(i)} \in \{\texttt{pending}, \texttt{completed}\}$，$\mathrm{result}^{(i)}$ 记录支持完成判定的观测文本。

在 episode 开始时，$\phi_{\mathrm{lang}}$ 根据指令与初始全景图生成初始清单；之后每一步，模型先输出四种更新操作之一，再基于更新后的清单选择动作：

- $\texttt{update}(i, \text{status}=\texttt{completed}, \text{result}=r)$：标记完成并记录依据；
- $\texttt{rewrite}(i, \text{content}=c)$：细化待完成子目标；
- $\texttt{add}(\text{content}=c, \text{index}=j)$：新增子目标；
- $\texttt{remove}(i)$：删除不再相关的子目标。

TDM 完全在 prompt 空间实现，不引入任何可训练参数。

### 3.6 Second Chance Backtrack（SCB）

SCB 将错误恢复提升为 first-class 动作。实现分为两步：

1. **显式回退**：Uni-LaViRA 在全局占用地图上标记每一次 Language Action 调用位置为航点，动作集合包含任意历史航点的 $\texttt{backtrack}(\mathrm{wp}_k)$。选中后，控制器规划返回路径。

2. **二次机会重决策**：回到 $\mathrm{wp}_k$ 后，系统向 $\phi_{\mathrm{lang}}$ 提供三类证据：
   - 原始任务 $\mathcal{T}$ 与航点处全景图 $\mathcal{O}^{(\mathrm{wp}_k)}$；
   - 当初离开航点时选择的失败方向 $\mathcal{A}_{\mathrm{prev}}^{\mathrm{lang}}$；
   - 从 $\mathrm{wp}_k$ 到死端收集的 egocentric 图像序列 $\Pi_k^{\mathrm{fail}}$。

模型被提示检查失败轨迹、诊断原因，并从剩余方向中选择新方向。SCB 与 TDM 正交：TODO 清单在回退后保留，重决策时还可更新失败子目标。

### 3.7 统一推理流程

单条 episode 的推理流程可概括为：

1. 初始化 TDM 清单与航点历史；
2. 每步调用 Language Action，联合输出 TDM 更新与动作；
3. 若动作为 stop，则终止；若为 backtrack，则回退并触发 SCB 重决策；
4. 调用 Vision Action 在选定方向图像上生成 bounding box；
5. Robot Action 反投影、规划并执行低层动作；
6. 更新观测、历史与航点，回到步骤 2。

## 四、实验

### 4.1 评测设置

Uni-LaViRA 在 6 个标准基准上评估，覆盖 4 个任务家族：

| 任务家族 | 基准 | 主要指标 |
|---|---|---|
| VLN-CE | R2R、RxR val-unseen | NE↓、OSR↑、SR↑、SPL↑、nDTW↑ |
| ObjectNav | HM3D-v2、HM3D-OVON | SR↑、SPL↑ |
| EQA | MP3D-EQA | ACC↑ |
| Aerial-VLN | OpenUAV UM | NE↓、OSR↑、SR↑、SPL↑ |

实验使用 Habitat-Sim（R2R/RxR/HM3D/MP3D-EQA）与 AirSim（OpenUAV）。地面机器人配备单个 640×480 RGB-D 相机，通过原地旋转获得四视角全景；UAV 配备五台固定相机（front/left/right/back/down）。Language Action 为 Gemini-3.1-Pro，Vision Action 为 Qwen3.5-27B，均不调参、纯推理。为控制评估成本，每个基准从 val-unseen 中抽取分层 100-episode 子集，运行 3 个随机种子。

### 4.2 主实验结果

表 2 给出 VLN-CE R2R 与 RxR 的结果：

| 方法 | R2R NE↓ | R2R OSR↑ | R2R SR↑ | R2R SPL↑ | RxR NE↓ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniNav（监督） | 3.74 | 74.6 | **69.5** | **66.1** | 3.77 | **73.6** | **62.0** | – |
| ABot-N0（监督） | 3.78 | 70.8 | 66.4 | 63.9 | 3.83 | 69.3 | 60.0 | – |
| SPAN-Nav（监督） | 4.07 | 75.3 | 66.3 | 59.3 | 4.20 | 69.7 | 60.1 | **67.9** |
| StreamVLN（监督） | 4.98 | 64.2 | 56.9 | 51.9 | 6.22 | 52.9 | 46.0 | 61.9 |
| LaViRA（零训练） | 6.54 | 48.7 | 38.3 | 28.3 | – | – | – | – |
| **Uni-LaViRA（零训练）** | **3.66** | **73.7** | 60.7 | 47.7 | 6.48 | 51.3 | 34.0 | 53.7 |

在 R2R 上，Uni-LaViRA 的 NE（3.66）低于所有列出的监督方法，SR（60.7%）接近 OmniNav 与 ABot-N0，且是所有零训练方法中的最佳。

表 3 给出 ObjectNav 与 EQA 结果：

| 方法 | HM3D-v2 SR↑ | HM3D-v2 SPL↑ | HM3D-OVON SR↑ | HM3D-OVON SPL↑ | MP3D-EQA ACC↑ |
|---|---:|---:|---:|---:|---:|
| Uni-NaVid（监督） | 73.7 | 37.1 | 39.5 | 19.8 | 47.3 |
| FiLM-Nav（监督） | 77.0 | 41.3 | – | – | – |
| OmniNav（监督） | – | – | 59.2 | 33.2 | – |
| ApexNav（零训练） | 76.2 | 38.0 | – | – | – |
| DSCD-Nav（零训练） | 73.0 | 38.7 | – | – | – |
| VLFM（零训练） | 62.6 | 31.0 | 38.5 | 22.2 | – |
| **Uni-LaViRA（零训练）** | **77.7** | **46.1** | **60.0** | **40.5** | **54.7** |

Uni-LaViRA 在 HM3D-v2、HM3D-OVON、MP3D-EQA 上均超越所有列出的零训练方法，并在 HM3D-v2、HM3D-OVON、MP3D-EQA 上超过最佳监督方法。

表 4 给出 OpenUAV 结果：

| 方法 | Full NE↓ | Full OSR↑ | Full SR↑ | Full SPL↑ |
|---|---:|---:|---:|---:|
| TravelUAV（监督） | 139 | 20.8 | 4.18 | 3.84 |
| AerialVLA（监督） | 67.4 | 52.9 | 37.6 | 28.2 |
| LongFly（监督） | 108 | 30.3 | 11.3 | 9.32 |
| **Uni-LaViRA（零训练）** | **84.29** | **67.33** | **40.00** | **30.37** |

Uni-LaViRA 是 OpenUAV 上首个零训练条目，SR 40.0% 超过 AerialVLA 的 37.6%。

### 4.3 子集有效性验证

论文在 100-episode 分层子集上复现了每个任务的两个公开基线。在 34 个有完整集对照的指标单元中，21 个（62%）与子集结果相差 ≤ 2 绝对点，31 个（91%）相差 ≤ 5 绝对点。OpenUAV 上存在较大方差，因为轨迹长达数百米且部分基线成功率极低。总体而言，100-episode 子集保留了官方验证集的难度分布。

### 4.4 机制消融与失败分析

论文提供了 TDM 与 SCB 的逐任务消融、1,800-trial 失败模式分类以及推理成本分析。核心发现包括：

- TDM 显著改善长指令任务（如 RxR、OpenUAV）的子目标跟踪；
- SCB 将错误子轨迹转化为推理上下文，减少重复犯错；
- 上层 MLLM 推理成本主要集中在 Language Action 调用，Vision Action 相对轻量。

## 五、优势与局限

### 5.1 主要优势

1. **零训练泛化**：无需机器人轨迹数据、无需 VLA 训练，直接继承 MLLM 的泛化能力。
2. **任务统一**：同一 prompt 与动作集合覆盖四种导航任务，接口高度一致。
3. **跨形态迁移**：上层 MLLM 与 TDM/SCB 完全不变，仅替换底层控制器即可部署到轮式、人形、四足、飞行机器人。
4. **可解释性强**：Language Action 的 JSON 输出暴露推理轨迹，TDM 清单显式展示子目标状态。

### 5.2 局限与待改进

1. **依赖 MLLM API**：Gemini-3.1-Pro 与 Qwen3.5-27B 的调用延迟与成本限制了实时性；完全本地部署尚需探索。
2. **全景图获取**：地面机器人需原地旋转获得四视角图像，真实场景中旋转时间不可忽视。
3. **接触场景假设**：论文明确将导航定位为“mostly contact-free”；对于需要复杂接触交互的操作-导航耦合任务，该分解是否仍然成立需要额外验证。
4. **长程效率**：虽然 SCB 提升了成功率，但回退会增加路径长度，SPL 在部分任务上仍低于端到端监督模型。

## 六、历史意义与后续影响

### 6.1 对导航基础模型范式的反思

Uni-LaViRA 与当前“越大数据、越大模型”的导航 VLA scaling law 路线形成鲜明对照。它证明：当任务的输出空间落在预训练 MLLM 的自然流形内时，**结构设计本身即可带来强泛化**，而不必将一切押注在机器人数据规模上。这一观点为资源受限的研究者提供了一条可行路径。

### 6.2 对 Agentic Robotics 的启示

TDM 与 SCB 不仅是导航专用技巧，也体现了 agentic 系统的两个通用原则：

- **外部化工作记忆**：将计划写成结构化清单并反复回读，比单纯拉长上下文更有效；
- **错误即信息**：把失败轨迹作为重决策的上下文，而非简单丢弃。

这些思想可迁移到其他长程、可逆、需自我纠错的 embodied 任务中。

### 6.3 与相关工作的关系

- ** vs. Uni-NaVid / NaVILA**：Uni-NaVid 与 NaVILA 均采用端到端 VLA 训练路线，依赖大量导航数据；Uni-LaViRA 则完全零训练，二者在哲学上互补。
- ** vs. LaViRA**：Uni-LaViRA 是 LaViRA 的系统扩展，从单任务到四任务、从两平台到四平台，并新增 TDM 与 SCB。
- ** vs. 监督导航基础模型**：在多个基准上，Uni-LaViRA 以零训练代价逼近或超越 OmniNav、ABot-N0 等监督模型，挑战了“导航必须大规模训练”的默认假设。

## 七、总结

Uni-LaViRA 将具身导航重新定义为**Language-Vision-Robot Actions Translation**，利用预训练 MLLM 在“方向语言”与“像素目标”上的自然输出能力，构建了一个零训练、跨任务、跨机器人的统一 agentic 导航系统。通过 **TODO List Memory** 管理长程子目标，通过 **Second Chance Backtrack** 把失败转化为推理上下文，该系统在 VLN-CE、ObjectNav、EQA、Aerial-VLN 四类任务的六个主流基准上取得了与最新监督导航基础模型相当甚至更优的性能。

其核心结论可以概括为一句话：**对于以空间运动为主的导航任务，结构正确的分解足以释放基础模型的预训练泛化能力，而不必将数百万条机器人轨迹作为前提。** 这一工作不仅为导航研究提供了新的零训练范式，也为更广泛的 agentic embodied AI 系统设计提供了重要参考。
