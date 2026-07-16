---
title: StreamVLN
date: 2025-07-07
categories: [VLN]
---

# StreamVLN：面向流式视觉-语言导航的 SlowFast 上下文建模

> **论文**：*StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling*
> **作者**：Meng Wei*, Chenyang Wan*, Xiqian Yu*, Tai Wang†, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, Xihui Liu‡, Jiangmiao Pang‡（* 同等贡献，† 项目主导，‡ 通讯作者）
> **单位**：Shanghai AI Laboratory、The University of Hong Kong、Zhejiang University、Shanghai Jiao Tong University
> **发布时间**：2025-07-07（arXiv）
> **arXiv**：[https://arxiv.org/abs/2507.05240](https://arxiv.org/abs/2507.05240)
> **项目主页**：[https://streamvln.github.io/](https://streamvln.github.io/)

## 摘要

StreamVLN 是首个面向**流式视觉-语言导航（Streaming VLN）**的端到端框架，旨在让智能体在持续视频流上以低延迟、连贯多轮对话的方式执行语言指令。论文指出，现有基于 Video-LLM 的 VLN 方法往往在**细粒度视觉理解、长程上下文建模与计算效率**之间顾此失彼：要么固定采样少量帧而丢失时序细节，要么对视觉 token 做特征级压缩而牺牲空间精度，且每步都刷新对话历史导致冗余计算。为此，StreamVLN 提出**快慢双轨（SlowFast）上下文建模**策略：

- **Fast stream**：通过固定大小的滑动窗口缓存最近 \(N\) 轮对话的 KV 状态，实现快速响应的动作解码；
- **Slow memory**：将历史窗口的视觉上下文以 3D 感知的方式进行 token 剪枝，压缩为稀疏记忆 token，支持跨窗口 KV cache 复用。

该设计使得模型在训练时使用短片段（16 帧），测试时却能稳定处理长视频流，上下文长度与推理成本均有界。在 VLN-CE 的 R2R 与 RxR Val-Unseen 上，StreamVLN 在仅使用 RGB 输入的方法中达到最优；同时还在 Unitree Go2 机器狗上完成了真实世界部署。

## 一、研究背景与动机

### 1.1 从离散 VLN 到连续流式 VLN

视觉-语言导航（VLN）要求智能体根据自然语言指令在环境中移动。早期研究集中在离散场景图（如 R2R、RxR），智能体在预定义节点间“瞬移”；随后发展到连续环境 VLN-CE（Habitat 仿真器），需要低层连续控制。近年来，Video-LLM 的进步催生了 **Vision-Language-Action（VLA）** 导航模型，将视觉编码、语言理解与动作预测统一为端到端框架。

### 1.2 现有 Video-LLM 导航方法的瓶颈

真实世界导航要求智能体持续处理 incoming 视频流，这对现有方法提出三重挑战：

| 挑战 | 具体表现 | 现有做法的缺陷 |
|------|----------|----------------|
| 长程上下文 | 视觉 token 随步数线性增长 | 固定帧采样（如 NaVILA、MapNav）时序分辨率不足 |
| 细粒度感知 | 低层动作需要精细时空线索 | 特征池化/合并（如 NaVid、UniNaVid）损失空间细节 |
| 计算效率 | 每步都重新预填充全部历史 | 独立对话导致大量冗余计算，延迟随回合增加 |

### 1.3 StreamVLN 的解决思路

StreamVLN 将导航过程重新建模为**交错视觉-语言-动作的多轮对话**。与每步重置上下文不同，它通过可复用的 KV cache 在连续对话间传递状态，并引入快慢双轨机制平衡响应速度与长程记忆。

## 二、核心贡献

1. **流式 VLN 框架**：首次将 Video-LLM 扩展为交错 VLA 模型，支持对连续视频流进行多轮对话式交互。
2. **SlowFast 上下文建模**：
   - Fast：滑动窗口 KV cache 保留最近 \(N\) 轮对话，实现低延迟动作生成；
   - Slow：基于 3D 体素的空间剪枝将历史窗口压缩为记忆 token，控制显存增长。
3. **训练可扩展性**：模型只需在 16 帧短片段上训练，即可泛化到长视频流，避免上下文长度爆炸。
4. **SOTA 性能与真实部署**：在 VLN-CE R2R/RxR 上取得 RGB-only 最优；并在 Unitree Go2 上完成物理世界验证。

## 三、方法详解

### 3.1 预置：连续多轮自回归生成

每个对话回合 \(d_i = (o_i, a_i)\) 包含一次新观测 \(o_i\) 与模型生成的动作响应 \(a_i\)。到第 \(i\) 步时，完整输入序列为：

$$
o_1 a_1 o_2 a_2 \cdots o_{i-1} a_{i-1} o_i
$$

Transformer LLM 每轮先执行 prefill 阶段编码输入 token 并缓存 KV，再自回归解码动作 token。若跨回合不复用 KV，则每轮都要对所有历史 token 重新预填充，计算量巨大。

### 3.2 Fast-Streaming Dialogue Context

StreamVLN 采用**固定大小滑动窗口**管理活跃对话上下文：

$$
W_j = [o_{i-N+1} a_{i-N+1} \cdots o_i a_i]
$$

- 窗口容量 \(N\) 固定，文中取 \(N=8\)；
- 当窗口满时，将其 KV 状态卸载；其中非观测 token（prompt、动作等）直接丢弃；
- 视觉 token 经过 3.3 节的压缩后成为记忆 token，供后续窗口使用。

生成当前动作时，解码器同时接收：

$$
a_i^{W_{j+1}} = \text{Decoder}\bigl(o_i, \{\mathcal{M}_0, \ldots, \mathcal{M}_j\}, \{k_{i-N+1}v_{i-N+1}, \ldots, k_{i-1}v_{i-1}\}\bigr)
$$

其中 \(\{\mathcal{M}\}\) 为历史记忆，\(\{kv\}\) 为当前窗口缓存。

### 3.3 Slow-Updating Memory Context

#### 3.3.1 时序采样

为减少时间冗余，先对历史帧做固定数量采样；同时避免可变长度记忆带来的“时长偏置”。

#### 3.3.2 3D 体素空间剪枝

核心思想：将 2D 图像 patch token 通过深度反投影到统一 3D 空间，再离散化为体素。若多个帧中的 token 落入同一体素，则只保留**最新观测**对应的 token。

算法流程（对应论文 Algorithm 1）：

1. 构建体素图 \(V \in \mathbb{Z}^{T \times H \times W}\)，步长 \(K\)，阈值 \(\theta\)；
2. 初始化剪枝掩码 \(M = 0\)，维护字典 `latest[(period, voxel)]`；
3. 对每个有效 token \((t, x, y)\)，计算 period \(p = \lfloor t/K \rfloor\) 与体素 \(v = V_{t,x,y}\)；
4. 若该 \((p, v)\) 未被记录或当前 \(t\) 更新，则更新 `latest`；
5. 最终被保留的 token 置 \(M_{t,x,y} = 1\)；
6. 若某帧保留 token 数少于 \(\theta \cdot H \cdot W\)，则丢弃该帧全部 token。

剪枝优势：
- 不改变已计算 token 的值，因此历史 KV cache 可被**离线复用**；
- 相比特征池化，保留更高图像分辨率，维持 Video-LLM 的可迁移性；
- 平均可减少约 20% 输入 token。

### 3.4 多源数据联合训练

训练分为两阶段：

1. **第一阶段**：仅在 oracle VLN 轨迹上微调 1 个 epoch；
2. **第二阶段**：用模型收集 DAgger 纠错轨迹，再与通用多模态数据联合训练 1 个 epoch。

数据配方如表所示：

| 数据类型 | 来源 | 规模 | 作用 |
|----------|------|------|------|
| VLN oracle | R2R、R2R-EnvDrop、RxR（MP3D 60 场景） | 450K | 基础导航模仿学习 |
| ScaleVLN | HM3D 700 场景子集 | 150K | 提升场景泛化 |
| DAgger | 模型 rollout + shortest-path 专家 | 240K | 增强新场景与错误恢复 |
| VideoQA | LLaVA-Video-178K、ScanQA | 248K | 保持时空/几何推理 |
| 交错图文 | MMC4 | 230K | 增强多轮对话能力 |

实现细节：
- 基座模型：LLaVA-Video 7B（语言模型 Qwen2-7B）；
- 学习率：LLM 2e-5，视觉编码器 5e-6；
- batch：每步 128 个视频片段；
- 训练耗时：约 1500 A100 GPU 小时。

## 四、实验

### 4.1 评测设置

- **仿真基准**：VLN-CE 的 R2R-CE 与 RxR-CE Val-Unseen；
- **指标**：Navigation Error（NE↓）、Oracle Success（OS↑）、Success Rate（SR↑）、Success weighted by Path Length（SPL↑），RxR 额外报告 nDTW↑；
- **真实世界**：Unitree Go2 机器狗 + Intel RealSense D455（朝上安装），远程 RTX 4090 推理，平均推理延迟 0.27s/4 actions。

### 4.2 VLN-CE 主实验

表 1 汇总了 R2R/RxR Val-Unseen 的结果。

| 方法 | 输入 | R2R Val-Unseen NE↓ | OS↑ | SR↑ | SPL↑ | RxR Val-Unseen NE↓ | SR↑ | SPL↑ | nDTW↑ |
|------|------|--------------------|-----|-----|------|--------------------|-----|------|-------|
| NaVid | RGB | 5.47 | 49.1 | 37.4 | 35.9 | - | - | - | - |
| MapNav | RGB | 4.93 | 53.0 | 39.7 | 37.2 | - | - | - | - |
| NaVILA | RGB | 5.37 | 57.6 | 49.7 | 45.5 | - | - | - | - |
| **StreamVLN** | RGB | **5.43** | **62.5** | **52.8** | **47.2** | **6.72** | **48.6** | **42.5** | **60.2** |
| NaVILA† | RGB | 5.22 | 62.5 | 54.0 | 49.0 | 6.77 | 49.3 | 44.0 | 58.8 |
| UniNaVid† | RGB | 5.58 | 53.3 | 47.0 | 42.7 | 6.24 | 48.7 | 40.9 | - |
| **StreamVLN†** | RGB | **4.98** | **64.2** | **56.9** | **51.9** | **6.22** | **52.9** | **46.0** | **61.9** |

> † 表示使用 R2R-CE/RxR-CE 之外额外训练数据的方法。

关键结论：
- StreamVLN 在 RGB-only 方法中取得 SOTA；
- 即使与使用额外数据的 NaVILA†、UniNaVid† 相比，StreamVLN† 仍在 R2R 上领先（SR 56.9 vs 54.0/47.0）；
- 与使用 waypoint predictor 的 ETPNav（NE 4.71，SR 57.0）相比，StreamVLN† 达到可比的 SR（56.9），但无需全景/路点监督。

### 4.3 3D 场景理解迁移实验

为验证模型保留的通用视觉推理能力，作者在 ScanQA 上评测：

| 方法 | Bleu-4↑ | Rouge↑ | Meteor↑ | Cider↑ | EM↑ |
|------|---------|--------|---------|--------|-----|
| NaVILA (16 frames) | 15.2 | 48.3 | 99.8 | 19.6 | 27.4 |
| **StreamVLN (16 frames)** | **15.7** | 48.3 | **100.2** | **19.8** | **28.8** |

StreamVLN 在通用 3D VQA 上优于 NaVILA 等导航模型，说明其视觉推理能力未被导航微调破坏。

### 4.4 真实世界定性结果

论文在家庭、办公空间、商场、户外四类场景中进行了物理部署。StreamVLN 能够在多变光照、多 landmark 的长程指令下完成导航，并维持较低推理延迟。

### 4.5 消融实验

#### 4.5.1 数据成分

| DAgger | VL Data | ScaleVLN | VideoQA | MMC4 | NE↓ | OS↑ | SR↑ | SPL↑ |
|--------|---------|----------|---------|------|-----|-----|-----|------|
| ✓ | ✓ | - | - | - | 6.05 | 53.8 | 45.5 | 41.6 |
| ✓ | ✓ | - | VideoQA | - | 5.47 | 57.8 | 50.8 | 45.7 |
| ✓ | ✓ | - | VideoQA | ✓ | 5.43 | 62.5 | 52.8 | 47.2 |
| ✓ | ✓ | ✓ | VideoQA | ✓ | 5.10 | 64.0 | 55.7 | 50.9 |
| - | ✓ | ✓ | VideoQA | ✓ | 5.73 | 56.4 | 50.2 | 47.1 |
| ✓ | - | ✓ | - | - | 5.90 | 55.9 | 47.9 | 43.6 |

- 加入 VideoQA/MMC4 分别带来 +5.3/+2.0 SR 的提升；
- ScaleVLN 提供 +2.9 SR 的场景多样性收益；
- DAgger 数据至关重要，移除后 SR 下降 5.5。

#### 4.5.2 记忆上下文与滑动窗口大小

| Memory Context | Window | NE↓ | OS↑ | SR↑ | SPL↑ |
|----------------|--------|-----|-----|-----|------|
| 2×196 | 8 | 6.96 | 48.2 | 37.3 | 34.2 |
| 4×196 | 8 | 6.62 | 49.1 | 38.9 | 35.4 |
| 8×196 | 8 | 6.05 | 53.8 | 45.5 | 41.6 |
| all | 8 | 6.76 | 49.5 | 40.0 | 36.4 |
| 8×196 | 4 | 6.31 | 51.1 | 41.4 | 37.5 |
| 8×196 | 2 | 6.16 | 52.8 | 43.7 | 40.3 |

- 记忆从 2×196 增加到 8×196，SR 从 37.3 提升到 45.5，说明细粒度长程记忆的重要性；
- “all” 记忆反而性能下降，说明过长上下文会引入训练/测试偏置；
- 窗口大小 \(N=8\) 在性能与训练成本间取得最佳平衡。

#### 4.5.3 3D 体素剪枝

| Pruning | R2R NE↓ | OS↑ | SR↑ | SPL↑ | RxR NE↓ | SR↑ | SPL↑ | nDTW↑ |
|---------|---------|-----|-----|------|---------|-----|------|-------|
| ✗ | 5.10 | 64.0 | 55.7 | 50.9 | 6.16 | 51.8 | 45.0 | 62.1 |
| ≈20% | 4.98 | 64.2 | 56.9 | 51.9 | 6.22 | 52.9 | 46.0 | 61.9 |

剪枝在减少约 20% token 的同时，R2R SR/SPL 提升 +1.2/+1.0，RxR SR/SPL 提升 +1.1/+1.0，说明去除空间冗余有助于模型聚焦关键 token。

#### 4.5.4 KV Cache 复用对延迟的影响

论文 Figure 5 显示：
- **Full Turns（完全复用）**：延迟始终最低，仅需对当前观测做 prefill；
- **Sliding Window**：每到一个新窗口起点时 latency 会有小幅尖峰；
- **Single Turn（不复用）**：延迟随回合数线性增长。

### 4.6 动作 token 设计

补充材料中比较了不同动作表示：

| Action Type | Token 数 | Generate Time(s) | NE↓ | OS↑ | SR↑ | SPL↑ |
|-------------|----------|------------------|-----|-----|-----|------|
| forward/left/right/stop | 4 | 0.27 | 6.25 | 52.2 | 44.4 | 41.0 |
| natural language | 23 | 1.00 | 5.74 | 55.4 | 47.2 | 42.9 |
| ↑←→ + stop (StreamVLN) | 4 | 0.27 | 6.05 | 53.8 | 45.5 | 41.6 |

StreamVLN 采用稀有符号（↑、←、→）表示动作，兼顾效率与避免过拟合常见词汇。

## 五、优势与局限

### 5.1 主要优势

- **低延迟**：通过 KV cache 复用，避免每轮重新预填充；
- **有界显存**：滑动窗口 + token 剪枝使上下文长度与 KV cache 显存可控；
- **数据高效**：仅用 150K ScaleVLN 子集即超越在 3M 轨迹上训练的 HMAT；
- **泛化能力强**：多源联合训练保留通用视觉推理，支持跨任务（ObjectNav）零样本迁移；
- **真实可部署**：已在四足机器人上完成端到端验证。

### 5.2 局限与未来方向

论文在第 6 节明确列出三点局限：

1. **低层动作鲁棒性**：直接从原始观测生成低层动作对视角变化、遮挡较敏感；
2. **更长程场景**：当前混合记忆在超长序列上保持一致推理仍有挑战；
3. **动作历史同步**：显式动作历史作为对话上下文的一部分，给异步推理与部署带来同步复杂度。

## 六、历史意义与后续影响

### 6.1 在 VLN 演进中的位置

StreamVLN 处于第三代 VLA-based VLN 的关键节点：

| 阶段 | 代表工作 | 特点 |
|------|----------|------|
| 离散 VLN | R2R、RxR | 预定义节点图，强调高层决策 |
| 连续 VLN-CE | CMA、ETPNav | 低层控制，常依赖 waypoint predictor |
| Video-LLM VLA | NaVid、NaVILA、UniNaVid、**StreamVLN** | 端到端视频理解，但 StreamVLN 首次解决流式长上下文与延迟问题 |

### 6.2 对后续研究的启发

- **SlowFast 思想**：将“活跃上下文”与“压缩记忆”解耦，可迁移至其他长程视频-语言任务；
- **KV cache 工程化**：展示了如何在多轮 embodied 交互中高效复用 KV cache；
- **训练数据配方**：DAgger + 通用 VL 数据的联合训练成为提升导航模型泛化的标准范式；
- **真实世界部署**：证明 7B 视频 VLA 可在消费级 GPU 上支撑机器狗实时导航。

## 七、总结

StreamVLN 通过 **SlowFast 混合上下文建模** 解决了流式 VLN 中长程记忆、细粒度感知与低延迟之间的三角矛盾。其核心设计——滑动窗口 KV cache 作为 Fast stream、3D 体素剪枝压缩历史作为 Slow memory——使得模型能够以有界的上下文和显存成本，持续处理长视频流并生成连贯动作。在 VLN-CE 基准与 Unitree Go2 真实环境中，StreamVLN 均展现出领先的性能与部署可行性，为构建可实时运行的视觉-语言-动作导航模型提供了重要参考。
