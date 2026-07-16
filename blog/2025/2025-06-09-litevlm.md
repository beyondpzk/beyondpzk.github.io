---
title: LiteVLM：面向嵌入式设备的低延迟 VLM 推理流水线
date: 2025-06-09
categories: [VLM]
---

# LiteVLM：面向嵌入式设备的低延迟 VLM 推理流水线

> **论文**: *LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments*  
> **作者**: Jin Huang, Yuchao Jin, Le An, Josh Park  
> **机构**: NVIDIA  
> **arXiv**: [arXiv:2506.07416](https://arxiv.org/abs/2506.07416)  
> **发布时间**: 2025-06-09

---

## 导语：VLM 上车的最后一公里

近年来，多模态大语言模型（MLLM / VLM）在视觉理解、推理和交互方面展现出惊人能力。从 GPT-4V、Claude 3 到开源的 Qwen-VL、InternVL、LLaVA，这些模型不仅能看懂图像，还能用自然语言描述场景、回答视觉问题、甚至辅助自动驾驶决策。然而，一个现实问题始终横亘在眼前：**这些模型大多运行在云端 GPU 集群上，而机器人、无人机、自动驾驶汽车等边缘设备却对延迟、功耗和内存有着严苛限制**。

如何让 VLM 在资源受限的嵌入式设备上实时运行？这是 VLM 从实验室走向真实物理世界的“最后一公里”。2025 年 6 月，NVIDIA 的研究者发布了 **LiteVLM**，提出了一套面向嵌入式设备的低延迟 VLM 推理流水线。通过在 NVIDIA DRIVE Thor 平台上的实测，LiteVLM 在保持任务精度的同时，将端到端推理延迟降低了 **2.5×**，配合 FP8 量化后更是达到 **3.2×** 加速。

---

## 一、为什么 VLM 在边缘设备上这么慢？

一个典型的 VLM 通常包含三个部分：

1. **Vision Encoder（通常是 ViT）**：将输入图像编码为大量视觉 token；
2. **Alignment Module（MLP / Q-Former / Perceiver）**：将视觉特征对齐到 LLM 的语义空间；
3. **LLM Decoder**：以视觉 token 和文本 token 为输入，自回归生成输出。

瓶颈主要来自两个方面：

- **Prefill 阶段**：视觉 token 数量庞大（尤其高分辨率图像会被切成大量 patch），导致 ViT 编码和 LLM 第一次前向传播的二次方注意力计算非常昂贵。
- **Decode 阶段**：自回归生成每次只输出一个 token，重复调用 LLM 的开销很大。

对于自动驾驶等安全关键应用，端到端延迟往往需要控制在数百毫秒以内。传统 VLM 在高端服务器 GPU 上或许还能接受，但在车载嵌入式平台（如 NVIDIA DRIVE Thor）上则难以满足实时性要求。

---

## 二、LiteVLM 的核心思路

LiteVLM 并没有重新设计一个从零开始的小模型，而是提出了一套**即插即用、可叠加**的推理加速流水线，在标准 VLM 基础上引入三个关键模块：

1. **Patch Selection Module（视角/图像块选择）**：在 ViT 编码之前，根据文本查询动态选择相关的相机视角，减少无关图像块的编码；
2. **Token Selection Module（视觉 token 选择）**：在 LLM Prefill 阶段进一步压缩视觉 token；
3. **Speculative Decoding Head（投机解码）**：用轻量 draft model 加速自回归 token 生成。

这三个模块分别作用于推理的不同阶段，可以联合使用，实现端到端延迟的大幅降低。

---

## 三、方法详解

### 3.1 Patch Selection Module：先选视角，再编码

在自动驾驶等多视角感知场景中，一个查询通常只与部分相机视角相关。例如：

- “前方是否有行人？” → 主要依赖前视相机；
- “左侧车道是否有车辆？” → 主要依赖左前/左后相机。

传统 VLM 会将所有视角的图像统一编码，造成大量冗余计算。LiteVLM 的 Patch Selection Module 在 ViT 编码之前，先根据文本查询判断哪些相机视角是相关的，只编码相关视角的图像块。

**实现方式**：

- 使用一个 4 层的小型 transformer encoder 处理文本查询；
- 通过 cross-attention 让查询特征与 $N$ 个视角相关的可学习 latent query 交互；
- 输出每个视角的二分类 logits，判断该视角是否应被保留；
- 训练时采用 BCE 损失，真值标签通过两种策略生成：
  - 对于显式包含视角词的查询（如 “front left”），使用词汇匹配；
  - 对于隐式查询，使用 GPT-3.5-Turbo 作为评估器判断需要哪些视角。

推理时，最终分数是词汇映射分数与模型 logits 的加权和，高于阈值的视角才会进入后续 ViT 编码。

**效果**：在 DRIVE Thor 的 8 目配置下，平均只需要编码 3.5 个视角的图像块，ViT 延迟从 136.9 ms 降至 45.1 ms。

### 3.2 Token Selection Module：再压缩进入 LLM 的视觉 token

即使经过视角选择，进入 LLM 的视觉 token 数量仍然可观。LiteVLM 进一步引入 Token Selection Module，在 LLM 的 prefill 阶段压缩视觉 token。

**设计特点**：

- 从已微调好的 LLM 中提取第一层 decoder，实例化为一个独立的轻量模块；
- 单独微调该模块用于 token 剪枝，而不改变主 LLM 的权重；
- 训练信号来自两部分：
  1. 利用 VLM 自身的 attention hidden features 作为视觉 token 重要性分数；
  2. 结合 nuScenes 数据集中行人、车辆等关键物体的边界框，确保这些关键 token 不被错误剪除。

这种独立设计的好处是：**不破坏主 LLM 的执行图**，便于与 TensorRT、vLLM 等优化推理引擎集成。

### 3.3 Speculative Decoding Head：加速生成阶段

Speculative decoding 是 LLM 推理加速的常用技术：使用一个小的 draft model 快速生成多个候选 token，然后由大模型一次性验证，可能一次接受多个 token，从而减少 LLM 前向次数。

LiteVLM 基于 **Eagle-2** 方法实现投机解码：

- 使用 LLM 最后一层 hidden state 作为 draft model 的输入；
- draft model 仅包含单层 decoder，参数量极小；
- 生成的候选 token 通过 LLM 一次前向验证，利用 KV cache 加速。

在 DRIVE Thor 上，投机解码将 decode 阶段延迟从 16.9 ms 降至 7.7 ms。

---

## 四、实验结果

### 4.1 NVIDIA DRIVE Thor 平台实测

论文在 NVIDIA DRIVE Thor 上对一个 2B 参数的 VLM 进行了评测，任务为自动驾驶场景理解。下表展示了不同配置下的延迟与精度：

| Model | Avg. Image Patches | ViT Latency (ms) | Avg. Input Tokens | Prefill Latency (ms) | Extend-One Latency (ms) | Decode Latency (ms) | Total Latency (ms) | Speed-Up | Combined Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VLM 2B | 12 | 136.9 | 3214 | 163.2 | 16.9 | 229.6 | 529.7 | 1.0× | 0.6618 |
| + FastV (r=0.7) | 12 | 136.9 | 2292 | 117.9 | 16.4 | 222.8 | 477.6 | 1.1× | 0.6610 |
| + FastV (r=0.3) | 12 | 136.9 | 1063 | 59.5 | 15.9 | 216.1 | 412.5 | 1.3× | 0.6468 |
| + Eagle | 12 | 136.9 | 3214 | 163.2 | 9.0 | 121.4 | 421.5 | 1.3× | 0.6618 |
| **LiteVLM** | **3.5** | **45.1** | **858** | **54.2** | **7.7** | **104.1** | **213.6** | **2.5×** | **0.6602** |
| **LiteVLM (FP8)** | **3.5** | **29.1** | **948** | **39.8** | **6.2** | **84.0** | **163.1** | **3.2×** | **0.6450** |

关键观察：

- **LiteVLM 端到端延迟 213.6 ms**，相比基线 529.7 ms 降低 **2.5×**；
- 配合 FP8 后量化，延迟进一步降至 **163.1 ms**，加速比 **3.2×**；
- 任务精度（Combined Accuracy）仅从 0.6618 微降至 0.6602，几乎可以忽略；
- 单独使用 FastV 或 Eagle 只能获得 1.1–1.3× 加速，而 LiteVLM 通过**多阶段联合优化**实现了更显著的收益。

### 4.2 各模块的贡献拆解

从表中可以看出：

- **Patch Selection** 主要降低 ViT 延迟（136.9 → 45.1 ms）和 LLM prefill 延迟（输入 token 从 3214 降至 858）；
- **Token Selection** 进一步降低 prefill 成本；
- **Speculative Decoding** 主要降低 decode 延迟（16.9 → 7.7 ms）。

三者叠加，而非简单替代，是 LiteVLM 高效的关键。

---

## 五、与相关工作的关系

LiteVLM 的工作处于“高效 VLM / 边缘 VLM”这一更大的研究脉络中。以下是一些代表性方向：

### 5.1 轻量化 VLM 架构

- **MobileVLM / MobileVLM V2**（2023–2024）：针对移动设备设计，使用 Lightweight Downsample Projector（LDPv2）减少视觉 token，配合 MobileLLaMA 实现快速多模态处理。
- **Qwen-VL 1.5B/3B、InternVL 1B/2B**：开源小尺寸 VLM，在边缘设备上已具备一定可用性。
- **MiniCPM-V、TinyLLaVA、OmniVLM、Flash-VL 2B**：进一步探索轻量架构与训练优化，追求效率与性能平衡。

这些工作主要从**模型设计**角度降低计算量，而 LiteVLM 则从**推理流水线**角度进行系统级优化，两者可以互补。

### 5.2 视觉 token 压缩

- **FastV（ECCV 2024）**：基于 LLM 早期层的 attention 分数剪枝视觉 token，是一种 plug-and-play 方法。
- **LLaVA-PruMerge / PyramidDrop / FasterVLM**：结合 token 剪枝与合并，实现自适应压缩。
- **FastVLM（CVPR 2025，Apple）**：通过新的混合视觉编码器 FastViTHD 减少高分辨率图像的 token 数量。

LiteVLM 的 Token Selection Module 与这些工作思路相近，但特别强调了**独立模块设计**和**关键物体保护**，便于在工程化推理引擎中部署。

### 5.3 投机解码

- **Eagle / Eagle-2**：针对 LLM 的投机解码方法，利用上层 hidden state 生成 draft token。
- **Spec-VLA**：将投机解码引入 Vision-Language-Action 模型，用于机器人控制。
- **VisSpec、Lantern**：针对 VLM / 视频 LLM 的视觉感知投机解码。

LiteVLM 直接基于 Eagle-2 在 VLM 上实现了生成加速。

---

## 六、关键洞察与技术启示

### 6.1 系统级优化优于单一技巧

LiteVLM 最重要的启示是：**VLM 边缘部署需要端到端的系统优化**。单独做 token 剪枝或投机解码，收益有限；只有将输入过滤、prefill 压缩、decode 加速三个阶段联合起来，才能实现数量级的延迟降低。

### 6.2 “先选择，再计算”是一种高效范式

Patch Selection Module 体现了一个重要思想：**根据查询语义预先决定计算分配**。这与人类视觉注意机制类似——我们不会同时高分辨率处理整个视野，而是根据任务关注相关区域。在 VLM 中，这种“查询感知的稀疏计算”有望在未来进一步扩展到 patch 级别、token 级别乃至 layer 级别。

### 6.3 工程可部署性与算法创新同样重要

LiteVLM 的 Token Selection Module 被设计为独立模块，不修改主 LLM 权重，这是出于工程集成的考虑。在实际部署中，能否与 TensorRT、vLLM、TensorRT-LLM 等推理框架无缝对接，往往决定了算法能否真正落地。LiteVLM 在这方面的设计值得借鉴。

---

## 七、局限与未来方向

### 7.1 局限

1. **任务场景以自动驾驶为主**：论文实验集中在自动驾驶多视角感知，对于通用 VQA、机器人操作等任务的泛化性尚未充分验证。
2. **视角选择依赖查询语义**：对于需要全局理解的任务（如“描述周围所有物体”），过度筛选视角可能丢失信息。
3. **FP8 量化带来一定精度损失**：虽然 0.6450 vs 0.6602 的差距不大，但在安全关键场景中仍需权衡。
4. **未开源完整实现**：截至论文发布，代码和模型细节尚未完全公开，社区复现存在一定门槛。

### 7.2 未来方向

- **更细粒度的稀疏计算**：从视角级别扩展到 patch 级别乃至 token 级别的动态稀疏；
- **查询自适应的分辨率**：根据任务难度动态调整输入图像分辨率；
- **与 VLA（Vision-Language-Action）结合**：将 LiteVLM 的加速技术扩展到机器人控制、自动驾驶规划等动作输出任务；
- **硬件-算法协同设计**：针对 NPU、DSP 等边缘加速器定制 VLM 推理图；
- **长视频与多帧推理**：将 patch/token 选择扩展到时序维度，服务视频理解任务。

---

## 八、结语

LiteVLM 为 VLM 在嵌入式设备上的实时部署提供了一套切实可行的解决方案。它通过 **Patch Selection + Token Selection + Speculative Decoding** 的三阶段流水线，在 NVIDIA DRIVE Thor 上实现了 **2.5×–3.2×** 的端到端加速，同时几乎不损失任务精度。更重要的是，LiteVLM 展示了**系统级推理优化**在 VLM 边缘部署中的巨大潜力——未来的 VLM 不仅要“聪明”，还要“快”和“省”，才能真正走进机器人、汽车和各类智能设备。

---

## 参考资料

1. Huang, J., Jin, Y., An, L., & Park, J. *LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments*. arXiv:2506.07416, 2025. [https://arxiv.org/abs/2506.07416](https://arxiv.org/abs/2506.07416)
2. Chu, X., et al. *MobileVLM: A Fast, Reproducible and Strong Vision Language Assistant for Mobile Devices*. arXiv:2312.16886, 2023.
3. Chu, X., et al. *MobileVLM V2: Faster and Stronger Baseline for Vision Language Model*. arXiv:2402.03766, 2024.
4. Chen, L., et al. *FastV: An Image is Worth 1/2 Tokens After Layer 2*. ECCV 2024.
5. Vasu, P.K.A., et al. *FastVLM: Efficient Vision Encoding for Vision Language Models*. CVPR 2025.
6. Li, Y., Wei, F., Zhang, C., & Zhang, H. *Eagle-2: Faster Inference of Language Models with Dynamic Draft Trees*. arXiv:2406.16858, 2024.
7. Sharshar, A., et al. *Vision-Language Models for Edge Networks: A Comprehensive Survey*. IEEE Internet of Things Journal, 2025.
8. nanoVLM: [https://github.com/huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)

---

> 本博客以 LiteVLM 论文发布日期 2025-06-09 作为时间标记，归类为 VLM。
