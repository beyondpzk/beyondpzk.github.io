---
title: Jetson Orin 部署实战：算力、显存与模型切分的决策
date: 2026-07-04
categories: [Deploy]
---

# Jetson Orin 部署实战：算力、显存与模型切分的决策

NVIDIA Jetson Orin 是目前边缘 AI 和机器人领域最主流的嵌入式计算平台之一。从无人机、机械臂到自动驾驶原型车，都能看到它的身影。但在 Orin 上部署模型时，开发者经常会遇到几个问题：

- 这个模型能不能跑在 Orin 上？
- 该用 GPU 跑还是 CPU 跑？
- 模型太大放不进显存，是不是要切分？
- Orin Nano / NX / AGX 该怎么选？

这篇博客从 Orin 的硬件架构出发，结合部署中的实际决策场景，系统回答这些问题。

---

## 一、Jetson Orin 是什么？

Jetson Orin 是 NVIDIA 面向边缘 AI、机器人和自主机器推出的嵌入式 SoC（System on Chip）。它把 **ARM CPU、NVIDIA Ampere 架构 GPU、Tensor Core、DLA（深度学习加速器）、PVA（视觉加速器）** 集成在一个低功耗模块里，并共享统一的 LPDDR5 内存。

和桌面 GPU 最大的区别是：**Orin 没有独立的显存**。CPU 和 GPU 共享同一块 LPDDR5，也就是说所谓的「显存」就是系统内存。这简化了编程模型，但也意味着内存竞争更激烈。

> **注意**：这里说的是「内存共享」，不是「没有 CPU 和 GPU 之分」。Orin 里仍然有独立的 ARM CPU 核心、NVIDIA GPU CUDA 核心、Tensor Core 和 DLA 加速器。它们只是共用同一块物理内存，而不是像桌面 PC 那样 CPU 用 DDR、GPU 用 GDDR6。

---

## 二、Orin 产品线与规格对比

截至 2026 年，Jetson Orin 系列主要有三条线：Orin Nano、Orin NX、AGX Orin。

| 型号 | AI 算力（INT8） | GPU | CPU | 内存 | 内存带宽 | DLA | 典型功耗 |
|---|---|---|---|---|---|---|---|
| **Orin Nano 4GB** | ~34 TOPS | 512 CUDA / 16 TC | 6-core A78AE | 4GB | 51 GB/s | 无 | 5W–10W |
| **Orin Nano 8GB** | ~67 TOPS | 1024 CUDA / 32 TC | 6-core A78AE | 8GB | 102 GB/s | 无 | 7W–15W |
| **Orin NX 8GB** | ~117 TOPS | 1024 CUDA / 32 TC | 6-core A78AE | 8GB | 102 GB/s | 1x | 10W–25W |
| **Orin NX 16GB** | ~157 TOPS | 1024 CUDA / 32 TC | 8-core A78AE | 16GB | 102 GB/s | 1x | 10W–25W |
| **AGX Orin 32GB** | ~200 TOPS | 1792 CUDA / 56 TC | 8-core A78AE | 32GB | 205 GB/s | 2x | 15W–60W |
| **AGX Orin 64GB** | ~275 TOPS | 2048 CUDA / 56 TC | 12-core A78AE | 64GB | 205 GB/s | 2x | 15W–60W |

> 注：TOPS 为稀疏 INT8 峰值，实际可达吞吐受模型结构、内存带宽、batch size、量化方式影响，通常只能达到理论值的 30%–70%。

### 三个关键结论

1. **算力差异巨大**：AGX Orin 64GB 的峰值算力是 Orin Nano 4GB 的 8 倍以上。
2. **内存是瓶颈**：Orin NX 16GB 和 AGX Orin 32GB 的内存带宽都是 102–205 GB/s，远不及桌面 RTX 4090 的 1008 GB/s。大模型推理时，内存带宽往往比算力更先成为瓶颈。
3. **功耗可配置**：同一模块可以通过 `nvpmodel` 设置不同功耗墙，牺牲峰值性能换取续航和散热空间。

---

## 三、Orin 的软件栈

在 Orin 上跑模型，通常会用到以下软件栈：

```text
应用层
   ↓
推理框架 / 服务层
   - TensorRT、ONNX Runtime、PyTorch for Jetson
   - TensorRT-LLM、vLLM（部分支持 Jetson）
   - Triton Inference Server（AGX Orin 上更常见）
   ↓
加速库
   - CUDA、cuDNN、cuBLAS
   - TensorRT plugin、DLA runtime
   ↓
系统层
   - JetPack（Ubuntu + Linux kernel + BSP）
   - Jetson Linux Driver Package (L4T)
```

**JetPack** 是 Orin 的「操作系统 + SDK 合集」，包含 CUDA、TensorRT、cuDNN、OpenCV、Vulkan 等。部署前一定要先确认 JetPack 版本与模型的依赖兼容。

---

## 四、什么时候在 GPU 上跑？

Orin 的 GPU 基于 NVIDIA Ampere 架构，支持 FP32、FP16、INT8，第三代 Tensor Core 还支持结构化稀疏性（structured sparsity）。以下情况优先用 GPU：

### 1. 计算密集型的深度学习模型

- CNN（ResNet、YOLO、EfficientNet）
- Transformer（ViT、BERT、GPT、VLM）
- 大规模矩阵乘法、卷积、注意力计算

这些操作的计算密度高，GPU 的并行能力能发挥最大价值。

### 2. 需要低延迟、高吞吐的推理

比如机器人实时目标检测、自动驾驶感知网络。TensorRT 优化后的模型在 GPU 上通常能比 CPU 快 5–20 倍。

### 3. 可以使用 TensorRT 或 TensorRT-LLM

TensorRT 是 Orin 上最高效的推理引擎。它能做算子融合、FP16/INT8 量化、动态 shape、多 stream 等优化。如果模型能转成 TensorRT engine，优先跑在 GPU 上。

### 4. Batch 较大

GPU 的优势在 batch size > 1 时更明显。即使是小模型，如果一次要处理多路摄像头输入，GPU 也更合适。

---

## 五、什么时候在 CPU 上跑？

Orin 的 CPU 是 ARM Cortex-A78AE，性能不弱，但和 GPU 相比并行计算能力差距明显。以下情况适合 CPU：

### 1. 轻量级模型或控制逻辑

- 简单的规则判断、状态机；
- 小型传统 ML 模型（SVM、随机森林、轻量决策树）；
- 传感器数据预处理（滤波、坐标转换）。

### 2. GPU 不支持的算子

某些特殊算子（如某些 PyTorch 自定义 op、动态控制流、复杂索引）在 TensorRT 中没有直接实现。如果：

- 算子计算量不大；
- 改模型结构成本高；

可以把这个算子 fallback 到 CPU 执行，其他部分继续在 GPU 上跑。

### 3. 前处理与后处理

图像解码、Resize、归一化、NMS（非极大值抑制）、结果格式化等操作，很多时候在 CPU 上做更灵活，也能减少 CPU ↔ GPU 之间的数据传输。

### 4. 节省功耗

GPU 一启动就会显著增加功耗。如果任务对延迟不敏感，或者只需要偶尔推理一次，用 CPU 可以延长续航。

### 5. 模型实在太小

如果一个模型只有几 MB，且输入尺寸很小，CPU  inference 可能只需要几毫秒。此时 GPU 的启动和拷贝开销反而得不偿失。

---

## 六、CPU 与 GPU 协同：混合部署

实际项目中，最常见的不是「全 CPU」或「全 GPU」，而是**混合部署**：

```text
摄像头数据
   ↓
CPU：图像解码、Resize、归一化
   ↓
GPU：TensorRT 模型推理
   ↓
CPU：后处理 / NMS / 业务逻辑
   ↓
输出控制指令
```

关键点：

- 尽量减少 CPU ↔ GPU 之间的数据拷贝；
- 使用 Zero-Copy、pinned memory、CUDA 流等方式重叠计算与传输；
- 前处理尽量用 GPU（例如用 CUDA/OpenCV CUDA 模块做 resize）。

---

## 七、什么时候需要切分模型？

模型切分（model splitting）通常指把一个模型拆成多个部分执行，原因无外乎三种：**内存不够、算力不够、延迟要求**。

### 场景 1：单块 GPU 内存放不下

这是最常见的切分原因。Orin 的 GPU 和 CPU 共享内存，但 TensorRT / PyTorch 在 GPU 上运行时仍然需要把模型权重、激活值、KV cache 等分配在 GPU 可访问的内存中。

假设你要在 **Orin NX 16GB** 上部署一个 **7B 大语言模型**：

- FP16 权重：约 14 GB；
- KV cache + 激活：额外 2–6 GB；
- 系统、ROS、其他应用：占用 2–4 GB。

14 + 4 + 3 = 21 GB > 16 GB，直接跑会 OOM。

解决方案：

1. **量化**：FP16 → INT8/INT4/AWQ/GPTQ，权重降到 3.5–7 GB；
2. **切分到 CPU 执行**：把部分层的计算放到 ARM CPU 核心上执行，中间结果在共享内存中传递；
3. **多设备切分**：如果有多个 Orin，可以把模型按层或按 tensor 并行切到不同设备上。

### 场景 2：需要同时运行多个大模型

比如一个机器人要同时跑：

- 视觉感知模型（YOLOv8，~100 MB）
- VLM 多模态模型（~4 GB）
- LLM 决策模型（~7 GB）

单块 AGX Orin 64GB 也许能放下，但内存和算力都会很紧张。这时可以：

- 把不同模型分配到不同 Orin 模块（多节点）；
- 或者把一个大模型切出部分层到 CPU 执行，给其他模型让出 GPU 计算资源和可用内存。

### 场景 3：降低单次推理延迟

某些实时性要求极高的任务，可以把模型按 pipeline 拆成多个 stage：

```text
Stage 1（GPU）：特征提取
Stage 2（CPU/DLA）：轻量分类/决策
Stage 3（GPU）：生成输出
```

通过流水化，掩盖单个 stage 的延迟。

### 场景 4：利用 DLA 卸载

AGX Orin 和 Orin NX 带有 **NVDLA v2**（Deep Learning Accelerator）。DLA 是专门做 INT8 推理的硬件加速器，功耗比 GPU 更低。可以把模型中适合 DLA 的层切出来用 DLA 跑，不适合的层回退到 GPU。

TensorRT 支持自动做这种切分，只需在 build engine 时指定 DLA 设备。

---

## 八、DLA 与 GPU 的算子分工

DLA 不是通用 GPU，它只支持固定种类的算子。理解 DLA 和 GPU 各自擅长什么，是决定是否使用 DLA 的前提。

### DLA 支持的典型算子（基于 TensorRT DLA 文档）

| 算子类型 | DLA 支持情况 | 关键限制 |
|---|---|---|
| **Convolution / Fully Connected** | ✅ 支持 | 2D 卷积；kernel [1,32]；stride [1,8]；输入/输出通道 [1,8192]；支持分组卷积和空洞卷积（有限制） |
| **Deconvolution** | ✅ 支持 | 2D；padding 必须为 0；不支持分组/空洞；kernel 和 stride 有特定限制 |
| **Activation** | ✅ 支持 | ReLU、Sigmoid、TanH、Clipped ReLU、Leaky ReLU；仅 2D spatial 操作 |
| **Pooling** | ✅ 支持 | MAX / AVERAGE；window [1,8]；stride [1,16]；仅 2D |
| **ElementWise** | ✅ 部分支持 | Sum、Sub、Product、Max、Min、Div、Pow、Equal、Greater、Less；仅 2D；广播有限 |
| **Concatenation** | ✅ 支持 | 只能沿 channel 轴拼接；所有输入空间尺寸相同 |
| **Scale / BatchNorm** | ✅ 支持 | 2D；Uniform、Per-Channel、ElementWise 模式 |
| **LRN** | ✅ 支持 | window 3/5/7/9；仅 ACROSS_CHANNELS |
| **Softmax** | ✅ 支持 | 仅 Orin，非 Xavier；axis 维度 ≤ 1024（优化模式） |
| **Slice / Shuffle / Resize / Reduce / Unary** | ✅ 有限支持 | 4D 输入、静态参数、尺寸范围 [1,8192]；Resize 缩放因子有严格限制 |
| **动态 Shape** | ❌ 不支持 | min/opt/max 必须相同，运行时 shape 必须与构建时一致 |
| **注意力 / Transformer** | ❌ 不支持 | Attention、MatMul（非卷积形式）会 fallback 到 GPU |
| **RNN（LSTM/GRU）** | ❌ 不支持 | 回退到 GPU |
| **自定义 Plugin** | ❌ 不支持 | CUDA plugin 只能在 GPU 上跑 |
| **Gather / Scatter / TopK / NonZero** | ❌ 不支持 | 回退到 GPU |

### GPU 承担的算子

GPU 是通用加速器，TensorRT GPU backend 支持几乎所有算子，包括但不限于：

- 所有 DLA 不支持的算子（Attention、LSTM、GRU、Gather、TopK、动态 shape 等）；
- 3D 卷积、超大 kernel 卷积、任意分组/空洞卷积；
- 自定义 CUDA plugin；
- 需要动态 shape 的模型；
- Transformer、Diffusion、VLM、LLM 中的矩阵乘法、softmax、rope、layernorm 等。

### 典型模型的归宿

| 模型类型 | DLA 适合程度 | 说明 |
|---|---|---|
| **ResNet / MobileNet / VGG** | ⭐⭐⭐⭐⭐ | 经典 CNN，大部分层可在 DLA 上跑 |
| **YOLOv5/v8/v11（无 attention）** | ⭐⭐⭐⭐ | 骨干网大部分可跑 DLA，head 中部分 reshape/concat/gather 会 fallback |
| **YOLO 带 attention / Transformer backbone** | ⭐⭐⭐ | attention 部分会 fallback 到 GPU |
| **ViT / DeiT / Swin** | ⭐⭐ | Patch embedding 后全是 attention，DLA 利用率低 |
| **BERT / GPT / LLaMA / VLM** | ⭐ | 几乎全在 GPU 上跑，DLA 基本不参与 |

> **重要澄清**：GPU 支持的算子范围比 DLA 大得多。DLA 能跑的算子只是 GPU 的一个**子集**，而不是超集。所以不存在「这个算子不能跑 DLA，但跑 GPU 会更快」的反面情况——DLA 快不快取决于它能不能跑、以及模型结构是否适合它。
>
> 另外，**DLA 的主要优势是低功耗，不是绝对速度**。在单帧延迟上，DLA 通常比 GPU 慢 2–4 倍。只有在功耗受限、需要释放 GPU、或批量/并发很大的场景下，DLA 的综合收益才明显。

### 什么时候用 DLA，什么时候用 GPU？

**用 DLA 的场景**：

- 模型以经典 CNN 为主；
- 对功耗敏感（DLA 单核约 2–5W，GPU 推理 10–25W）；
- 需要同时跑多个模型，希望把 GPU 释放出来跑其他任务；
-  batch size 固定、输入 shape 固定；
- 主要用 INT8 或 FP16 精度。

**用 GPU 的场景**：

- 模型包含 Transformer、Attention、LSTM、自定义 plugin；
- 需要动态 shape；
- 追求最低的单次推理延迟；
- 模型中有大量 DLA 不支持的算子，fallback 比例高；
- 需要 FP32 精度或特殊数据类型。

### 一个实用的判断规则

用 `trtexec --useDLACore=0 --allowGPUFallback --buildOnly` 构建 engine，看日志里各层被分配到哪里：

- 如果 **70% 以上的计算量**（按 FLOPs 或层耗时）被分配到 DLA，DLA 通常能带来功耗和吞吐收益；
- 如果大量层频繁在 DLA 和 GPU 之间切换，整体 latency 可能反而变差，此时不如纯 GPU。

### 举个具体例子：ViT 的 Patch Embedding

ViT 前面的 `patch_embed` 本质上是一个 `Conv2d(kernel_size=16, stride=16)`，这个卷积本身 DLA 是支持的。但问题在于：

1. `patch_embed` 之后立刻要做 **reshape、permute、加 cls token、加 position embedding**——这些操作 DLA 都不支持；
2. 后面紧跟着的 **Transformer Block（Attention + MLP）** 也全在 GPU 上跑。

如果强行把 `patch_embed` 放到 DLA，输出必须从 DLA 搬回 GPU，而这个卷积的计算量通常只占整个 ViT 的 **5%–10%**。结果很可能是：

- DLA 跑 patch embed 省了一点功耗；
- 但 DLA→GPU 的数据同步和 kernel 切换开销把这点收益吃掉了；
- 整体 latency 反而比纯 GPU 更长。

**结论**：像 ViT 这种「前面一小段 CNN、后面全是 Transformer」的模型，`patch_embed` 也建议直接放 GPU，和后续 Attention 一起跑。DLA 更适合 ResNet、YOLO backbone 这种连续大段都是卷积的网络。

---

## 九、模型切分的三种方式

### 1. Pipeline Parallelism（流水线并行）

把模型按层切成若干段，每段放在不同设备上。前一段的输出作为下一段的输入。

```text
Layer 0-10   →  GPU
Layer 11-20  →  CPU / 另一块 Orin
Layer 21-30  →  GPU
```

优点：实现简单，适合层间依赖强的网络。
缺点：设备间通信开销大，吞吐量受限于最慢的一段。

### 2. Tensor Parallelism（张量并行）

把同一层的权重或激活拆成多份，分别放在多个设备上并行计算。常见于大模型多卡训练/推理。

```text
Attention Head 0-15  →  GPU 0
Attention Head 16-31 →  GPU 1
```

优点：可以处理超大单层，减少单设备内存压力。
缺点：需要设备之间有高速互联。Orin 模块之间通常通过 PCIe 或以太网连接，带宽有限，tensor parallel 的通信开销会很大。因此在 Orin 上较少使用纯 tensor parallel，除非是多卡 AGX 平台或外部 GPU 扩展。

### 3. CPU Offload（CPU 卸载）

这是最轻量的「切分」。当 GPU 可用内存不足时，把部分层的计算或 KV cache 放到 ARM CPU 核心上处理，需要时再交回 GPU。虽然 Orin 的 CPU 和 GPU 共享物理内存，但数据从 GPU 计算域换到 CPU 计算域仍然涉及缓存同步和映射开销。

```text
GPU：当前正在计算的层
CPU：暂时不用的层 / 历史 KV cache
```

优点：不需要额外硬件。
缺点：CPU ↔ GPU 搬运会带来延迟，适合对延迟不敏感或内存优先的场景。

在 LLM 推理中，**offload 常用于长上下文**，把较早的 KV cache 换出到由 CPU 管理的内存区域，甚至落到磁盘。

---

## 十、Orin 部署的决策流程

遇到一个模型时，可以按下面的流程判断怎么部署：

```text
1. 模型能不能转成 TensorRT / TensorRT-LLM？
   ├── 能 → 优先 GPU
   └── 不能 → 看下一步

2. 模型大小 + 激活 + 其他应用内存 < 可用内存？
   ├── 是 → GPU 直接跑
   └── 否 → 看下一步

3. 量化后能不能放下？
   ├── 能 → 量化 + GPU
   └── 否 → 看下一步

4. 模型是否可以切分？
   ├── 能 → Pipeline / CPU offload / 多设备
   └── 否 → 考虑换更小模型、升级硬件（AGX Orin 64GB）、或云端协同

5. 是否有 latency 不敏感的轻量任务？
   ├── 是 → 放到 CPU
   └── 否 → 尽量优化 GPU 部分，减少 CPU-GPU 拷贝
```

---

## 十一、实战案例

### 案例 1：YOLOv8 目标检测在 Orin NX 16GB

- 模型大小：~80 MB（FP32）
- 输入：640×640
- 决策：使用 TensorRT，GPU 推理，CPU 做 NMS
- 预期性能：INT8 量化后单帧延迟 5–15 ms，单 Orin NX 可跑 60 FPS 以上

不需要切分，因为模型很小，内存占用远小于可用容量。

### 案例 2：InternVL2.5-1B VLM 在 Orin NX 16GB

- FP16 权重：约 2 GB
- 视觉编码器 + LLM decoder + KV cache：总计 4–8 GB
- 决策：
  - 视觉编码器用 TensorRT / PyTorch CUDA 跑 GPU；
  - LLM 部分用 TensorRT-LLM 或 llama.cpp 量化到 INT4/INT8；
  - 前后处理放 CPU。

通常不需要切分到多设备，但需要严格控制 batch size 和上下文长度。

### 案例 3：LLaMA-2-7B 在 AGX Orin 64GB

- FP16 权重：约 14 GB
- KV cache（2k 上下文）：约 2–4 GB
- 决策：
  - INT4/INT8 量化，权重降到 3.5–7 GB；
  - 使用 TensorRT-LLM 或 llama.cpp；
  - 如果上下文很长，启用 KV cache offload。

AGX Orin 64GB 基本可以单卡运行 7B 模型，但若同时跑感知模型，可能需要把 LLM 部分层切到 CPU 或做更激进的量化。

### 案例 4：多模型并发机器人系统

系统同时运行：

- 3 路 YOLOv8 目标检测；
- 1 个语义分割模型；
- 1 个 VLM 用于指令理解；
- 1 个 LLM 用于决策。

单 AGX Orin 64GB 可能吃紧。可行方案：

- YOLO 和分割用 GPU + TensorRT；
- VLM 和 LLM 各自量化，必要时 LLM 部分层 offload 到 CPU；
- 如果预算允许，把 LLM 放到第二块 AGX Orin，通过 ROS/网络通信。

---

## 十二、常见坑与最佳实践

### 1. 内存不是无限的

即使 AGX Orin 64GB，也要留足系统和其他进程的内存。建议模型 + 激活 + KV cache 峰值占用不超过总内存的 70%。

### 2. 注意 JetPack 版本

不同 JetPack 的 CUDA、cuDNN、TensorRT 版本不同。一个 Engine 文件通常和 JetPack/GPU 架构强绑定，换版本可能需要重新构建。

### 3. 量化要验证精度

INT8/INT4 量化可能显著降低模型效果。部署前务必用真实数据做精度对比，必要时做 QAT（Quantization Aware Training）或 AWQ/GPTQ 等更精细的量化。

### 4. CPU-GPU 拷贝是大忌

每帧都反复把大 tensor 从 CPU 搬到 GPU 会严重影响性能。尽量让数据流待在 GPU 上，后处理也尽量用 CUDA 或 GPU 友好的库。

### 5. 功耗墙会限制性能

Orin 默认可能运行在低功耗模式。用 `nvpmodel` 和 `jetson_clocks` 查看和调整功耗配置：

```bash
# 查看当前模式
sudo nvpmodel -q

# 设置为最大性能模式（AGX Orin 60W）
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 6. DLA 不是万能的

DLA 只支持部分算子和 INT8。如果模型里有大量 DLA 不支持的层，TensorRT 会把它**回退到 GPU**（不是 CPU）执行，频繁的 GPU↔DLA 切换反而不如纯 GPU 跑得快。

---

## 十三、小结

Jetson Orin 是一个算力、功耗、体积平衡得很好的边缘 AI 平台，但它的资源依然有限。部署模型的核心不是「能不能跑」，而是「怎么跑得最划算」。

关键决策原则：

- **GPU**：计算密集型深度学习任务，能用 TensorRT 就用 TensorRT；
- **CPU**：轻量任务、前后处理、不支持 GPU 的算子、省电场景；
- **混合部署**：最常见，合理分配 CPU/GPU 任务，减少数据搬运；
- **量化**：内存/算力不够时的第一选择；
- **切分**：单设备内存/算力不足时的兜底方案，常用 pipeline parallel 或 CPU offload；
- **多设备**：多个大模型并发、超大模型、高可靠场景。

理解 Orin 的硬件边界，并根据模型大小、延迟要求、功耗预算做取舍，是把模型从实验室推向产品化的关键一步。

---

> **一句话总结**：在 Orin 上部署模型，GPU 是主力，CPU 是补充，量化是首选优化，切分是内存/算力不足时的兜底手段。
