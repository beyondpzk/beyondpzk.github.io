---
title: NVIDIA Orin NX 硬件原理与边缘 AI 芯片通识
date: 2026-09-11
categories: [Deploy]
---

# NVIDIA Orin NX 硬件原理与边缘 AI 芯片通识

> 一份给想系统理解"模型为什么在边缘设备上跑这么慢"的硬件入门文档。

## 一、Orin NX 是什么

Orin NX 是 NVIDIA 在 2022 年发布的一款面向**嵌入式与边缘端**的 AI 计算模块。它本质上是一台运行完整 Linux 系统（L4T/JetPack）的微型计算机，上面集成了一颗代号为 **GA10B** 的 GPU，基于 NVIDIA 的 **Ampere 架构**（与桌面端 RTX 30 系列同代）。

把它想象成一台没有显示器接口、没有风扇的迷你 GPU 工作站，大小约 70×45 mm 的电路板，功耗只有 10-25W。

**关键规格**（以 16GB 版本为准）：

| 参数 | 数值 | 解读 |
|---|---|---|
| GPU 架构 | Ampere（GA10B） | 第三代 Tensor Core |
| CUDA Core 数量 | 1024 | 通用并行计算单元 |
| Tensor Core 数量 | 32 | 专门加速矩阵乘法的硬件 |
| CPU | 6 核 ARM Cortex-A78AE | 不同于 x86 桌面 CPU |
| 内存 | 16 GB LPDDR5 | 统一内存，CPU/GPU 共享 |
| 内存带宽 | 102.4 GB/s | 数据在芯片和内存之间搬移的速度上限 |
| 最大功耗 | 10-25W | 被动散热，无风扇 |
| 操作系统 | L4T（Linux for Tegra） | Ubuntu 定制版 |
| **DLA** | **2 颗** | Deep Learning Accelerator，独立于 GPU 的 CNN 专用推理硬件，功耗极低（<1W/颗）。Transformer/LLM 不可用，但可并行跑 ViT 或目标检测，不占 GPU 资源 |

> **DLA（Deep Learning Accelerator，深度学习加速器——NVIDIA 对标 NPU 的硬件）**：GA10B SoC 内部**独立于 GPU** 的专用推理硬件，不占用 CUDA Core 或 Tensor Core。两颗 DLA，每颗功耗 <1W。专为 CNN 推理设计（支持卷积、池化、激活函数等），**不支持 Transformer 的自注意力机制**。理想 VLM 部署场景：DLA 跑 ViT/Vision 模块，GPU 专跑 LLM decode——两路并行，不抢占。当前两份 Orin NX 报告均未使用 DLA（全部 Vision 走 GPU TensorRT FP16），是因为 VLM 的 ViT 模块未经 DLA 量化适配。

## 二、统一内存架构——为什么 Orin NX 没有"显存"这个概念

这是理解 Orin NX 最重要的起点。

### 2.1 桌面 GPU 是"分居"的

一台装了 NVIDIA RTX 4090 的桌面电脑，硬件上是这样的：

```
CPU ← 内存总线 → 系统 RAM（32 GB DDR5）
                      │
GPU ← PCIe 4.0 →  显存 VRAM（24 GB GDDR6X）
```

数据和模型从系统 RAM 出发，走 PCIe 总线，到达显卡的独立 VRAM，GPU 再从 VRAM 里读写。这个过程有三段：**系统 RAM → PCIe → VRAM → GPU 缓存 → GPU 计算单元**。

> **PCIe** = **P**eripheral **C**omponent **I**nterconnect **E**xpress，外设组件互连高速总线。它是 CPU 和外部设备（显卡、SSD、网卡等）之间的**数据传输高速公路**。每一代 PCIe 带宽翻倍：PCIe 3.0 每条通道 ~1 GB/s，PCIe 4.0 每条 ~2 GB/s，PCIe 5.0 每条 ~4 GB/s。桌面 GPU 通常用 ×16 通道（16 条并行），所以 PCIe 4.0 ×16 的总带宽约 32 GB/s。Orin NX 之所以"省掉 PCIe"——是因为 CPU 和 GPU 焊在同一颗 SoC 里、共享同一根内存总线，不需要通过外部插槽通信。E300 官网提到 PCIe 5.0 ×14 Lane，但那是指 SoC 对外连接其他设备的 I/O 能力，不是 CPU 和 GPU 之间的通信方式（M1000 是 igpu，CPU 和 GPU 同样在 SoC 内部直连）。

> **RAM** = **R**andom **A**ccess **M**emory，随机存取存储器。和硬盘/SSD 不同，RAM 可以以任意顺序读写任意位置的数据，速度远超硬盘但断电即丢失。DDR5、LPDDR5、GDDR6X 都是 RAM 的不同类型——DDR（Double Data Rate，双倍数据速率）表示每个时钟周期传输两次数据。LPDDR 的 LP = Low Power（低功耗），专为手机和嵌入式设备优化。

> **HBM** = **H**igh **B**andwidth **M**emory，高带宽内存。和普通 DRAM 芯片平铺在电路板上不同，HBM 把多层 DRAM 芯片**垂直堆叠**在一起，通过穿过硅片的 TSV（Through-Silicon Via，硅通孔）连接，再通过一层硅中介层（Interposer）和 GPU 核心封装在同一个基板上。结果是惊人的带宽：HBM2e 约 1.5-2 TB/s，HBM3 约 3 TB/s，HBM3e 约 4.8 TB/s——是 LPDDR5（Orin NX：102.4 GB/s）的 15-50 倍，也是桌面 GDDR6X（RTX 4090：~1 TB/s）的 3-5 倍。代价是贵、不可扩展、只能焊死在芯片旁边。Orin NX 用的 LPDDR5 走的是**低成本低功耗**路线——不需要金字塔般的 3D 堆叠，带宽够用就行。E300（M1000）同样走 LPDDR5/LPDDR5X 路线。HBM 目前只在数据中心 GPU（NVIDIA A100/H100/B200）和高端自动驾驶芯片（如 NVIDIA Thor、高通 Snapdragon Ride Flex）上出现。

### 2.2 Orin NX 是"同居"的

```
CPU ──┐
      ├── 内存总线 ── LPDDR5（16 GB）── GPU 缓存 ── GPU 计算单元
GPU ──┘
```

CPU 和 GPU **挂在同一根内存总线上，共享同一块 LPDDR5 芯片**。好处是——没有 PCIe 传输，省掉了一段巨大的延迟和带宽瓶颈。代价是——这 16 GB 是两个人分的，谁也跑不掉。

### 2.3 "统一内存"不等于"同一块硅片"

上面说的"共享同一块 LPDDR5"容易让人以为 LPDDR5 和 GPU 都在一颗芯片里。实际上不是——它们是两颗物理上分开的元件：

```
┌────────── Orin NX 模块（70×45 mm 电路板）──────────┐
│                                                      │
│  ┌──────────────────┐    ┌──────────────────────┐    │
│  │  SoC 硅片         │    │  LPDDR5 内存颗粒 ×4   │    │
│  │  (GA10B)          │←──→│  (每颗 4 GB)         │    │
│  │                  │铜箔 │                      │    │
│  │  CPU + GPU +     │走线 │  速度上限：          │    │
│  │  内存控制器 +    │    │  102.4 GB/s          │    │
│  │  L2/L1 缓存      │    │                      │    │
│  └──────────────────┘    └──────────────────────┘    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- **SoC 硅片（GA10B）**：里面集成了 CPU、GPU、Tensor Core、内存控制器、L2 缓存。这就是日常说"Orin NX 的芯片"时指的那个东西，**指甲盖大小**。
- **LPDDR5 内存颗粒**：四颗独立的 DRAM 芯片，焊在 SoC 旁边。**它们不在 SoC 内部。**

> **102.4 GB/s 是 4 颗加在一起的总带宽，不是每颗。** Orin NX 的内存总线宽 128 bit，由 4 颗颗粒各出 32 bit 的通道并行拼成。LPDDR5 速率为 6400 MT/s，所以：128 bit ÷ 8 × 6400 MT/s = **102.4 GB/s（总和）**，摊到每颗只有 ~25.6 GB/s。焊 4 颗不只是为了容量（4×4 GB = 16 GB），更是为了**位宽**——少焊一颗，带宽直接掉 1/4。同理，AGX Orin 的 204.8 GB/s 也不是颗粒更快，而是总线翻倍到 256 bit（8 颗 × 32 bit），同款 6400 MT/s 颗粒，带宽 ×2。

所以"102.4 GB/s 的带宽上限"指的是**数据从 LPDDR5 颗粒通过电路板上的铜箔走线传到 SoC 内部的速度极限**，不是 SoC 内部的速度。SoC 里面的 L2 缓存、L1 缓存的速度是 TB/s 级别的，但没有任何数据能绕过 LPDDR5→SoC 这段物理连接——它就是瓶颈。

这和你手机主板上的 DRAM 内存芯片与处理器不在同一颗硅片里是一样的道理。Orin NX 只是把 SoC + LPDDR5 + 电源管理全部焊在同一块小电路板上，**省掉了插槽和长走线，但硅片和内存芯片物理上仍然分开**。

### 2.4 对部署的意义

在桌面电脑上，你可以说"模型权重 8 GB，系统还剩 24 GB 内存，够用"。在 Orin NX 上，这 8 GB 权重就是从 16 GB 池子里直接扣除的。操作系统、摄像头驱动、ROS、Docker 容器，全都要从这同一块 LPDDR5 里分——没有任何额外的显存可以加。

**这就是为什么所有分析始终围绕 RSS（操作系统统计的进程物理内存）展开——它就是这个池子里的真实消费。**

## 三、内存的层级——LPDDR5 不是"内存"，是"车库"

即使统一内存省掉了 PCIe，但**数据还是要从 LPDDR5 搬进 GPU 的片上缓存，Tensor Core 才能用**。

### 3.1 为什么要有缓存

LPDDR5 的 102.4 GB/s 听起来很快，但 Tensor Core 的计算吞吐是这个数字的几百倍。如果 Tensor Core 直接等 LPDDR5 喂数据，99% 的时间都浪费在等数据到达上。

解决方法是**缓存层级**——在 LPDDR5 和计算单元之间插入几层越来越快也越来越小的存储：

```
LPDDR5（16 GB, 102.4 GB/s）
        ↕
GPU L2 Cache（~256 KB, ~500 GB/s）
        ↕
L1 Cache / Shared Memory（每 SM 192 KB, ~1 TB/s）
        ↕
寄存器（每 SM 65536×32bit, 寄存器级速度）
        ↕
Tensor Core / CUDA Core（计算单元）
```

> **SM** = **S**treaming **M**ultiprocessor，流式多处理器。NVIDIA GPU 由多个 SM 组成——可以把每个 SM 理解为一个独立的"迷你 GPU"，有自己的 CUDA Core、Tensor Core、L1 缓存、共享内存和寄存器文件。Orin NX 的 GA10B 有 4 个 SM，每个 SM 包含 256 个 CUDA Core 和 8 个 Tensor Core（总计 1024 CUDA + 32 TC）。
>
> **"寄存器 每 SM 65536×32bit"** 的意思是每个 SM 拥有 65536 个 32 位寄存器（合计 256 KB）。在执行一条指令时，操作数必须在寄存器里——这是 GPU 内部最快的一层存储，延迟 <1 个时钟周期，作为对比，L1 缓存要 ~30 个周期，LPDDR5 要几百个周期。所以编译器会尽可能把频繁使用的变量放在寄存器里，寄存器不够时才溢出到 L1 或更慢的层级。

Tensor Core 只和寄存器及 L1 打交道。每次计算前，需要的数据必须提前**从 LPDDR5 → L2 → L1 → 寄存器**搬过去。这个搬运动作用时，由 LPDDR5 的 102.4 GB/s 决定。

### 3.2 为什么 decode 每个 token 都要"重新搬一遍"

自回归生成的 decode 阶段，每生成一个 token，整个 LLM 的前向传播都要执行一次。每一层 Transformer 的权重（Q、K、V、FFN 等矩阵）都需要从 LPDDR5 加载到缓存里参与计算。

虽然有 KV-cache（见第六节）省了重复计算注意力的开销，但**"把每层权重从 LPDDR5 搬进缓存"这件事无论如何都省不掉**。4B 模型 8 GB 权重，在 102.4 GB/s 的带宽下，纯搬运就需要：

```
8 GB ÷ 102.4 GB/s ≈ 78 ms
```

这就是 decode 每 token **92 ms** 的由来——78 ms 是硬地板，剩下 14 ms 是实际计算和 KV-cache 读写时间。不是算得慢，是搬得慢。

### 3.3 prefill 为什么相对快

prefill 阶段要处理 210 个输入 token（448 分辨率下的 visual tokens + prompt tokens）。同样需要把 8 GB 权重搬一遍，但这一次处理直接处理了 210 个 token——**摊到每个 token 上，搬运成本只有 ~0.37 ms**。

这就解释了首 token（126 ms）和后续 token（92 ms × 63 ≈ 5.8 秒）之间的巨大剪刀差。

## 四、Tensor Core——为什么有些精度比其他精度快

### 4.1 Tensor Core 是什么

CUDA Core 是"全能选手"——什么运算都能做，张量、标量、逻辑运算都行。

Tensor Core 是"专攻一项的疯子"——它只做一件事：**4×4 矩阵乘法累加**，但做得极快。一个 Tensor Core 在一个时钟周期里能完成一个 `D = A×B + C` 的小矩阵运算。

大语言模型的注意力计算和 FFN 前向传播本质上就是一大堆矩阵乘法堆起来的，所以 Tensor Core 是 LLM 推理的核心加速硬件。

> **4×4 是单次运算粒度，不是矩阵大小上限。** 实际推理中的大矩阵乘法（比如 4096×4096）是通过"切块"实现的：把大矩阵切成无数个 4×4 子块，32 个 Tensor Core 每个时钟周期各吃一块并行计算，最后拼回完整结果。这个切块和调度对开发者透明——你写 `torch.matmul()`，底层自动完成。

**CUDA Core 和 Tensor Core 在同一颗硅片里，不是分开的芯片。** 它们都在同一个 SM（Streaming Multiprocessor）内部：

```
┌─────────── 一个 SM ───────────┐
│                               │
│  CUDA Core ×4×16 = 64 个    ←── 通用计算
│                               │
│  Tensor Core ×8             ←── 矩阵乘法专精
│                               │
│  L1 Cache / Shared Memory    │
│  寄存器文件（65536×32bit）   │
│                               │
└───────────────────────────────┘
```

Orin NX 的 GA10B 有 4 个这样的 SM，所以总共 1024 CUDA Core + 32 Tensor Core，全在同一颗硅片里。区别不在"在哪里"，而在**做什么**：

| | CUDA Core | Tensor Core |
|---|---|---|
| 出现时间 | 2006（G80 起所有 NVIDIA GPU 都有） | 2017（Volta 起才有） |
| 功能 | 通用：整数、浮点、逻辑、分支 | 只做矩阵乘加 `D = A×B + C` |
| 计算粒度 | 标量：一次算一个数 | 矩阵块：一次算一小块矩阵 |
| LLM 推理中担任 | element-wise 操作、激活函数、归一化等杂活 | 注意力矩阵乘法、FFN 投影，占 >90% 计算量 |
| 类比 | 瑞士军刀 | 液压机 |

#### 4.1.1 CUDA Core：一个"标量乘加"单元

一个 CUDA Core 本质上就是一个 **ALU（算术逻辑单元）**，核心电路是一个 FMA（Fused Multiply-Add，乘加融合）单元：每个时钟周期完成一次 `a × b + c`，其中 a、b、c 都是**标量**（单个数）。

Orin NX 有 1024 个 CUDA Core，意味着理想情况下每个时钟周期全芯片能并行完成 1024 次标量乘加。它能做所有类型的运算——整数、浮点、比较、分支跳转——所以叫"通用"。

CUDA Core 不是单独干活的，GPU 以 **warp（线程束，32 个线程）** 为单位调度：一条指令同时发给 32 个 CUDA Core，每个 Core 处理一个线程的数据。这就是 GPU"单指令多数据"（SIMT）的工作方式——你写 `c[i] = a[i] + b[i]`，1024 个元素分给 32 个 warp，每个 warp 里 32 个 CUDA Core 一人加一个元素。

#### 4.1.2 Tensor Core：把矩阵乘法"焊死在电路上"

矩阵乘法有个特点：**每个数据会被复用很多次**。计算 `C = A × B` 时，A 的每一行要和 B 的每一列逐个相乘再求和——如果用 CUDA Core 算，这些数据要在寄存器和 ALU 之间来来回回搬运，大量时间花在"取数"而不是"算数"上。

Tensor Core 的思路是把整个 4×4 矩阵乘法的**数据通路直接做成硬件电路**：矩阵块一次性流入，内部的乘法器阵列和加法树在一个时钟周期内直接产出结果，中间不需要软件参与调度。这就是为什么同样一个时钟周期，Tensor Core 的等效乘加次数是 CUDA Core 的几十倍。

以 Orin NX 粗略估算 FP16 矩阵乘法的理论吞吐差距：

```
CUDA Core 路线：1024 个 FMA/周期 × 2 次运算 ≈ 2,048 FLOP/周期
Tensor Core 路线：等效 ~8,000+ FLOP/周期（FP16）
→ 矩阵乘法走 Tensor Core 大约快 4-8 倍
```

这也是为什么"模型用了 FP16"不等于"变快了"——关键是 FP16 让矩阵乘法**有资格走 Tensor Core**，如果算子没被映射到 Tensor Core 上，FP16 和 FP32 一样慢。

#### 4.1.3 一层 Transformer 里，谁干什么

以前向传播中的一层 Transformer 为例，两类核心的分工是：

| 算子 | 执行者 | 原因 |
|---|---|---|
| Q/K/V 投影（大矩阵乘） | Tensor Core | 稠密矩阵乘法 |
| 注意力分数 Q×K^T、×V | Tensor Core | 矩阵乘法 |
| Softmax | CUDA Core | 逐元素求 max、exp、除法，不是矩阵乘 |
| RoPE 位置编码 | CUDA Core | 逐元素旋转（sin/cos） |
| RMSNorm / LayerNorm | CUDA Core | 逐元素归一化 |
| SiLU / GELU 激活 | CUDA Core | 逐元素非线性函数 |
| 残差连接（相加） | CUDA Core | 逐元素加法 |
| FFN 的 up/gate/down 投影 | Tensor Core | 三个大矩阵乘法 |
| KV-cache 读写 | 内存系统（CUDA Core 协助） | 纯数据搬运，几乎不算 |

规律很清晰：**凡是"矩阵 × 矩阵"或"矩阵 × 向量"的重活都给 Tensor Core；凡是"逐元素"的轻活都给 CUDA Core**。

#### 4.1.4 为什么 CUDA Core 的"杂活"也不能忽视

虽然 >90% 的计算量在 Tensor Core 上，但剩下的杂活如果效率低，照样拖慢整体——这就是阿姆达尔定律。实际工程中有两个对应手段：

- **Kernel 融合（Fusion）**：把"矩阵乘 + 后面的逐元素操作"合并成一个 kernel。典型例子是 FlashAttention——把 Q×K^T、softmax、×V 三步融为一个 kernel，softmax 部分由 CUDA Core 在数据还在寄存器/共享内存里时就地算掉，避免中间结果写回 LPDDR5 再读回来。省的不是计算，是**搬运**。
- **算子覆盖率**：TensorRT 这类推理引擎的核心工作之一，就是分析计算图，把能走 Tensor Core 的算子尽量映射上去，并把相邻的 CUDA Core 杂活融合进同一个 kernel。第三方 NPU 工具链"算子覆盖不全"的意思就是：某些算子它不认识，只能回退到慢路径甚至 CPU 上跑。

#### 4.1.5 对开发者来说，这个分工是透明的

你几乎不需要（也很难）直接指挥 Tensor Core：

- 写 `torch.matmul()` / `nn.Linear` → 底层调用 cuBLAS → 自动走 Tensor Core 的 `mma.sync` 指令
- 写普通的 CUDA kernel（比如自定义激活函数）→ 编译成 CUDA Core 的标量指令
- 用 TensorRT 编译模型 → 引擎自动决定每一层走哪个核心、做什么融合

所以日常优化 LLM 推理的思路不是"怎么用 Tensor Core"，而是反过来：**消除那些让数据离开 Tensor Core 的环节**——减少精度来回转换、融合零散算子、避免小算子把数据写回内存。

### 4.2 Ampere Tensor Core 的精度支持

Ampere 架构（Orin NX 的 GPU）上的 **第三代 Tensor Core** 对以下精度格式有原生加速：

| 精度 | 每元素大小 | Tensor Core 相对吞吐 | 权重内存占用（每 1B 参数） |
|---|---|---|---|
| FP32（标准 float） | 4 字节 | 1×（基准） | ~3.73 GB |
| TF32（Tensor Float） | 19 bit 内部 | ~8× | ~3.73 GB（存储不变） |
| FP16（半精度） | 2 字节 | ~2×（相对 FP32） | ~1.86 GB |
| BF16（Brain Float） | 2 字节 | ~2×（同 FP16） | ~1.86 GB |
| INT8（8-bit 整型） | 1 字节 | ~4×（相对 FP32） | ~0.93 GB |
| INT4（4-bit 整型） | 0.5 字节 | ~8×（相对 FP32） | ~0.47 GB |
| FP64（双精度） | 8 字节 | ~1/64× | ~7.45 GB |

> **"每元素大小"是怎么来的：** 一个 FP32 元素 = 32 位（1 符号 + 8 指数 + 23 尾数），而 1 字节 = 8 位，所以 32 ÷ 8 = **4 字节**。同理：FP16/BF16 = 16 位 = **2 字节**，INT8 = 8 位 = **1 字节**，INT4 = 4 位 = **0.5 字节**。这就是精度表中"每元素大小"列的数字来源。

几个关键结论：

- **FP16 和 BF16 速率完全相同**。差异在数值表示上（见第五节），不在速度上。
- **INT8 不仅算得快（吞吐翻倍），内存也减半**。对 Orin NX 这种内存带宽受限的平台，后者比前者更关键。
- **从 FP16 到 INT8，收益不是线性的**：权重从 8 GB → 4 GB（搬运时间从 78 ms → 39 ms），计算从 FP16→INT8（Tensor Core 更快），两端同时获益。

## 五、FP16 vs BF16——为什么两个"看起来差不多"的格式在报告里同时出现

### 5.1 两者的位布局

```
FP32： [s] [eeeeeeee] [mmmmmmmmmmmmmmmmmmmmmmm]   ← 1 符号 + 8 指数 + 23 尾数
FP16： [s] [eeeee] [mmmmmmmmmm]                    ← 1 + 5 + 10
BF16： [s] [eeeeeeee] [mmmmmmm]                    ← 1 + 8 + 7
```

**FP16**：指数位少（只有 5 位），尾数位多（10 位）。意味着它能表示的数值范围小（max ≈ 65504），在小数值范围内精度高。

**BF16**：指数位和 FP32 一样多（8 位），尾数位少（7 位）。意味着它能表示和 FP32 一样大的数（max ≈ 3e38），但每一步的精确度更低。

### 5.2 在推理中哪个更好

- **训练阶段**：BF16 优势大——动态范围宽 = 不需要 loss scaling，梯度不会溢出成 NaN。
- **推理阶段**：两者差距极小，质量差异通常 <0.1%。选择取决于框架集成偏好，而非精度。

### 5.3 本报告中的实际选择

| 路线 | 精度 | 原因 |
|---|---|---|
| InternVL → TensorRT | FP16 | TensorRT 生态原生偏好 |
| Qwen3-VL → vLLM Jetson | BF16 | vLLM/PyTorch 生态默认 autocast 到 BF16 |

两者在硬件速率上完全平权，不是"谁更好"的问题。报告里的精度选择是框架路径决定的，不是硬件限制决定的。

## 六、KV-cache——为什么 decoder 不必从头算

### 6.1 没有 KV-cache 的世界

自回归生成的过程是每次生成一个 token，然后把它追加到输入序列末尾，重新跑一遍模型：

```
Step 1: 输入 [token_1, token_2, ..., token_210] → 输出 token_211
Step 2: 输入 [token_1, ..., token_210, token_211] → 输出 token_212
Step 3: 输入 [..., token_211, token_212] → 输出 token_213
...
```

每步都要对**整个历史**重新计算注意力（Q×K^T），计算量 O(n²) 增长。

### 6.2 有 KV-cache 的世界

Transformer 的注意力机制中，**Key 和 Value 矩阵只依赖于当前 token 之前的历史，不受未来 token 影响**。所以前一步算出来的 K 和 V 可以直接存起来，下一步只算新 token 的 Q，和缓存的 K 做点乘就行：

```
Step 1: 算 K[1:210], V[1:210]，全量计算
        用 Q_211 × K[1:210] 得到注意力权重
        缓存：K[1:210], V[1:210]

Step 2: 只算新 token 的 K_211, V_211
        用 Q_212 × (缓存 K[1:210] + K_211) 得到注意力权重
        追加缓存：K[1:211], V[1:211]

...
```

每步的计算量从 O(n²) 变成 O(n)——只需算当前 token 的一行。

### 6.3 KV-cache 到底省了什么，没省什么

**省了**：注意力计算的重复工作。不需要每次重新算全序列的 Q、K、V。

**没省**：
- 每层 FFN（前馈网络）的权重加载——这占了 decode 时间的大头
- 注意力时 Q、K、V 投影矩阵的加载
- 层归一化、激活函数的执行

**这就是 decode 仍然要 92 ms 的原因**——KV-cache 让注意力变快了，但 Transformer 里还有大把别的矩阵乘法（FFN 的 up/gate/down 投影等），每步都要把对应的权重完整加载一遍。

## 七、RSS——为什么 `ps aux` 看到的数字不完整

### 7.1 RSS 是什么

RSS = **Resident Set Size**，Linux 内核统计的一个进程当前占用了**多少物理内存页**的指标。它只数那些已经映射到进程 CPU 虚拟地址空间的页面。

### 7.2 RSS 在 Jetson 上的盲区

在标准 x86 服务器上，PyTorch 的 CUDA 内存统一走 CPU 端分配 + 映射的路径，`cudaMalloc` 返回的 GPU 内存通常会创建一个 CPU 可见的地址映射——所以 RSS 能看到。

在 Jetson 平台上，`cudaMalloc` 底层走的是 **nvmap**（NVIDIA 的统一内存管理子系统），GPU 内存通过 IOMMU 直接映射，不一定经过进程的 CPU 页表。这意味着：

| 分配方式 | CPU 可寻址 | 计入 RSS |
|---|---|---|
| `malloc` / `mmap` | ✅ | ✅ |
| PyTorch CUDA allocator（Jetson 路径） | ✅（PyTorch 会创建映射） | ✅ 大部分计入 |
| vLLM 直接 `cudaMalloc` | ❌（走 nvmap/IOMMU） | ❌ 部分或全部遗漏 |

### 7.3 对本报告的影响

- InternVL mixed KV 路线（PyTorch 管 LLM 内存）→ RSS ≈ 实际占用 ✅
- Qwen3-VL vLLM 路线（vLLM 自己走 cudaMalloc）→ RSS 严重低估 ❌

**Qwen3-VL-4B 的 RSS 显示 2.9 GB，但 vLLM 自己记录的 model loading 是 8.57 GiB。** 5.7 GB 的差距不是"没用"，是 RSS 没抓到。

正确的做法是：TRT-LLM 路线信任 RSS，vLLM 路线用 vLLM 自己的记录或 `tegrastats` 交叉验证。

## 八、解码阶段为什么比首 token 慢这么多——一个数字推演

以 **Qwen3-VL-4B、448 分辨率、vLLM FP16 compiled** 为例：

### 首 token（TTFT = 126 ms）

```
Vision encoder（ViT）：处理 448×448 图片 ~80-90 ms
LLM prefill：210 个 prompt token × 8 GB 权重矩阵加载 ≈ 78 ms
            （prefill 阶段 210 个 token 并行，权重搬一次全用完）
TTFT ≈ 90 + 36 ≈ 126 ms
```

### 后续每个 token（decode 92 ms/token × 63 ≈ 5,809 ms）

```
第 1 步：搬 8 GB 权重 → 算 1 个 token → 写 KV-cache ≈ 92 ms
第 2 步：搬 8 GB 权重 → 算 1 个 token → 写 KV-cache ≈ 92 ms
...
第 63 步：搬 8 GB 权重 → 算 1 个 token → 写 KV-cache ≈ 92 ms
──────────────────────────────────────────────
合计 ≈ 5,809 ms
```

**首 token 和 decode 的比例 = (1 次搬运 ÷ 210 个 token) vs (63 次搬运 ÷ 63 个 token) = 1:210**。这就是为什么它们能差 40 倍以上。

## 九、与 Orin NX 类似的边缘 AI 芯片

### 9.1 同家族的 NVIDIA Jetson 产品线

| 型号 | GPU（CUDA/Tensor Core） | 内存 | 内存带宽 | 功耗 | 定位 |
|---|---|---|---|---|---|
| **Jetson Orin Nano** | Ampere, 512/1024 CUDA + 16/32 TC | 4/8 GB LPDDR5 | 68 GB/s | 5-15W | 入门级，适合单小模型 |
| **Jetson Orin NX** | Ampere, 1024 CUDA + 32 TC | 8/16 GB LPDDR5 | 102.4 GB/s | 10-25W | 中端，本报告测试平台 |
| **Jetson AGX Orin** | Ampere, 2048 CUDA + 64 TC | 32/64 GB LPDDR5 | 204.8 GB/s | 15-60W | 旗舰，可同时跑多个大模型 |
| **Jetson AGX Xavier** | Volta, 512 CUDA + 64 TC | 32 GB LPDDR4x | 136.5 GB/s | 10-30W | 上一代旗舰 |

关键区别是**内存带宽**——它直接决定了 decode tok/s：

| 型号 | 带宽 | 4B FP16 decode 理论下限 |
|---|---|---|
| Orin Nano | 68 GB/s | ~118 ms/token |
| Orin NX | 102.4 GB/s | ~78 ms/token |
| AGX Orin | 204.8 GB/s | ~39 ms/token |

**Orin NX 的 decode 瓶颈在带宽，不在算力**。AGX Orin 虽然 CUDA Core 只翻倍，但因为带宽翻倍，decode 速度几乎线性翻倍。

### 9.2 非 NVIDIA 的平行方案

| 芯片 | 架构特点 | 内存 | 对比 Orin NX |
|---|---|---|---|
| **Qualcomm QCS8550**（骁龙 8 Gen 3 的嵌入式版） | Hexagon NPU + Adreno GPU | 12-16 GB LPDDR5x | NPU 走 INT8/INT4 量化为主，生态偏 ONNX 和 Qualcomm AI Engine。内存带宽相近，但 GPU 通用算力弱于 Orin NX。适合已量化的专用模型，不适合灵活 PyTorch/TensorRT 原型 |
| **Intel Meteor Lake / Core Ultra** | CPU + GPU + NPU 三合一 | 16-32 GB LPDDR5x | x86 生态，NPU 做低功耗 AI 推理，GPU（Arc 架构）做中等负载。适合 PC 端本地推理，但对嵌入式机器人（功耗/体积/振动）不友好 |
| **Hailo-8 / Hailo-10** | 专用 NPU 加速器 | 外挂 LPDDR4（芯片无集成内存） | 纯推理加速器，功耗极低（~2.5W）。但必须走 Hailo 自己的量化工具链，不支持 PyTorch 原生模型直接部署，灵活性较差 |
| **地平线征程 6 (J6)** | 自研 BPU（Bernoulli 架构） | 外挂 LPDDR4x/5 | 国内自动驾驶主流方案，INT8 推理效率极高（~560 TOPS）。专为车载视觉+BEV 优化，不适合通用 LLM/VLM 部署 |
| **RK3588**（Rockchip） | ARM Mali GPU + 自研 NPU（6 TOPS） | 4-16 GB LPDDR4x | 极低功耗、极低价位，适合轻量 CNN（如 YOLO、人脸检测）。LLM 几乎不可行——6 TOPS NPU 远不够跑 Transformer decoder |

### 9.3 关键规律

所有边缘 AI 芯片的瓶颈都在**内存带宽**，不是 TOPS。一颗标称 200 TOPS 的芯片，如果只有 50 GB/s 的内存带宽跑 LLM decode，实际计算利用率可能不到 5%——Tensor Core 大部分时间在等权重搬进来。

这也是为什么 NVIDIA 在 Jetson 产品线上坚持用自家 GPU 架构：**TensorRT + CUDA 生态让带宽利用率能达到 80-90%**，而第三方 NPU 往往因为工具链限制和算子覆盖不全，实际利用率远低于标称。

## 十、W8 INT8 量化的真正价值——不是"算得快"，是"搬得少"

结合以上硬件背景，W8 量化的收益就能清晰理解了：

| | FP16 mixed KV | W8 INT8 |
|---|---|---|
| 权重内存 | 8 GB（4B）/ 1.86 GB（1B） | 4 GB（4B）/ 0.93 GB（1B） |
| 每 decode step 搬运时间 | 78 ms / 18 ms | 39 ms / 9 ms |
| decode tok/s（4B） | ~10.9 | 理论上限 ~25 |
| RSS（1B 实测） | 5.8 GB | **1.78 GB** |
| E2E（1B 实测） | 1,442 ms | **584 ms** |

**W8 让 1B 模型内存减半、decode 搬运时间减半，E2E 快了 2.5 倍。** 这个收益在桌面 GPU 上（带宽充裕）不明显，在 Orin NX 上（带宽紧张）是决定性的。

## 十一、如果要在 16 GB 的 Orin NX 上稳定部署，应该怎么配

```
预算分配（16 GB 总内存）：

L4T 系统 + JetPack：           ~2.0 GB
Docker / 摄像头驱动 / ROS 2：  ~2.0 GB
──────────────────────────────────
推理可用：                     ~12.0 GB

场景 A：快速动作 + 场景描述（双模型）
  SmolVLM2-500M TRT：          ~2.3 GB
  InternVL2.5-1B W8：           ~1.8 GB
  KV-cache 余量 + buffer：      ~2.0 GB
  ────────────────────────────────
  已用：                        ~8.1 GB ✅ 有余量

场景 B：高质量长描述（单模型）
  InternVL3-1B mixed KV：       ~5.9 GB
  KV-cache + buffer：           ~2.0 GB
  ────────────────────────────────
  已用：                       ~9.9 GB ⚠️ 紧张但可行

场景 C：4B 模型（风险）
  Qwen3-VL-4B vLLM：            实际 ~8.6 GB（加载峰值 ~11.5 GB）
  ────────────────────────────────
  ❌ 加载阶段已超过 12 GB 可用预算，存在 OOM 风险
```

**选型结论**：Orin NX 16GB 上的安全策略是 **1B 模型 + W8 量化**。4B 模型在加载阶段已接近上限，生产环境不建议长期依赖。

## 十二、一张图片从相机曝光到 VLM 输出文字：完整链路

把前面所有硬件知识串起来，看一条真实的数据链路：Orin NX 接了一个相机，一张图片从曝光到变成一段文字，中间发生了什么。

```
相机传感器曝光 → MIPI CSI-2 传输 → ISP 硬件处理 → 统一内存帧 buffer
→ GPU 预处理（resize/归一化）→ ViT 视觉编码 → 多模态投影
→ LLM prefill → decode 逐 token 生成 → detokenize → 文字
```

### 12.1 曝光与采集（相机传感器端）

光子打到 CMOS 传感器上，曝光时间和增益由自动曝光算法控制。传感器输出的不是彩色图，而是 **RAW 图（Bayer 格式，10/12 bit）**——每个像素只记录一个颜色通道。RAW 数据通过 **MIPI CSI-2** 排线（每 lane 数 Gbps）送进 Orin NX SoC，由 V4L2 / NVIDIA Argus 相机驱动接管。

### 12.2 ISP 硬件处理（SoC 内，不占 GPU）

Orin NX 的 SoC 里有专用 **ISP（Image Signal Processor，图像信号处理器）**，硬件电路直接完成：

- **Demosaic**：Bayer 插值，把单通道 RAW 还原成每个像素都有 RGB 的彩色图
- 去噪、自动白平衡、色调映射（tone mapping）

输出 NV12/YUV420 或 RGB 帧，直接写进**统一内存**的 buffer。ISP 和 DLA 一样，是 SoC 里独立于 GPU 的专用硬件，这些处理不消耗任何 CUDA Core / Tensor Core 资源。

### 12.3 CPU 拿到帧（零拷贝）

应用程序（GStreamer / ROS 2 节点）拿到帧。统一内存架构的红利在这里体现：ISP 写入的 buffer 就是 GPU 可以直接读取的内存——**没有 PCIe、没有 memcpy，相机帧零拷贝进入 GPU 视野**。GStreamer 里这套机制叫 NVMM buffer。对比桌面平台：相机帧先进系统 RAM，要过 PCIe 拷进显存才能给 GPU 用，平白多出一段几毫秒、还占 CPU 的搬运。

### 12.4 预处理（CUDA Core，~1-2 ms）

ViT 要求 448×448 的固定输入，所以要在 GPU 上跑几个 CUDA kernel：

- **resize**：双线性插值到 448×448
- **色彩空间转换**：BGR → RGB
- **归一化**：`x / 255`，再减均值、除方差
- **排布转换**：HWC → CHW（模型要的内存布局）

全是逐元素操作——正是 4.1.3 节说的 CUDA Core 杂活，总共不到 1-2 ms。

### 12.5 ViT 视觉编码（Tensor Core，~80-90 ms）

图片被切成 14×14 像素的 patch（448÷14 = 32×32 = 1024 个 patch；很多 VLM 再做一次 pixel shuffle，压缩到 ~256 个 visual token），经过 patch embedding + 多层 Transformer，输出 visual tokens。这一段的矩阵乘法全部走 Tensor Core——这就是第八节 TTFT 里那 80-90 ms 的去处。

### 12.6 多模态投影 + LLM prefill（Tensor Core，~36 ms）

MLP projector 把 visual tokens 映射到 LLM 的 embedding 空间，和文字 prompt 的 token 拼接成 ~210 个 token 的完整输入序列。LLM prefill 并行处理这 210 个 token（权重只搬一遍，见 3.3 节），生成 KV-cache，并产出**第一个文字 token**。

到这里累计 ≈ **126 ms，这就是 TTFT（Time To First Token）**——从"看见"到"说出第一个字"的延迟。

### 12.7 Decode 逐 token 生成（每步 ~92 ms）

进入自回归循环，每生成一个 token：

```
搬 8 GB 权重（78 ms 硬地板）→ 前向传播算 1 个 token
→ 追加写 KV-cache → softmax 采样出 token id → 进入下一步
```

63 个 token ≈ 5,809 ms。瓶颈不在算力，在 LPDDR5 的 102.4 GB/s（见第三、八节）。

### 12.8 Detokenize 输出（CPU，≈0 ms）

token id 流式地查 BPE 词表，还原成 UTF-8 字符串片段，拼成你看到的连续文字；遇到 EOS token 停止。纯 CPU 查表，耗时可以忽略。

### 12.9 全链路耗时与硬件分工一览

| 环节 | 执行硬件 | 耗时量级 |
|---|---|---|
| 曝光 + RAW 输出 | 相机 CMOS 传感器 | 10-33 ms（帧率决定） |
| CSI-2 传输 + ISP 处理 | SoC 内专用 ISP 硬件 | <5 ms |
| 帧 buffer 交接 | 统一内存零拷贝 | ~0 |
| 预处理（resize/归一化） | CUDA Core | 1-2 ms |
| ViT 视觉编码 | Tensor Core | 80-90 ms |
| 投影 + LLM prefill | Tensor Core | ~36 ms |
| decode × 63 token | Tensor Core，受 LPDDR5 带宽限制 | ~5,809 ms |
| detokenize | CPU | ≈0 |

**结论**：从"看见"到第一个字只要 ~130 ms，但写完这段话要 ~6 秒。整条链路里相机、ISP、预处理、detokenize 加起来不到 5%——**95% 以上的时间花在 decode 阶段反复搬运权重上**。这就是为什么整份文档的优化结论（W8 量化、选 1B 模型）都围绕内存带宽展开。

---

## 十三、看懂算力标称：稠密/稀疏/INT8/FP16，以及怎么估算模型能不能跑

厂商宣传页上的 TOPS 数字（比如 Orin NX Super 的 157 TOPS）需要打几次折扣才是你实际能用到的算力。本节把各种算力口径讲清楚，并给出一个"两步估算法"：给定一个模型参数量，估算它在 Orin NX 上能不能跑、能跑多快。

### 13.1 各种算力口径是什么

- **稠密算力（Dense）**：矩阵里每个数都参与计算时的真实峰值，是硬件的硬实力。
- **稀疏算力（Sparse）**：Ampere 起的 Tensor Core 支持"2:4 结构化稀疏"——每 4 个权重强制剪 2 个为 0，硬件跳过这些 0 不算，**标称算力直接 ×2**。代价是要剪枝 + 重训，LLM 实际部署几乎不用稀疏，所以这是宣传口径，不是你拿到的算力。
- **INT8 算力**：8 位整数乘加的吞吐，单位 TOPS（每秒万亿次整数运算）。Tensor Core 上 INT8 吞吐是 FP16 的 2 倍。
- **FP16 / BF16 算力**：半精度浮点吞吐，单位 TFLOPS。两者速度完全相同（见第五节），差异只在数值格式。

**Orin NX Super 的 157 TOPS 拆解**——它是**稀疏 INT8** 口径：

| 口径 | Orin NX Super | 对 LLM 的意义 |
|---|---|---|
| 稀疏 INT8（宣传口径） | 157 TOPS | 几乎用不上 |
| 稠密 INT8 | ~78 TOPS | W8 量化部署时的算力上限 |
| 稠密 FP16 / BF16 | ~39 TFLOPS | 原生精度部署时的算力上限 |

> **关键事实：Super 模式只超了计算频率，内存带宽仍是 102.4 GB/s 没变。** LLM decode 是带宽瓶颈，所以 Super 的 157 TOPS 对 decode 速度**几乎没有提升**；它只加速计算瓶颈段——ViT 视觉编码和 prefill（TTFT 会变短一些）。

### 13.2 为什么 decode 速度只看带宽（算术强度视角）

判断"算力瓶颈还是带宽瓶颈"有一个定量工具——**算术强度（Arithmetic Intensity）= 每搬 1 字节数据能做多少次运算**。

硬件的"平衡点"（Roofline 拐点）= 峰值算力 ÷ 带宽：

```
Orin NX Super FP16：39 × 10¹² FLOPS ÷ 102.4 × 10⁹ B/s ≈ 383 FLOP/字节
```

意思是：每搬进 1 字节数据，至少要做 383 次运算，Tensor Core 才不会有空闲。

而 decode 阶段的算术强度：每 token 计算量 = 2×参数量 FLOP，搬运量 = 参数量×精度字节数，FP16 下强度 = **2 FLOP/字节**——比平衡点低了近 200 倍。结论：**decode 永远带宽瓶颈，Tensor Core 99% 时间在等数据**。INT8 也一样（平衡点 ~762，强度还是 2）。这就是"估算 decode 速度只需要一个除法"的理论依据。

### 13.3 两步估算法

**第 1 步：内存能不能装下（决定"能不能跑"）**

```
权重内存 = 参数量 × 每参数字节数（FP16 = 2B，INT8 = 1B）
总需求 ≈ 权重 × 1.3（KV-cache + 运行时开销）+ 系统预留 4 GB ≤ 16 GB
```

**第 2 步：带宽决定 decode 速度（决定"跑多快"）**

```
decode tok/s ≈ 102.4 GB/s ÷ 权重 GB 数 × 利用率（0.7~0.8）
```

### 13.4 1B / 2B / 4B 在 Orin NX 16GB 上的估算表

| 模型 | 精度 | 权重 | 内存总需求 | 能否跑 | decode 理论 | decode 实际估计 |
|---|---|---|---|---|---|---|
| 1B | FP16 | 1.86 GB | ~4 GB | ✅ 轻松 | 55 tok/s | ~40 tok/s |
| 1B | INT8 | 0.93 GB | ~3 GB | ✅ 轻松 | 110 tok/s | ~70-80 tok/s |
| 2B | FP16 | 3.73 GB | ~7 GB | ✅ 可以 | 27 tok/s | ~20 tok/s |
| 2B | INT8 | 1.86 GB | ~5 GB | ✅ 可以 | 55 tok/s | ~40 tok/s |
| 4B | FP16 | 7.45 GB | ~11 GB | ⚠️ 贴上限，有 OOM 风险 | 13.7 tok/s | ~10 tok/s（实测 10.9） |
| 4B | INT8 | 3.73 GB | ~7 GB | ✅ 可以 | 27 tok/s | ~20 tok/s |

4B FP16 那行的实测值（10.9 tok/s）和估算法几乎重合——这套两步估算是可信的。

**换算成 Hz（每秒产出几段完整描述）**：以 63 token 的场景描述为例，E2E = TTFT + 63 × TPOT：

- 1B INT8：TTFT ~110 ms + 63 × 12 ms ≈ 0.9 s → **~1.1 段/秒**
- 4B FP16：TTFT ~126 ms + 63 × 92 ms ≈ 5.9 s → **~0.17 段/秒**

### 13.5 这套方法怎么推广到其他芯片

换任何边缘芯片，只需要换两个数字就能估算：**内存容量**（第 1 步）和**内存带宽**（第 2 步）。TOPS 只在判断 prefill / ViT / CNN 检测这类计算瓶颈负载时才需要看。这也是为什么选边缘 AI 芯片跑 LLM/VLM 时，**带宽比 TOPS 更能预测体验**——标称 157 TOPS 的 Orin NX Super 和标称 100 TOPS 的普通版，decode 速度完全一样。

---

## 十四、多模型并行：内存是叠加的，带宽是分时的，算力是共享的

单模型的估算法推广到多模型，三个维度要分别过一遍。瓶颈优先级：**内存（硬门槛）> 带宽（决定速度）> 算力（通常够用）**。

### 14.1 第 1 步：内存——硬门槛，不满足直接 OOM

所有模型的权重**同时常驻**统一内存，互相挤压：

```
Σ 各模型（权重 × 1.3）+ 系统预留 4 GB ≤ 16 GB
```

这条不满足，后面都不用算。

### 14.2 第 2 步：带宽——分时共享，不满足不会失败，只会变慢

102.4 GB/s 只有一根内存总线，谁 decode 谁占用。Orin NX 只有 4 个 SM，多模型"并发"本质也是时间片轮转，所以**多模型并行的总带宽需求可以直接相加**。设模型 i 权重 W_i GB、每次输出 n_i 个 token、运行频率 f_i 段/秒：

```
每段输出搬运量 = (n_i + 1) × W_i      （1 次 prefill + n 次 decode，每次都要搬全部权重）
模型带宽需求   = f_i × (n_i + 1) × W_i
可行条件       = Σ 各模型带宽需求 ≤ 102.4 × 0.75 ≈ 76.8 GB/s
```

注意"超了"的后果和内存不同——**不是跑不了，而是实际频率被压下来**：

```
实际频率 = 76.8 GB/s ÷ Σ (n_i + 1) × W_i
```

另外，用 DLA 分担 CNN 检测能释放 GPU 的 Tensor Core 时间，但 **DLA 同样从 LPDDR5 搬权重，带宽预算里照样要算它一份**。

### 14.3 第 3 步：算力——只检查计算瓶颈段

decode 反正带宽瓶颈，不占算力。要检查的是 ViT 编码、prefill、CNN 检测这些**计算瓶颈**负载的叠加：

```
Σ 各模型每秒计算量 ≤ 39 TFLOPS（稠密 FP16）× 0.5（实际利用率）
```

这条通常非常宽裕，检测类小模型一般只占百分之几。

### 14.4 两个具体组合

**组合 A：1B VLM 场景描述（1 段/秒）+ YOLOv8n 检测（30 FPS）**

| 模型 | 权重 W | 每段 token n | 频率 f | 带宽需求 | 算力需求 |
|---|---|---|---|---|---|
| 1B INT8 VLM | 0.93 GB | 63 | 1 Hz | 64 × 0.93 = **59.5 GB/s** | prefill + ViT，少量 |
| YOLOv8n INT8 | ~3 MB | — | 30 FPS | **~0.5 GB/s** | 8.7 GFLOP × 30 ≈ 0.3 TFLOPS（~1%） |
| **合计** | | | | **60 GB/s ≤ 76.8 ✅** | **<5% ✅** |

✅ 可行——这正是第十一节"场景 A 双模型"的定量依据。

**组合 B：2B VLM（1 段/秒）+ 同样检测**

```
VLM 带宽需求 = 64 × 1.86 = 119 GB/s > 76.8 ❌
实际频率 = 76.8 ÷ 119 ≈ 0.65 段/秒
```

不会报错，但描述产出从 1 段/秒掉到 0.65 段/秒。要么接受降频、要么砍输出长度（减小 n）、要么退回 1B。

### 14.5 工程验证与余量

估算只是预算，上机后用 `tegrastats` 交叉验证三个数：`RAM`（内存余量）、`EMC_FREQ`（内存控制器占用率 ≈ 带宽利用率）、`GR3D_FREQ`（GPU 占用率）。另外 25W 功耗墙下多模型持续满载可能触发动态降频，估算时留 20% 余量更稳。

---

## 十五、只有"2 核 CPU + 30 TOPS"这种参数时，怎么判断能跑什么模型

看低算力芯片（RK3588、地平线、各类 NPU 方案）的规格页时，通常只给你 CPU 核数和 TOPS 两个数字。这两个数字**不够下结论**——决定体验的参数有四个，规格页往往只写两个。先凑齐这张清单：

| 必查参数 | 为什么 | 规格页常见猫腻 |
|---|---|---|
| 内存容量 | 决定能不能装下模型 | 系统要切走 1-2 GB |
| **内存位宽 × 速率** | 决定 decode 速度 | **经常不写，要自己算** |
| TOPS 的口径 | 决定算力上限 | 稀疏还是稠密？INT8 还是 FP16？ |
| 算子覆盖 | 决定 TOPS 能不能兑现 | NPU 多为 CNN 优化，Transformer 可能回退 CPU |
| CPU 核数与架构 | 决定系统开销和回退路径 | 2 核小核连系统本身都吃力 |

### 15.1 把 TOPS 翻译成你能用的数

问三个问题：**稀疏还是稠密？**（稀疏 ÷2）；**什么精度？**（INT8 转 FP16 再 ÷2）；**支持哪些算子？** 第三个问题最致命——很多低算力 NPU 只为 CNN 优化，Transformer 需要的 LayerNorm / Softmax / 大矩阵乘不在算子列表里，LLM 只能整体回退 CPU，那 30 TOPS 对你等于 0。

### 15.2 自己算出内存带宽

带宽 = 位宽 ÷ 8 × 速率。例如 64-bit LPDDR4x @ 4266 MT/s：64 ÷ 8 × 4.266 ≈ **34 GB/s**——只有 Orin NX 的 1/3。规格页只写"LPDDR4x"时，按这个公式自己补出带宽。

### 15.3 算例：2 核 ARM + 30 TOPS INT8 NPU + 4 GB LPDDR4x（34 GB/s）

**① 内存**：系统切走 ~1.5 GB，可用 ~2.5 GB → 1B INT8（约 1.2 GB）✅；2B INT8（约 2.4 GB）⚠️ 贴脸；任何 FP16 模型和 4B 一律 ❌。

**② CNN 检测（走 NPU）**：YOLOv8n 每帧 8.7 GFLOP，30 TOPS 按 30% 实际利用率 → 理论上千帧/秒，实际 30-60 FPS 毫无压力。✅ 这是这类芯片的主场。

**③ LLM decode**：分两路——

- NPU 若能跑 Transformer：34 ÷ 0.93 ≈ 理论 36 tok/s，实际 ~25 tok/s
- 若回退 CPU（CPU 推理利用率只有 30-50%）：34 ÷ 0.93 × 0.4 ≈ **~15 tok/s，经验值只有 5-10 tok/s**

**④ LLM prefill（CPU 回退时的噩梦）**：1B 模型 prefill 210 个 token 需要 2×10⁹×210 ≈ 420 GFLOP，2 核小核合计约 40 GFLOPS → **首 token 要等 10 秒级**。NPU 不支持 Transformer 时，LLM 在这类芯片上基本不可用。

### 15.4 结论

| 模型类型 | 该配置可行性 |
|---|---|
| CNN 检测/分割（YOLO 等） | ✅ 流畅，30 TOPS 的主场 |
| 1B INT8 LLM/VLM | ⚠️ 取决于 NPU 算子覆盖；回退 CPU 则 decode ~5-10 tok/s、首 token 数秒 |
| 2B+ 或 FP16 | ❌ 内存和带宽都不够 |

一句话规律：**TOPS 决定"这类芯片擅长什么"（通常是 CNN），带宽和内存决定"能不能跑 LLM"**——两组数字必须分开看，只看 TOPS 选芯片是边缘部署最常见的踩坑方式。

---

## 十六、变长输入、CUDA Graph 与 Padding——推理引擎的三个工程细节

VLM 实际部署中，每次推理的输入都不一样：图片分辨率变 → visual token 数变，用户 prompt 长短变 → 总输入变。这一节讲清楚变长输入的影响，以及推理引擎应对它的两个核心技术：CUDA Graph 和 Padding。

### 16.1 每次 prompt 不一样，对推理有什么影响

影响主要在 **prefill 侧**，decode 侧基本无感：

1. **prefill 时间随 token 数变化**：成本大致随 token 数线性增长（attention 部分甚至是平方），210 token 和 500 token 的首字延迟能差一倍多。
2. **推理引擎的 shape 问题**：TensorRT engine 构建时绑定输入 shape（或几个预设的 optimization profile）；CUDA Graph 要求 shape 完全固定（见 16.2）。输入长度每次都变，有三条路：为每个长度重新构建 engine（不现实）；预设几档 profile（档位外傻眼）；**padding 到定长**（工程上最常用）。
3. **KV-cache 按最大长度预分配**：不管这次实际用多少。
4. **decode 几乎不受影响**：每步只算 1 个新 token，大头是搬权重（不变），只是 KV-cache 读取随历史变长缓慢增加。

### 16.2 CUDA Graph：把几百次 kernel 启动压缩成一次

**问题：kernel launch 开销。** 一次 LLM forward 不是"一次计算"，而是几百上千次 GPU kernel 启动——每层 Transformer 有 QKV 投影、attention、norm、FFN 等十来个 kernel，乘上 28-36 层。每次启动，CPU 要花 5-10 µs 做参数准备和驱动调用。桌面 x86 CPU 快，GPU 上一个 kernel 还没算完下一个已经喂进来，开销被掩盖。但 **Jetson 的 ARM Cortex-A78AE 弱得多**：decode 小模型时会出现"GPU 算完了在等 CPU 发下一个 kernel"的空转——1B 模型 decode 理论 18 ms，launch 开销漏进来 3-5 ms，就是白丢 20-30% 速度。

**CUDA Graph 的做法**：把整段 kernel 序列**录制一次成一张图**，之后用**一次 launch 重放整张图**。CPU 开销从几百次启动变成 1 次提交，GPU 端 kernel 背靠背执行、无气泡。vLLM、TensorRT-LLM 默认都对 decode 步开 CUDA Graph。

**约束（细节所在）**：录制时 **shape、内存地址、控制流全部固定**。推论：

- decode 步天然适合进图（每步 shape 恒为 1 个 token）
- prefill 步变长，不能进图，走普通 eager 执行
- KV-cache 必须**预分配在固定地址**——vLLM 的 PagedAttention 为此专门做了页表间接寻址来兼容图
- 引擎启动时要 warm-up 跑几遍完成录制，所以第一次推理特别慢是正常的

### 16.3 Padding：为定长付出的代价

为了配固定 shape 的 engine/graph，把输入补齐到定长（比如 210 → 512），pad 部分用 attention mask 屏蔽——**结果完全正确，但白算了**。

- **代价定量**：pad 210→512，prefill 计算量 ×2.4。但注意在带宽瓶颈平台上，prefill 的权重搬运不变（还是搬一遍），增加的主要是计算时间——所以 padding 的浪费在 Orin NX 上有时反而可接受。
- **工程折中是分桶（bucketing）**：预设 128/256/512/1024 几档，输入归入最近的桶。桶越细浪费越少，但每桶要存一份 engine/graph，内存占用变多——又回到 16 GB 预算的权衡。

### 16.4 辨析："NX 的 INT8 算力只有 38 TOPS"是真是假

把换算链摆出来：

```
标称 100 TOPS（稀疏 INT8，宣传口径）
  ÷ 2（去掉 2:4 稀疏）
= 50 TOPS（稠密 INT8，硬件真实峰值）
  × 75% 左右（实际推理的利用率）
≈ 35-40 TOPS（工程上能兑现的有效值）
```

所以这句话**半真半假**：

- 说"硬件峰值只有 38 TOPS"——不准确，稠密峰值是 ~50 TOPS
- 说"实际跑起来能兑现的也就 38 TOPS 上下"——合理，CNN 实测 70-80% 利用率后就是这个数
- 还有一种可能：Orin NX **8GB 版**标称 70 TOPS（稀疏）→ 稠密 35 TOPS，如果测的是 8GB 版，38 还偏高了

Super 版同理：157（稀疏）→ 78（稠密）→ 有效 ~60。

以及对 LLM 的老话重提：**38 也好 50 也好 157 也好，decode 速度都一样**——由 102.4 GB/s 带宽决定。这个数字只在 prefill、ViT、CNN 检测这些计算瓶颈负载上才有意义。

---

## 附录：本文涉及的核心概念速查

| 概念 | 一句话解释 |
|---|---|
| **统一内存** | CPU 和 GPU 共用同一块物理 LPDDR5，无独立显存 |
| **RSS** | Resident Set Size，常驻内存集。Linux 内核统计的进程当前占用的物理内存页数，只计入了 CPU 页表映射的页面，Jetson 上可能遗漏 GPU 侧分配 |
| **SM** | Streaming Multiprocessor，流式多处理器。GPU 内部的"迷你 GPU"单元，包含 CUDA Core、Tensor Core、L1 缓存和寄存器。Orin NX (GA10B) 有 4 个 SM |
| **LPDDR5 带宽** | 102.4 GB/s，数据从内存到 GPU 缓存的搬运速度上限 |
| **RAM** | Random Access Memory，随机存取存储器。和硬盘不同，可任意顺序读写，速度快但断电丢失。DDR/LPDDR/GDDR 都是 RAM 的不同类型 |
| **DLA** | Deep Learning Accelerator，深度学习加速器。SoC 内部独立于 GPU 的 CNN 专用推理硬件，功耗极低。不支持 Transformer，可并行跑 ViT 不占 GPU |
| **ISP** | Image Signal Processor，图像信号处理器。SoC 内专用硬件，负责把相机 RAW 图（Bayer）处理成彩色图，不占 GPU |
| **零拷贝（NVMM buffer）** | Jetson 统一内存下，ISP 输出的相机帧 GPU 可直接读取，无需 memcpy/PCIe 搬运 |
| **GPU 缓存层级** | LPDDR5 → L2 → L1 → 寄存器，每层更快更小 |
| **Tensor Core** | GPU 中专做矩阵乘法的硬件单元，LLM 推理的核心加速器 |
| **KV-cache** | 缓存历史 token 的 Key/Value，省掉重复注意力计算 |
| **Prefill vs Decode** | Prefill = 并行处理全量输入（快），Decode = 串行逐 token 生成（慢） |
| **内存带宽瓶颈** | Decode 每步都要搬全部权重，102.4 GB/s 成为硬上限 |
| **W8 INT8 量化** | 权重压缩为 8 位整数，内存减半 + decode 搬运时间减半 |
| **nvmap** | NVIDIA Jetson 的 GPU 内存管理子系统，可能导致 RSS 漏算 |
| **TTFT** | Time To First Token，从请求到首 token 的延迟（vLLM 术语） |
| **TPOT** | Time Per Output Token，后续每个 token 的平均生成时间 |
| **TOPS / TFLOPS** | 每秒万亿次整数 / 浮点运算。TOPS 一般指 INT8，TFLOPS 一般指浮点 |
| **稠密 vs 稀疏算力** | 稠密 = 全部权重参与计算的真实峰值；稀疏 = 2:4 结构化剪枝后硬件跳过零值的标称值（×2，LLM 实际用不上） |
| **算术强度 / Roofline** | 每搬 1 字节数据能做的运算次数。硬件平衡点 = 算力 ÷ 带宽（Orin NX Super FP16 ≈ 383），decode 只有 ~2，故永远带宽瓶颈 |
| **CUDA Graph** | 把整段 kernel 启动序列录制一次、之后一次 launch 重放，消除 CPU 端 launch 开销（Jetson 的弱 ARM CPU 上收益尤其大）。代价：shape/内存地址必须固定，故只适合 decode 步 |
| **Padding / 分桶** | 把变长输入补齐到定长以适配固定 shape 的 engine/graph，mask 保证结果正确，代价是 pad 部分的计算白做；分桶（128/256/512…）在浪费和内存占用之间折中 |
