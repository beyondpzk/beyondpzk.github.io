---
title: NVIDIA Sparse TOPS 与 Dense TOPS：部署时必须看懂的算力数字游戏
date: 2026-07-08
categories: [Deploy]
---

# NVIDIA Sparse TOPS 与 Dense TOPS：部署时必须看懂的算力数字游戏

在选购 NVIDIA GPU 或 Jetson 模组时，规格表上最醒目的数字往往是 **"XXX TOPS"** 的 AI 算力。但同一个型号常会同时出现两个版本：

- **Dense TOPS**（稠密算力）
- **Sparse TOPS**（稀疏算力，通常比 Dense 高出 1 倍）

比如 Jetson AGX Orin 标称 **275 TOPS**，但如果你仔细看白皮书，会发现这其实是 **Sparse INT8** 的成绩；它的 **Dense INT8** 算力大约是 **137.5 TOPS** 甚至更低。两篇测评文章若分别用 Sparse 和 Dense 做对比，结论可能完全相反。

这篇文章从硬件机制、软件条件和部署实践三个层面，讲清楚这两个数字到底代表什么、为什么 NVIDIA 喜欢用 Sparse 来宣传、以及工程师在选型时应该怎么看。

---

## 一、TOPS 是什么？为什么它不等于实际速度

**TOPS** = Tera Operations Per Second，即每秒万亿次运算。它通常用来衡量 AI 推理芯片在整数精度（主要是 INT8）下的峰值吞吐能力。

计算公式很直接：

```text
TOPS = Tensor Core 数量 × 每个 Tensor Core 每周期运算次数 × 运行频率 × 利用系数
```

但这个数字是**理论峰值**，实际模型很难跑到 100%，原因包括：

1. **内存墙**：算力再强，权重和激活值喂不进去也是白搭。
2. **算子开销**：Softmax、LayerNorm、KV Cache、数据搬运会吃掉大量周期。
3. **利用率**：并非所有层都能持续占满 Tensor Core。
4. **精度与稀疏度**：Sparse 模式下还需要满足特定稀疏结构，否则硬件不会加速。

所以 **Sparse TOPS 和 Dense TOPS 的区别，本质上是对“一次运算”是否该被计为一次运算”的不同定义**。

---

## 二、Dense TOPS：老老实实数乘法

**Dense TOPS（稠密算力）**指的是芯片在**不依赖稀疏性**的情况下，对普通稠密矩阵乘法（GEMM）的峰值吞吐。

假设一个矩阵乘法：

```text
C = A × B
A: M × K
B: K × N
```

稠密情况下，每个元素相乘都要做一次运算，没有跳过的余地。

Dense TOPS 的特点是：

- **最保守、最可复现**：任何模型都能按这个上限估算。
- **跨厂商可比性强**：AMD、Intel、高通通常也报 Dense 数字。
- **更接近端到端体验**：真实模型大多没有 2:4 结构化稀疏，实际吞吐往往接近 Dense 上限。

如果你只是做一个粗略的选型对比，**先看 Dense TOPS**。

---

## 三、Sparse TOPS：靠“跳过零”换来的 2 倍峰值

### 3.1 什么是稀疏性

神经网络中的许多权重非常接近 0。如果我们把这些接近 0 的权重真的变成 0，矩阵中就出现了大量零元素，这样的矩阵叫**稀疏矩阵**。

稀疏矩阵乘法有一个显而易见的优化：

> 任何数乘 0 都等于 0，所以遇到 0 就可以跳过这次乘法。

理论上，如果一半权重都是 0，计算量直接减半；在相同硬件上，可以把省下来的资源用来算别的东西，从而**让有效吞吐翻倍**。

### 3.2 NVIDIA 的 2:4 结构化稀疏

NVIDIA 从 **Ampere 架构**（A100、RTX 30 系列、Jetson Orin）开始在 Tensor Core 中支持一种叫 **2:4 结构化稀疏** 的硬件加速。

它的规则非常严格：

- 每 4 个连续权重中，**必须有 2 个是 0，另外 2 个非零**。
- 0 的位置可以任意，但总数必须是 2 个。
- 硬件以 4 个权重为一组进行压缩，只存储 2 个非零值和它们的索引。

满足这个条件后，Tensor Core 可以在同一个周期内处理两倍的有效数据，于是峰值算力翻倍。

> **关键点**：Sparse TOPS 不是“所有模型都能自动享受的加成”，它要求模型权重必须经过稀疏化训练或剪枝，并严格满足 2:4 结构。

### 3.3 为什么 Sparse 通常是 Dense 的 2 倍

NVIDIA 的硬件设计是：只要满足 2:4 稀疏，Tensor Core 的**有效 MAC（乘加）数量翻倍**。因此：

```text
Sparse TOPS ≈ 2 × Dense TOPS
```

这个关系在 Ampere、Ada Lovelace、Hopper、Blackwell 几代架构中基本保持一致。

---

## 四、硬件世代的稀疏支持

| 架构 | 代表产品 | 是否支持 2:4 结构化稀疏 | 备注 |
|---|---|---|---|
| Volta | V100 | 否 | 只有 Dense 数字 |
| Ampere | A100、RTX 3090、Jetson Orin | 是 | 引入第二代稀疏 Tensor Core |
| Ada Lovelace | RTX 4090、L40S | 是 | FP8 也支持稀疏翻倍 |
| Hopper | H100、H200 | 是 | Transformer Engine + FP8 稀疏 |
| Blackwell | B200、RTX 50 系列 | 是 | 继续 2:4 稀疏，部分场景引入更细粒度稀疏 |

可以看到，**从 Ampere 开始，NVIDIA 消费级和数据中心级产品都支持稀疏加速**。这也是为什么近几年规格表上 Sparse 数字越来越常见。

---

## 五、典型产品的 Dense vs Sparse 对比

### 5.1 数据中心 GPU

| 型号 | FP16/BF16 Dense | FP16/BF16 Sparse | INT8 Dense | INT8 Sparse | FP8 Dense | FP8 Sparse |
|---|---|---|---|---|---|---|
| A100 SXM | ~312 TFLOPS | ~624 TFLOPS | ~624 TOPS | ~1248 TOPS | — | — |
| H100 SXM | ~989 TFLOPS | ~1979 TFLOPS | ~1979 TOPS | ~3958 TOPS | ~1979 TFLOPS | ~3958 TFLOPS |
| H200 SXM | ~989 TFLOPS | ~1979 TFLOPS | ~1979 TOPS | ~3958 TOPS | ~1979 TFLOPS | ~3958 TFLOPS |
| B200 | ~2250 TFLOPS | ~4500 TFLOPS | — | — | — | — |

> 注：B200 的公开数据常以 Sparse 形式给出，Dense 数字约为 Sparse 的一半。

### 5.2 消费级 GPU

| 型号 | INT8 Dense | INT8 Sparse | FP16/BF16 Dense | FP16/BF16 Sparse |
|---|---|---|---|---|
| RTX 3090 | ~284 TOPS | ~568 TOPS | ~142 TFLOPS | ~284 TFLOPS |
| RTX 4090 | ~661 TOPS | ~1321 TOPS | ~330 TFLOPS | ~660 TFLOPS |
| RTX 5090 | ~1150 TOPS | ~2300 TOPS | ~575 TFLOPS | ~1150 TFLOPS |

### 5.3 Jetson 边缘模组

Jetson 的标称算力通常是 **GPU Tensor Core + DLA 加速器** 的总和，并且默认报 Sparse。

| 型号 | 标称 AI 算力（Sparse INT8） | 近似 Dense INT8 | 组成 |
|---|---|---|---|
| Orin Nano 4GB | 20 TOPS | ~10 TOPS | GPU only |
| Orin Nano 8GB | 40 TOPS（标准）/ 67 TOPS（Super） | ~20 / ~33 TOPS | GPU only |
| Orin NX 8GB | 70 TOPS | ~35 TOPS | GPU + 1x DLA |
| Orin NX 16GB | 100 TOPS | ~50 TOPS | GPU + 1x DLA |
| AGX Orin 32GB | 200 TOPS | ~100 TOPS | GPU + 2x DLA |
| AGX Orin 64GB | 275 TOPS | ~137.5 TOPS | GPU + 2x DLA |

> 注：AGX Orin 的 275 TOPS 具体拆分为 GPU ~170 TOPS（Sparse）+ 2×DLA ~52.5 TOPS（Sparse），Dense 时整体约 137.5 TOPS。

---

## 六、Sparse 为什么看起来更快，但实际未必

### 6.1 稀疏化不是免费的

要让模型享受 Sparse 加速，必须满足：

1. **权重满足 2:4 结构**：不是任意剪枝都行，必须每 4 个权重恰好 2 个 0。
2. **推理框架支持**：TensorRT、ONNX Runtime、PyTorch 等需要调用稀疏 GEMM 内核。
3. **精度损失可控**：剪枝后通常需要微调，否则精度会掉。
4. **激活值也最好稀疏**：如果只有权重稀疏、激活值稠密，实际加速可能远低于 2 倍。

### 6.2 端到端加速通常小于 2 倍

即使权重完全满足 2:4 稀疏，真实推理也很难达到理论 2 倍提升，原因包括：

- **非矩阵乘法算子不加速**：Softmax、Reshape、Concat、NMS 等不会变快。
- **内存带宽瓶颈**：大模型推理时，读取权重的速度往往比计算速度更慢。
- **GPU 占用率**：小 batch 时 Tensor Core 可能喂不饱。
- **DLA 限制**：Jetson 的 DLA 对稀疏支持有限，有时反而不如 GPU 快。

根据 NVIDIA 和社区的一些实测，**稀疏化后的端到端延迟提升通常在 1.2×–1.6× 之间**，极少接近 2×。

### 6.3 并非所有模型都适合稀疏

- **Transformer / LLM**：注意力层权重相对规则，稀疏化效果较好，但 KV Cache 会引入额外内存压力。
- **CNN / 检测模型**：某些层剪枝后精度下降明显，需要大量微调。
- **小模型**：本来计算量就小，稀疏带来的绝对收益有限，不值得额外工程投入。

---

## 七、NVIDIA 为什么喜欢用 Sparse 数字宣传

这个问题没有阴谋论，原因很简单：**数字更大、更好看**。

- 275 TOPS 比 137.5 TOPS 更容易印在包装盒上。
- 在竞争激烈的 AI 芯片市场，Sparse 数字能让产品在参数表上领先对手。
- 对于已经使用稀疏训练流程的客户，Sparse 数字确实能反映最佳情况。

但这对部署工程师意味着：

> **不要只看标题里的 TOPS，要看括号里的小字，看它到底是 Sparse 还是 Dense。**

如果你用 Sparse 数字去评估一个稠密模型，会严重高估实际性能；反之，如果你用 Dense 数字去评估一个已经稀疏化的模型，又会低估优化空间。

---

## 八、部署时怎么选、怎么看

### 8.1 选型阶段

- **横向对比不同厂商芯片**：统一看 **Dense TOPS/TFLOPS**，否则相当于拿苹果的峰值比橘子的常规值。
- **同厂不同代对比**：可以看 Sparse 数字，但要确认两者都是 Sparse，且精度一致（INT8 vs INT8，FP16 vs FP16）。
- **关注内存带宽**：大模型推理往往是带宽受限，算力数字再高也受限于 `带宽 / 模型大小`。

### 8.2 落地阶段

- **先跑 Dense 基线**：用 TensorRT、ONNX Runtime 或 vLLM 跑出原始模型的端到端延迟和吞吐。
- **再评估稀疏收益**：如果业务对延迟敏感，可以尝试 2:4 结构化剪枝，但必须重新校准精度。
- **看实际 FPS / token/s**：最终验收指标应该是 `帧率` 或 `每秒生成 token 数`，而不是 TOPS。

### 8.3 一句话原则

> **Dense 是底线，Sparse 是上限；部署看实测，TOPS 仅供参考。**

---

## 九、总结

| 维度 | Dense TOPS | Sparse TOPS |
|---|---|---|
| 定义 | 稠密矩阵乘法的峰值吞吐 | 利用 2:4 结构化稀疏后的峰值吞吐 |
| 与硬件关系 | 任何模型都能达到的理论上限 | 需要模型满足稀疏结构才能触发 |
| 典型数值关系 | 基准 | 约为 Dense 的 2 倍 |
| 适用场景 | 通用选型、跨厂商对比 | 已稀疏化模型的最佳情况估算 |
| 可信度 | 更保守、更真实 | 更理想化、容易高估 |

NVIDIA 的 Sparse TOPS 不是假的，但它是一种“在理想条件下才能兑现”的算力。对于部署工程师来说：

- 如果你还没做稀疏化，**用 Dense 数字做容量规划**。
- 如果你已经做了 2:4 稀疏训练，**可以用 Sparse 数字来估算优化后的天花板**。
- 无论用哪个数字，**最终都要回到实测 latency 和 throughput**。

在 AI 部署这件事上，规格表上的大数字只是起点，工程上的真实表现才是终点。

---

## 参考与延伸阅读

- NVIDIA A100 Tensor Core GPU Architecture Whitepaper
- NVIDIA H100 Tensor Core GPU Datasheet
- Jetson Orin Series Technical Brief
- [INT8 TOPS Standardization: Dense Values for Fair Cross-Vendor Comparison](https://gpupoet.com/news/int8-tops-standardization)
- [TensorRT vs DLA on Jetson Orin: When to Use Each](https://proventusnova.com/blog/tensorrt-vs-dla-jetson-orin/)
- [四. TensorRT 模型部署优化 - pruning(sparse-tensor-core)](https://jishuzhan.net/article/1812125990655102977)
