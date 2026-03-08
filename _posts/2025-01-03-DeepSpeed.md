---
layout: post
title: DeepSpeed
date: 2025-01-03
categories: [Understandings]
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# DeepSpeed


# DeepSpeed 技术原理详解（以大模型训练/推理为主）

DeepSpeed 是微软开源的一套**分布式深度学习系统**，核心目标是让你用更少的显存、更高的吞吐，在更多 GPU/多机环境下训练和推理超大模型。它不是单一算法，而是一组可组合的系统能力，最核心的几块是：

- **ZeRO 系列（显存优化/状态分片）**：解决“模型太大放不下”
- **通信与计算重叠、分层通信、bucket 化**：解决“多卡多机效率”
- **Offload（CPU/NVMe）**：用带宽换显存
- **3D 并行（DP/TP/PP）与调度**：解决“规模进一步扩大”
- **推理优化（DeepSpeed-Inference）**：KV cache、并行、量化、kernel 融合等

下面按“原理 → 机制 → 关键组件 → 常见配置与取舍”详细说明。

---

## 1. DeepSpeed 在训练中到底要管理哪些东西？

一次训练迭代里，显存主要被这些占用：

1. **Parameters（参数）**
2. **Gradients（梯度）**
3. **Optimizer States（优化器状态）**
   - AdamW 通常有 `m` 和 `v` 两份状态（与参数同大小），再加上 master fp32 权重等
4. **Activations（激活）**（随 batch/seq/网络结构变化）
5. 临时 buffer、通信 buffer、碎片等

DDP 的问题：**每张卡都保存完整参数 + 完整优化器状态**，显存爆炸。  
DeepSpeed 最核心的贡献：用 ZeRO 把这三类训练状态做“分片”，显存随 GPU 数接近线性下降。

---

## 2. ZeRO（Zero Redundancy Optimizer）：DeepSpeed 的核心技术

ZeRO 的思想是：在数据并行里，不必让每张 GPU 都保存一份完整训练状态；只要把它们**切分到各张卡**，并在需要时通信聚合即可。

### 2.1 ZeRO 的三个阶段（Stage 1/2/3）

> 记 N 为数据并行 GPU 数量（ZeRO 的分片是在 data parallel group 内）

#### ZeRO-1：分片 Optimizer States
- 每张 GPU：保存 **1/N 的优化器状态**
- 参数仍然完整复制在每张卡
- 梯度仍然完整存在每张卡（反向后 all-reduce）

收益：Adam 场景下，显存明显下降（优化器状态常是大头）。  
通信：与 DDP 类似。

#### ZeRO-2：分片 Optimizer States + Gradients
- 优化器状态：1/N
- 梯度：1/N（用 reduce-scatter 形式规约并分片保存）
- 参数：仍然完整复制

收益：进一步省显存。  
通信：反向阶段更多使用 reduce-scatter，更新时各卡只用自己的 grad shard。

#### ZeRO-3：分片 Optimizer States + Gradients + Parameters（全分片）
- 参数：1/N（每卡只常驻参数分片）
- 梯度：1/N
- 优化器状态：1/N

这是“最省显存”的模式，也是大模型训练的关键。  
代价：前向/反向需要频繁 **all-gather 参数分片** 才能计算。

> 直观总结：  
> - ZeRO-1 解决“优化器太大”  
> - ZeRO-2 再解决“梯度太大”  
> - ZeRO-3 再解决“参数本身也放不下”

---

## 3. ZeRO-3 的一次迭代：前向/反向/更新在干什么？

以某一层的参数 `W` 为例（简化描述）：

### 3.1 常驻状态（平时）
- rank0 只存 `W_shard0`
- rank1 只存 `W_shard1`
- …
- 以及各自的 optimizer state shard

### 3.2 前向（Forward）
对要计算的这一层：
1. **AllGather 参数**：把所有 shard 拼成 `W_full`（临时）
2. 做本层 forward
3. （可选）立刻释放 `W_full`（减少峰值显存），回到只保留 shard

### 3.3 反向（Backward）
1. 若 forward 后释放了 full 参数，反向前会再 **AllGather** 一次
2. 计算得到本层梯度（概念上是 full 梯度）
3. **ReduceScatter 梯度**：把梯度规约并切回 shard，使每卡只留下 `dW_shard`

### 3.4 参数更新（Optimizer Step）
每张卡本地只对自己的 `W_shard` 用 `dW_shard` + state_shard 更新即可，无需再拿全量参数。

**关键理解**：  
ZeRO-3 的通信模式核心是：
- 参数：AllGather（按需拿 full 参数算）
- 梯度：ReduceScatter（直接得到全局规约后的梯度分片）

这和 PyTorch FSDP 的机制非常接近（两者理念同源）。

---

## 4. DeepSpeed 如何进一步“系统化”：通信、bucket、重叠、分层

仅仅做分片还不够，分布式效率往往卡在通信上。DeepSpeed 在工程上做了大量优化：

### 4.1 Bucket 化通信（通信分桶）
- 不把每个小张量都单独通信（开销巨大）
- 把很多参数/梯度拼成大 bucket 再通信
- 降低通信启动开销，提高带宽利用率

### 4.2 通信与计算重叠（overlap）
- 反向传播是逐层的：后面层先出梯度
- DeepSpeed 可以边算边通信（例如 reduce-scatter 提前开始）
- 目标：让通信隐藏在计算里，减少“等通信”的时间

### 4.3 分层通信（hierarchical all-reduce）
在多机环境：
- 同机内（NVLink）快
- 跨机（InfiniBand / RoCE）慢
DeepSpeed 可以先机内归约，再跨机归约，提升整体效率。

---

## 5. Offload：用 CPU / NVMe 换显存

当显存依旧不够，DeepSpeed 提供 ZeRO-Offload / ZeRO-Infinity：

### 5.1 CPU Offload
把优化器状态（甚至参数）放到 CPU 内存：
- GPU 只保留计算时必需部分
- 通过 PCIe/NVLink 在需要时搬运

优点：显存压力大幅下降  
缺点：带宽/延迟成本高，训练吞吐可能显著下降（取决于硬件）

### 5.2 NVMe Offload（ZeRO-Infinity）
进一步把状态放到 NVMe SSD：
- 显存更省
- 但性能更依赖 IO，吞吐可能更受限
适合极限规模、但对速度要求没那么苛刻或能做强优化的场景。

---

## 6. 与激活相关的显存优化：Activation Checkpointing / Partitioning

ZeRO 主要解决参数/梯度/优化器状态，但大模型中 **activation** 也可能是显存大头。

DeepSpeed 常与：
- **Activation Checkpointing（重计算）**：少存激活，多做重算
- **Activation Partitioning（激活分片）**：把激活也在 DP/TP 组中切分

组合使用可以进一步突破显存限制。

---

## 7. 3D 并行：把“更大模型”拆到更多维度

当单纯靠 ZeRO 仍不足（或效率不够）时，就要引入模型并行：

- **DP（Data Parallel）**：不同卡处理不同数据
- **TP（Tensor Parallel）**：把矩阵乘的维度切开（Megatron-LM 风格）
- **PP（Pipeline Parallel）**：把网络按层切成 stage 流水线

DeepSpeed 支持把它们组合成所谓 **3D parallelism**：

- 例如：TP 解决单层矩阵太大，PP 解决层太多，ZeRO/DP 解决优化器与冗余

代价：实现更复杂，通信模式更多样，需要更细的调参（micro-batch、pipeline schedule 等）。

---

## 8. DeepSpeed 推理（DeepSpeed-Inference）在做什么？

训练是一个重点，但 DeepSpeed 也有推理加速组件，常见能力包括：

- **Kernel 融合**（attention/MLP 融合、layernorm 融合等）
- **Tensor parallel 推理**
- **KV cache 管理**
- **权重量化**（FP16/INT8/INT4 等，视版本/后端而定）
- **更高效的 transformer 实现**（替换部分算子/模块）

需要注意：生产推理生态里 vLLM/TensorRT-LLM 也很强，DeepSpeed-inference 更偏“与训练栈一体化”的路线。

---

## 9. 一个典型 DeepSpeed ZeRO 配置项怎么映射到原理？

以常见 json 里的概念对应一下（不展开具体语法，讲原理映射）：

- `zero_optimization.stage = 1/2/3`  
  → 对应 ZeRO 分片到哪一类状态
- `overlap_comm`  
  → 通信与计算重叠
- `reduce_scatter`  
  → 梯度规约用 reduce-scatter（更适合 ZeRO-2/3）
- `contiguous_gradients`、`bucket_size`  
  → 梯度存储与通信分桶策略
- `offload_optimizer / offload_param`  
  → 把优化器状态/参数 offload 到 CPU/NVMe

---

## 10. 选型与取舍建议（非常实用）

### 10.1 什么时候用 ZeRO-1/2？
- 模型能放下参数，但 Adam 状态/梯度很大
- 想改动小、保持较高吞吐
- 通常 ZeRO-2 是很常见的“甜点位”（省显存明显、性能也不错）

### 10.2 什么时候必须 ZeRO-3？
- 参数本身都放不下（大模型常见）
- 或你希望用更多 DP 卡让显存按 1/N 下降

代价：
- 更多 all-gather 参数通信
- 对网络更敏感
- wrap/分桶/overlap 对性能影响更大

### 10.3 什么时候考虑 offload？
- 显存严重不足、但可以接受更慢
- CPU 内存充足 / NVMe IO 很强
- 或作为“应急”让实验先跑起来

---

## 11. DeepSpeed 与 FSDP（PyTorch 原生）怎么理解区别？

两者在“全分片”思想上非常接近（FSDP ≈ ZeRO-3 思路），但差异在系统生态与工程侧重点：

- **DeepSpeed**：系统更完整（ZeRO + offload + pipeline + inference 等），配置驱动，生态偏工程化
- **FSDP**：PyTorch 原生集成更自然，和 torch.compile、state_dict、PT2 等结合更直接

实际选型常看：
- 你是否需要 DeepSpeed 的 offload / pipeline / 现成工程能力
- 你是否更偏 PyTorch 原生与可维护性

---

## 12. 如果你希望“更落地”，我建议你补充这几项信息

我可以根据你的场景给出更针对性的 DeepSpeed 方案（stage 选择、关键参数、吞吐/显存预估、常见坑）：

1. 模型规模（参数量）与结构（Transformer 层数/hidden/heads）
2. GPU 型号与数量（单机还是多机，是否 NVLink/IB）
3. batch size、seq length、是否梯度累积
4. 使用的优化器（AdamW/Adafactor/SGD）
5. 目标：更省显存 vs 更高吞吐？

如果你愿意，我也可以下一步用一个“具体数值例子”说明：在 DDP、ZeRO-2、ZeRO-3 下，**每张 GPU 上参数/梯度/优化器状态分别占多少显存**，以及通信在一次迭代里发生在哪些点。
