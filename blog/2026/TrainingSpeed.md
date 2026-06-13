---
title: TrainingSpeed
date: 2026-01-01
categories: [Understandings]
---

# TrainingSpeed

# 大模型训练加速方法（聚焦FSDP与Gradient Checkpointing等系统级技术）

## 前言：重新定义讨论范畴

您在反馈中提到希望了解**FSDP、梯度检查点**这类技术，而非FlashAttention那样侧重算子内存访问优化的方法。两者的核心区别在于：

- **算子级优化**（如FlashAttention）：通过改进单次计算的内存访问模式来减少I/O，本质上不改变模型状态在分布式设备间的存储方式。
- **系统级并行与重计算技术**（如FSDP、梯度检查点）：通过**跨设备分片模型状态**或**牺牲计算换取显存**，从根本上改变训练过程的内存占用曲线，从而允许在给定硬件上运行更大规模的模型或更大的batch size，间接提升训练吞吐量。

本文将详细阐述这两大类技术的原理、数学基础和工程实现细节。

---

## 1. FSDP：全分片数据并行

### 1.1 从数据并行到全分片

**标准数据并行（DDP，DistributedDataParallel）** 是分布式训练的基础范式：每个GPU持有完整的模型副本，处理不同的输入数据子集，然后在反向传播后对梯度进行AllReduce同步。其内存占用公式为（以字节计）：

$$
M_{DDP} = \underbrace{2\Psi}_{\text{FP16参数+梯度}} + \underbrace{2\Psi}_{\text{FP32参数副本}} + \underbrace{K \cdot 2\Psi}_{\text{Adam状态}} \approx 16\Psi
$$

对于 $N$ 个GPU，集群总内存冗余高达 $16\Psi \times N$，而实际有效模型状态仅为 $16\Psi$。

**FSDP（Fully Sharded Data Parallel）** 是PyTorch原生实现的ZeRO-3等效技术，其核心思想是：**在数据并行的基础上，将模型参数、梯度和优化器状态在数据并行维度上进行分片（shard），每个GPU仅持有分片后的部分状态。** 需要计算时，通过AllGather通信临时重建完整参数，计算完成后立即释放非本地分片。

### 1.2 FSDP的工作流程

FSDP将每个Transformer层（或任意子模块）包装为一个`FlatParameter`，并将其在数据并行组内切分。一次完整的前向-反向传播过程如下：

1. **前向传播准备**：对于即将计算的层，FSDP触发AllGather通信，从所有GPU收集该层完整的参数分片，在本地拼接为完整参数。
2. **前向计算**：使用临时组装的完整参数执行该层的前向计算，生成激活值。
3. **参数释放**：前向计算完成后，立即释放非本地的参数分片（`free_full_weights`）。
4. **反向传播**：当梯度传播到该层时，再次触发AllGather重建完整参数，计算梯度。
5. **梯度分片归约**：计算得到的梯度先在本地按分片切分，然后执行ReduceScatter通信，使每个GPU仅保留其负责分片的聚合梯度。
6. **优化器更新**：每个GPU使用本地分片的梯度更新其对应的优化器状态和参数分片。

### 1.3 数学分析：内存与通信的权衡

设数据并行组大小为 $N$，模型参数总量为 $\Psi$。

**内存占用**：每个GPU仅保留 $\frac{1}{N}$ 的参数、梯度和优化器状态，因此单GPU模型状态内存为：

$$
M_{FSDP} \approx \frac{16\Psi}{N}
$$

对于175B参数的GPT-3，若使用64张A100（80GB），单卡模型状态内存从约2.8TB（不可行）降至约44GB，完全容纳于单卡显存中。

**通信开销**：每次前向和反向均需执行一次AllGather（收集完整参数）和一次ReduceScatter（归约梯度）。总通信量为：

- AllGather：每个GPU发送 $\frac{\Psi}{N}$，接收 $(N-1)\frac{\Psi}{N}$，总通信 $\Psi$。
- ReduceScatter：每个GPU发送 $\Psi$ 梯度，接收 $\frac{\Psi}{N}$ 归约后梯度，总通信 $\Psi$。
- 每次前向+反向总通信量：约 $2\Psi$（参数AllGather） + $2\Psi$（梯度ReduceScatter） = $4\Psi$。

与标准DDP相比，DDP仅在反向结束时做一次AllReduce（通信量 $2\Psi$），FSDP的通信量增加了约一倍。**因此FSDP本质是用额外的通信开销换取显存的急剧降低**，适合显存受限场景。

### 1.4 FSDP的优化策略

#### 1.4.1 混合精度与分片粒度

FSDP支持将参数保持在低精度（如BF16）用于前向/反向计算，而将分片的优化器状态保持在FP32。进一步，可以设置`sharding_strategy`控制分片范围：

- `FULL_SHARD`：ZeRO-3，参数、梯度、优化器状态全分片。
- `SHARD_GRAD_OP`：ZeRO-2，仅分片梯度和优化器状态。
- `NO_SHARD`：标准DDP。

#### 1.4.2 通信与计算重叠

FSDP提供`limit_all_gathers=True`选项，允许在反向传播期间预取下一层的参数AllGather操作，使其与当前层的计算重叠，隐藏通信延迟。

#### 1.4.3 CPU Offload

当GPU显存仍不足时，可将优化器状态或参数分片卸载到CPU内存（`cpu_offload`），进一步扩展模型容量。

### 1.5 FSDP vs DeepSpeed ZeRO-3

| 特性 | PyTorch FSDP | DeepSpeed ZeRO-3 |
|------|--------------|------------------|
| 原生集成 | PyTorch核心库，无需额外依赖 | 需安装DeepSpeed库 |
| API风格 | 与`nn.Module`包装器结合 | 配置文件驱动或API调用 |
| 性能 | 近年在原生支持上优化显著，接近ZeRO-3 | 长期优化，通信内核高效 |
| 灵活性 | 更紧密地与PyTorch其他特性（如`torch.compile`）协同 | 提供更丰富的Offload选项和MoE支持 |

两者算法等价，实际选择取决于团队技术栈偏好。

---

## 2. 梯度检查点（Gradient Checkpointing）

### 2.1 激活值内存问题

在深度神经网络的反向传播中，需要保存前向传播产生的中间激活值（activations）以计算梯度。对于Transformer模型，激活值的内存占用量常超过模型参数本身。

考虑一个 $L$ 层的Transformer，隐藏维度 $d$，序列长度 $S$，batch size $B$。单层激活值大小约为 $O(B \cdot S \cdot d)$。若使用标准训练，需保存所有 $L$ 层的激活值，内存占用为 $O(L \cdot B \cdot S \cdot d)$。对于GPT-3规模（$L=96, d=12288$），激活值内存可达数百GB。

### 2.2 基本思想：计算换内存

**梯度检查点**的核心是：**在前向传播时仅保存部分层的激活值作为“检查点”（checkpoint），在反向传播需要其他层的激活值时，从最近的检查点开始重新计算前向过程。** 通过增加约33%的前向计算量，将激活值内存复杂度从 $O(L)$ 降低至 $O(\sqrt{L})$（最优检查点放置策略）或 $O(1)$（每层都重算的极端情况）。

### 2.3 数学描述与实现

设模型由 $L$ 个顺序模块 $f_1, f_2, \dots, f_L$ 组成，输入为 $x_0$：

$$
x_i = f_i(x_{i-1}), \quad i = 1, \dots, L
$$

损失函数 $\mathcal{L}$ 对输入 $x_0$ 的梯度可通过链式法则计算：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_{L-1}} \cdots \frac{\partial x_1}{\partial x_0}
$$

计算 $\frac{\partial x_i}{\partial x_{i-1}}$ 需要中间激活值 $x_{i-1}$ 和模块参数。标准训练中 $x_{i-1}$ 已在内存中。梯度检查点策略如下：

- **前向传播时**：只对指定的检查点层保存 $x_i$，其余层的 $x_i$ 在计算后立即丢弃。
- **反向传播时**：从最近的检查点 $x_k$ 开始，重新执行 $f_{k+1}, \dots, f_i$ 来恢复所需的 $x_i$，然后计算梯度。

### 2.4 检查点放置策略

#### 2.4.1 均匀放置
将 $L$ 层等分为 $\sqrt{L}$ 个段（segment），每段包含 $\sqrt{L}$ 层。在每个段的边界设置检查点。前向传播保存 $\sqrt{L}$ 个检查点，反向时每段内最多重算 $\sqrt{L}$ 层，总重计算量为 $O(\sqrt{L} \cdot \sqrt{L}) = O(L)$，即额外一倍的前向计算。

#### 2.4.2 递归放置（树形检查点）
通过递归二分策略，可以将内存复杂度降至 $O(\log L)$，但实现复杂且重计算量稍大。

#### 2.4.3 PyTorch实现
PyTorch提供`torch.utils.checkpoint.checkpoint`函数，对指定的模块或函数进行包装：

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # 将计算量大的部分用checkpoint包裹
    x = checkpoint(self.layer1, x, use_reentrant=False)
    x = checkpoint(self.layer2, x, use_reentrant=False)
    return x
```

参数`use_reentrant=False`启用非重入式检查点，在PyTorch 2.0+中性能更佳。

### 2.5 选择性检查点

并非所有层对内存的贡献相同。实际应用中可**选择性**地对内存占用大、计算量相对较小的层启用检查点。典型实践：

- **Transformer层中的MLP块**：激活值大，计算量中等，启用检查点。
- **Attention块**：激活值小（若已用FlashAttention），可不启用。
- **LayerNorm / Dropout**：内存占用极小，无需检查点。

在Llama等模型的训练中，通常对每个Transformer层的MLP部分启用检查点，而Attention部分保持全激活值，平衡内存与速度。

### 2.6 梯度检查点与FSDP的协同

FSDP已通过参数分片大幅降低模型状态内存，此时激活值内存成为新的瓶颈。**结合FSDP与梯度检查点**是当前千亿参数模型训练的标配：

- **FSDP**：解决参数、梯度、优化器状态的内存瓶颈。
- **梯度检查点**：解决激活值的内存瓶颈。

两者结合后，训练内存主要由 $\frac{16\Psi}{N}$（模型状态）和 $\sqrt{L} \cdot B \cdot S \cdot d$（检查点激活值）决定，使得在普通GPU集群上训练千亿参数模型成为现实。

---

## 3. 混合并行策略的系统视图

当模型规模继续增长，单一数据并行（即便全分片）也无法容纳时，需引入**模型并行**的多个维度。

### 3.1 张量并行（Tensor Parallelism, TP）

将单个Transformer层的权重矩阵按列或行切分到多个GPU上，每个GPU计算部分结果后通过通信合并。典型实现如Megatron-LM：

- 对于Attention层的 $Q,K,V$ 投影，按列切分权重，每个GPU计算部分头的投影。
- 对于MLP的第一层按列切分，第二层按行切分。

TP的通信量较大，要求GPU间具有高带宽互联（如NVLink）。

### 3.2 流水线并行（Pipeline Parallelism, PP）

将模型的不同层组分配到不同GPU上，形成流水线。通过将batch拆分为多个micro-batch，实现计算与通信的重叠。典型调度算法：

- **GPipe**：所有micro-batch前向完成后统一反向，激活值内存高。
- **1F1B（One-Forward-One-Backward）** ：交替执行前向和反向，降低激活值峰值。

### 3.3 3D并行：TP+PP+FSDP

现代大模型训练的标准配置是三者结合：

- **张量并行**：在单机多卡内使用，利用高带宽NVLink。
- **流水线并行**：跨节点使用，利用IB/RoCE网络。
- **FSDP/ZeRO-3**：在流水线并行的每个stage内部，再对数据并行组进行全分片，进一步降低内存。

通过合理配置，可在数千GPU上线性扩展训练吞吐量。

---

## 4. 总结与对比

| 技术 | 核心思想 | 内存节省 | 额外计算/通信 | 适用场景 |
|------|----------|----------|---------------|----------|
| **FSDP** | 模型状态跨数据并行分片 | 约 $N$ 倍（$N$为GPU数） | 增加约2倍通信量 | 所有规模分布式训练 |
| **梯度检查点** | 重算激活值而非存储 | 激活值内存降至 $O(\sqrt{L})$ | 增加约33%前向计算 | 深层Transformer必选 |
| **张量并行** | 层内权重切分 | 单层内存降低，但通信量极大 | 每层两次AllReduce | 单机多卡超大规模模型 |
| **流水线并行** | 层间切分形成流水线 | 模型参数内存分摊 | 流水线气泡，需micro-batch填充 | 跨节点极深模型 |

**最终实践范式**：

- 对于10B~100B模型：**FSDP + 梯度检查点 + BF16混合精度** 可在单机8卡A100/H100上完成。
- 对于100B~1T模型：在上述基础上增加 **流水线并行（跨节点）**。
- 对于1T以上模型：进一步引入 **张量并行** 和 **专家并行**（MoE）。

FSDP与梯度检查点作为基础组件，已成为大模型训练栈中不可或缺的标配技术。理解其数学原理与实现细节，有助于在实际工程中合理权衡内存、计算与通信，实现最优的训练效率。
