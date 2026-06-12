---
title: 3DpointCloudGenerative
date: 2023-10-10
---

# 3DpointCloudGenerative

# Copilot4D 训练流程详解

### 整体架构概览

Copilot4D 的训练分为两个阶段：**Tokenizer 训练**和**World Model 训练**。整体流程是先将点云观测token化，然后在离散token空间上应用离散扩散模型预测未来。

---

## 1. Tokenizer 训练

### 输入数据维度

- **输入**: 点云观测，覆盖区域为 $[-80\text{m}, 80\text{m}] \times [-80\text{m}, 80\text{m}] \times [-4.5\text{m}, 4.5\text{m}]$
- **体素大小**: $15.625\text{cm} \times 15.625\text{cm} \times 14.0625\text{cm}$
- **初始特征体积**: $1024 \times 1024 \times 64 \times 64$ (经过初始 PointNet 编码后)
- **BEV 特征图**: $128 \times 128 \times 256$ (经过 Swin Transformer 编码和下采样后)
- **输出**: $128 \times 128$ 个离散 token，每个 token 是 codebook 中的一个索引 [Tokenizer Architecture](https://alphaxiv.org/abs/2311.01017?page=16)

### 训练目标

Tokenizer 的损失函数由两部分组成:

$$\mathcal{L} = \mathcal{L}_{\text{vq}} + \mathcal{L}_{\text{render}}$$

**向量量化损失** $\mathcal{L}_{\text{vq}}$:
$$\mathcal{L}_{\text{vq}} = \lambda_1 \|\text{sg}[E(o)] - \hat{z}\|_2^2 + \lambda_2 \|\text{sg}[\hat{z}] - E(o)\|_2^2$$

其中 $\lambda_1 = 0.25$, $\lambda_2 = 1.0$ [VQ Loss](https://alphaxiv.org/abs/2311.01017?page=17)

**渲染损失** $\mathcal{L}_{\text{render}}$:
$$\mathcal{L}_{\text{render}} = \mathbb{E}_r \left[ \|D(r, \hat{z}) - D_{\text{gt}}\|_1 + \sum_i \mathbb{1}(|h_i - D_{\text{gt}}| > \epsilon) \|w_i\|_2 \right] + \text{BCE}(v, v_{\text{gt}})$$

包含深度渲染的 L1 损失和空间跳跃分支的二元交叉熵损失 [Rendering Loss](https://alphaxiv.org/abs/2311.01017?page=5)

### 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 0.001 |
| Warmup | 4000 迭代 (线性) |
| Batch Size | 16 |
| 最大梯度范数 | 0.1 |
| 衰减策略 | Cosine decay (0.4M 迭代) |

---

## 2. World Model 训练

### 输入输出维度

- **输入序列**: 过去 $T_{\text{past}}$ 帧的 token + 未来 $T_{\text{future}}$ 帧的 token
- **每帧 token 数**: $128 \times 128 = 16384$ 个离散 token
- **时间维度**:
  - **NuScenes**: 1s 预测 = 2 帧过去 + 2 帧未来; 3s 预测 = 6 帧过去 + 6 帧未来
  - **KITTI/Argoverse2**: 1s/3s 预测 = 5 帧过去 + 5 帧未来 [Dataset Settings](https://alphaxiv.org/abs/2311.01017?page=8)

### 离散扩散算法

Copilot4D 将 MaskGIT 改进为离散扩散模型，关键改进如下:

**训练算法 (Algorithm 1)**:
1. 从数据分布中采样 $x_0 \in \{1, \cdots, |\mathcal{V}|\}^N$
2. 随机 mask 掉 $\lceil \gamma(u_0)N \rceil$ 个 token ($\gamma(u) = \cos(u\pi/2)$)
3. 对剩余 token 注入最多 $\eta\% = 20\%$ 的均匀噪声
4. 用交叉熵损失重建 $x_0$ [Training Algorithm](https://alphaxiv.org/abs/2311.01017?page=5)

**采样算法 (Algorithm 2)**:
1. 从全 mask token 开始 $x_K$
2. 迭代 $K-1 \to 0$:
   - 预测 $\tilde{x}_0 \sim p_\theta(\cdot | x_{k+1})$
   - 计算 logit: $l_k = \log p_\theta(\tilde{x}_0 | x_{k+1}) + \text{Gumbel}(0, 1) \cdot k/K$
   - 在非 mask 位置设 $l_k \leftarrow +\infty$
   - 选择 top-$M$ 个位置解码 ($M = \lceil \gamma(k/K)N \rceil$) [Sampling Algorithm](https://alphaxiv.org/abs/2311.01017?page=5)

### 混合训练目标

World Model 使用三种训练目标的混合 (见 Figure 4):

| 目标类型 | 比例 | 描述 |
|---------|------|------|
| 1. 条件未来预测 | 50% | 给定过去，去噪未来 |
| 2. 联合建模 | 40% | 联合去噪过去和未来 (更难的预训练任务) |
| 3. 单帧建模 | 10% | 独立去噪每帧 (用于 classifier-free guidance) [Training Objectives](https://alphaxiv.org/abs/2311.01017?page=6) |

数学形式为最大化:
$$\mathbb{E}_{q(\tau), k_1,\cdots,k_T \sim \text{SampleObj}(\cdot)} \left[ \log p_\theta(x_0^{(1)}, \cdots, x_0^{(T)} | x_{k_1}^{(1)}, \cdots, x_{k_T}^{(T)}, a^{(1)}, \cdots, a^{(T-1)}) \right]$$

### World Model 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 0.001 |
| Warmup | 2000 迭代 (线性) |
| Batch Size | 8 |
| 最大梯度范数 | 5.0 |
| 衰减策略 | Cosine decay (0.75M 迭代) |
| 标签平滑 | 0.1 |
| 优化器 | AdamW ($\beta_1=0.9$, $\beta_2=0.95$) |
| 权重衰减 | 0.0001 (bias/LayerNorm 除外) [Training Hyperparameters](https://alphaxiv.org/abs/2311.01017?page=20) |

---

## 3. 推理流程

推理时使用**自回归预测** + **Classifier-Free Diffusion Guidance (CFG)**:

1. 在每一步 $t$, 给定过去历史 $c^{(t-1)}$
2. 用 Algorithm 2 采样下一帧，每帧 10 个扩散步
3. CFG 放大条件信息: $\text{logits}_{\text{CFG}} = \text{logits}_{\text{cond}} + w \cdot (\text{logits}_{\text{cond}} - \text{logits}_{\text{uncond}})$
4. 实验表明 $w=2.0$ 效果最佳，Chamfer 距离降低 60% [CFG Results](https://alphaxiv.org/abs/2311.01017?page=9)

---

## 关键创新点总结

1. **Tokenization**: 用 VQVAE 将复杂点云空间转换为离散 token 序列
2. **离散扩散改进**: 在 MaskGIT 基础上增加噪声注入和 token 重采样机制，Chamfer 距离降低 29%
3. **混合训练目标**: 联合建模过去和未来，提升长视野推理能力
4. **并行解码**: 每扩散步并行解码约 1600 个 token，远快于自回归解码 [Parallel Decoding](https://alphaxiv.org/abs/2311.01017?page=9)
