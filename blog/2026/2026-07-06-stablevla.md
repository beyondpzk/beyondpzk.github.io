---
title: StableVLA：不增加数据，把 VLA 的视觉鲁棒性提升 30%
date: 2026-07-06
categories: [VLA]
---

# StableVLA：不增加数据，把 VLA 的视觉鲁棒性提升 30%

> **论文**：*StableVLA: Towards Robust Vision-Language-Action Models without Extra Data*  
> **作者**：Yiyang Fu, Chubin Zhang, Shukai Gong, Yufan Deng, Kaiwei Sun 等（PKU、Tsinghua、Astribot、NJU、Nankai）  
> **发表会议**：ICML 2026  
> **arXiv**：https://arxiv.org/abs/2605.18287  
> **项目主页**：https://dagroup-pku.github.io/StableVLA/  
> **代码与权重**：https://github.com/DAGroup-PKU/HumanNet/tree/main/src/model/StableVLA

---

## 摘要

StableVLA 是一项针对 **Vision-Language-Action（VLA）模型视觉鲁棒性** 的研究。作者发现，当前 SOTA VLA 模型在干净环境下游刃有余，但一旦遇到训练时未见过的视觉扰动（噪声、模糊、雾气、遮挡等），性能会断崖式下跌——某些严重腐败下甚至从 96% 成功率掉到 0%。

为此，论文提出 **Information Bottleneck Adapter（IB-Adapter）**，一个基于信息瓶颈理论的轻量级适配器：
- **无需额外数据、无需数据增强**，直接替换 VLA 中的投影模块即可；
- 仅增加 **<10M 参数**；
- 在合成腐败和真实机器人扰动上平均提升约 **30%**；
- 用 **0.5B 参数** 的 StableVLA 就能在鲁棒性上比肩 **7B 级 OpenVLA-OFT** 和 **3B 级 OpenPi-0.5**。

---

## 一、问题背景：为什么 VLA 需要「稳定」？

VLA 模型通过把预训练 VLM 的视觉-语言表征对齐到机器人动作空间，实现了跨任务、跨本体的泛化。然而，现有基准（LIBERO、CALVIN）大多在**理想视觉条件**下评估，而真实世界 deployment 中难免遇到：

- 传感器噪声、运动模糊；
- 雾气、雨滴、强光、昏暗；
- 镜头油污、遮挡、灰尘；
- 其他训练分布外的视觉扰动。

论文作者首先对 VLA-Adapter 做了压力测试：在 LIBERO 上原本成功率 **96%** 的模型，加入 ImageNet-C 风格的视觉腐败后，严重情况下直接掉到 **0%**。更令人担忧的是，这种脆弱性并非 VLA-Adapter 独有，OpenVLA、OpenVLA-OFT、OpenPi-0.5 等主流模型都出现明显下降。

传统提升鲁棒性的做法有两种：
1. **数据增强**：把各种腐败加到训练集里；
2. **海量预训练**：用 Open X-Embodiment 等大数据集「见多识广」。

但两者都有明显缺陷：
- 真实世界的腐败组合是无限的，增强无法覆盖全部；
- 模型容易记住增强模式，而不是学到不变特征；
- 大数据预训练成本极高，小团队难以复现。

因此作者提出一个核心问题：**能不能不依赖额外数据，仅靠架构设计让 VLA 模型 inherently robust？**

---

## 二、关键发现：脆弱性来自「投影模块」

作者对 VLA-Adapter 的各层特征进行了可视化分析，发现视觉腐败导致的特征退化主要集中在**连接视觉编码器和 LLM 的 projector（适配器/投影模块）** 上。

现有 VLA 通常：
- **冻结视觉编码器**以保持语义先验；
- 使用简单的 **MLP projector** 把视觉特征映射到 LLM 输入空间。

从信息瓶颈（Information Bottleneck, IB）的角度看，MLP 这类投影器相当于一个**全通滤波器**：它无差别地最大化输入特征与输出表征之间的互信息 `I(X;Z)`，结果把任务相关的语义和任务无关的噪声一起传给了下游策略。

这启发了 StableVLA 的核心思路：**在投影阶段引入信息瓶颈，显式过滤噪声通道，只保留对动作预测有用的语义信息。**

---

## 三、方法：IB-Adapter 与 Fused IB-Adapter

### 3.1 信息瓶颈视角下的模态对齐

标准 VLA 的模态对齐可以写成：

```
I → E(I) = X_v ∈ R^{N×D_v}
X_v → φ(X_v) = Z ∈ R^{N×D}
Z + T → LLM → a
```

其中 `E` 是视觉编码器，`φ` 是投影器，`T` 是文本指令，`a` 是动作。

StableVLA 把投影过程建模为信息瓶颈优化问题：

```
min_{φ(Z|X_v)}  L_IB = I(X_v; Z) - β·I(Z; S)
```

- `I(X_v; Z)`：投影后的表征保留输入信息的程度；
- `I(Z; S)`：投影后的表征与「干净语义 `S`」的互信息；
- `β`：控制压缩与保留之间的 trade-off。

直观理解：我们要找一个**既紧凑、又与任务语义高度相关**的表征 `Z`，把噪声和冗余信息挤出去。

### 3.2 IB-Adapter：通道级的噪声过滤

IB-Adapter 把特征压缩从常见的「空间维度」搬到了「通道维度」。它假设：
- 视觉编码器输出的不同通道携带不同语义；
- 噪声通道与语义通道之间的协方差很低；
- 通过通道级注意力可以抑制噪声、放大语义。

具体实现分为三步：

#### 1. 子空间协方差建模（Multi-head Covariance）

输入特征 `X' ∈ R^{N×D}` 被分成 `H` 个头，每个头有 `d = D/H` 个通道。对每个头计算 Gram 矩阵：

```
G_h = Q_h^T K_h ∈ R^{d×d}
```

其中 `Q_h = X'_h W_q`，`K_h = X'_h`（identity key，保留原始几何结构）。`G_h[i,j]` 表示通道 `i` 和通道 `j` 在整个空间 token 上的协方差。

#### 2. Sigmoid 门控（Subspace Gating）

对 Gram 矩阵做可学习的 Sigmoid 门控：

```
A_h = σ(G_h · τ_h) ∈ [0,1]^{d×d}
```

`τ_h` 是可学习温度。与 Softmax 不同，Sigmoid 不对通道做竞争性归一化，而是**独立地打开/关闭每个通道**。这正好对应信息瓶颈中「独立 Bernoulli 潜在结构」的假设：噪声通道与语义通道协方差低，门控值接近 0，被自动抑制。

#### 3. 非线性特征变换

输入特征先经过两层 GELU MLP 得到 value 特征 `V_h`，再与门控图 `A_h` 相乘完成重构：

```
V_h = Norm(GELU(X_h W_v1) W_v2)
Z_h = V_h A_h
```

最终把所有头的输出拼接，得到过滤后的视觉表征。

### 3.3 Fused IB-Adapter：鲁棒语义 + 高频细节

纯 IB-Adapter 虽然能过滤噪声，但也可能抑制对精细操作至关重要的高频空间细节。为了兼顾两者，StableVLA 提出 **Fused IB-Adapter**：

```
Z = MLP(X) + tanh(λ) · IB-Adapter(X)
```

- **MLP 通路**：保留原始高保真特征，负责精细的空间定位与动作执行；
- **IB-Adapter 通路**：提取鲁棒的语义特征，负责抗干扰；
- **λ**：可学习的融合系数。

为了进一步让模型学会依赖鲁棒特征，作者在 fine-tuning 时引入 **Stochastic Pathway Dropout（SPD）**：以概率 `p_drop` 随机丢弃 MLP 通路，迫使策略从 IB-Adapter 通路学习。不同任务对 SPD 的需求不同：
- **长程精细操作任务**（如 LIBERO-Long）：保留 MLP 通路，`p_drop ≈ 0`；
- **需要稳定语义识别的任务**（如 CALVIN、LIBERO-Object）：适度 dropout，`p_drop ≈ 0.3`。

---

## 四、训练与实现

### 4.1 训练流程

StableVLA 基于 VLA-Adapter 框架，把标准 MLP projector 替换为 Fused IB-Adapter，训练分为两个阶段：

1. **VLM Alignment 阶段**：用 LLaVA-LVIS4V-LRV 数据集训练 Fused IB-Adapter projector，使其与 LLM 输入空间对齐；
2. **机器人 Fine-tuning 阶段**：在 LIBERO 或 CALVIN 上微调，适配具体任务。

**关键细节**：训练过程中**不加入任何视觉腐败或鲁棒性增强**，只在测试时 zero-shot 评估腐败场景，因此所有提升都来自架构本身。

### 4.2 开源实现

论文已完全开源代码和权重：

```bash
# 环境
git clone https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet/src/model/StableVLA
pip install -e .

# VLM 预训练（可选，可直接下载 HF 权重）
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "prism-qwen25-extra-dinosiglip-224px+0_5b+fusedfan-projector" \
  --dataset.type "llava-lvis4v-lrv" ...

# LIBERO fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  vla-scripts/finetune_fused.py \
  --vlm_path pretrained_models/stablevla-fusedfan-projector \
  --dataset_name libero_spatial_no_noops \
  --use_lora True --lora_rank 64 --learning_rate 2e-4 ...
```

模型权重托管在 HuggingFace：`DAGroup-PKU/StableVLA`（GitHub README 中也有 `beikui12345/stablevla` 的下载链接）。

### 4.3 显存配置参考

| 显存 | 推荐配置 |
|---|---|
| 10–12 GB（如 RTX 3080） | `--batch_size 1 --grad_accumulation_steps 16` |
| 24 GB（如 RTX 3090/4090） | `--batch_size 4 --grad_accumulation_steps 4` |
| 40–80 GB（如 A100/H100） | `--batch_size 16 --grad_accumulation_steps 1` |

---

## 五、实验结果

### 5.1 基准设置

- **LIBERO**：Spatial / Object / Goal / Long 四个任务套件，每个套件 10 个子任务，每个子任务评估 50 个 episode；
- **CALVIN**：在训练时未见过的环境上执行 1000 个连续任务链，每个任务 5 个子任务；
- **腐败协议**：采用 ImageNet-C 的 19 种腐败类型，评估 Severity 3/4/5；
- **Baseline**：OpenVLA-7B、OpenVLA-OFT-7B、OpenPi-0.5-3B、VLA-Adapter-0.5B。

### 5.2 LIBERO 与 CALVIN 结果

| 训练范式 | 方法 | 参数量 | Spatial C/S3/S4/S5 | Object C/S3/S4/S5 | Goal C/S3/S4/S5 | Long C/S3/S4/S5 | CALVIN C/S3/S4/S5 |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenX 预训练 | OpenVLA | 7B | 80.0 / 40.9 / 24.6 / 14.7 | 69.6 / 18.2 / 10.4 / 2.7 | 74.0 / 38.7 / 27.0 / 16.3 | 55.5 / 20.5 / 12.4 / 7.0 | – |
| OpenX 预训练 | OpenVLA-OFT | 7B | 92.6 / 89.3 / 84.0 / 72.1 | 98.4 / 82.5 / 69.2 / 52.8 | 96.8 / 94.5 / 84.6 / 70.3 | 94.4 / 77.6 / 61.9 / 40.3 | – |
| OpenX+Web 共训 | OpenPi-0.5 | 3B | 98.4 / 88.3 / 79.0 / 62.4 | 99.4 / 97.1 / 88.4 / 76.4 | 97.2 / 87.2 / 82.5 / 64.2 | 92.0 / 76.1 / 65.6 / 47.7 | – |
| VLM 直接 FT | VLA-Adapter | 0.5B | 96.0 / 93.7 / 83.3 / 58.5 | 96.8 / 71.0 / 44.1 / 29.3 | 97.4 / 79.5 / 64.7 / 47.3 | 94.4 / 63.5 / 41.0 / 26.2 | 4.14 / 2.56 / 1.89 / 1.44 |
| VLM 直接 FT | **StableVLA** | **0.5B** | **96.2 / 94.4 / 92.1 / 82.0** | **98.8 / 92.4 / 83.6 / 70.2** | **98.0 / 93.4 / 85.0 / 71.9** | **93.6 / 76.3 / 62.4 / 45.3** | **4.17 / 2.77 / 2.11 / 1.51** |

几点观察：
- **干净数据上基本持平**：StableVLA 没有因为引入压缩而牺牲 clean 性能；
- **腐败场景下优势巨大**：在 Severity 5 下，StableVLA 相比 VLA-Adapter 在四个 LIBERO 套件上分别提升 **40.2%–139.6%**；
- **以小博大**：0.5B 的 StableVLA 在严重腐败下的表现接近甚至超过 7B OpenVLA-OFT 和 3B OpenPi-0.5。

### 5.3 真实机器人部署

作者在 Astribot S1 双臂机器人上验证了 StableVLA 的真实世界鲁棒性，设计了四个任务：Pick and Place、Throw Basketball、Pour Water、Pack Doll。腐败类型包括：
- 数字腐败：高斯噪声、失焦模糊；
- 物理腐败：镜头涂油（Oil）、塑料遮挡（Shelter）。

| 任务 | 方法 | Clean | Noise Δ | Blur Δ | Oil Δ | Shelter Δ | Avg Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Pick and place | π0.5 | 100.0 | -63.3 | -16.7 | -10.0 | -30.0 | -30.1 |
| Pick and place | VLA-Adapter | 80.0 | -66.7 | -40.0 | -30.0 | -60.0 | -49.2 |
| Pick and place | **StableVLA** | 80.0 | **-30.0** | **-10.0** | **-10.0** | **-20.0** | **-17.5** |
| Throw basketball | π0.5 | 80.0 | -60.0 | -33.3 | -20.0 | -30.0 | -35.8 |
| Throw basketball | VLA-Adapter | 60.0 | -53.0 | -40.0 | -20.0 | -40.0 | -38.3 |
| Throw basketball | **StableVLA** | 60.0 | **-36.7** | **-16.7** | **-10.0** | **-10.0** | **-18.4** |
| Pour water | π0.5 | 70.0 | -60.0 | -20.0 | -20.0 | -20.0 | -30.0 |
| Pour water | VLA-Adapter | 40.0 | -40.0 | -30.0 | -10.0 | -20.0 | -25.0 |
| Pour water | **StableVLA** | 40.0 | **-23.3** | **-16.7** | **-0.0** | **-10.0** | **-12.5** |
| Pack doll | π0.5 | 80.0 | -63.3 | -33.3 | -30.0 | -40.0 | -41.7 |
| Pack doll | VLA-Adapter | 50.0 | -40.0 | -26.7 | -30.0 | -30.0 | -31.7 |
| Pack doll | **StableVLA** | 60.0 | **-16.7** | **-10.0** | **-20.0** | **-10.0** | **-14.2** |

StableVLA 在所有任务上都表现出**最小的性能下降**，尤其在物理遮挡和镜头油污这种真实场景中优势明显。

---

## 六、消融实验

| 架构 | LIBERO Clean | LIBERO Avg 腐败 | CALVIN Clean | CALVIN Avg 腐败 |
|---|---:|---:|---:|---:|
| IB-Adapter | 96.3 | 76.0 | 1.64 | 1.44 |
| Fused IB-Adapter | 96.6 | 79.1 | 4.17 | 2.13 |
| Fused IB-Adapter + Softmax | 89.6 | 62.8 | 0.46 | 0.46 |

消融表明：
1. **双通路必不可少**：去掉 MLP 通路后 CALVIN 性能从 2.13 降到 1.44；
2. **Sigmoid 显著优于 Softmax**：Softmax 强制通道竞争，破坏了独立噪声过滤机制，性能大幅下滑；
3. **Fused IB-Adapter 同时保留了 clean 性能和腐败鲁棒性**。

---

## 七、可迁移性：作为即插即用模块

IB-Adapter 是一个**模型无关的 projector 替换方案**。论文不仅在 VLA-Adapter 上验证，还在 OpenVLA 等模型上进行了测试，发现替换 projector 后一致提升。这意味着：
- 如果你已经在某个 VLA 上训练好了模型，可以只换 projector 做二次微调；
- 不需要重新收集机器人数据；
- 参数量增加 <10M，几乎可以忽略。

---

## 八、局限与未来方向

### 8.1 局限

1. **任务范围**：当前主要在 LIBERO、CALVIN 等桌面操作任务上验证，尚未覆盖移动机器人、导航、多机协同等场景；
2. **腐败类型**：虽然覆盖了 ImageNet-C 的 19 种腐败，但真实世界的光照变化、动态遮挡、快速运动模糊等仍需进一步验证；
3. **严重腐败仍有下降**：即使 StableVLA 在 Severity 5 下远强于 baseline，但相比 clean 仍有明显性能损失，说明问题并未完全解决；
4. **长程任务的 trade-off**：Fused IB-Adapter 的两个通路需要针对任务调 `p_drop`，工程上增加了超参搜索成本。

### 8.2 未来方向

1. **把 IB-Adapter 扩展到更大规模的 VLA**：验证其在 7B+ 模型上的收益；
2. **结合在线自适应**：在 deployment 时根据当前视觉质量动态调整门控；
3. **多模态输入**：把 IB 思想推广到深度图、点云、触觉等额外传感器；
4. **与数据增强结合**：虽然论文强调无需额外数据，但 IB-Adapter 与轻量增强可能是互补的。

---

## 九、总结

StableVLA 的核心贡献在于：**用架构设计解决 VLA 的视觉鲁棒性问题，而不是靠堆数据**。它揭示了 VLA 脆弱性的一个重要来源——视觉-语言投影模块，并提出基于信息瓶颈的 IB-Adapter 来显式过滤噪声。

通过 Fused IB-Adapter，StableVLA 在仅增加 <10M 参数的情况下，同时保留了精细操作所需的高频细节和面对视觉扰动时的语义稳定性。0.5B 参数的版本就能在多个基准上媲美甚至超过 3B–7B 的 SOTA 模型，为资源受限的真实机器人部署提供了一个极具吸引力的方案。

对于正在做 VLA 落地的研究者或工程师来说，StableVLA 提供了一个值得尝试的**即插即用模块**：只改 projector，不改数据，可能就能让模型在现实世界的「不完美」视觉条件下稳定很多。

---

*参考：StableVLA: Towards Robust Vision-Language-Action Models without Extra Data (ICML 2026, arXiv:2605.18287)*
