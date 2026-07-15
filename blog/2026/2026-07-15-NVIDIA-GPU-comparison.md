---
title: NVIDIA主流AI显卡价格与算力对比
date: 2026-07-15
categories: [others]
---

# NVIDIA主流AI显卡价格与算力对比

> 调研日期：2026-07-15  
> 说明：价格为公开市场参考价（云厂商按需/二手/零售），实际成交价因供应链、批量采购、地区限制（如出口管制型号）差异较大。算力以 Tensor Core 峰值为主，仅供参考。

---

## 一、快速对比表

| GPU | 架构 | 显存 | 显存带宽 | FP16/BF16 (Tensor) | FP8 (Tensor) | TDP | 形态 | 参考价格区间 |
|-----|------|------|----------|---------------------|--------------|-----|------|--------------|
| **L4** | Ada Lovelace | 24 GB GDDR6 | 300 GB/s | 121 TFLOPS | 242 TFLOPS | 72 W | PCIe | $0.29–0.44 / 时 |
| **L20** | Ada Lovelace | 48 GB GDDR6 | 864 GB/s | 239 TFLOPS | 478 TFLOPS | 275 W | PCIe | $0.80–1.10 / 时 |
| **L40S** | Ada Lovelace | 48 GB GDDR6 | 864 GB/s | 362 TFLOPS | 724 TFLOPS | 350 W | PCIe | $0.85–1.50 / 时 |
| **A100 80GB** | Ampere | 80 GB HBM2e | ~2.0 TB/s | 312 / 624 TFLOPS | 不支持 | 300–400 W | PCIe / SXM | $1.49–2.50 / 时；购置 $8K–17K |
| **H800** | Hopper | 80 / 94 GB HBM2e/HBM3 | 2.0–3.9 TB/s | ~989 / 1979 TFLOPS | ~1979 / 3958 TFLOPS | 350–400 W | PCIe | 中国市场为主，约等同或略低于 H100 |
| **H100 SXM5** | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 / 1979 TFLOPS | 1979 / 3958 TFLOPS | 700 W | SXM5 | $2.50–3.95 / 时；购置 $25K–30K |
| **H200 SXM5** | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 / 1979 TFLOPS | 1979 / 3958 TFLOPS | 700 W | SXM5 | $2.39–4.00 / 时；购置 $30K–40K |
| **B300** | Blackwell Ultra | 288 GB HBM3e | ~8.0 TB/s | ~2250 TFLOPS | ~4500+ TFLOPS；FP4 ~15 PFLOPS | 1000–1400 W | SXM | $5.65–9+ / 时；购置 $30K–40K |

数据来源：[InferenceBench L20](https://inferencebench.io/gpus/nvidia-l20/)、[getdeploying H800](https://getdeploying.com/gpus/nvidia-h800)、[server-parts B300](https://www.server-parts.eu/post/nvidia-b300-gpu-specs)、[spheron H100](https://www.spheron.network/blog/nvidia-h100-specs/)、[spheron H200](https://www.spheron.network/blog/nvidia-h200-specs/)、[vessl GPU guide](https://vessl.ai/en/blog/gpu-workload-guide-en)。

---

## 二、按定位分类

### 2.1 推理与边缘：L4 / L20 / L40S

这三张卡都采用 **Ada Lovelace** 架构，面向推理、视频、图形渲染和中轻量级微调。

- **L4**：24 GB 显存，功耗仅 72 W，适合小模型推理（<13B）、视频转码、边缘部署。单卡成本低，但无 NVLink，多卡扩展受限。
- **L20**：48 GB GDDR6，FP8 478 TFLOPS，TDP 275 W。性价比高于 L40S，是 70B 级模型 INT4/FP8 单卡推理的常见选择。
- **L40S**：48 GB GDDR6，算力比 L20 高约 50%（FP8 724 TFLOPS），TDP 350 W。适合 Vision-Language 模型、AIGC 推理和小规模 LoRA 微调。

共同特点：
- 使用 **GDDR6**，显存容量和带宽远低于 HBM 系列。
- 仅支持 **PCIe**，多卡训练时 GPU 间通信是瓶颈。
- 不支持 MIG（Multi-Instance GPU）。

### 2.2 上一代训练主力：A100

A100 是 2020–2023 年 AI 训练的事实标准。

- 80 GB HBM2e，~2.0 TB/s 带宽。
- FP16/BF16 312 TFLOPS（dense），支持 TF32。
- **不支持 FP8**，在新型 LLM 训练/推理效率上落后 H100 一代。
- 支持 MIG，最多划分 7 个独立实例，适合多租户共享。

2026 年的 A100 已经从“训练首选”变成“高性价比推理/微调”选择。云厂商报价已降至 $1.5 / 时左右，二手卡价格也明显下滑。

### 2.3 当前训练主力：H100 / H200 / H800

**H100** 是目前大模型训练最主流的卡：
- FP8 Transformer Engine，原生支持 FP8 训练与推理。
- SXM5 版本提供 3.35 TB/s 带宽和 900 GB/s NVLink。
- 80 GB 显存对 70B+ 模型全参数训练仍显紧张，通常需要 8 卡起步或配合 ZeRO/TP。

**H200** 是 H100 的“显存升级版”：
- 计算性能与 H100 几乎相同（同 GH100 die）。
- 显存升级到 **141 GB HBM3e**，带宽提升到 **4.8 TB/s**。
- 对 LLM 推理（大 batch、长序列）增益明显， reportedly 比 H100 推理快 45%。

**H800** 是 NVIDIA 针对中国市场推出的 H100 阉割版：
- CUDA/Tensor Core 算力与 H100 基本一致。
- 主要阉割在 **NVLink 带宽**（从 900 GB/s 降至 400 GB/s）和 **PCIe/互联规格**。
- 对大规模分布式训练（尤其是跨节点通信重的 workload）影响较大；单卡或小张量并行场景影响较小。

### 2.4 下一代旗舰：B300 / Blackwell Ultra

B300 属于 Blackwell 架构的进一步演进（Blackwell Ultra），单卡规格极为夸张：

- **288 GB HBM3e**，几乎是 H200 的两倍。
- **~8 TB/s** 显存带宽。
- **FP4 Transformer Engine**，对推理量化友好。
- TDP 高达 1000–1400 W，必须配合液冷基础设施。

B300 适合：
- 万亿参数模型的训练与推理。
- 单卡部署大模型，减少 GPU 数量与通信开销。
- 高密度的 DGX/GB300 NVL72 机架级系统。

主要问题：
- 价格高、功耗高、供应紧张。
- 软件和生态（FP4、新 NVLink 5）仍在快速迭代。

---

## 三、选型建议

### 3.1 按场景

| 场景 | 推荐 GPU | 理由 |
|------|----------|------|
| 小模型推理 (<13B) | L4 / RTX 4090 | 成本低，功耗小 |
| 70B 模型单卡推理 | L20 / L40S / RTX Pro 6000 | 48–96 GB 显存可装下量化模型 |
| 7B–30B 微调 | A100 80GB / H100 | 显存充足，生态成熟 |
| 70B+ 全参数训练 | H100 / H200 / B300 | 需要 NVLink 与大显存 |
| LLM 高吞吐推理 | H200 / B300 | 大带宽 + 大显存 |
| 中国地区合规部署 | H800 / H20 | 受出口管制影响的可选型号 |
| 多租户共享平台 | A100 / H100 | MIG 支持 |

### 3.2 按预算

- **极低预算**：L4 / RTX 4090（$0.3–0.4 / 时）
- **性价比优先**：A100 80GB / L40S（$1.0–2.0 / 时）
- **性能优先**：H100 / H200（$2.5–4.0 / 时）
- **不计成本**：B300 / GB300 系统（$5.5+ / 时，购置 $30K+ / 卡）

---

## 四、几个值得注意的趋势

1. **显存比算力更稀缺**  
   对 LLM 推理而言，能放下多大的模型、多大的 batch、多长的 KV cache，往往比峰值 TFLOPS 更决定实际吞吐。H200 和 B300 的核心优势都在显存容量与带宽。

2. **FP8 / FP4 成为标配**  
   从 H100 开始，FP8 训练/推理生态成熟；B300 进一步推动 FP4。A100 因不支持 FP8，在新型 workload 中逐渐边缘化。

3. **互联带宽决定扩展性**  
   单卡性能再强，跨卡通信跟不上也会成为瓶颈。H800 的阉割版 NVLink 就是典型案例。选购 8 卡以上训练集群时，必须关注 NVLink / NVSwitch / InfiniBand 配置。

4. **功耗与散热成为数据中心主要成本**  
   B300 单卡 TDP 超过 1000 W，整机架功率密度骤增。液冷、供电、机房改造成本会快速超过 GPU 本身。

---

## 五、总结

- **L20 / L40S / L4**：推理与轻量微调，价格低，无 NVLink。
- **A100 80GB**：上一代训练主力，2026 年性价比突出，但不支持 FP8。
- **H100 / H200**：当前大模型训练与推理的主流，H200 靠显存/带宽领先。
- **H800**：中国特供版，算力同 H100，互联缩水。
- **B300**：下一代旗舰，288 GB 显存 + FP4，适合超大模型与高密度集群。

如果只能给一个通用建议：**训练选 H100/H200，推理看模型大小选 L40S/A100/H200，未来 proof 或超大模型选 B300。**

---

> 参考来源：
> - [NVIDIA B300 Specs - server-parts.eu](https://www.server-parts.eu/post/nvidia-b300-gpu-specs)
> - [H800 Cloud Pricing - getdeploying.com](https://getdeploying.com/gpus/nvidia-h800)
> - [L20 Specs - InferenceBench](https://inferencebench.io/gpus/nvidia-l20/)
> - [NVIDIA H100 Specs - spheron.network](https://www.spheron.network/blog/nvidia-h100-specs/)
> - [NVIDIA H200 Specs - spheron.network](https://www.spheron.network/blog/nvidia-h200-specs/)
> - [2026 GPU Selection Guide - vessl.ai](https://vessl.ai/en/blog/gpu-workload-guide-en)
