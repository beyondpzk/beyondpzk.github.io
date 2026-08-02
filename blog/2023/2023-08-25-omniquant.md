---
title: OmniQuant：可学习裁剪与等价变换的全方位 LLM 量化
date: 2023-08-25
categories: [Deploy]
---

# OmniQuant：可学习裁剪与等价变换的全方位 LLM 量化

> **论文**：OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models  
> **发布时间**：2023-08-25  
> **会议**：ICLR 2024  
> **arXiv**：https://arxiv.org/abs/2308.13137

## 一句话总结

OmniQuant 用少量可学习参数同时优化权重量化范围和学习等价变换，在不需要重训 LLM 的前提下把量化精度提升到 W4A4 等极端低比特。

## 动机

早期 PTQ 方法通过手工设计裁剪阈值或固定缩放策略来适配 LLM 分布，但：

- 权重和激活的 outlier 分布复杂；
- 固定策略无法适应不同模型/层；
- W4A4 这种权重+激活都量化时，手工参数很容易失效。

## 核心方法

### 1. Learnable Weight Clipping（LWC）

LWC 把量化前的裁剪阈值变成可学习参数，通过直通估计器（STE）优化，让模型自己找到每层最优的裁剪范围。

### 2. Learnable Equivalent Transformation（LET）

LET 学习计算等价的输入/输出变换，例如：

$$ Y = (X \cdot s)(W \cdot s^{-1}) $$

以及其他可吸收到权重或 LayerNorm 的线性变换。它不需要修改原始模型语义，但能让量化分布更适合低比特。

### 3. Block-Wise 误差最小化

OmniQuant 按 Transformer block 进行参数优化，目标是让量化块的输出接近全精度块输出，而不是只优化局部 scale。

## 关键结果

- 在 LLaMA、OPT 等模型上，W4A4 性能显著提升；
- 相比 SmoothQuant、GPTQ 等基线，极端低比特精度更稳定；
- 仍是 PTQ，不需要端到端重训 LLM。

## 部署视角

OmniQuant 适合：

- 想同时压缩权重和激活的低比特部署；
- 希望保留量化参数少、可导出标准 INT4 kernel 的场景；
- 作为 GPTQ/AWQ 无法满足 W4A4 精度时的替代。

## 局限

- 学习过程仍需少量校准和优化步骤；
- 可学习变换需要导出到推理框架；
- 小模型上 W4A4 的收益波动较大。

## 总结

OmniQuant 是 LLM PTQ 从“手工设计”走向“少参数可学习校准”的代表作，它让 W4A4 不再必须依赖完整 QAT。
