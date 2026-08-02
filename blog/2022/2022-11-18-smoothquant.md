---
title: SmoothQuant：把激活 Outlier 平滑进权重的 W8A8 量化
date: 2022-11-18
categories: [Deploy]
---

# SmoothQuant：把激活 Outlier 平滑进权重的 W8A8 量化

> **论文**：SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models  
> **发布时间**：2022-11-18  
> **会议**：ICML 2023  
> **arXiv**：https://arxiv.org/abs/2211.10438

## 一句话总结

SmoothQuant 观察到 LLM 权重容易量化、激活很难量化，于是通过数学等价的逐通道缩放，把激活中的 outlier 迁移到权重侧，从而实现 W8A8 的全激活量化。

## 动机

weight-only 量化能省显存，但矩阵乘仍按 FP16/高精度计算，无法真正利用 INT8 Tensor Core。要做 W8A8，必须把激活也量化为 INT8。

但 LLM 激活中存在 outlier，直接 INT8 量化会大幅掉点。

## 核心方法

### 1. 数学等价的平滑变换

把原线性层：

$$ Y = X W $$

改写成：

$$ Y = (X \cdot s) (s^{-1} \cdot W) $$

其中 $s$ 是逐通道缩放向量。

### 2. 离线吸收到权重

量化时：

- $X \cdot s$ 变得更容易量化；
- $s^{-1} \cdot W$ 在部署前预先算好并写入权重。

这样推理时没有额外缩放算子，只是标准 INT8 GEMM。

### 3. 超参数 $\alpha$

缩放向量通常按激活和权重的幅值比例确定：

$$ s_j = \frac{\max_j(|X_j|)^\alpha}{\max_j(|W_j|)^{1-\alpha}} $$

$\alpha$ 控制 outlier 迁移程度，通常在 0.5 左右。

## 关键结果

- OPT-175B 等模型在 W8A8 下几乎无损；
- 相比 weight-only 方法，能够真正减少激活和计算开销；
- 在 INT8 硬件上实测吞吐提升，例如约 1.5× 量级。

## 部署视角

SmoothQuant 是 INT8 全量化部署的核心方法：

- 被 TensorRT-LLM、vLLM 等主流推理栈采用；
- 适合需要激活量化的 W8A8 服务；
- 变换可以离线完成，推理时无额外开销。

## 局限

- 仅 8-bit 全量化，4-bit 激活仍非常困难；
- $\alpha$ 需要按模型校准；
- 极端 outlier 超出平滑能力时仍需配合旋转/剪枝。

## 总结

SmoothQuant 解决了 LLM W8A8 的核心矛盾：激活 outlier 不再通过更宽动态范围硬扛，而是“平滑”到权重侧，让标准 INT8 硬件可以高效执行。
