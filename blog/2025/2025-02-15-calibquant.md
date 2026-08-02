---
title: CalibQuant：多模态 LLM 的 1-Bit KV Cache 量化
date: 2025-02-15
categories: [Deploy]
---

# CalibQuant：多模态 LLM 的 1-Bit KV Cache 量化

> **论文**：CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs  
> **发布时间**：2025-02-15  
> **arXiv**：https://arxiv.org/abs/2502.14882  
> **代码**：https://github.com/insuhan/calibquant

## 一句话总结

CalibQuant 面向多模态大语言模型（MLLM）的长上下文推理，把 KV Cache 压到 1-bit，并结合后缩放和校准技术保留精度，在 InternVL 上实现约 10× 吞吐提升。

## 动机

MLLM 的 KV Cache 比纯文本 LLM 增长更快：长图像、视频帧和文档都会产生大量视觉/文本 token。KV Cache 过大会造成：

- 显存占用持续增长；
- 吞吐下降；
- 长时间部署受限。

现有 4-bit/8-bit KV Cache 量化仍不够激进。

## 核心方法

### 1. 极端 1-bit KV 量化

CalibQuant 将 KV Cache 中每个值量化为 1-bit 表示，大幅降低显存带宽需求。

### 2. Post-Scaling

单纯 1-bit 量化误差太大。CalibQuant 在量化后引入额外的缩放/补偿参数，恢复 KV 值的关键统计信息。

### 3. 针对 KV 模式的校准

KV Cache 的分布和普通激活不同，CalibQuant 针对键、值以及不同位置/token 的内在模式设计校准策略，而不是复用通用 activation calibration。

### 4. Triton 运行时优化

论文使用 Triton 实现高效的 kernel 和运行时，把 1-bit KV 的访存优势真正转化为吞吐收益。

## 关键结果

- InternVL 上吞吐提升约 **10×**；
- 显著降低 KV Cache 显存占用；
- 无需修改模型架构，可即插即用。

## 部署视角

CalibQuant 适合：

- 长图、长视频、长文档等长上下文 MLLM 服务；
- 显存受限的推理服务器；
- 需要高并发吞吐但无法容纳完整 KV Cache 的场景。

需要注意，这里的 1-bit 是 KV Cache 的极端压缩，不等于整个模型权重都是 1-bit。

## 局限

- 1-bit 压缩对长序列精度仍有风险，需要任务级验证；
- post-scaling 参数会随模型和输入分布变化；
- 吞吐收益依赖 Triton/kernel 实现，跨框架迁移需要重做优化。

## 总结

CalibQuant 是 MLLM 部署中 KV Cache 极致压缩的代表工作，核心贡献是把“1-bit 可行性”从论文推进到可运行的 Triton 推理路径。
