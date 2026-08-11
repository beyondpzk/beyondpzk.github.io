---
title: ZeroQuant：面向大规模 Transformer 的高效后训练量化
date: 2022-06-04
categories: [Deploy]
---

# ZeroQuant：面向大规模 Transformer 的高效后训练量化

> **论文**：ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
> **作者**：Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He (Microsoft)
> **发布时间**：2022-06-04
> **arXiv**：https://arxiv.org/abs/2206.01861

## 一句话总结

ZeroQuant 是较早系统化验证大规模 Transformer 可以做 INT8 训练后量化的工作，提出了一套端到端量化推理流水线，首次把 GPT-J6B 和 GPT-NeoX20B 拉到 INT8 精度。

## 动机

2022 年初，大模型推理面临严重的显存和计算压力。虽然 INT8 量化在视觉模型上已很成熟，但在 Transformer 上——尤其是 GPT-3 这类自回归模型——精度损失仍然显著。ZeroQuant 的目标是给出一个实用、便宜的端到端方案，不需要重训练，也不需要原始训练数据。

## 核心方法：三管齐下

### 1. 细粒度、硬件友好的量化

激活值做 per-token 量化（每个 token 独自一个 scale），权重做 per-group 量化（每 K 个权重一组，各自 scale）。这种粒度比 per-tensor 细，能显著降低 outlier 影响，同时核函数实现友好。

### 2. 逐层知识蒸馏（LKD）

在没有任何原始训练数据的情况下，以原模型为 teacher，用合成数据逐层对齐量化模型的输出分布。这步让 INT4 权重量化成为可能：注意力层权重保持 INT8，全连接层权重降到 INT4，激活统一 INT8，最终实现约 3× 的内存压缩。

### 3. 优化的量化推理后端

在 DeepSpeed 推理引擎中实现了 INT8 GEMM kernel，消除了量化/反量化的额外开销。

## 关键结果

- BERT 模型 INT8 延迟加速最高 5.19×（vs FP16）；
- GPT-3 风格模型 INT8 最高 4.16× 加速；
- GPT-J6B / GPT-NeoX20B 首次拉到 INT8，精度与 FP16 持平，效率提升最高 5.2×；
- INT4 + LKD 下全连接权重降到 4-bit 后，精度仍可接受。

## 历史意义

ZeroQuant 处于 LLM 量化的早期探索阶段，为后来的 GPTQ/AWQ 铺了路：

- GPT 类自回归模型可以安全做 INT8 PTQ；
- 细粒度量化和蒸馏是弥补精度损失的有效手段；
- 需要专用推理后端才能真正吃到量化红利。

## 参考

- ZeroQuant: https://arxiv.org/abs/2206.01861
- DeepSpeed 推理: https://github.com/microsoft/DeepSpeed
