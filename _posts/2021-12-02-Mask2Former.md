---
layout: post
title: Mask2Former
date: 2021-12-02
categories: [Segmentation]
tags: Segmentation
---
<!--more-->


- [paper地址](https://arxiv.org/abs/2112.01527)

# Masked-attention Mask Transformer for Universal Image Segmentation

## sketch the main points

框架可以参考MaskFormer, 主要更改在于 transformer decoder这里, 由之前的 公式(1)

$ X_l = softmax(Q_lK_l^T)V_l +X_{l-1} $

到现在的公式(2) .


$ X_l = softmax(M_{l-1} + Q_lK_l^T)V_l +X_{l-1} $


$M_{l-1}$ 来源于之前一层Transformer decoder的输出,如果概率超过一个thresh(0.5)了就是0, 否则就是 $-\infty$, 因为

$\infty $ 加上一个有限的仍然是 $\infty$ 所以经过softmax之后, 这块儿就对应着0. 所以叫localized features.
这样局部region会不断地加强.

## 这样做有什么好处


1. restricts the attention to localized features centered around predicted segments, (做法)
2. leads to faster convergence and improved performance. (这个是好处)
