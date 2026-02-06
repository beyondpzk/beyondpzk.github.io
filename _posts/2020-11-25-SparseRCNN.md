---
layout: post
title: SparseRCNN
date: 2020-11-25
categories: [Detection]
tags: Detection
---
<!--more-->


- [paper地址](https://arxiv.org/abs/2011.12450)

# Sparse R-CNN: End-to-End Object Detection with Learnable Proposals

## sketch the main points

回忆一下之前的目标检测大概是怎么做的,大致分成了两类,即dense的和dense-to-sparse的.

dense的比如RetinaNet, 直接在dense的features上面接class和box的分支;
还有像Faster-RCNN一样,先有一堆anchors,然后得到proposals, 从proposals进行筛选,再进入最后的head.
而本文提出的SparseRCNN则是有更少的proposals进入head, 而且proposals是可学习的. 思想我认为还是有DETR的object-query在里面.
主要的motivation就是想用可学习的proposals-box来取代RPN得到的很多的proposals.

大致的流程是:
输入: image, proposals-box, proposals-features, 后面两个是可学习的. 类似object-query
1. backbone提图像的features,
2. 根据proposals-box获取到了roi-feats, 
3. roi-feats与proposals-features做dynamic instance
   交互(可以理解为局部的Cross-attention,只不过实现上没有Cross-attention那么复杂.)
4. 得到的N个feats，输入给head来做分类与回归.

这里 proposals-box与proposals-features 可以理解为都是query的一部分, 前面4维表示box, 后面表示context的信息,
因为只有box的话，确实是太弱了，没有了语义和场景信息.
4维的box,都Normalize到了0-1,作者也发现这个和初始化的方式关系并不大.

最终的效果是训练和推理速度都很快,而且指标与之前的方法有可比性.
