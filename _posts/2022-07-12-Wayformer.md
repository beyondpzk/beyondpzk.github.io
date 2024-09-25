---
layout: post
title: Wayformer
date: 2022-07-12
categories: [Prediction]
tags: Prediction
---
<!--more-->


- [paper地址](https://arxiv.org/abs/2207.05844)

# Wayformer: Motion Forecasting via Simple & Efficient Attention Networks

## sketch the main points

自动驾驶的运动预测是一项具有挑战性的任务，因为复杂的驾驶场景会导致静态和动态输入的异构混合。如何最好地表示和融合有关道路几何形状、车道连通性、交通灯状态以及一组动态agents及其相互作用的历史信息，并将其转化为有效的编码，这是一个悬而未决的问题。为了对这组不同的输入特征进行建模，许多方法提出设计一个具有不同模态特定模块的同样复杂的系统。这导致系统难以扩展、扩展或以严格的方式调整以权衡质量和效率。

Wayformer提供了一个紧凑的模型描述，由基于注意力的场景编码器和解码器组成。在场景编码器中,证明了早期融合，尽管其结构简单，但不仅与模态无关，而且效果好.
由于许多原因，这种场景理解所需的建模具有挑战性。首先，输出是高度非结构化和多模态的.
例如，开车的人可以实现观察者未知的许多潜在意图之一，并且需要代表多样化和不相交的可能未来的分布。第二个挑战是输入由模态的异构组合组成，包括智能体过去的物理状态、静态道路信息（例如车道位置及其连通性）和多变的交通灯信息。

### 输入信息理解

1. Agent History $[A, T, 1, D_h]$ 可以理解为batch size是$A$, 历史桢有$T$桢, $D_h$ 是这个agent的feature长度,
   比如,如果是位置, 那就是三维,或者二维.

2. Agent Interactions $[A, T, S_i, D_i]$, 可以理解为 batch size 是 $A$, 每桢里面有 $S_i$ 个agent,
   每个agent的feature长度是$D_i$, 这里 $S_i, D_i$ 可以设成是一个比较大的数字,这样就可以做到batch内部统一.
   比如每个其他的agent,用历史的$（x,y,z, l, h, w, yaw, vx, vy）$ 等来表示的话, $D_i$ 就可以取为9.

3. Roadgraph $[A, 1, S_r, D_r]$ 因为静态的道路不会随时间变化,所以可以理解成是当前道路有 $S_r$ 个 segments,
   每个segments的feature的度度是 $D_r$, 比如每个segment用100个点来表示,$D_r$ 就可以取为200.

4. Traffic light State $[A, T, S_{tls}, D_{tls}]$ 可以理解为每个时刻有 $S_{tls}$ 个交通灯, 每个灯的feature的长度是 $D_{tls}$

好像少了交通标识牌，限速标那些.

事先会对输入数据做处理,转成 ego-centric的. 
各个模态的数据 也会最终统一到相同的feature 维度,这样便于concat处理.
即由$[A, T, S_m, D_m] -> [A, T, S_m, D]$, 此外针对不同的模态也有模态的position embedding,用于区分不同的模态.

(输入的每个模态的token长度会有影响么, 即 $S_m$)
这里 projection 非常简单,用的就是 $relu(Wx_i+b)$, 或许直接用 linear更好呢.

### Fusion

由于输入的信息源多种多样,fusion起来确实不容易, 有的是几何信息,有的是其他信息, 所以想得到统一的信息很难.
作者这里没有花精力搞每个模态的计算速度以及超参之类的,而是把重点放在了研究fusion的方式. 

1. late fusion

每个模态有自己的encoder做self-attention, 然后经过各自的模态encoder得到embedding,之后再concatenation,
我认为目前的视觉语言多模态大模型就是这种方式, 即文本由文本的encoder, vision有vision的encoder. 而且各自的backbone都不小.

2. early fusion
每个模态内部并没有self-attention, 仅有一个简单的projection layers. 这种框架下, scene encoder 包含了一个 self-attention
encoder(跨模态的encoder), 给了网络最大的自由度来学习到底哪个模态比较重要.

3. Hierarchical fusion
这种有每个模态内部的 self-attention, 也有concat起来之后的 cross-modal的 self Attention.

### Attention

这里主要讨论的是速度问题,因为本身self attention和position embedding会随着输入序列的长度变长而变得计算复杂度高.

#### Factorized attention

这个比较好理解,比如输入是 $T\times I$, 如果对所有位置都搞pos-embedding的话, pos-embedding的长度是 $T\times
I$,而对两个轴分别做的话,比如时间维度和空间维度，只需要 $T+I$ 个 pos-embeddings. 

这个工作中做了两种对比实验, 一种是前$N/2$层是时序, 后$N/2$层是空间; 一种是时序与空间交差着搞.

#### Latent query attiontion

我理解这个有点类似于 DETR中的object query. 这个query的大小,可以自已设定,所以也是减小计算复杂度的一种方式.


### trajectory decoding

方法类似于mutlipath, 首先会有k个可学习的initial queries, 这个与scene-embedding进行cross-attention, 最终得到query
embeddings, 后面用于产生分类的概率以及trajectory.

#### trajectory aggregation

如果输出的GMM有太多modes的话,比如实验设定的是64, 不好聚合,因为输出的时候有限制,比如就8种可能性. 这个需要看[3]. 
