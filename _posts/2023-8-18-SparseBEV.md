---
layout: post
title: SparseBEV
date: 2023-8-18
categories: [Perception]
tags: Perception
---
<!--more-->


- [paper地址](https://arxiv.org/pdf/2308.09244)

# SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos

## sketch the main points

之前基于dense bev features的工作有下列问题:
1. 需要复杂的2d->3d, 3d->2d的 view transformations.
2. 计算慢.
但是基于sparse的工作,效果不太好. 
这篇工作的贡献就是 用sparse的方式做到了不错的效果. 
这期中的gap,作者认为是detector对于bev空间和图像空间的适应能力.

## methods

整体pipeline仍然类似DETR，不过这里是3D任务,输入是环视相机的Multi-view videos. 得到image features之后, 
对于decoder部分， query刚开始是 bev空间里的 sparse 的pillars集合.  decoder出来后, 接了head来出预测. 
下面是几个重要的部分:

1. query formulation

query 初始化成bev空间中的 pillars, 具体来说, 是用 $(x,y,z=0,w,l,h, \theta, v_x=0, v_y=0, query feature)$ 作为初始化.

2. scale-adaptive self Attention

基于dense bev feature的方法,都是基于bev encoder先建立bev feature, 但是这里并没有这么做,所以如何基于图像信息get到空间信息很重要, 
作者认为, self attention本身可以充当 bev encoder, 因为query是定义在bev空间中的.  具体是这么做的. 
假设有 $N$ 个query, 对这 $N$ 个query先做距离,得到一个 $N \times N$ 的距离矩阵, 然后把这个矩阵放进了 self
attention公式里的softmax项中, 见公式2. 
此时，对于距离较远的两个query, D就大, logits就会越小, softmax也就会越小, 即attention就会越小, 
通过这个D的引入,相当于把query之间的距离作为了attention的权重一部分进行考虑.
其中公式2中的系数 $\tau$ 也是学出来的, 见公式3. 即根据query的不同, 系数也不同.

3. adaptive spatio-temporal sampling

这个操作是图像空间和bev空间联系的纽带. 根据 每组query的feaeture,网络会学习产生一组sampling offsets,
然后基于这个offsets,和query的初始位置(或上一层时的位置),用公式(4) 就得到了3D sampling points. (有点deformable attenion的思想)

同时作者还考虑了运动信息, 基于公式(5)和(6), 相当于是假设了每个采样点与query具有相同的速度. 从而算出来,时序之后的位置.
对于自车的运动, 用公式(7), 之间从pose层面来做.
对3D sampling points 做完空间和时间上的变换之后, 就可以去对应时间戳的图像上拿image feature了, 这就是公式(8)
做的事情,因为图像不同的view之间可能有overlap所以会有一个求和与平均的操作. 公式(8)得到的东西是sampling point $i$
得到的feaeture,  假设有T桢, 每桢采样S个点, 那就会得到 $T\times S$ 个 point features, 假设每个feature是$D$ 维的. 
接下来就是这个大的feature怎么用.

4. adaptive Mixing

参考的是 [AdaMixer]
首先上面得到的大的feature可以记为 $f$, 是 $P\times C$ 维的, 下面的channel Mixing就是分别对这两个维度分别处理的.
一种是channel Mixing,即基于query的feautre, 学习一个权重,这个权重和points feature相乘再经过layernorm和relu. 见公式9和10. 
point mixing,刚好反过来，对于另一个维度做的处理.

