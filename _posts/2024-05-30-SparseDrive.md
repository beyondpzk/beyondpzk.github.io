---
layout: post
title: SparseDrive
date: 2024-05-30
categories: [e2e]
tags: e2e
---
<!--more-->

- [paper地址](https://arxiv.org/abs/2405.19620)

# SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation


模块化自动驾驶系统被解耦为感知，预测和规划. 缺点是模块的信息丢失和错误积累。
相比之下，端到端范式将多任务统一到一个完全可区分的框架中，允许以规划为导向的精神进行优化。
尽管端到端范式潜力巨大，但现有方法的性能和效率都不令人满意，特别是在规划安全方面。

作者认为有两方面的原因:
1. 计算昂贵的BEV（鸟瞰）特性
2. 预测和规划的直接设计

针对原因1，探索稀疏表示并回顾端到端自动驾驶的任务设计，提出了一种名为SparseDrive的新范式。
具体而言，SparseDrive由对称稀疏感知模块和并行运动规划器组成。稀疏感知模块将检测、跟踪和在线mapping与对称模型架构相结合，学习驾驶场景的完全稀疏表示。

针对原因2, 对于运动预测和规划, 由于这两个任务之间的巨大相似性(这两个任务的本质不同在哪里,能不能用预测的方法来做planning?)，从而导致了运动规划器的并行设计。
基于这种将规划建模为多模态问题的并行设计，我们提出了一种分层规划选择策略，该策略结合了碰撞感知rescore模块，以选择合理且安全的轨迹作为最终规划输出。


为了充分利用端到端范式的潜力，回顾现有方法的任务设计，认为运动预测和规划之间的三个主要相似之处被忽略如下：
（1）旨在预测周围智能体和自车的未来轨迹，运动预测和规划应考虑道路智能体之间的高阶和双向交互。然而，以前的方法通常采用顺序设计进行运动预测和规划，忽略了自车对周围智能体的影响。
（2）对未来轨迹的准确预测需要语义信息用于场景理解，几何信息用于预测智能体的未来运动，这既适用于运动预测，也适用于规划。虽然这些信息是在上游感知任务中为周围智能体提取的，但对于自车却被忽略了。
（3）运动预测和规划都是具有固有不确定性的多模态问题，但以前的方法只预测确定性轨迹用于规划。

如paper-1b所示,对比之前bev-centric的方案来说

(a): bev-centric的方法: 先得到dense的bev-features, 然后分多个heads来做 detecion、tracking、online mapping, 然后再做motion prediction,最后是planning.

而作者的改进是

(b): sparse-centric paradigm: 先得到sparse的信息, 然后也是分两路分别做 detection & tracking 与 online mapping, 之后是prediction与planning一块儿做; (这里有个问题,能不能prediction也不做了?)

效果上最明显的是速度问题,比如训练时快了7.2倍,推理时快了1.8倍.

## overview of SparseDrive

1. encodes multi-view images into feature maps.
2. learns sparse scene representation (through symmetric sparse perception)
3. perform motion prediction and planning

时序上,这里用了一个叫做 instance memory queue 的东西. 这里面包括了

1. instance feature & anchor box (可以认为是other agents)
2. ego instance feature & anchor box
3. map instance feature & anchor polyline

相当于是得到了dynamic 和 static 以及ego 的重要信息.

### image encoder

Given multiview images, the image encoder, including a backbone network and a neck, first encodes images to multi-view multi-scale feature maps I.


### symmetric sparse perception module

In symmetric sparse perception module, the feature maps I are aggregated into two groups of instances to learn the sparse representation of the driving scene. These two groups of instances, representing surrounding agents and map elements respectively, are fed into parallel motion planner to interact with an initialized ego instance.

这一块儿我认为整体的结构类似于detr或者是petr, 初始化的map instances 和detection instances可以看成是detr里的object query. 通过这个操作,最后得到的就是对应的query, 而不是dense的feature map. 

#### sparse detection 

sparse detection分两个东西, 一个是 instance features, 一个是anchor boxes. (框和features有没有重复,可不可以只用一个?) (框和features这里都作为query是如何处理的?).

#### sparse online mapping

sparse online mapping的分支和 sparse detection 的分支结构一样,但不share weights, 此时用的是N个points来代表一个polyline. 所以用Nm个featuers和Nm个polylines来作为query.

#### NOTE

因为都是 Nm个,虽然是显式地建模成了两组query, 其实仍然可以认为是一组query, 比如可以看成在feature-dim维度上把feature加长, 所以说可以认为是detr中的object query.



### prediction and planning (Parallel Motion Planner)

The motion planner predicts multi-modal trajectories of surrounding agents and ego vehicle simultaneously, and selects a safe trajectory as the final planning result through hierarchical planning selection strategy.

分成了三步

1. ego instance initizlization
2. spatial-temporal interactions
3. hierarchical planning selection

#### ego instance initizlization

这个和 sparse detecion 类似,只不过N取为1. 另外一点, 并没有和环视feature map 进行交互, 因为很显然,所有的相机都看不到自车, 即自车在所有的相机盲区中.
这里采用了最直观的做法, 即和前视相机的最小的feature map 做了pooling 之后作为 ego instance feature的初始化.

这么做的优势是:
1. 使用前视相机的最小的feature,确实是视觉的场景信息.
2. dense feature map的加入, 对于sparse detection 是一个辅助. 因为sparse detecion只能检whitelist里面的.

ego anchor box的初始化, 位置,size, yaw, 可以用自车的信息设置 ,但是速度的初始化要注意. (看参考文献 27)。 为此还专门加了一个task来decode当前自车的速度 ，加速度,角速度，以及steering angle. 时序层面,用上一桢预测的速度作当前桢的初始化.

#### spatial-temporal interactions

为了考虑所有道路代理之间的high-level interactions, 把ego 与 other agents的信息concat起来. 

由于ego instance 是在没有时间线索的情况下初始化的，但是ego对规划很重要，这里搞了一个 instance memory queue. 
然后执行三种类型的交互来聚合时空上下文：
1. agent-temporal CA、
2. agent-agent SA. 
3. agent-map CA. 

NOTE，在稀疏感知模块的时间交叉注意中，当前帧的实例与所有时间实例交互，称之为场景级交互。
而对于agent-temporal 交叉注意，采用实例级交互，使每个实例都专注于自身的历史信息。

然后就是 predict multi-modal 轨迹及对应的scores, 这里是对自车和其他agent都有预测, 对于其他每个agent，每个预测的都有Km个可能性,而对于自车, 却预测了 (NxKp) 个scores, 这里的N是command的数量,这里用的就只有三种, 左转、右转、执行. (这里会不会有问题,因为command与轨迹是有依赖关系的, 比如左转情况下，对应的可能性中，往左边的轨迹的概率要大于其它,我认为这样才是合理的; 另外一个问题是,为什么不直接预测轨迹的多个模式,而还要有comannds的预测, 这里可能需要看一下[15, 20]这两个工作)

#### Hierarchical planning selection

做法是根据每个command, 得到 Kp 个轨迹的subset, 然后根据对其他agents的motion prediction的结果, 对于那些可能发生碰撞的轨迹, 强制调整其score为0，最终剩下的分高的作为最终的输出.

paper里面没有详细说明 a novel collision-aware rescore module, 是如何设计的.

### training

1. 先训练sparse perception
2. 再全部放开一起训练.

loss 函数中除了每个任务的loss外,在训练sparse perception 的时候还加了深度估计在里面.

L = Ldet + Lmap + Lmotion + Lplan + Ldepth .


# Review and reflection

核心思想是 用sparse query 替代 dense futures 往下游送. 对于backbone的计算确实快了很多, 这也给出了一种和其他模态的token相融合的方法.不过这里没有使用occ, 可能用sparese的方式来做occ太难了.
如何自监督地训练 sparse perception ?

可以当做一个可以在车上运行的baseline.
