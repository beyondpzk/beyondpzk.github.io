---
layout: post
title: SparseOcc
date: 2023-12-28
categories: [Occupancy]
tags: Occupancy
---
<!--more-->


- [paper地址](https://arxiv.org/abs/2312.17118)

# Fully Sparse 3D Occupancy Prediction

## sketch the main points

整体框架看paper里的figure2,类似于MaskFormer, 不同点在于,现在是3d的了, 因此 MaskFormer里的 pixel-decoder 变成了 Sparse
Voxel decoder来得到Sparse Voxel embeddings, 这个相当于MaskFormer里面的pxiel-embeddings.
另外MaskFormer里的N个query可以理解为是N个regions, 而现在的K个sparse voxel embeddings充当的是这个角色,它会做binary
的分类,即每个grid是否会被占据. 从而最终得到的是在整体空间中是否被占据的信息. 
而基于图像的features 进入到Transformer-decoder做cross-attention得到的N个query embeddings用来做global的分类. 
所以最终并不是对整个空间进行的分类，而是对sparse voxel做的分类. 

看figure3,细节上,voxel-decoder输入时voxel queries 首先是均匀分布在3d空间中,
每层输入的时候，都会首先进行上采样2倍，即由之前的一个voxel,变成更小的8个voxel,
然后会估计每个voxel的occ-score,然后根据occ-score选取top-k做剪枝, (k是根据数据集统计出来的最大占用voxel数).

最重要的一点是, N个query是由两部分组成, 即可以认为维度是(K+C)的, K是sparse voxel的个数. 
前半部分(NxK) 是binary mask query, 用来代表K个voxel-query中是否被占据的信息, 这个才对应maskformer中对应region的那部分,
后半部分(NxC) 是contents vectors,用来学global的类别信息.
这样理解就通顺了

## Other

1. 相机内外参在这个pipline里面有没有用到.

肯定是用到的,具体需要看一下 SparseBEV 这个工作.
