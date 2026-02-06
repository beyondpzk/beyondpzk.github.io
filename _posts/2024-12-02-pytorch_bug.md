---
layout: post
title: pytorch_bug
date: 2024-12-02
categories: [pytorch]
tags: pytorch
---
<!--more-->

# pytorch 常见问题解决

## torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate more than 1EB memory.

这个问题是由于多机多卡的时候,每个卡上的dataloader的长度不一样,导致有的已经训完了,有的还在等待, 会出现这个错误, 解决方法是强行把每个卡上的dataloader长度调成一致.
