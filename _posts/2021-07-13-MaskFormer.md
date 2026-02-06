---
layout: post
title: MaskFormer
date: 2021-07-13
categories: [reading]
tags: reading
---
<!--more-->


- [paper地址](https://arxiv.org/abs/2107.06278)

# Per-Pixel Classification is Not All You Need for Semantic Segmentation

## 回忆语义分割,实例分割,全景分割 

1. 语义分割：对图像中的每个像素进行分类，不区分同一类别的不同实例。例如，在一幅包含人和狗的图像中，语义分割会将所有的人标记为 “人” 这个类别，所有的狗标记为 “狗” 这个类别，不区分不同的个体。

2. 实例分割：不仅要区分不同的类别，还要区分同一类别的不同实例。继续以人和狗的图像为例，实例分割会将不同的人分别标记为 “人 1”“人 2” 等，不同的狗分别标记为 “狗 1”“狗 2” 等。

3. 全景分割: 为图像中的每个像素点都分配一个语义标签和一个实例 ID。语义标签表明物体的类别，实例 ID 则对应同类物体的不同编号。如果无法确定具体的类别或实例，可以给予空标注。所有语义类别要么属于不可数目标（stuff 类别），如天空、草地等没有固定形状的背景类；要么属于可数目标（things 类别），如人、动物、工具等相互独立的具体物体类，不能同时属于二者，且 stuff 类别没有实例ID

与语义分割相比, 全景分割在语义分割的基础上，还需要区分同一类别的不同实例。例如，在语义分割中，可能只将图像中的所有狗都标记为 “狗” 这个类别，而全景分割则要进一步区分出每只不同的狗，并为它们分配不同的实例 ID。
与实例分割相比, 全景分割要求每个像素只能有一个类别和 ID 标注，不能出现实例分割中的重叠现象。
(不能重叠是全景分割区别于实例分割的重要标志!)

语义分割的做法比如FCN， U-Net等, 可以实现像素级别的分类,
实例分割如Mask-RCNN,在Faster-RCNN的基础之上增加了一个分支来预测每个实例的mask(即所占的domain), 从而实现实例分割 


## sketch the main points

之前的方法对于 语义分割与实例分割 可以说是采用的两套不同的方案, 而MaskFormer把这些分割
任务从框架设计上进行了统一,并且效果达到了sota. MaskFormer会预测一组binary masks,
同时每个mask都有global的类别预测,这样global的类别预测就对应于语义分割,
binary的mask就代表了实例分割.同时因为每个pixel只有一个id, 所以也做到了全景分割.

## per-pixel分类法与mask分类法

1. 基于 per-pixel分类的方法

这个很好理解,比如FCN, 总共有K个类别,那就是每个像素点的K类别分类问题.

2. 基于 mask 分类的方法

这种方法会把分割 任务分成两步, 第一步是把image 分成N个regions, (N不一定等于K), 这N个regions, 每个大小仍然是
HxW,会有0,1来区分每个区域,像岛屿一样, 第二步是给每个region预测一个整体的类别. 因此为了做 mask的分类, 输出的结果其实是
N个pairs, 每个pair, 有global的类别(第二步的那个类别), 同时还有用来区分其他区域0,1的标志(第一步的那个region的标识).

3. 不同. 相比于per-pixel的分类而言, 基于mask的分类的global类别会是K+1, 因为多了一个 "no object" 的类别.
   即不在这K个白名单里面的。 另外一点是 mask分类的方法允许多个mask分类, 举个例子就是说,
   比如一个图片里面只有两个类别,人和狗,
   但是是多个人和多条狗，如果用pixel-分类法来做的话，只能分人和狗,而用mask分类的方法来做的话，因为mask的数量N是可以自定的, 这样虽然类别数量K有限,但是N可以多个, 这样就可以同时来做语义分割 和实例分割了.

4. 要点. 为了训练mask分类的模型, 需要给预测和gt之间做匹配,就类似于DETR一样. 一般会设预测N不小于真实的GT数量, 这样保证
   one-one.

## methods

想法来源于DETR,回忆DETR中的object query ,可以接task-head来做detections, 那么这个object query
既然最终可以转化成为image上的一个目标region, 那它也可以用来代表一个region. MaskFormer就是这种思想.

它分成了三块儿, 第一块儿是 pixel-level module, 这个就是一个encoder-decoder的结构, encoder 的输出是image features,
decoder的输出是per-pixel-embeddings. 
第二块是 transformer decoder,这个可以认为是DETR中的decoder,输入的是N个queries,以及第一步的image featuers,
这两个做Cross-Attention, 输出之后就进入了第三块，即分割模块,
这里分成了两支,一支经过MLP会得到N个query的分类预测,所以是 Nx(K+1). 另外一支先经过MLP得到 N个mask-embedding, 然后与per-pixel embeddings 做矩阵乘法,得到NxHxW的 mask predictions, 即相当于N个regions的预测.
因此在训练的时候会有global的分类loss,也有region的binary mask loss. 

推理的时候, 这两个预测做矩阵乘法,消掉N，并去掉 "no object", 可以得到KxHxW的prediction,这就是 语义分割的输出.

每个query的类别,类似于DETR中的label-assign方法.

## thinking

整体来看，我觉得是一个很不错的工作,核心思想是发扬DETR中的object query的思想.
