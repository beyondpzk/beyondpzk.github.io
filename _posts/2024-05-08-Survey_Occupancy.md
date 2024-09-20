---
layout: post
title: Survey_Occupancy
date: 2024-05-08
<!--categories: [reading]-->
tags: Occupancy
---
<!--more-->


- [paper地址](https://arxiv.org/abs/2405.05173)

# A Survey on Occupancy Perception for Autonomous Driving: The Information Fusion Perspective

3D占用感知技术旨在观察和理解自动驾驶汽车的密集3D环境。由于其全面的感知能力，该技术正在成为自动驾驶感知系统中的一种趋势，并且正在引起工业界和学术界的重大关注。与传统的鸟瞰（BEV）感知类似，3D占用感知具有多源输入的性质和信息融合的必要性。然而，不同之处在于它捕获了2D
BEV忽略的垂直结构或者高度信息。

对于可靠和安全的自动驾驶来说，一个至关重要的能力是理解周围环境，即感知观察到的世界。目前，BEV感知是主流的感知方案，它具有绝对尺度和不考虑遮挡的优点。BEV感知为多源信息融合（例如，来自不同视点、模态、传感器和时间序列的信息）和众多下游应用（例如，可解释的决策制定和运动规划）提供了统一的表示空间。然而，BEV感知不会监控高度信息(比如马路牙子)，从而无法为3D场景提供完整的表示。我认为另一方面基于白名单的感知技术cover的类别总是有限
为了解决这些问题，占用感知被提出用于自动驾驶，以捕捉现实世界的密集3D结构。这种新兴的感知技术旨在为体素化世界推断每个体素的占用状态，其特点是对开集物体、不规则形状车辆和特殊道路结构具有很强的泛化能力[3,4]。与透视图和鸟瞰视图等2D视图相比，占用感知具有3D属性的性质，使其更适合3D下游任务，例如3D检测[5,6]、分割[4]和跟踪[7]。

在学术界和工业界，整体3D场景理解的占用感知产生了有意义的影响。从学术角度考虑，从复杂的输入格式中估计现实世界的密集3D占用是一项挑战，包括多个传感器、模态和时间序列。此外，进一步推理占用体素的语义类别[8]、文本描述[9]和运动状态[10]是有价值的. 在每辆自动驾驶车辆上部署LiDAR套件成本高昂。随着摄像头成为LiDAR的廉价替代品，以视觉为中心的占用感知确实是一种具有成本效益的解决方案，可以降低车辆设备制造商的制造成本。(这确实也是一个因素,因为lidar扫出来的天然具有深度以及高度, 那基于lidar的方案还需要做occ么? 我理解应该不需要了吧, 但可能作为结构化的输出被下游消费可能还是需要.)

占用感知的要点在于理解完整和密集的3D场景，包括理解遮挡区域。然而，来自单个传感器的观察只捕获场景的一部分。例如，图像或点云不能提供3D全景或密集的环境扫描。为此，研究来自多个传感器[11,12,13]和多个帧[4,8]的信息融合将有助于更全面的感知。这是因为，一方面，信息融合扩大了感知的空间范围，另一方面，它使场景观察变得密集。此外，对于遮挡区域，集成多帧观察是有益的，因为同一场景由大量视点观察，这些视点为遮挡推断提供了足够的场景特征。

此外，在光线和天气条件变化的复杂户外场景中，对稳定的占用感知的需求至关重要。这种稳定性对于确保驾驶安全至关重要。在这一点上，多模态融合的研究将通过结合不同数据模式的优势来促进稳健的占用感知[11,12,14,15]。例如，激光雷达和雷达数据对光照变化不敏感，可以感知场景的精确深度。这种能力在夜间驾驶或阴影/眩光掩盖关键信息的场景中尤为重要。相机数据擅长捕捉详细的视觉纹理，擅长识别基于颜色的环境元素（例如，路标和交通信号灯）和长距离物体。因此，来自激光雷达、雷达和相机的数据融合将呈现对环境的整体理解，同时抵御不利的环境变化。

(其实文章前面就是在解释为什么要做occ,多举一些实际的例子就能说明了,实际中有许多情景可以解释。)


(history) 占用感知源自移动机器人导航中的经典课题占用网格映射（OGM）[24]，旨在从嘈杂和不确定的测量中生成网格地图，该地图中的每个网格都被分配了一个值，该值对网格空间被障碍物占用的概率进行评分。语义占用感知源自SSCNet[25]，预测占用状态和语义。然而，研究室外场景中的占用感知对于自动驾驶来说势在必行。MonoScene[26]是仅使用单目摄像头的室外场景占用感知的开创性工作。与MonoScene同时代，特斯拉在CVPR 2022自动驾驶研讨会上宣布了其全新的仅限摄像头的占用网络[27]。这个新网络根据环绕视图RGB图像全面了解车辆周围的3D环境。随后，占用感知引起了广泛关注，催化了近年来自动驾驶占用感知研究的激增。
室外占用感知的早期方法主要使用LiDAR输入来推断3D占用[28,29,30]。然而，最近的方法已经转向更具挑战性的以视觉为中心的3D占用预测[31,32,33,34]。目前，占用感知研究的一个主导趋势是以视觉为中心的解决方案，并辅以以以LiDAR为中心的方法和多模态方法。占用感知可以作为端到端自动驾驶框架内3D物理世界的统一表示[8,35]，其次是跨越检测、跟踪和规划等各种驾驶任务的下游应用。占用感知网络的训练严重依赖于密集的3D占用label，从而导致了多样化街景占用数据集的开发[11,10,36,37]。最近，利用大模型的强大性能，大模型与占用感知的集成在减轻繁琐的3D占用label需求方面显示出希望[38]。

(task-definition) 
占用感知旨在从多源输入中提取观察到的3D场景的体素表示。具体来说，这种表示涉及将连续的3D空间离散成由密集体素组成的网格volume。每个体素的状态由值{1,0}或者多类别语义id表示.

这种体素化表示提供了两个主要优势：
- 它能够将非结构化数据转换为体素体积，从而促进卷积[39]和transformer[40]架构的处理；
- 它为3D理解场景提供了灵活和可扩展的表示，在空间颗粒度和内存消耗之间取得了最佳平衡。

输入往往是当前桢t以及历史的几桢. 产生t-th的voxel-wise的表示.
(既然输入的时候就有当前桢的信息，为什么叫occ-prediction,而不是叫occ-perception呢.)

(相关工作)
根据输入数据，BEV感知主要分为三种：
1. BEV相机[42,43,44]、
2. BEV LiDAR[45,46]
3. BEV融合[47,48]。

目前的研究主要集中在BEV相机上，其关键在于从图像空间到BEV空间的有效特征转换。为了应对这一挑战，一种类型的工作采用显式转换，首先估计前视图像的深度，然后利用相机的内在和外在矩阵将图像特征映射到3D空间，然后进行BEV池化[43,48,49]。
相反，另一种类型的工作采用隐式转换[44,50]，它通过交叉注意力机制隐式模拟深度，并从图像特征中提取BEV特征。值得注意的是，基于相机的BEV感知在下游任务中的性能现在与基于LiDAR的方法相当[49].相比之下，占用感知可以看作是BEV感知的延伸，占用感知构建了3D体积空间而不是2D
BEV平面，从而导致对3D场景的更完整描述。 
(其实我认为BEV感知还应该算上另一个的方法,即petr系列的,
只不过bevformer与lss这类方法是先构建bev空间或者bev features,然后基于bev
features再来做具体的task, 而petr跳过了bev空间或bevfeatures建立的这一步.)

3D重建是计算机视觉和机器人社区中的一个传统但重要的主题[63,64,65,66]。从图像进行3D重建的目标是基于从一个或多个视点捕获的2D图像构建对象或场景的3D。
早期的方法利用了structure-from-motion[68]。之后，神经辐射场（NeRF）[69]引入了一种用于3D重建的新型范式，它学习了3D场景的密度和颜色场，产生了具有前所未有的细节和保真度的结果。然而，这样的性能需要大量的训练时间和资源来渲染[70,71,72]，特别是对于高分辨率输出。最近，3D高斯飞溅（3D GS）[73]通过重新定义场景表示和渲染的范式转换方法来解决这个问题。具体而言，它以显式的方式用数百万个3D高斯函数表示场景表示，实现更快、更高效的渲染[74]，3D重建强调场景的几何质量和视觉外观，相比较而言，体素级占用感知具有较低的分辨率和视觉外观要求，转而关注场景的占用分布和语义理解。


## methods

### lidar-centric occupancy Perception

基于lidar的主要是语义分割, 得到每个点的语义类别. 这里面核心的两个问题是,需要处理
sparce 到 dense的 场景理解, 因为lidar点云其实也还是比较稀疏,另外一个是遮挡问题，
即也需要补全。 比如障碍物的补全.
整体框架,类似于基于lidar的感知任务,

1. voxel 化
2. voxel feature extraction
3. feature enhance 
4. task-head, 可以infer {0, 1}, 也可以是semantic 类别.

### camera-centric 

1. 基于lss
2. 基于bevformer中的cross-attention.

(这两种方式有什么优劣?)




## Challenges and Opportunities

### Occupancy 的应用

1. segmentation

即 3D semantic segmentation task.

2. detection

比如, OccupancyM3D, SOGDet. 方法是基于Ocuupancy来做检测,
或者是加入了occupancy的任务在里面，同时训练检测和分割.

3. dynamic Perception

这个是用来获得动态物体及他们的motion信息, 输出是以 occupancy flows的形式.
比如Cam4DOcc, LOF(抽空补一下.).

4. worldmodels

OccWorld, OccSora

5. 自驾框架上

它将不同的传感器输入集成到一个统一的占用表示中，然后将占用表示应用于广泛的驾驶任务，如3D目标检测、在线映射、多目标跟踪、运动预

测和运动规划。相关作品包括OccNet[8]、DriveWorld[7]和UniScene[61]。

但是(这也是我关心的问题,即如何消费OCC更好地应用于decision及planning)
现有的基于占用的应用主要关注感知层面，而较少关注决策层面。鉴于3D占用比其他感知方式（例如鸟瞰视角感知和透视视角感知）更符合3D物理世界，理认上3D占用为自动驾驶中更广泛的应用提供了机会。在感知层面，它可以提高现有场所识别[166,167]、行人检测[168,169]、事故预测[170]和车道线分割[171]的准确性。在决策层面，它可以帮助更安全的驾驶决策[172]和导航[173,174]，并为驾驶行为提供3D可解释性。  (这几个工作看一看.)

### 部署效率

有几个工作是做这方面的,比如 FastOcc, FlashOcc, SparseOcc等,
以及 GaussianFormer (用一系列的3D Gaussians 表示 sparse的感兴趣区域.)
(这个工作要看.)

### Robust & Generalized 3D Occupancy Perception

3D标签成本高昂，并且针对现实世界的大规模3D注释不切实际。在有限的3D标记数据集上训练的现有网络的泛化能力尚未得到广泛研究。为了摆脱对3D标签的依赖，自监督学习代表了通往广义3D占用感知的潜在途径。它从广泛的未标记图像中学习占用感知。然而，当前自监督占用感知[31,38,87,91]的性能较差。
在Occ3D-nuScene数据集上，自监督方法的精度在很大程度上不如强监督方法。我认为一方面是因为学术集不够大,
数据量足够大, 模型也大时,自监督的方法应该能够体现出作用.

此外，当前的3D占用感知只能识别一组预定义的对象类别，这限制了其通用性和实用性。LLM和
VLM展示了一种很有前途的推理和视觉理解能力。集成这些预训练的大型模型已被证明可以增强感知的泛化[9]。POP-3D[9]利用强大的预训练视觉语言模型[192]来训练其网络，并实现了开放词汇的3D占用感知。 (利用现有的VLM或者large vision model，确实是一个很值得思考的事情)


