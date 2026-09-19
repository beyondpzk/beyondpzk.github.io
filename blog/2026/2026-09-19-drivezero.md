---
title: DriveZero 深读：用强化学习教师与视觉蒸馏，突破人类驾驶轨迹的限制
date: 2026-09-19
categories: [自动驾驶]
---

# DriveZero 深读：用强化学习教师与视觉蒸馏，突破人类驾驶轨迹的限制

小米这项工作的名字是 **DriveZero**，论文全名为 *DriveZero: End-to-End Driving Beyond Human Demonstrations*，由 Xiaomi L3 Team 发布，arXiv v1 提交于 **2026 年 9 月 5 日**。

它研究的问题很直接：如果端到端驾驶策略一直拟合人类日志里的那一条轨迹，那么日志没有展示过的选择、纠错和交互行为，从哪里来？DriveZero 的回答是：**先在结构化仿真中通过强化学习训练一个能生成驾驶行为的教师，再把教师的行为蒸馏给视觉学生。** 视觉能力则通过另一条预训练路线，融合多个视觉基础模型。

这套方案值得关注，但有三个边界必须先说清楚：

- **闭环 RL 发生在 DriveRL 教师上。** DriveZero 视觉学生使用配对日志帧做离线蒸馏，本身没有进入交互环境做 RL。
- **“没有人类轨迹监督”不等于没有使用人类采集的数据。** 场景、图像、背景交通以及训练目标点的构造仍然利用驾驶日志。
- **截至 2026-09-19，公开内容主要是 DriveRL 推理代码和权重。** DriveZero 学生、DriveVFM 以及完整训练链路还不能仅凭当前仓库复现。

本文依据[原论文 v1](https://arxiv.org/html/2609.06055v1)、[官方项目页](https://xiaomiautol3.github.io/DriveZero/)和[官方代码仓库](https://github.com/XiaomiAutoL3/DriveZero)撰写。实验数字为作者报告；代码部分进行了静态阅读，没有重新训练模型或运行 nuPlan 评测。源码链接固定在提交 `24959547edac549d64c34b2aab62d5c52599d078`，避免后续更新改变本文所指的实现。

## 1. 先把三个名字和训练关系分开

![DriveZero 原论文总体架构：DriveRL、DriveVFM 与视觉学生](/images/drivezero/teaser.png)

*图 1：原论文 Figure 1。左侧分别学习行动能力和视觉表征，中间将两者结合为视觉策略。图中成绩对应不同输入与评测协议，不能直接跨榜比较。来源：Hao He 等 / Xiaomi L3 Team，CC BY-SA 4.0。*

| 组件 | 输入 | 如何学习 | 输出与作用 |
| --- | --- | --- | --- |
| **DriveRL** | 自车状态、周围交通参与者、矢量地图、信号灯、目标点 | 在交互式仿真中从随机初始化做 PPO | 动作分布、价值估计；提供驾驶轨迹教师 |
| **DriveVFM** | 通用图像与驾驶图像 | 蒸馏多个冻结视觉基础模型的特征 | 可用于驾驶规划的共享视觉骨干 |
| **DriveZero** | 四路相机、自车运动状态、导航指令 | 模仿 DriveRL 生成的轨迹，并学习候选评分 | 多条未来轨迹，选择其中一条执行 |

DriveRL 使用的是 **privileged observation，即特权结构化观测**。例如，周围车辆的位置、速度和车道几何已经以结构化形式给出，教师不需要先从像素中识别它们。DriveZero 则必须从图像中提取规划所需信息，因此两者面对的感知难度并不相同。

这里的“端到端”描述的是最终学生从视觉输入到轨迹的推理接口。训练过程实际上进行了明确分工：行动策略适合在可交互环境中学习，视觉骨干适合从大规模图像和已有基础模型中学习，最后再建立二者的联系。

![DriveZero 三阶段训练、蒸馏和部署的中文说明图](/images/drivezero/training-boundaries.svg)

*图 2：依据论文 §2 重绘。特别注意第三阶段：教师可以向前 rollout，但学生的训练输入仍然是与初始状态配对的日志图像。*

## 2. 为什么要换掉人类轨迹这个监督目标？

普通行为克隆常写成：

$$
\mathcal L_{\mathrm{BC}}
=\mathbb E_{(o,g,\tau_h)\sim\mathcal D}
\left[\ell\bigl(\pi_\theta(o,g),\tau_h\bigr)\right],
$$

其中 $o$ 是观测，$g$ 是导航意图，$\tau_h$ 是日志中的人类轨迹。一次路口驾驶只留下了其中一种行为，但这不代表它是唯一合理、最优或最适合当前评测目标的行为。

更困难的是，模型部署后会改变自己的状态分布。人类日志大多展示“驾驶员保持在正常位置时如何开”，而模型一旦产生横向偏差，就需要回答“已经偏了以后怎样恢复”。继续拟合正常状态下的动作，并不能自动解决这个问题。

DriveRL 把监督来源改成闭环回报：策略执行动作，仿真器更新车辆和交通状态，再根据碰撞、越界、到达目标和驾驶质量给奖励。教师可以在训练中经历自己的错误，也可以在同一个初始场景下尝试不同目标。

因此，DriveZero 的潜在收益有两部分：**更好的行为标签，以及可主动生成的新行为覆盖。** 后面的消融结果会表明，第二部分非常关键；仅仅把教师名字从“人类”换成“RL”并不能保证更强。

同时，论文没有消除所有数据分布问题。教师在仿真中学过偏离日志后的恢复，并不意味着视觉学生已经见过那些偏离状态对应的图像。两种能力之间仍然隔着观测和数据覆盖的差距。[论文 §1、§2.3](https://arxiv.org/html/2609.06055v1)

## 3. DriveRL：一个 570 万参数的结构化驾驶教师

### 3.1 观察什么：时序参与者、地图与目标

教师的主要输入规格如下：

| 输入 | 论文配置 | 设计意义 |
| --- | --- | --- |
| 自车及交通参与者 | 最多 96 个 agent token，包含自车 | 建模局部交互对象 |
| 历史状态 | 共 5 帧，5 Hz，包含当前帧 | 覆盖当前时刻及此前 0.8 秒的运动信息 |
| 地图 | 最多 256 个矢量元素，其中最多 64 个车道中心线元素 | 表达道路结构及相关信号灯状态 |
| 导航目标 | 自车坐标系中的两个二维目标点 | 让策略知道应该往哪里去 |
| 网络宽度 | 256，4 个 attention head | 用相对小的网络处理结构化输入 |

策略以自车 token 为 query，先经过两层 ego-to-agent cross-attention，再经过一层 ego-to-map cross-attention。随后将交互信息、目标、自车运动学特征及地图信息拼接，交给 MLP 和动作头。

这里不是让所有交通参与者之间都做昂贵的完整注意力，再从中寻找自车答案。网络围绕“当前自车怎样行动”组织信息聚合。这也是小模型能承载相当复杂驾驶行为的一个原因，但它依赖上游已经提供有效的结构化表征。[论文 §2.1.1、Table 1、Appendix A.2](https://arxiv.org/html/2609.06055v1#S2.SS1)

### 3.2 两个目标点是集合，不是有顺序的两个 waypoint

两个目标分别通过正弦位置编码和共享 MLP，然后做平均：

$$
z_g=\frac12\left[
f\bigl(\operatorname{PE}(g_1)\bigr)+
f\bigl(\operatorname{PE}(g_2)\bigr)
\right].
$$

交换 $g_1$、$g_2$ 不改变结果。因此，这个表示不能简单解释成“必须先经过第一个点，再经过第二个点”的有序路径编码。

训练时，目标来自日志中的未来自车位置，并包含投影到车道的构造方式；有时将同一点重复两次，有时使用两个不同点。部署时则沿当前导航路线选近、远两个锚点，随速度调整前视距离，并在每次规划时刷新。训练和部署的目标来源不同，但编码接口一致。

这个细节说明“没有人类轨迹监督”的准确范围：**不以人类未来轨迹作为驾驶动作的模仿标签，但导航意图的构造仍然可以利用人类日志。** 部署时的目标来自路线，不需要知道未来人类会怎样开。

### 3.3 输出什么：jerk 和转向角速度

DriveRL 每步输出两个归一化控制量，分别对应纵向 jerk 和轮胎转向角速度。jerk 是加速度对时间的导数；转向角速度描述方向盘对应的轮胎角度如何变化。

每个控制维度采用独立的 Beta 分布：

$$
u\sim\operatorname{Beta}(\alpha,\beta),\qquad
\alpha=1+\operatorname{softplus}(a),\quad
\beta=1+\operatorname{softplus}(b).
$$

由于 $\alpha,\beta>1$，分布的众数位于区间内部，避免 U 形分布把概率集中在两个极端控制上。训练时采样，评测时取解析众数：

$$
u_{\mathrm{mode}}=\frac{\alpha-1}{\alpha+\beta-2}.
$$

注意它与均值 $\alpha/(\alpha+\beta)$ 不同。随后将归一化动作映射为物理量：

$$
j=-8+13u_j\quad[\mathrm{m/s^3}],\qquad
\dot\delta=-0.8+1.6u_\delta\quad[\mathrm{rad/s}].
$$

实际执行还经过动力学限制：例如当纵向加速度非负时，正 jerk 进一步限制为不超过 $1\,\mathrm{m/s^3}$；加速度上限为 $4\,\mathrm{m/s^2}$，轮胎转角上限为 $\pi/3$。运动学自行车模型还包含加速度与转向的响应时间常数。

所以“网络输出动作”并不等于“将两个数直接发送给底盘”。归一化范围、车辆参数、积分周期和状态限制共同定义了策略的执行语义。**教师输出底层控制，学生输出未来轨迹，两者也不是同一种 action head。**[论文 §2.1.1、Appendix A.3](https://arxiv.org/html/2609.06055v1)

### 3.4 混合交通仿真：既要交互，也要维持场景合理性

DriveRL 从真实 nuPlan 日志初始化世界，再为不同交通参与者分配行为提供者：日志回放、IDM 等规则模型，或者学习策略。这样，同一场景里的车辆可以由不同机制控制。

论文基线训练使用 1:1 的日志回放和 IDM 场景配比。背景车辆可以按 IDM 运动，一小部分还可执行前车急刹行为；行人等非车辆对象保持日志回放。框架支持 self-play，让同一个策略控制多个车辆，每辆车使用各自坐标系下的观测和目标，但**默认教师训练只控制自车，实车部署章节才明确采用 self-play 配置**。

一个容易被忽视的工程问题是“晚进入场景的演员”。原日志里，一辆车可能在第 5 秒进入可见范围；如果自车已经偏离原轨迹，此时强行把它放回日志坐标，可能会突然出现在自车车身里。DriveRL 对插入设置了条件：自车与日志位置差不超过 5 米、朝向差不超过 0.35 rad，且新演员不会与可见演员碰撞。

这不是网络架构上的创新，却决定了 RL 是否在学习真实驾驶规律，还是在应对数据重放制造的不合理事件。[论文 §2.1.2](https://arxiv.org/html/2609.06055v1#S2.SS1)

### 3.5 奖励：硬事件加惩罚，软质量做乘积

论文的标量奖励为：

$$
r_t=h_t+(1-d_t)\left(g_t+\frac1H\prod_{k\in\mathcal K}q_{t,k}\right),
\qquad H=110.
$$

$h_t$ 表示碰撞、驶离道路等硬事件惩罚，$d_t$ 表示硬终止，$g_t$ 是一次性的到达目标奖励。六个 $q_{t,k}\in[0,1]$ 分别对应跨车道、中心线偏离、路缘间距、舒适性、TTC 和超速等质量指标。TTC 是按相对运动估计的碰撞时间相关指标。

软项采用乘积，意味着其中一项接近零，整体软奖励就会很低。相比简单加权求和，这减少了“通过更高速度或更好舒适度补偿另一项严重不足”的空间；代价是多个因素同时较差时，软信号会变小，训练仍需合适的硬事件、目标奖励和探索机制。

到达目标不会立即结束 episode，碰撞、驶离道路等硬事件会结束。论文基线没有额外的距离进度 shaping 或生存奖励，而且**交通灯违规并未纳入基线标量奖励**，尽管输入包含信号灯信息。因此不能把“包含合规相关软项”扩大解释成“奖励已经覆盖所有交通规则”。

还有一个细节：论文采用分解式 critic，预测硬事件、目标和六个软项对应的 **8 个价值通道**。但六个软通道并不是把奖励重新改成六项相加，而是将已经算出的乘积软奖励 $f_t$ 分配出去：

$$
r_{t,k}^{\mathrm{soft}}=w_{t,k}f_t,\qquad
w_{t,k}\propto\max(1-q_{t,k},0),\qquad
\sum_k w_{t,k}=1.
$$

所有软指标都完美时，使用均匀分配。这样各通道相加仍然还原原始标量奖励。使用相同折扣和 GAE 参数时，优势估计也可保持相应的加和关系。这是在改善价值学习的结构，而不是悄悄换掉 PPO 的优化目标。

**公开推理实现实际预留了第 9 个 `soft_other` 通道。** 论文公式和发布版张量维度之间的区别，后面的代码章节会具体说明。[论文 Appendix A.4](https://arxiv.org/html/2609.06055v1)

### 3.6 小模型不代表低训练成本

教师使用约 92.27 万个 nuPlan 场景、96 张 GPU，每个 rank 运行 2,048 个世界，总计 **196,608 个并行世界**。每次 rollout 为 110 步、5 Hz，即 22 秒；训练 2,400 次更新，约需 21 小时。

这是约 $96\times21=2,016$ GPU 小时的教师训练规模，还没有计入 DriveVFM 预训练和视觉学生训练。网络只有 570 万参数，主要说明结构化行动网络可以很小，不能据此推断完整方案能低成本从零复现。

PPO 使用 $\gamma=0.99$、GAE $\lambda=0.95$，每轮四个优化 epoch。值得注意的是，附录列出的教师优化器是 **Muon**，而视觉预训练与学生训练使用 AdamW，三阶段的优化配置不能混用。[论文 Table A1](https://arxiv.org/html/2609.06055v1#A1.T1)

## 4. TTS：用短期模拟和 critic，重新选择第一步动作

DriveRL 学到了动作分布和价值函数，但普通评测只取 Beta 众数，没有利用其他可能动作，也没有利用 critic 做推理决策。论文加入的 test-time search，简称 TTS，就是把这两部分重新用起来。

具体流程是：

1. 保留策略众数作为候选 0，再采样 $N-1$ 个首步动作。
2. 每个候选分别模拟 $L$ 步。只有第一步使用候选动作，之后在各自到达的状态上重新调用同一个策略并取众数。
3. 计算这段模拟的折扣奖励，并用末端 critic 补上未来回报估计。
4. 只有最佳采样动作比候选 0 高出阈值 $\delta$，才改变原动作。

未终止时，评分可简写为：

$$
S^{(i)}=\sum_{\ell=0}^{L-1}\gamma^\ell r_\ell^{(i)}
+\gamma^L V\bigl(O_L^{(i)}\bigr).
$$

出现终止时，终止那一步的奖励必须保留；后续奖励和末端价值则清零。论文配置 $L=5$，在 5 Hz 下只向前看约 1 秒，切换阈值为 $0.03$。

背景交通使用受限的恒定转率与加速度模型（CTRA）做短期外推。因此这是一种**基于当前策略的首动作搜索**，没有生成未来视频，也没有构建 MCTS 式的多层分支树。短期外推和 critic 都有误差，保留基线动作并设置切换阈值，是对此作出的保守处理。

增加 $N$ 主要扩大批量并行计算，增加 $L$ 则增加串行的策略、动力学与奖励调用。它们对延迟的影响不同。论文报告 $N=64$ 时 nuPlan 六项平均分从 93.01 提升到 93.57，但没有给出足够的端到端延迟信息，不能只根据分数把它称为“无代价提升”。[论文 §2.1.3、Table 3](https://arxiv.org/html/2609.06055v1#S2.SS1)

## 5. DriveVFM：把四种视觉教师压进一个骨干

### 5.1 每个教师负责什么？

DriveZero 的视觉部分借鉴 RADIO 的多教师蒸馏思路，避免为驾驶骨干另行组织一整套检测、分割、车道和深度标注。

| 冻结教师 | 提供的主要信息 | 蒸馏接口 |
| --- | --- | --- |
| DINOv3 | 空间结构、视觉对应关系 | summary token + patch token |
| SigLIP2 | 图像级语义与开放词汇相关表征 | summary token |
| SAM | 对目标边界敏感的表征 | patch token |
| Depth Anything V2 | 几何相关表征 | 视觉特征；最终方案没有选直接回归深度预测 |

共享 ViT 输出每个教师对应的 summary token，以及公共空间 patch token。通过轻量 adaptor 将它们投影到各教师的特征空间。summary 使用余弦距离，patch 使用 MSE。

这里蒸馏的是教师的**表示能力**。不能把它描述成部署时同时运行 DINOv3、SAM、SigLIP2 和深度模型；预训练结束后，教师和 adaptor 都移除，只保留融合后的 DriveVFM。

“无需任务标注”也限于这个蒸馏阶段：原始图像即可产生教师特征目标，但上游基础模型自身仍有各自的数据和训练来源。[论文 §2.2](https://arxiv.org/html/2609.06055v1#S2.SS2)

### 5.2 PHI-S：不同教师的特征不能直接等权相加

假设教师 A 的特征幅值比教师 B 大十倍，直接计算 MSE 时，A 的损失可能大两个数量级。即使设置教师级 loss weight，一个教师内部不同维度的方差不平衡仍然存在。

DriveVFM 使用 PHI Standardization，简称 PHI-S：离线估计均值和协方差，对特征中心化，通过 PCA–Hadamard 旋转把方差分散到各维，再用共同标量缩放。

重点是**调整回归目标的统计条件**，让教师的重要性不由其激活幅值决定。它不是普通的逐维 whitening：正交旋转加全局缩放不会把协方差的所有特征值都变成一样，也不能解释成“删除所有维度间相关性”。

### 5.3 QKClip：控制注意力 logits 的异常增大

在大规模视觉预训练中，某些 attention head 的 QK 点积可能不断变大，导致 softmax 过度尖锐、loss 突增。论文采用 QKClip：监控 head 的最大注意力 logit，当超过阈值时，在优化步骤后缩放相应的 Q、K 投影权重。

直观上，若某个 head 的最大值为 $m$、阈值为 $c$，同时把 $W_Q$ 和 $W_K$ 乘以 $\sqrt{c/m}$，就会把相应点积按 $c/m$ 缩放。这里是说明缩放关系的简化解释，不是对尚未公开的 DriveVFM 训练代码的逐行复现。

它与逐元素裁剪 attention logits、或者在每次前向中增加 QK normalization 都有区别。附录的阈值为 1,000；作者还比较了精度变化下的特征偏差，QKClip 的余弦一致性更好。但这类特征数值结果不等价于低精度部署后的驾驶安全结论。[论文 §2.2、Table A14](https://arxiv.org/html/2609.06055v1#A2.T14)

### 5.4 特征图应该怎样读？

![DINOv3 与 DriveVFM 的 patch 特征 PCA 和地面相似度可视化](/images/drivezero/pca.png)

*图 3：原论文 Figure 3。依次为输入、DINOv3 PCA、DriveVFM PCA、DINOv3 地面相似度热图和 DriveVFM 地面相似度热图；蓝色表示与地面原型更相似，红色表示更不相似。方框标出长尾障碍物。来源：Xiaomi L3 Team，CC BY-SA 4.0。*

作者用这张图说明融合后的表征能更好地区分某些地面和障碍物。读图时应关注“目标是否被错误地归入地面相似区域”，而不是把 PCA 颜色当成语义类别标签。

这些图既不是深度真值，也不是像素分割精度统计。它们可以解释模型可能在关注什么，真正的效果仍需通过规划消融来判断。

预训练先在 $256\times256$ 分辨率下运行 60 万次迭代，再在 $512\times512$ 下运行 20 万次，两阶段全局 batch 都为 2,048。数据混合 LAION-2B、ImageNet-21K、SA-1B，以及 OpenDV、nuPlan、Waymo 驾驶图像。这是一条相当重的视觉预训练路线。[论文 §2.2、§3.2](https://arxiv.org/html/2609.06055v1)

## 6. DriveZero：如何把结构化教师蒸馏成视觉学生？

### 6.1 最关键的桥梁是“同一帧的两种观测”

日志在时刻 $t$ 同时提供多视角图像 $I_t$ 与结构化状态 $O_t$。先让冻结教师从 $O_t$ 出发 rollout，得到未来轨迹 $\tau_T$，再用它监督输入 $I_t$ 的视觉学生。

教师 rollout 期间，学生不需要看到每一个模拟中间状态的图像：它只需根据起始帧预测整段未来轨迹。这就是基础日志帧蒸馏为什么不需要额外渲染。

但这句话有明确范围。**DriveZero-Scale 加入了 SimScale 仿真数据，不能把“基础配对帧蒸馏不需新渲染”推广成“整套系统不使用渲染数据”。** 此外，蒸馏标签的背景交通使用日志回放，也不同于教师 RL 阶段的混合交互训练。

![DriveZero 视觉学生轨迹蒸馏与候选评分架构](/images/drivezero/distill.png)

*图 4：原论文 Figure 2。教师生成 20 步轨迹；学生生成多个候选，通过 WTA 接收轨迹监督，再通过独立分支学习 PDM 评分。来源：Xiaomi L3 Team，CC BY-SA 4.0。*

### 6.2 从四路图像到 64 条四秒轨迹

附录给出的学生配置为：

| 项目 | 配置 |
| --- | --- |
| 相机 | 前、后、左、右四路当前帧 |
| 单路输入 | $960\times512$，宽 × 高 |
| 视觉骨干 | DriveVFM ViT-L，冻结基础权重，通过 Q/V LoRA 适配 |
| LoRA rank | 32 |
| 场景压缩 | 每个相机 16 个 register token，共 64 个 |
| 轨迹 decoder | 4 层，隐藏维度 256 |
| 候选数 | 64 条 |
| 每条轨迹 | 20 个未来位姿，5 Hz，即 4 秒；包含 $(x,y,\psi)$ |
| 参数量 | 总计 338.46M，其中可训练 18.58M |

视觉 token 加上 3D 位置编码，帮助网络理解不同相机中的空间位置。register token 将大量图像特征压缩成较少的场景表示，供轨迹 decoder 做 cross-attention。

自车运动学和导航命令编码后，加入 64 个可学习轨迹 query，生成不同候选。这里的“camera-only”指环境感知使用相机，不意味着忽略自车状态、导航命令或相机几何信息。[论文 §2.3、Table A12](https://arxiv.org/html/2609.06055v1#A2.T12)

### 6.3 WTA：不要让所有候选都拟合同一条轨迹

设学生生成 $M$ 条候选，轨迹损失采用 winner-takes-all：

$$
\mathcal L_{\mathrm{traj}}
=\min_{m\in\{1,\ldots,M\}}
\frac1T\sum_{t=1}^T
\left\|\hat\tau_{m,t}-\tau_{T,t}\right\|_1.
$$

如果每个候选都对同一条标签做回归，所有候选容易趋同。WTA 只更新最接近教师的那个候选，为不同 query 学习不同模式留下空间。

例如，面对障碍物时“左侧通过”和“右侧通过”都可能合理，直接平均轨迹却可能穿过障碍物。多候选表示希望保留这些分支。不过，WTA 本身并不保证 64 个 query 都被充分使用，更不保证所有合理模式都被覆盖。

### 6.4 评分分支：用什么选出最后一条？

学生的另一个 decoder 把候选轨迹编码成 query，再去关注视觉 token，预测候选的 PDM 分项得分：无碰撞、可行驶区域合规、进度、TTC、舒适性和驾驶方向合规等六项。

这些分项用 BCE 训练，按评测规则聚合后选择轨迹。BCE 的目标可以是 $[0,1]$ 中的软分数，不必都是二元标签。送入评分分支的候选轨迹会 **detach**，阻止评分损失沿这条输入路径反向改变轨迹生成结果。

因此，系统区分了两个问题：轨迹生成分支学习“教师会怎样开”，评分分支学习“这些候选在当前场景中的评价如何”。预测的无碰撞分数只是模型估计，不能直接解释为真实世界已校准的无碰撞概率。

### 6.5 Goal augmentation 才是新监督的来源

对于同一帧图像，日志里只有一个人类意图对应的未来。但目标条件教师可以被重新查询：换一个有效导航目标，再 rollout 一条与新目标一致的轨迹。

这样可以产生类似的数据：

```text
同一场景图像 + 直行指令 → 教师的直行轨迹
同一场景图像 + 转弯指令 → 教师的转弯轨迹
```

这里表达的是目标增强的概念，不表示任意道路都能任意转向。目标必须与路线和场景约束相容；改变目标之后，也必须同步更新学生导航指令及候选轨迹的评价标签。

下面是根据论文整理的**教学伪代码**，用于说明数据依赖。它不是官方已发布的 DriveZero 训练实现：

```python
# I 与 O 来自同一个日志时刻；两个教师相关模块均已预训练。
images, symbolic_state, ego_state, original_goal = batch
goal = augment_valid_goal(symbolic_state, original_goal)
command = command_from_goal(goal)

with torch.no_grad():
    target = rollout_frozen_driverl(
        symbolic_state, goal, steps=20, background="log_replay"
    )

visual_tokens = drivevfm_with_lora(images)
scene_tokens = registers(visual_tokens)
proposals = trajectory_decoder(scene_tokens, ego_state, command)
# proposals: [B, 64, 20, 3]，每个未来位姿是 x、y、yaw。

per_proposal_l1 = (proposals - target[:, None]).abs().mean((-1, -2))
loss_traj = per_proposal_l1.min(dim=1).values.mean()

score_logits = score_decoder(proposals.detach(), visual_tokens)
with torch.no_grad():
    score_targets = evaluate_pdm(proposals.detach(), symbolic_state, goal)
loss_score = binary_cross_entropy_with_logits(score_logits, score_targets)

loss = w_traj * loss_traj + w_score * loss_score
loss.backward()
```

`evaluate_pdm` 表示用场景和评测规则构造监督，不是“在真实未来已经发生之后才规划”。权重 `w_traj`、`w_score` 在这里保持符号形式，不补造论文未明确给出的配置。[论文 §2.3](https://arxiv.org/html/2609.06055v1#S2.SS3)

Scale 版本使用约 100K navtrain 加 237K SimScale 场景，在 16 张 H20 上训练 25 个 epoch，约 38 小时。这是学生训练耗时，不包含前面两个预训练阶段。

## 7. 实验究竟证明了什么？

### 7.1 nuPlan：结构化教师很强，TTS 带来小幅增益

下表摘录论文 Table 2，NR 为非反应式背景交通，R 为反应式背景交通。均为分数越高越好。

| 方法 | Val14 NR | Val14 R | Hard NR | Hard R | Random NR | Random R | 六项均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Log-Replay | 93.53 | 80.32 | 85.96 | 68.80 | 94.03 | 75.86 | 83.08 |
| DriveRL | 95.16 | 94.25 | 89.97 | 89.18 | 94.50 | 95.00 | 93.01 |
| DriveRL + TTS，64 候选 | 95.54 | 94.53 | 91.13 | 89.15 | 95.93 | 95.12 | **93.57** |

TTS 提高了平均分，但 Hard R 从 89.18 变成 89.15，**并不是每个子集都提升**。更重要的是，这张表评价的是拿到结构化状态的 DriveRL，不能当作纯视觉 DriveZero 的成绩。

Log-Replay 在反应式环境中还受到一个结构性限制：其他车会改变行为，而自车仍坚持原来的录制轨迹。因此它是有意义的基线，却不等同于“一个现场人类驾驶员看到新的交互以后重新作出反应”。[论文 Table 2、Table 3](https://arxiv.org/html/2609.06055v1#S3.T2)

### 7.2 NAVSIMv1：“超过人类”主要是综合评分结论

| 方法 | 环境输入 | 无碰撞 NC | 可行驶区域 DAC | TTC | 舒适性 | 进度 EP | PDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human Driver | 日志/GT 参考 | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| DrivoR-Scale | 相机 | 99.1 | 99.2 | 96.9 | 100 | 91.6 | 94.6 |
| DriveRL | 结构化 GT | 99.8 | 99.9 | 99.1 | 99.0 | 91.5 | 95.8 |
| DriveZero | 相机 | 99.0 | 99.2 | 96.0 | 100 | 93.1 | 94.8 |
| DriveZero-Scale | 相机 | 99.2 | 99.4 | 97.3 | 99.9 | 92.5 | **95.3** |

DriveZero-Scale 的综合分比人类参考高 0.5 分，但 NC、DAC、TTC 都没有超过人类参考，其优势包含更高的前进进度。基础版 DriveZero 在表中按一位小数报告为 94.8，不应写成明显领先人类。

所以这里支持的结论是：**在 NAVSIMv1 的特定综合指标下，Scale 版本优于记录的人类轨迹参考。** 它没有证明模型在每个安全维度、所有真实路况或人类现场驾驶能力上都更强。

![DriveZero-Scale 与人类日志轨迹在 NAVSIM 中的定性比较](/images/drivezero/drivezero_vs_human.png)

*图 5：原论文 Figure 4。红色为模型轨迹，绿色为人类轨迹。图中示例展示模型取得更高基准评分的情形；这是作者挑选的案例，不能代替总体统计。来源：Xiaomi L3 Team，CC BY-SA 4.0。*

NAVSIMv1 属于 pseudo closed-loop 评测：会模拟候选轨迹、检查驾驶指标，但不是让视觉策略持续接收自己执行后产生的新视角，再长时间反复决策。它比只算轨迹 L2 误差更接近行为评价，仍与持续视觉闭环有区别。[论文 §3.2、Table 4](https://arxiv.org/html/2609.06055v1#S3.T4)

### 7.3 NAVSIMv2 与 HUGSIM：更值得看偏移状态和持续闭环

| 方法 | NAVSIMv2 navhard EPDMS | HUGSIM HD-Score | HUGSIM 路线完成度 RC |
| --- | ---: | ---: | ---: |
| DrivoR-Scale | 54.6 | 38.1 | 46.4 |
| GigaPixel | 50.1 | 38.5 | 50.1 |
| DriveZero | 51.5 | 39.4 | 53.9 |
| DriveZero-Scale | **57.1** | **46.6** | **54.5** |

不同列是不同协议和指标，只能在同一列内比较。NAVSIMv2 使用更严格的两阶段评价，第二阶段包含自车状态扰动对应的渲染观测。Scale 的优势很有解释力：第一阶段分数从基础版的 84.1 降到 82.3，第二阶段却从 60.8 升到 69.1，最终综合分提高。这与增加仿真数据改善偏移状态覆盖的解释一致，而不只是拟合正常日志帧更好。

HUGSIM 则让策略在渲染环境中持续闭环行动。论文将 NAVSIMv1 的学生直接用于 HUGSIM，不做该基准上的微调，因此这是更有价值的跨环境行为证据。

不过，Scale 在 HUGSIM 的 Hard、Extreme 子集 HD-Score 仍只有 24.7 和 27.6；Extreme 还略低于基础版的 28.7。平均分提升不代表困难交互已经解决。[论文 Table 5、Table 6](https://arxiv.org/html/2609.06055v1#S3.T5)

![HUGSIM 中 DriveZero-Scale 与 DrivoR-Scale 的连续驾驶画面对比](/images/drivezero/drivezero_vs_sota_hugsim_0.png)

*图 6：原论文 Figure 6 的第一组图，上为 DrivoR-Scale，下为 DriveZero-Scale，横轴为连续执行时间。可以观察车辆间距和后续行为的变化；单个案例无法说明平均失败率。来源：Xiaomi L3 Team，CC BY-SA 4.0。*

### 7.4 最有信息量的消融：RL 标签本身并不自动更好

在论文 Table 7 的同组监督方式消融中：

| 监督来源 | NAVSIMv1 PDMS |
| --- | ---: |
| 人类轨迹 | 93.92 |
| 仅 DriveRL 轨迹 | 93.61 |
| DriveRL 轨迹 + goal augmentation | **94.41** |

**仅替换为 RL 轨迹下降 0.31 分，加入目标增强后提升 0.80 分，并比人类监督高 0.49 分。** 因此，这篇论文更有力地支持“可查询教师带来的行为扩展”，而不是“RL 产生的每条轨迹天然比人类好”。

视觉方面，同为 ViT-S 的 DINOv3 为 93.88，DriveVFM 为 94.41。附录中 DINOv3 ViT-L 为 94.55，DriveVFM ViT-L 为 94.83，相差 0.28 分。多教师表征有贡献，但收益需要结合骨干规模、行为监督、目标增强和额外数据一起解释，不能将所有提升归于一个视觉模块。

这些对比给出了组合有效的证据；小数点后的小差距如果缺少多次运行的方差，也不宜扩大成稳定的巨大优势。[论文 Table 7、Table A13](https://arxiv.org/html/2609.06055v1#S3.T7)

## 8. 源码阅读：当前真正能读到什么？

### 8.1 先确认开放范围

官方仓库 2026-09-15 公告发布的是 **DriveRL inference code and checkpoints**。当前包含 `driverl-teacher-u2400`、`driverl-selfplay-u3800` 两组推理配置与权重，配套 nuPlan 适配及评测脚本。

DriveZero 和 DriveVFM 仍在 TODO 中；当前发布内容不能提供视觉学生从图像到轨迹的完整运行实例，也不是完整的教师 PPO 训练工程。发布 manifest 的评测状态还写着 `artifact_bundled_pending_public_eval`。这表示文件已打包，不能据此宣称公开权重已经逐项复现论文表格。

以下阅读路径针对 DriveRL。所有示例都应配合固定提交，而不是假定未来 main 分支一直相同。[仓库 README](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/README.md)、[权重 manifest](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/release/driverl_checkpoints.yaml)

### 8.2 推荐沿着一次规划调用向下读

下面路径都相对于仓库内的 `DriveRL/`：

| 入口 | 职责 | 阅读重点 |
| --- | --- | --- |
| `src/driverl/nuplan/planner.py` | 接收 nuPlan 观测，调用策略或 TTS，适配规划接口 | `compute_planner_trajectory` 中如何分流普通推理和 TTS |
| `src/driverl/nuplan/feature_builder.py` | 组装历史、自车、地图、路线目标 | 坐标系、采样间隔、padding 与 mask |
| `src/driverl/agents/learning/vanilla_net_agent.py` | 将场景转换为网络需要的张量 | 预处理、动作编码与推理采样 |
| `src/driverl/agents/learning/networks/vanilla_net.py` | attention、目标编码、Beta 动作头、critic | `forward` 及 `enable_value_decomposition` |
| `src/driverl/nuplan/tts.py` | 首动作候选生成、模拟、评分、保守选择 | `propose_candidate_actions`、`select_action`、`_rollout_candidates` |
| `src/driverl/env/engine/reward_decomposition.py` | 将乘积软奖励分配到价值通道 | 总奖励守恒，以及额外 `soft_other` 通道 |
| `src/driverl/nuplan/controller.py` | 执行策略控制与车辆状态更新 | 不要将 nuPlan 轨迹接口包装误当作学生的轨迹生成头 |

推荐先读[planner.py](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/planner.py#L213)，再读[网络 forward](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/agents/learning/networks/vanilla_net.py#L269)，最后进入[TTS 实现](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/tts.py#L264)。这样每个类都能对应到真实数据流，而不会陷入配置文件细节。

### 8.3 从 `vanilla_net.py` 看目标编码和动作头

代码先把展平的目标恢复为 `[..., goal_count, 2]`，共享位置 MLP 后沿目标数量维度求平均。默认多目标路径的关键语义可以简化为：

```python
# 按公开代码整理的简化片段；省略旧版与可选编码分支。
goal_xy = goal_features.reshape(B, 1, goal_count, 2)
goal_tokens = goal_mlp(sine_position_encoding(goal_xy))
goal_embedding = goal_tokens.mean(dim=2)  # 交换目标点不改变结果

raw_alpha, raw_beta = action_head(fused_ego).chunk(2, dim=-1)
alpha = softplus(raw_alpha) + 1.0
beta = softplus(raw_beta) + 1.0
```

实际预处理还包含目标状态等字段，不能把整个原始 `goal_features` 不加区分地 reshape。这里为了突出坐标编码，已经省略那些字段。

网络输出的动作参数形状是 `[B, 1, 4]`：自车一个 token，两个动作维度各有 alpha、beta。它不是一条 `[B, T, 3]` 的预测轨迹。价值输出则是 `[B, 1, value_dim]`。

这两点直接从[目标编码](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/agents/learning/networks/vanilla_net.py#L346)和[Beta head](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/agents/learning/networks/vanilla_net.py#L540)就可以核对，也说明了教师与视觉学生接口的根本不同。

### 8.4 从 `tts.py` 看终止掩码和保守选择

下面是对应源码控制流程的教学伪代码，省略批量维度、随机种子和数值检查：

```python
first_actions = [policy_mode(obs)] + sample_policy(obs, count=N - 1)
state = repeat_state(obs, N)
alive = ones(N, dtype=bool)
returns = zeros(N)

for step in range(5):
    action = first_actions if step == 0 else policy_mode(state)
    state, reward, done = rollout_one_step(state, action, background="CTRA")
    returns += (gamma ** step) * alive * reward
    alive = alive & (~done)  # 在累加奖励后更新，保留终止事件的惩罚

bootstrap = critic(state).sum(dim=-1)  # 对价值通道求和
returns += (gamma ** 5) * alive * bootstrap
best = returns.argmax()
chosen = best if returns[best] > returns[0] + 0.03 else 0
return first_actions[chosen]
```

如果先把 `alive` 更新为 false，再乘到当前 reward 上，就会把碰撞终止那一步的惩罚也抹掉。这种小错误足以改变动作排序。公开实现明确先保留当前步奖励，再处理终止。

真实代码还检查候选动作、rollout 奖励、bootstrap 和总分是否有限。基线候选出现非有限值会报错；无效的其他候选不会参与比较。这些逻辑和搜索公式一样，属于复现结果的必要部分。[候选评分与选择](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/tts.py#L300)、[rollout 终止处理](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/tts.py#L461)

### 8.5 三处论文与发布配置的差异

**第一，价值通道是 8 还是 9。** 论文描述硬事件、目标、六个软项，共 8 通道。公开 `VALUE_COMPONENT_NAMES` 还包含 `soft_other`，所以 `DECOMPOSED_VALUE_DIM` 实际为 9。它用于容纳未映射到六个标准名称的软项等情况。自行重建网络时硬编码 8，会与发布结构不一致。[reward_decomposition.py](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/env/engine/reward_decomposition.py#L7)

**第二，默认 TTS 不等于论文最高分配置。** 发布 planner 默认关闭 TTS，开启后的默认候选数为 8；论文 93.57 的结果对应 64 候选。仅仅使用默认命令，不能期待自动得到论文最强配置。[config.py](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/config.py#L120)

**第三，不同 release 的奖励配置确实不同。** 当前 `driverl_teacher.yaml` 中，off-road 权重为 -1.0、到达阈值为 3.0 米、目标奖励为 0.3；`driverl_selfplay.yaml` 对应为 -2.0、1.5 米和 0.5，后面这组数值与论文所述基线更接近。TTS 会从加载的 engine 配置构建奖励计算器，所以配置差异会影响候选排名。

这不构成对论文成绩的否定，但说明应当以 **checkpoint + YAML + 评测脚本 + 数据划分** 为一组复现单位，不能将论文参数、teacher 权重和 self-play 配置任意混搭。公开推理包也不足以反推出这些 checkpoint 完整的训练历史。[teacher 配置](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/release/configs/driverl_teacher.yaml)、[self-play 配置](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/release/configs/driverl_selfplay.yaml)、[TTS 构建奖励引擎](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/tts.py#L171)

### 8.6 如果继续跑代码，应该从哪里开始？

先固定本文阅读的版本，并进入真正的 Python 工程目录：

```bash
git clone https://github.com/XiaomiAutoL3/DriveZero.git
cd DriveZero
git checkout 24959547edac549d64c34b2aab62d5c52599d078
cd DriveRL
```

然后按照仓库的[环境文档](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/docs/driverl_environment.md)准备 Python 3.11、PyTorch 2.7.1、NumPy 1.26.4；GPU 文档使用 CUDA 12.8 对应的 PyTorch 构建。项目本体和配套 `nuplan-devkit` 都需要安装。

在安装依赖、准备 nuPlan 数据和地图之后，从[复现文档的 preflight](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/docs/driverl_repro_eval.md)开始，再做普通六项评测，最后切换到 TTS。明确设置 candidate 数量和 seed，并保存逐场景日志。

这条路径能检验公开 DriveRL 推理包；它还不能运行未公开的 DriveZero 视觉学生。本文没有在本机安装该研究环境或下载 nuPlan 数据，以上属于源码与文档对应的复现入口。

## 9. 实车演示和技术迁移，应该怎样判断？

### 9.1 实车章节展示的是 DriveRL 接入感知系统

论文 §3.3 展示了 real-world deployment，但系统链路是：目标检测、在线地图与导航模块输出结构化信息，再交给 DriveRL。该策略使用自有日志、self-play、随机化以及感知退化相关训练设置。

这支持“结构化 RL 行动策略可以接入真实车辆”的可行性。它并不等价于证明前面 NAVSIM 的 DriveZero 纯视觉学生已经完成同样的实车部署，也没有提供足以推导道路安全水平的里程和事故统计。[论文 §3.3](https://arxiv.org/html/2609.06055v1#S3.SS3)

### 9.2 对机器人导航最有用的启发

以下是基于论文机制的技术推断，不是作者已经验证的机器人实验。

**先在结构化空间学行为，可以显著降低闭环训练负担。** 对移动机器人，可以用占据、障碍物、相对速度、局部路径与目标等表示构造教师输入，而不必一开始就在高保真视觉渲染里完成所有探索。真正需要保证的是，这些信息与最终视觉策略可推断的信息之间存在可学习的联系。

**目标条件教师可以成为主动数据生成器。** 同一观测下改变有效目的地、路线分支或绕行意图，能补充被动日志无法提供的行为监督。导航命令、目标和评价规则必须一起更新，否则增强出来的只是互相矛盾的样本。

**恢复行为要同时覆盖状态和视觉。** 如果教师在结构化仿真里学会从障碍物边缘退出，但学生只见过走廊中心线上的相机图像，能力转移就可能失败。需要真实偏移数据、渲染视角增强或其他能保持几何一致性的观测补充。DriveZero-Scale 在 NAVSIMv2 第二阶段的提升，正是这类问题值得单独投入数据预算的线索。

**教师奖励与学生评分器应分别审查。** 教师在学怎样行动，学生评分器在估计候选是否合适。两者都可能适应某个 benchmark 的偏好；高综合分仍需要拆解到碰撞、可通行区域、进度、舒适性和困难场景。

DriveZero 最值得借鉴的，是将“学会看”“学会行动”和“把行为交给视觉策略”拆成可分别扩展的训练问题。它的实验尤其提醒我们：可查询的教师与行为覆盖，比单纯更换模仿标签来源更有价值；而视觉偏移覆盖、评价目标及公开复现之间的差距，仍然需要独立解决。

## 参考资料与图像说明

- [DriveZero 论文 v1：完整方法、实验与附录](https://arxiv.org/html/2609.06055v1)
- [arXiv 摘要页：作者、提交日期与版本](https://arxiv.org/abs/2609.06055)
- [Xiaomi L3 Team 官方项目页：架构和演示](https://xiaomiautol3.github.io/DriveZero/)
- [官方代码：本文核对的固定提交](https://github.com/XiaomiAutoL3/DriveZero/tree/24959547edac549d64c34b2aab62d5c52599d078)
- [DriveRL 推理复现指南](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/docs/driverl_repro_eval.md)

文中论文原图来自 Hao He 等 / Xiaomi L3 Team 的 arXiv v1，遵循原文 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可；图 2 为本文依据论文重绘，也按同一许可提供。原图未经内容修改，图 6 使用原论文 Figure 6 的第一组独立图片。本文作为中文解读与改编，同样按 CC BY-SA 4.0 提供。图像均保存在本站，方便随博客一起发布。
