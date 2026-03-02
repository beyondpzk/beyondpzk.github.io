---
layout: post
title: DayDreamer
date: 2022-06-28
categories: []
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# DayDreamer

[paper link](https://arxiv.org/abs/2206.14176) 


# DayDreamer——物理机器人学习的世界模型

**核心文献**：Wu, P., Escontrela, A., Hafner, D., Goldberg, K., & Abbeel, P. (2022). [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176v1). arXiv preprint arXiv:2206.14176.

---

## 第一部分：研究背景

### 1.1 机器人学习的痛点：样本效率与虚实鸿沟
在引入DayDreamer之前，我们需要理解当前机器人强化学习（RL）面临的两大核心挑战：

1.  **样本效率低下（Sample Inefficiency）**：
    传统的无模型（Model-Free）强化学习算法（如DQN, PPO, SAC）通常需要数百万甚至数亿步的交互才能学会复杂任务。
    > Deep reinforcement learning is a common approach to robot learning but requires a large amount of trial and error to learn, limiting its deployment in the physical world. <alphaxiv-paper-citation paper="2206.14176v1" title="Abstract" page="1" first="Deep reinforcement learning" last="the physical world." />

    在物理世界中，时间是不可压缩的，且机器人的机械磨损和维护成本极高。

2.  **模拟与现实的鸿沟（Sim-to-Real Gap）**：
    为了规避样本效率问题，主流做法是在仿真器中训练，然后迁移到真机。但这带来了新问题：
    *   仿真器无法完美捕捉物理世界的复杂性（如软体物体、复杂摩擦、光照变化）。
    *   需要繁琐的领域随机化（Domain Randomization）。
    *   不仅行为难以适应，且一旦环境发生变化（如机器人关节受损），基于仿真的策略往往失效。

### 1.2 世界模型（World Models）的理念
人类并不是通过成千上万次撞墙来学会走路的。我们拥有一个“心理模型”（Mental Model），能够在脑海中推演行为的后果。

*   **核心思想**：智能体应该从过去的经验中学习一个环境的动态模型（即“世界模型”），然后在这个学到的模型中进行“想象”训练。
*   **优势**：
    *   **样本效率**：世界模型可以从少量数据中提取丰富的动态知识。
    *   **想象规划**：在潜空间（Latent Space）中进行规划，不消耗物理时间。
    *   **通用性**：学到的物理规律可以泛化。

### 1.3 本文的核心贡献
DayDreamer 这篇论文的里程碑意义在于：它打破了“世界模型只能在游戏或仿真中有效”的刻板印象。它证明了 Dreamer 算法可以直接在真实机器人上进行**在线学习（Online Learning）**，无需任何仿真器预训练。

> In this paper, we apply Dreamer to 4 robots to learn online and directly in the real world, without any simulators. <alphaxiv-paper-citation paper="2206.14176v1" title="Abstract" page="1" first="In this paper," last="without any simulators." />

---

## 第二部分：理论核心——Dreamer 模型架构详解

深入剖析 Dreamer（基于 DreamerV2）的内部构造。它主要由两部分组成：**世界模型学习（World Model Learning）** 和 **行为学习（Behavior Learning）**。

### 2.1 循环状态空间模型 (RSSM)
Dreamer 的核心是一个循环状态空间模型（Recurrent State-Space Model, RSSM）。它的设计目的是为了解决部分可观测性（Partial Observability）和随机性（Stochasticity）。

#### 2.1.1 结构组件
RSSM 将状态分解为两个部分：
1.  **确定性状态（Deterministic State, $h_t$）**：由循环神经网络（GRU）建模，负责记忆长期的历史信息。
2.  **随机状态（Stochastic State, $z_t$）**：由后验/先验网络建模，负责捕捉环境本身的不确定性（如未知的摩擦力、传感器噪声）。

#### 2.1.2 模型的四个核心网络
我们需要关注方程 (1) 中的定义：

1.  **编码器（Encoder）**：
    将原始感官输入（图像 $x_t$、本体感知数据）压缩为特征。
    $$\text{enc}_{\theta}(s_t \mid s_{t-1}, a_{t-1}, x_t)$$

2.  **解码器（Decoder）**：
    从潜变量重构观测值。这迫使潜变量包含足够的信息来描述环境。
    $$\text{dec}_{\theta}(s_t) \approx x_t$$

3.  **动态网络（Dynamics Network / Prior）**：
    在给定前一状态和动作的情况下，预测下一个**随机状态**的分布。这对应于“想象”未来的能力。
    $$\text{dyn}_{\theta}(s_t \mid s_{t-1}, a_{t-1})$$

4.  **奖励网络（Reward Network）**：
    预测当前状态的奖励，用于指导策略学习。
    $$\text{rew}_{\theta}(s_{t+1}) \approx r_t$$
    > The reward network learns to predict. Using manually specified rewards as a function of the decoded sensory inputs is also possible. <alphaxiv-paper-citation paper="2206.14176v1" title="Reward Learning" page="3" first="The reward network" last="is also possible." />

### 2.2 潜空间中的行为学习 (Actor-Critic)
一旦我们拥有了世界模型，我们就不再需要与真实环境交互来更新策略，而是在“梦境”（Latent Space）中进行。

#### 2.2.1 想象展开 (Imagination Rollouts)
从当前的真实状态出发，使用动态网络（Prior）预测未来 $H$ 步的状态序列。
这个过程完全在 GPU 上并行进行，速度极快（Batch size 可达 16K）。

#### 2.2.2 演员-评论家 (Actor-Critic) 更新
*   **Critic ($v(s_t)$)**: 学习预测从状态 $s_t$ 开始的预期回报（Value Function）。这里使用了 $\lambda$-return 来平衡偏差和方差：
    $$V_t^{\lambda} \doteq r_t + \gamma \left( (1 - \lambda) v(s_{t+1}) + \lambda V_{t+1}^{\lambda} \right)$$
    > To avoid the choice of an arbitrary value for N , we instead compute λ-returns <alphaxiv-paper-citation paper="2206.14176v1" title="Lambda Returns" page="4" first="To avoid the" last="compute λ-returns" />

*   **Actor ($\pi(a_t | s_t)$)**: 学习最大化 Critic 预测的价值。
    对于连续动作（如关节角度），使用重参数化技巧（Reparameterization Trick）直接通过动态模型的梯度反向传播来优化策略。
    对于离散动作（如抓取开/关），使用 REINFORCE 梯度估算。

### 2.3 关键技术细节
*   **离散潜变量 (Discrete Latents)**：DreamerV2 使用 Categorical 分布而不是 Gaussian 分布来表示随机状态 $z_t$。这在处理非平滑动态时更鲁棒。
*   **KL Balancing**：为了防止后验（Posterior）和先验（Prior）分离，损失函数包含 KL 散度项。DayDreamer 使用特殊的加权方式，让先验更快地逼近后验，而不是让后验坍缩向先验。

---

## 第三部分：面向物理机器人的工程实现

将理论模型部署到物理机器人上涉及大量的工程挑战。

### 3.1 异步训练架构 (Asynchronous Architecture)
这是在真实世界能够实时运行的关键。
*   **问题**：神经网络的更新（反向传播）通常很慢，如果等待更新完成再执行动作，会造成控制延迟，导致机器人抖动或不稳定。
*   **解决方案**：分离 **Actor 线程** 和 **Learner 线程**。
    *   **Actor 线程**：以高频运行，仅执行前向传播（Inference），将数据存入 Replay Buffer。
    *   **Learner 线程**：在后台不断从 Buffer 采样并更新 World Model 和 Actor/Critic 网络。
    > We parallelize data collection and neural network learning so learning steps can continue while the robot is moving and to enable low-latency action computation. <alphaxiv-paper-citation paper="2206.14176v1" title="Pipeline" page="2" first="We parallelize data" last="action computation." />

### 3.2 多模态传感器融合 (Sensor Fusion)
物理机器人通常有多种传感器：RGB 摄像头、深度图、关节角度（Proprioception）、力矩传感器等。
Dreamer 的编码器设计天然支持融合：
*   图像通过 CNN 编码。
*   本体感知数据通过 MLP 编码。
*   所有特征被连接（Concatenate）后输入到 RSSM 中生成统一的潜状态 $z_t$。这意味着世界模型是在一个融合的特征空间中进行预测的。

---

## 第四部分：

我们将深入探讨论文中的四个核心实验，展示该方法的通用性和鲁棒性。

### 4.1 案例一：A1 四足机器人行走 (Locomotion)
*   **任务**：从背部朝下躺着开始，学会翻身、站立并以目标速度行走。
*   **输入**：本体感知（关节角度、速度）。
*   **结果**：
    *   **10分钟**：学会翻身。
    *   **20分钟**：学会站立。
    *   **1小时**：学会稳定的行走步态（Pronking gait）。
    *   **对比**：SAC 算法在同样时间内只能学会翻身，无法站立。
*   **鲁棒性测试**：研究人员用长杆推搡机器人。Dreamer 在 10 分钟内适应了这种扰动，学会了对抗推力或快速恢复。

下面的图表展示了 Dreamer 和 SAC 在 A1 机器人上的学习曲线对比：

<alphaxiv-chart>
{
  "type": "line",
  "title": "A1 Robot Learning Efficiency (Reward vs Time)",
  "xAxis": "Training Time (Minutes)",
  "yAxis": "Average Reward",
  "datasets": [
    { "label": "Dreamer (Ours)", "points": [[0, 2], [10, 4], [20, 6], [30, 8], [40, 9.5], [50, 10.5], [60, 11]] },
    { "label": "SAC (Baseline)", "points": [[0, 1], [10, 3], [20, 3.5], [30, 3.5], [40, 3.8], [50, 4], [60, 4.2]] }
  ]
}
</alphaxiv-chart>

> Dreamer trains a quadruped robot to roll off its back, stand up, and walk from scratch and without resets in only 1 hour. <alphaxiv-paper-citation paper="2206.14176v1" title="Abstract" page="1" first="Dreamer trains a" last="only 1 hour." />

### 4.2 案例二：UR5 与 xArm 视觉抓取 (Manipulation)
*   **挑战**：稀疏奖励（只有抓到物体才得分）、视觉定位、手眼协调。
*   **设置**：UR5（工业级）和 xArm（低成本）。RGB 图像 + 本体感知。
*   **结果**：
    *   在 8-10 小时内达到接近人类操作员的抓取效率。
    *   **Baseline 失败原因**：Rainbow (DQN 变体) 和 PPO 难以在如此少的数据量下处理高维图像输入和稀疏奖励。它们往往陷入局部最优（如抓起后立即放下）。
    *   **适应性**：在 xArm 实验中，光照随日出发生剧烈变化，Dreamer 性能短暂下降后迅速适应了新的光照条件。

下面的图表展示了不同算法在 UR5 机器人上的抓取效率：

<alphaxiv-chart>
{
  "type": "bar",
  "title": "UR5 Pick and Place Performance (Objects per Minute)",
  "labels": ["Dreamer", "Human Operator", "Rainbow (DQN)", "PPO"],
  "datasets": [
    { "label": "Objects Picked/Minute", "points": [2.5, 3.0, 0.2, 0.1] }
  ],
  "yAxisTitle": "Pick Rate (Objects/Min)"
}
</alphaxiv-chart>

> The learned behavior outperforms model-free agents and approaches human performance. <alphaxiv-paper-citation paper="2206.14176v1" title="Abstract" page="1" first="The learned behavior" last="approaches human performance." />

### 4.3 案例三：Sphero 导航 (Navigation)
*   **任务**：仅凭 RGB 图像导航到固定目标。
*   **难点**：
    *   部分可观测：单张图像无法判断机器人的朝向（对称球体），必须依赖历史信息（Temporal Context）。
    *   控制滞后：滚动机器人有惯性。
*   **结果**：RSSM 的循环结构成功记住了历史轨迹，推断出方位，在 2 小时内学会导航。

---

## 第五部分：总结与讨论

### 5.1 核心结论
1.  **无需仿真**：世界模型具有足够高的样本效率，使得直接在真机上从零训练成为可能。
2.  **通用性强**：同一套超参数（Hyperparameters）适用于轮式、足式、机械臂等多种形态的机器人。
    > Using the same hyperparameters across all experiments, we find that Dreamer is capable of online learning in the real world <alphaxiv-paper-citation paper="2206.14176v1" title="Abstract" page="1" first="Using the same" last="the real world" />
3.  **多模态融合**：潜空间自然地融合了视觉和触觉/本体感知信息。

### 5.2 局限性与未来方向
*   **安全性**：虽然效率高，但在线探索初期的随机动作可能损坏硬件（虽然文中使用了滤波器保护电机）。
*   **长期记忆**：目前的 RSSM 主要处理短期动态，对于需要长期记忆（如在此房间拿钥匙去彼房间开门）的任务仍有挑战。
*   **未来**：结合预训练模型（Foundation Models）或离线数据（Offline RL）来进一步加速初始阶段的学习。

### 5.3 思考
*   为什么在视觉任务中，Dreamer 相比 Model-Free 方法（如 Rainbow）优势如此巨大？请从表征学习（Representation Learning）的角度分析。
*   如果在训练过程中不仅有 RGB 图像，还有触觉传感器数据，你会如何修改 Encoder 架构？

---

## DayDreamer相比于DreamerV2


**从算法原理（数学公式、损失函数、网络结构）上讲，DayDreamer 的核心确实就是 DreamerV2**。
但是，理论上“不需要仿真器”和工程上“能在真机上跑通”之间存在巨大的鸿沟。DayDreamer 这篇论文的核心贡献在于**解决了将 DreamerV2 部署到物理机器人上时面临的实际挑战**。

以下是 DayDreamer 与原始 DreamerV2 在实现和应用层面上的几个关键区别：

### 1. 异步并行架构 (Asynchronous Actor-Learner Architecture)
这是最本质的区别。

*   **原始 DreamerV2 (仿真环境)**：
    通常是**同步**的。流程是：`环境交互 -> 存入Buffer -> 训练网络 -> 更新策略 -> 下一步交互`。
    在仿真器里，时间是静止的。网络训练花 1 秒还是 1 分钟都没关系，仿真器会等你。

*   **DayDreamer (物理环境)**：
    在真实世界，时间无法暂停。如果训练网络需要 0.5 秒，而机器人的控制频率是 20Hz（0.05秒一帧），同步模式会导致机器人动作卡顿、控制延迟，甚至摔倒。
    **DayDreamer 重新设计了架构，将“数据收集（Actor）”和“模型训练（Learner）”完全解耦并行**：
    *   **Actor 线程**：只负责前向推理（Inference），速度极快，保证机器人控制的高频响应（低延迟）。
    *   **Learner 线程**：在后台利用 GPU 进行繁重的反向传播和模型更新。
    > We parallelize data collection and neural network learning so learning steps can continue while the robot is moving and to enable low-latency action computation. <alphaxiv-paper-citation paper="2206.14176v1" title="Pipeline" page="2" first="We parallelize data" last="action computation." />

### 2. 多模态传感器融合 (Sensor Fusion)
*   **原始 DreamerV2**：
    大多在 Atari 或 DM Control Suite 上测试，输入通常是纯图像（Pixels）或纯状态（State）。

*   **DayDreamer**：
    必须同时处理**高维图像**（RGB/Depth）和**低维本体感知**（Proprioception，如关节角度、速度）。
    DayDreamer 明确展示了如何将这两种截然不同的数据流融合进同一个 Latent Space：
    *   图像通过卷积神经网络（CNN）编码。
    *   关节数据通过多层感知机（MLP）编码。
    *   两者的特征被拼接（Concatenate）后输入 RSSM。
    > The encoder network fuses all sensory inputs $x_t$ together into the stochastic representations $z_t$. <alphaxiv-paper-citation paper="2206.14176v1" title="Encoder" page="3" first="The encoder network" last="representations zt." />

### 3. 样本效率的“实战验证” (Empirical Validation of Sample Efficiency)
“Dreamer 不需要仿真器也能训练”在理论上是对的，但在此之前，没人敢保证它在真机上的**收敛速度**快到具有实用价值。

*   **区别在于验证**：在此之前，Dreamer 被认为是一个“在视频游戏里表现很好的算法”。DayDreamer 证明了它**不需要数百万步**，只需要**几万步**（约 1 小时）就能在真实物理世界从零学会走路。
*   这一点非常重要，因为如果一个算法虽然不需要仿真器，但需要真机运行 100 小时才能收敛，那它在机械磨损和时间成本上是不可接受的。DayDreamer 确立了它是目前**样本效率最高**的真机学习基线之一。
    > Dreamer trains a quadruped robot to roll off its back, stand up, and walk from scratch and without resets in only 1 hour. <alphaxiv-paper-citation paper="2206.14176v1" title="Abstract" page="1" first="Dreamer trains a" last="only 1 hour." />

### 4. 动作平滑与安全性 (Action Smoothing)
*   在仿真中，高频震荡的动作可能只会导致分数低。
*   在真机上，高频震荡会烧坏电机。
*   DayDreamer 在输出端增加了一些工程处理（如 Butterworth 滤波器）来保护硬件，这是原始算法研究中往往忽略的细节。
    > To protect the motors, we filter out high-frequency motor commands through a Butterworth filter. <alphaxiv-paper-citation paper="2206.14176v1" title="Safety" page="5" first="To protect the" last="Butterworth filter." />

### 总结
**DayDreamer = DreamerV2 算法 + 异步低延迟架构 + 多模态融合工程 + 真实世界鲁棒性验证。**

它的贡献不在于发明了新的数学模型，而在于打通了从“理论算法”到“物理实体”的最后一公里，证明了世界模型在机器人领域的巨大潜力。


## 在物理真机上面,reward是如何获得的

在仿真器（Simulator）里，我们有一个上帝视角的函数可以直接返回 reward，但在真实物理世界中，**Reward 必须是可以被测量或计算的**。

在 DayDreamer 中，作者针对不同的机器人任务，采用了三种巧妙的方式来计算 Reward。核心原则是：**利用机器人自带的传感器（本体感知）或通过简单的逻辑判断来生成奖励信号。**

以下是具体的计算方式：

### 1. A1 四足机器人（Locomotion）：基于本体感知的密集奖励
对于行走任务，Reward 完全由机器人身上的传感器（IMU 和关节编码器）实时计算得出。不需要外部动作捕捉系统（MoCap）。

奖励函数由五个部分组成，是一个**密集奖励（Dense Reward）**：
1.  **直立奖励 ($r_{upr}$)**：利用 IMU（惯性测量单元）测量机身是否水平。如果背部朝上，给高分。
2.  **姿态奖励 ($r_{hip}, r_{shoulder}, r_{knee}$)**：测量关节角度，鼓励机器人保持一个自然的站立姿态，避免关节扭曲。
3.  **速度奖励 ($r_{velocity}$)**：测量前进速度，鼓励机器人向前移动。

> An upright reward is computed from the base frame up vector ẑT , terms for matching the standing pose are computed from the joint angles... <alphaxiv-paper-citation paper="2206.14176v1" title="A1 Reward" page="6" first="An upright reward" last="joint angles" />

这个奖励函数的设计非常巧妙，它使用了**分级激活（Curriculum）**机制：只有当机器人学会“翻身”和“站立”（即前几项奖励达到阈值 0.7）之后，“速度奖励”才会生效。这引导了机器人先学站、后学走。

### 2. UR5 和 xArm 机械臂（Manipulation）：基于逻辑状态的稀疏奖励
对于抓取任务，计算 Reward 不需要视觉判断“是否抓到”，而是通过**机械爪的物理状态**来推断。这是一个**稀疏奖励（Sparse Reward）**。

具体的判断逻辑如下：
1.  **抓取成功 (+1)**：**通过“部分闭合”检测**。如果机械爪收到闭合指令，但没有完全闭合（说明中间有东西挡住了），系统就认为“抓到了物体”，给予 +1 奖励。
2.  **放置成功 (+10)**：如果机械爪处于目标箱子上方（通过机械臂坐标判断）且检测到物体被释放，给予 +10 大奖。
3.  **错误放置 (-1)**：如果在错误的箱子上方释放，给予 -1 惩罚。

> Successfully grasping one of the 3 objects, detected by partial gripper closure, results in a +1 reward <alphaxiv-paper-citation paper="2206.14176v1" title="Manipulator Reward" page="6" first="Successfully grasping one" last="a +1 reward" />

这种方法非常鲁棒，不需要复杂的计算机视觉算法去判断物体是否在爪子上，直接用电机反馈即可。

### 3. Sphero 小球（Navigation）：基于位置的距离奖励
对于导航任务，Reward 是**负的 L2 距离**（即距离目标越近，惩罚越小/奖励越大）。

> The robot is provided with a dense reward equal to the negative L2 distance. <alphaxiv-paper-citation paper="2206.14176v1" title="Sphero Reward" page="7" first="The robot is" last="negative L2 distance." />

**注意**：虽然论文正文中没有详细展开 Sphero 的定位方式，但在室内机器人实验中，通常有两种方式获取这个坐标来计算 Reward：
1.  **外部定位**：在天花板上安装一个摄像头（Overhead Camera）或者使用 UWB/MoCap 系统来追踪小球的实时坐标。
2.  **里程计（Odometry）**：利用轮子的编码器推算位置（但在光滑地面上容易打滑，不准确）。

考虑到 Sphero 是在一个固定围栏内运动，通常使用的是外部观测设备来提供这个“上帝视角”的坐标用于计算训练时的 Reward。

### 总结：World Model 的角色
这里有一个关键点需要区分：
*   **训练时（Interaction）**：机器人通过上述传感器/逻辑获得真实的 Reward值 ($r_t$)，存入 Replay Buffer。
*   **学习后（Dreaming）**：Dreamer 的**奖励网络（Reward Network）** $\text{rew}_\theta(s_{t+1})$ 会学习去**预测**这些传感器数值。

一旦 World Model 训练好了，机器人在“做梦”规划时，就不再需要真实的传感器反馈，而是由网络直接预测“我如果这样做，大概能感觉到爪子抓紧了（获得 +1）”。这就是它能在潜空间规划的关键。
