---
title: "MotionWAM：让世界动作模型实时驱动人形机器人全身移动操作"
date: 2026-06-08
categories: [WAM]
description: "围绕MotionWAM：让世界动作模型实时驱动人形机器人全身移动操作整理研究背景、核心方法、实验结果与应用边界。"
---

# MotionWAM：让世界动作模型实时驱动人形机器人全身移动操作

> **论文**：*MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation*<br>
> **作者**：Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang<br>
> **单位**：Mondo Robotics、HKUST (GZ)、HKUST<br>
> **版本**：arXiv:2606.09215v1，2026-06-08，尚未注明正式会议录用<br>
> **链接**：[论文](https://arxiv.org/abs/2606.09215) · [项目页](https://dit4dit.github.io/MotionWAM/) · [演示视频](https://youtu.be/_U5gyj2cuSc)

---

## 摘要

MotionWAM 研究的是一个比桌面机械臂困难得多的问题：能否让同一个策略从头部单目相机出发，实时控制人形机器人的行走、蹲起、躯干、双臂、夹爪，甚至主动用脚踢球或踩踏板？

论文给出的答案是一套 **Video DiT + Motion DiT + SONIC 全身控制器** 的分层系统。Video DiT 不在部署时完整生成未来视频，而是在接近纯噪声的流时间步上只做一次前向传播，把中间去噪特征交给 Motion DiT；Motion DiT 再用 4 个流匹配推理步生成动作块。动作不直接表示成 29 个关节的裸控制量，而是表示成 SONIC 的全身运动 token 与连续末端执行器值，由低层控制器解码并维持动态平衡。

在 Unitree G1 的 9 个真实任务上，MotionWAM 平均成功率为 **76.1%**，最强对比方法 GR00T-N1.7 为 **43.9%**。它在单张 A100 上以 **4.9 Hz 的动作块频率**运行，是 Cosmos Policy 的 7 倍。论文最重要的价值并不是“生成的视频更漂亮”，而是说明：**视频生成模型中尚未被渲染成像素的中间动态表征，可以成为实时机器人策略的有效条件；统一的全身动作接口则决定了策略究竟能不能表达脚、腰和手的协同行为。**

![MotionWAM 展示的五类全身能力：腰部控制、高度调节、蹲伏移动、身体与手部协调、足部交互](/assets/motionwam/teaser.png)

*图 1：MotionWAM 在 Unitree G1 上展示的五类全身能力。图片来自论文 Figure 1，CC BY 4.0。*

---

## 1. 论文到底解决了什么问题？

### 1.1 Loco-manipulation 不是“机械臂加一双腿”

固定底座机械臂只需决定手臂如何运动；人形机器人在伸手、弯腰或搬重物时，每一个操作动作都会改变重心和支撑关系。它必须同时处理：

- **移动**：前进、后退、转向和重新站位；
- **平衡**：根据上身动作及时调整足端和躯干；
- **高度与腰部**：蹲下取低处物体、起身搬运、转腰扩大工作空间；
- **双手操作**：抓取、放置、推拉、擦拭；
- **任务性足部动作**：踢球、踩踏板，而不只是用脚保持平衡。

主流系统通常采用上下半身分层接口：高层操作策略输出精细的手臂关节目标，低层行走控制器只接收基座速度、躯干高度或朝向等粗粒度命令。这个设计有很强的工程合理性，但也形成了表达瓶颈：高层无法指定某一只脚何时、以什么轨迹接触物体，腿只能服务于平衡。

MotionWAM 的第一个关键判断是：**足部任务做不出来，未必只是模型能力不足，也可能是动作空间从一开始就没有给策略表达它的机会。**

### 1.2 VLA 知道“是什么”，WAM 还要学习“接下来会怎样”

VLA 通常学习：

$$
\pi_\theta(\mathbf a_t \mid \mathbf o_t, l),
$$

其中 $\mathbf o_t$ 是当前图像，$l$ 是语言指令，$\mathbf a_t$ 是动作。大规模图文预训练让 VLA 具有很强的物体和语义知识，但静态图文目标不会直接要求模型理解“推动购物车后画面如何变化”“脚接触球以后球往哪里滚”。

WAM 则把未来视觉变化纳入学习目标。MotionWAM 将其写成“先建模视频动力学，再反演动作”：

$$
\mathbf o_{t+1}\sim p_v(\cdot\mid\mathbf o_t,l),
\qquad
\mathbf m_t\sim p_a\!\left(\cdot\mid\mathbf o_t,p_t,
\mathcal H(\mathbf o_{t+1}^{\tau_v})\right),
$$

其中 $p_t$ 是本体状态，$\mathcal H(\cdot)$ 是视频生成过程的中间隐藏特征，$\mathbf m_t$ 是全身运动 latent。训练目标是联合建模：

$$
p_{va}(\mathbf o_{t+1},\mathbf m_t\mid\mathbf o_t,p_t,l).
$$

这里有一个容易误解的地方：MotionWAM 在部署时**不真的生成完整未来视频**。视频预测主要提供训练监督和中间动力学特征，策略读取这些特征后直接生成运动 latent。

### 1.3 两个瓶颈必须一起解决

MotionWAM 同时处理两类瓶颈：

| 瓶颈 | 传统做法 | MotionWAM 的处理 |
|---|---|---|
| 视频扩散太慢 | 迭代去噪并显式生成未来帧 | Video DiT 只前向一次，读取中间特征 |
| 上下半身动作空间割裂 | 手臂关节目标 + 腿部基座命令 | 用统一全身运动 latent 驱动 SONIC |

如果只解决速度，策略仍不能表达踢球；如果只统一动作空间，视觉模型又未必具备足够的动态先验。这也是论文把架构和动作表示放在同等重要位置的原因。

---

## 2. 核心方法总览

![MotionWAM 的双 DiT 架构与三阶段训练流程](/assets/motionwam/arch.png)

*图 2：MotionWAM 架构。Stage 1 只训练 Video DiT；Stage 2 接入 Motion DiT 并用异构人形数据联合训练；Stage 3 用目标任务的全身遥操作数据微调。图片来自论文 Figure 2，CC BY 4.0。*

从部署链路看，MotionWAM 可以拆成四步：

```text
头部 RGB + 语言指令
        │
        ▼
Video DiT（一次前向，读取中间去噪特征）
        │  h_t
        ▼
Motion DiT（结合本体状态，4 步流积分）
        │  离散运动 token + 连续手部值
        ▼
SONIC 全身运动解码器
        │
        ▼
Unitree G1 的 29-DoF 关节命令
```

### 2.1 统一全身运动 latent

MotionWAM 将动作写成：

$$
\mathbf m_t=(\mathbf m_t^{\mathrm{cont}},\mathbf k_t).
$$

- $\mathbf k_t$ 是 SONIC 的离散全身运动表示，编码移动、躯干、高度和足部交互；
- $\mathbf m_t^{\mathrm{cont}}$ 是 SONIC 未覆盖的连续通道，例如左右夹爪或灵巧手数值。

这样，高层策略不必直接学习每一个关节在高速闭环下如何维持平衡，但又比“前进速度 + 身高”拥有更完整的全身行为表达。SONIC 负责将运动 token 解码成 G1 的 29-DoF 关节角。

论文称 SONIC latent 使用两个 32-level 的 FSQ token，并把相关表示描述为 64 维离散向量；在 Stage 3 的公式和架构图中，又把输出写成一个取值为 $\{0,\ldots,K-1\}$ 的 motion-token 索引。就接口语义而言，可以把它理解成“**离散全身运动码 + 连续手部值**”；但 2×32 的 FSQ 表示、64 维码本和单一索引之间如何精确转换，论文记号不够统一，这是复现时需要向代码确认的细节。

Stage 3 没有为离散 token 单独设计分类头，而是把索引当成连续标量 $\tilde{k}_t$ 与其他连续通道一起做流匹配，推理结束后四舍五入：

$$
(\mathbf m_t^{\mathrm{cont}},\tilde{k}_t)
\xrightarrow{\text{flow matching}}
(\hat{\mathbf m}_t^{\mathrm{cont}},\hat{\tilde{k}}_t)
\xrightarrow{\mathrm{round}}
(\hat{\mathbf m}_t^{\mathrm{cont}},\hat{k}_t)
\xrightarrow{\mathrm{SONIC}}
\mathbf a_t.
$$

这个方案结构很简单，但也存在隐含风险：连续空间里的距离未必对应离散运动码之间的语义距离。如果编号 12 和 13 在码本中并不相似，回归到 12.4 再四舍五入并不具有自然的物理含义。论文没有单独消融分类头、向量量化损失或 soft codebook 解码。

### 2.2 Video DiT：只读取“一步想象”的中间特征

Video DiT 初始化自 **Cosmos-Predict2.5-2B**。它使用因果时空 VAE 把当前观测和未来帧编码成 latent，并在语言条件下用 flow matching 学习未来视频。

标准流匹配在干净未来 latent $\mathbf z_{t+1}^0$ 与高斯噪声 $\boldsymbol\epsilon_v$ 之间插值：

$$
\mathbf z_{t+1}^{\tau_v}
=(1-\tau_v)\mathbf z_{t+1}^{0}
+\tau_v\boldsymbol\epsilon_v,
\qquad \tau_v\in[0,1].
$$

因此 $\tau_v=0$ 对应干净未来，$\tau_v=1$ 对应纯噪声。Video DiT 学习速度场：

$$
\mathcal L_{\mathrm{video}}=
\mathbb E\left[
\left\|
v_\theta^{\mathrm{video}}
(\mathbf z_{t+1}^{\tau_v},\tau_v\mid\mathbf z_t^0,l)
-(\boldsymbol\epsilon_v-\mathbf z_{t+1}^0)
\right\|_2^2
\right].
$$

推理时，MotionWAM 固定 $\tau_f\approx1$，未来端直接输入高斯噪声，并在某个 Transformer block 上挂 hook：

$$
\mathbf h_t^{\tau_f}
=\mathcal H[v_\theta^{\mathrm{video}}]
(\mathbf z_{t+1}^{\tau_f},\tau_f\mid\mathbf z_t^0,l),
\qquad
\mathbf z_{t+1}^{\tau_f}\sim\mathcal N(0,I).
$$

Video DiT 只跑一次，不把 $\mathbf z_{t+1}^{\tau_f}$ 逐步去噪成可观看的视频。作者称这种模式为 **one-shot imagination**。更准确地说，它是“一次前向得到受未来预测任务训练过的特征”，而不是“一步生成准确未来帧”。

### 2.3 Motion DiT：从动态特征生成动作块

Motion DiT 接收三类输入：

1. Video DiT 的隐藏状态 $\mathbf h_t^{\tau_f}$；
2. 64 维本体状态 $p_t$；
3. 加噪后的 66 维运动 latent token，以及 embodiment tag $e$。

它通过交错的 self-attention / cross-attention 预测运动速度场：

$$
\mathcal L_{\mathrm{motion}}=
\mathbb E\left[
\left\|
v_\phi^{\mathrm{motion}}
(\mathbf m_t^{\tau_a},\tau_a\mid
\mathbf h_t^{\tau_f},p_t,e)
-(\boldsymbol\epsilon_m-\mathbf m_t^0)
\right\|_2^2
\right].
$$

部署时 Motion DiT 使用 **4 个推理时间步**。因此，“单次前向”只适用于最昂贵的 Video DiT 分支，整个策略并非只做一个神经网络 forward。这一区分对于评估延迟非常重要。

多 embodiment 训练时，共享 Motion DiT 主干的前后放置各自的输入/输出 projector。不同动作向量右侧补零到最大 66 维，并用 mask 标记有效通道。部署到 G1 时只保留对应 projector。

---

## 3. 三阶段训练：从视频动态到机器人动作

### 3.1 Stage 1：第一视角视频预训练

第一阶段只更新 Video DiT，在约 **2,136 小时**的人类第一视角和人形/机器人视频上训练未来帧预测，动作标签即使存在也会被忽略。VAE 和文本编码器始终冻结。

数据预算按领域分成：人类第一视角 30%、G1 类人形数据 50%、其他真实机器人 20%。域内再按 $\sqrt{\#\text{episodes}}$ 分配权重，避免超大数据集完全淹没其他来源。

| 数据源 | Embodiment | Stage 1 权重 |
|---|---|---:|
| EgoDex | 人类第一视角 | 30.0% |
| GR00T-X-Embodiment-Sim | Fourier GR1 仿真 | 25.5% |
| RoboCOIN（G1edu/Galbot/Leju） | 多种人形机器人 | 8.0% |
| GR00T-Teleop-GR1-Robot | Fourier GR1 真机 | 7.1% |
| Humanoid-Everyday | Unitree G1 | 4.7% |
| UnifoLM-WBT | Unitree G1 | 2.3% |
| PSI-Real / PSI-Simple | Unitree G1 | 2.4% |
| RoboCOIN（R1_Lite + RMC-AIDA-L） | 其他真实机器人 | 20.0% |

这一步背后的假设是：早期瓶颈是第一视角视觉动力学，而不是动作标签的多样性。无动作视频规模更大、更便宜，可以先把通用视频模型适配到“机器人眼睛看到的世界”。

### 3.2 Stage 2：跨 embodiment 动作后训练

第二阶段接入 Motion DiT，在异构的人形动作数据上联合更新 Video DiT 和 Motion DiT。不同数据集有不同末端执行器和动作标注格式，因此用 embodiment tag、专属 projector、padding 和 mask 对齐到共享主干。

联合损失为：

$$
\mathcal L_{\mathrm{Stage\ 2}}
=\mathcal L_{\mathrm{motion}}+\mathcal L_{\mathrm{video}}.
$$

保留视频损失的作用是防止大模型在接收稀疏动作监督后遗忘视频动力学先验。这个阶段建立的是“动态特征如何对应到动作”的 grounding。

### 3.3 Stage 3：目标任务全身微调

作者用 PICO VR 头显、两个脚踝 tracker 和两个手柄记录操作者动作，经 XRoboToolkit 转成 SMPL-24 全身姿态，再由 SONIC 重定向为 G1 的 29-DoF 关节角。视觉、状态、命令和语言目标以 LeRobot 格式按 50 Hz 保存。

9 个任务每个采集 **200 个 episode**，即约 1,800 条目标任务演示。Stage 3 端到端微调整个 Video DiT + Motion DiT 网络，但仍冻结 VAE 与文本编码器。

### 3.4 训练规模与复现成本

| 配置 | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| GPU 数量 | 128 | 32 | 8 |
| 每卡 batch size | 8 | 8 | 8 |
| 最大训练步数 | 100,000 | 50,000 | 15,000 |
| Video DiT 学习率 | $10^{-5}$ | $10^{-5}$ | $10^{-5}$ |
| Motion DiT 学习率 | - | $10^{-4}$ | $10^{-4}$ |
| Motion DiT 训练重复扩散步 | - | 8 | 4 |
| Motion DiT 推理步数 | - | 4 | 4 |

Video DiT 的隐藏维度为 2048；Motion DiT 采用 DiT-B，隐藏/输出维度写为 2560，最大序列长度 1024，cross-attention 维度 2048。优化器为 AdamW，使用 cosine 学习率调度和 1.0 梯度裁剪。

这不是一套普通实验室可以轻易从头复现的训练配方。128 卡的 Stage 1、未公开的目标任务数据、SONIC token 接口和真实 G1 遥操作系统共同构成了较高门槛。截至本文撰写时，项目页只给出论文和演示视频，没有公开训练代码或 checkpoint。

---

## 4. 实验：9 个真实世界全身任务

![MotionWAM 的 9 个 Unitree G1 真实世界评测任务](/assets/motionwam/task_suite.png)

*图 3：9 个真实世界 loco-manipulation 任务。图片来自论文 Figure 3，CC BY 4.0。*

实验平台为 Unitree G1，双臂末端安装 ALOHA 2 夹爪，头部使用单个 Intel RealSense D435i RGB 相机。策略服务器运行在单张 RTX 4090 上，通过 WebSocket 被机载控制器闭环查询。每种方法、每个任务测试 20 次。

| 任务 | 语言指令 | 主要能力 |
|---|---|---|
| PnP Bottle | 把瓶子放入篮子 | 站位、抓取、放置 |
| Kick Soccer | 把足球踢进球门 | 主动足部交互 |
| Retrieve Item | 把包放到桌上并关抽屉 | 蹲起、移动、连续操作 |
| Load Cart | 推车并把衣服装入车中 | 行走、推物、手脚协调 |
| Toss Garbage | 把垃圾扔进垃圾桶 | 躯干与手部协调 |
| Lift Basket | 把桌下篮子搬到桌上 | 蹲伏移动、双臂搬运 |
| Stock Shelves | 饮料放上层、蔬菜放下层 | 高度调节、长程多步 |
| Wipe Board | 擦净白板 | 大范围腰臂协调 |
| Do Laundry | 把衣服放进洗衣机 | 开阔移动、弯腰投放 |

### 4.1 与 VLA 和模仿学习基线的比较

![MotionWAM 与五个基线在 9 个真实任务上的成功率](/assets/motionwam/suc_rate.png)

*图 4：每个任务各测试 20 次。所有方法使用相同 Stage 3 演示，并通过同一个 SONIC 动作接口输出。图片来自论文 Figure 4，CC BY 4.0。*

| 方法 | PnP | Kick | Retrieve | Load | Toss | Lift | Stock | Wipe | Laundry | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MotionWAM** | **95** | **60** | **90** | **75** | **45** | **80** | **70** | **95** | **75** | **76.1** |
| GR00T-N1.7 | 80 | 20 | 50 | 35 | 30 | 45 | 40 | 50 | 45 | 43.9 |
| $\pi_{0.5}$ | 70 | 10 | 25 | 10 | 5 | 15 | 5 | 20 | 10 | 18.9 |
| Qwen3DiT | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 4.4 |
| Diffusion Policy | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.6 |
| ACT | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.1 |

MotionWAM 在 9 个任务上都取得最高成功率。相对 GR00T-N1.7，平均绝对提升为 **32.2 个百分点**。差距最大的任务是 Wipe Board（+45）、Kick Soccer（+40）、Retrieve Item（+40）、Load Cart（+40）和 Do Laundry（+30）。这些恰好是需要腿、腰、手联合改变工作空间的任务。

Qwen3DiT 是这里最有解释力的对照：它用参数量相近的 Qwen3-VL 2B 替换视频世界模型，但保留相同 Motion DiT、统一动作空间以及 Stage 2/3 训练流程。其平均成功率只有 4.4%。在这套实验条件下，静态 VLM 表征并没有替代视频动态先验。

不过，不能仅凭这个实验得出“所有 WAM 都优于所有 VLA”。比较仍有三个边界：

1. MotionWAM 额外经历了 2,136 小时视频预训练，数据和算力预算并不对称；
2. 最终策略还依赖 SONIC 提供的强全身控制先验，结果不是端到端裸关节控制器单独带来的；
3. 评测任务、相机视角和物体与训练设置接近，论文没有严格的新物体或新场景 OOD 测试。

### 4.2 三阶段训练消融

| 变体 | Stage 1 | Stage 2 | Lift | Retrieve | Load | Toss | Kick | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o Stage 2 | ✓ | - | 65 | 45 | 30 | 30 | 40 | 42.0 |
| w/o Stage 1 | - | ✓ | 70 | 75 | 60 | 35 | 55 | 59.0 |
| **Full** | **✓** | **✓** | **80** | **90** | **75** | **45** | **60** | **70.0** |

在 5 个代表任务上，去掉 Stage 1 后平均下降 11 个百分点，说明第一视角视频适配有效；去掉 Stage 2 后下降 28 个百分点，说明跨数据源的动作 grounding 更关键。两者作用不同：Stage 1 决定世界模型“看懂怎样的动态”，Stage 2 决定这些动态特征“如何落到机器人动作”。

这里缺少两个值得期待的消融：一是保留同等视频数据但只做表征学习、不做未来预测；二是完整保留 Video DiT，但在 Stage 3 冻结它。没有这两项，很难精确区分收益究竟来自未来预测目标、数据规模，还是大模型端到端微调。

### 4.3 实时性

| 模型 | 可训练参数 | A100 动作块频率 |
|---|---:|---:|
| GR00T-N1.7 | 1.6B | 6.5 Hz |
| Qwen3DiT | 2.3B | 9.0 Hz |
| Cosmos Policy | 2.0B | 0.7 Hz |
| **MotionWAM** | **2.5B** | **4.9 Hz** |

MotionWAM 比 Cosmos Policy 快约 7 倍，核心原因是取消了未来视频的完整迭代去噪。但这里的 4.9 Hz 是**输出整个动作块的频率**，不是 29 个关节的底层控制频率。真正维持平衡和跟踪动作的是更高频的 SONIC 控制器。

论文没有报告动作块长度、执行窗口、端到端延迟分位数和网络通信抖动。因此，“实时”应理解为这套高层策略能够参加真机闭环，而不是 2.5B 模型以 4.9 Hz 直接完成全部稳定控制。

---

## 5. 如何理解 MotionWAM 的真正创新？

### 5.1 它不是传统意义上的“先想象视频，再执行动作”

传统两阶段 WAM 是：先完整生成未来视频，再用逆动力学模型恢复动作。优点是中间结果可视化，缺点是慢，而且视频幻觉会传给动作。

MotionWAM 更接近：

> 用未来视频预测作为训练任务塑造表示，在推理时只读取生成过程早期的隐状态。

它保留视频世界模型的训练红利，却跳过昂贵的像素解码。代价是可解释性下降：系统无法展示“自己具体想象了什么未来”，我们只能通过动作成功率间接判断隐藏特征是否有用。

### 5.2 最大突破可能是动作接口，而不只是 backbone

踢球和踩踏板的能力来自两个条件同时成立：

- Video DiT 提供对未来变化更敏感的表示；
- 统一 motion latent 允许高层策略主动选择足部行为。

如果动作空间仍只有 base velocity，任何强大的世界模型都无法输出“抬右脚踢球”。反过来，统一 token 如果缺少低层全身控制器，也很难稳定落到真实关节。MotionWAM 的结果应被理解为**世界模型、动作表示和运动控制器共同构成的系统成果**。

### 5.3 三阶段训练体现了清晰的数据分工

三个阶段对应三种成本不同的数据：

| 数据 | 规模 | 提供的知识 |
|---|---|---|
| 无动作标签的第一视角视频 | 最大、最便宜 | 视觉变化与交互动态 |
| 异构机器人动作数据 | 中等 | 动态特征到动作的通用映射 |
| 目标平台全身遥操作 | 最小、最昂贵 | 具体硬件和任务行为 |

这种由宽到窄、由视觉到动作的训练顺序，比直接拿少量 G1 任务数据端到端训练 2.5B 模型更合理。它也给后续工作一个很明确的扩展方向：扩大视频数据不必同步扩大昂贵的真机动作标注。

---

## 6. 局限、复现风险与开放问题

### 6.1 论文已经承认的局限

- **只在 Unitree G1 上完成 Stage 3 验证**，尚未证明最后的全身策略可跨硬件迁移；
- **没有严格的新物体泛化实验**，训练和测试物体在视觉上相似；
- **单目头部相机容易丢失 grounding**：物体离开视野或头部视角偏离训练分布时，策略会停滞或走出错误轨迹。

### 6.2 还需要进一步回答的问题

1. **统一 token 的收益有多少？** 论文没有给出“同一 Video DiT + 传统上下半身解耦动作空间”的对照，因此无法量化统一动作接口和视频世界模型各自贡献多少。
2. **one-shot 特征为何有效？** 缺少对 hook 层、$\tau_f$、随机噪声种子和多次采样方差的系统消融。
3. **4.9 Hz 的控制裕度多大？** 没有给出感知、Video DiT、Motion DiT、网络和解码器的逐模块延迟，也没有外力扰动下的稳定性数据。
4. **离散索引连续回归是否稳健？** 缺少分类头、soft codebook 或不同量化方案的比较。
5. **语言泛化有多强？** 每个任务只给出一个固定英文 prompt，实验更接近多任务行为克隆，没有展示改写指令或组合新任务。
6. **安全性如何保证？** 论文中的机器人带有顶部安全绳；对于动态足部动作，尚未报告碰撞、跌倒、越界动作或安全过滤机制。
7. **代码何时开放？** 当前项目页未提供代码和 checkpoint，SONIC 接口的若干关键细节也不足以从论文独立复现。

### 6.3 对结果的合理结论边界

论文有力支持以下结论：在同一批 Stage 3 演示和 SONIC 接口下，经过大规模第一视角视频预训练、跨数据集动作后训练的 MotionWAM，显著优于被测 VLA 与模仿学习基线。

它还不能证明：WAM 在所有机器人任务上普遍优于 VLA，或显式/隐式未来预测本身就是全部性能来源。数据预算、模型预训练、统一动作空间和强低层控制器都参与了最终结果。

---

## 7. 与相关工作的关系

| 工作 | 与 MotionWAM 的关系 |
|---|---|
| Video Prediction Policy | 用视频扩散模型的未来表征指导逆动力学，是“预测动态再反演动作”的早期代表 |
| DiT4DiT | 提供双 DiT 和联合视频-动作建模思路，MotionWAM 将其扩展到人形全身控制 |
| Cosmos Predict / Cosmos Policy | 前者提供 Video DiT 初始化；后者是需要完整未来视频去噪的速度对照 |
| SONIC | 提供离散全身运动 latent 与低层 29-DoF 解码能力 |
| GR00T-N1.7 / $\pi_{0.5}$ | 代表通用 VLA 路线，是论文的主要真机对照 |
| Qwen3DiT | 参数和动作头匹配的静态 VLM 消融，用来隔离视频动态先验的作用 |

MotionWAM 与 FastWAM、GigaWorld-Policy 等工作呈现了相似趋势：未来视频仍是重要训练信号，但部署时未必需要把未来像素完整生成出来。机器人的实时控制真正需要的可能是**被视频预测任务塑造过的动态表示**。

---

## 8. 总结

MotionWAM 给出了一个完整的人形 WAM 系统答案：用 Cosmos 视频模型学习第一视角动态，用中间去噪特征条件化 Motion DiT，用统一 motion latent 表达腿、腰、手的协同意图，再由 SONIC 负责高速全身控制。它把高层 WAM 的动作块频率提升到 4.9 Hz，并在 9 个 G1 真机任务上把平均成功率从最强基线的 43.9% 提高到 76.1%。

这篇论文最值得记住的不是某个单一模块，而是三层接口的对齐：**视频世界模型提供动态先验，运动 token 提供全身行为语言，低层控制器把抽象行为稳定地落到真实机器人。** 它同时也留下了清晰的研究空间：拆分各模块贡献、验证 OOD 泛化、补齐延迟与安全评测，并公开足以复现 token 接口和训练流程的代码。

---

## 参考资料

1. Zheng, J. et al. [MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation](https://arxiv.org/abs/2606.09215), 2026.
2. MotionWAM authors. [MotionWAM Project Page](https://dit4dit.github.io/MotionWAM/).
3. Ma, T. et al. [DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control](https://arxiv.org/abs/2603.10448), 2026.
4. Luo, Z. et al. [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://arxiv.org/abs/2511.07820), 2025.
5. Kim, M. J. et al. [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163), 2026.
6. Hu, Y. et al. [Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations](https://arxiv.org/abs/2412.14803), 2024.

> 文中 Figure 1-4 均取自 MotionWAM 论文与项目页，并按论文的 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可使用。
