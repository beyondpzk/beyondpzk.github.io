---
title: MotionWAM
date: 2026-06-08
categories: [WAM]
---

# MotionWAM：面向实时人形 loco-manipulation 的基础世界动作模型

> **论文**：*MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation*  
> **作者**：Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang  
> **单位**：Mondo Robotics, HKUST (GZ), HKUST  
> **发布时间**：2026-06-08（arXiv）  
> **arXiv**：[https://arxiv.org/abs/2606.09215](https://arxiv.org/abs/2606.09215)

---

## 摘要

MotionWAM 提出了一个**实时的 World Action Model（WAM）**，用于单目第一视角下的人形机器人全身 loco-manipulation。它把视频世界模型的中间去噪特征注入策略，用**统一的全身运动 latent** 同时预测 locomotion、躯干运动、高度调节、足部交互和手部操作，打破了传统“上半身 manipulation + 下半身 locomotion”分层控制带来的动作空间不一致问题。训练采用**三阶段渐进式框架**：先在 2,136 小时 egocentric 人类/人形视频上预训练视频分支，再跨多种 embodiment 做动作后训练，最后在少量真实遥操作数据上微调。在 Unitree G1 的 9 项真实世界任务上，MotionWAM 总体成功率达到 **76.1%**，比最强的 VLA 基线 GR00T-N1.7（43.9%）高出 32% 以上，并首次展示了 WAM 驱动的闭环实时全身人形控制，包括踢球、踩踏板等任务驱动的足部行为。

---

## 一、研究背景与动机

### 1.1 人形 loco-manipulation 的独特挑战

人形机器人要在人类尺度环境中完成任务，需要同时协调：
- **平衡与移动**（locomotion）；
- **躯干姿态与高度调节**；
- **双臂/双手操作**；
- **任务驱动的足部交互**（踩踏板、踢球等）。

现有主流方案采用**分层结构**：高层 manipulation 策略只输出上半身关节目标，低层 locomotion 控制器只接收粗糙的 base 命令（速度、躯干高度、朝向）。这带来两个问题：

1. **动作空间不一致**：上下半身被强制分离，腿部只能被动维持平衡；
2. **无法做足部任务**：踩、踢、推等需要腿部主动参与的任务被排除在外。

### 1.2 World Action Models 的潜力与瓶颈

World Action Models（WAM）通过视频生成器提供 dynamics prior，再把策略条件化到世界模型的隐状态上，从而获得比纯 VLA 更强的时间一致性与物理合理性。但现有 WAM 大多用于桌面短程操作，因为：

- 高维视频-动作 latent 的迭代去噪太慢，难以闭环实时控制；
- 人形全身控制需要同时覆盖 locomotion 与 manipulation，动作维度更高。

MotionWAM 的核心问题是：**能否把 WAM 实时地、端到端地用于全身人形 loco-manipulation？**

---

## 二、核心贡献

| 贡献 | 内容 |
|------|------|
| **实时 WAM 用于人形全身控制** | 首个闭环实时、端到端的 WAM 驱动策略，在 Unitree G1 上运行 |
| **统一全身运动 latent** | 用单一动作空间同时表达 locomotion、躯干、高度、足部、手部行为，替代上下半身解耦 |
| **中间去噪特征条件化** | 通过单次前向 Video DiT 获取中间隐状态，避免迭代去噪，保证实时性 |
| **三阶段渐进训练框架** | 从 egocentric 视频预训练 → 跨 embodiment 动作后训练 → 任务微调，逐步适配 |
| **真实世界任务套件** | 设计 9 项需要全身参与的 loco-manipulation 任务，系统评估 waist/height/foot/hand 协调能力 |
| **显著性能提升** | 总体成功率 76.1%，比最强 VLA 基线高 32% 以上 |

---

## 三、方法详解

### 3.1 问题形式化：predict-video-dynamics, then invert

不同于 VLA 直接学习 $\pi_\theta(a_t \mid o_t, l)$，MotionWAM 遵循**先预测视频动态、再反演动作**的范式：

$$
o_{t+1} \sim p_v(\cdot \mid o_t, l), \quad m_t \sim p_a\left(\cdot \mid o_t, p_t, H(o_{t+1}^{\tau_v})\right)
$$

其中：
- $l$：语言目标；
- $o_t$：第一视角 RGB 观测；
- $p_t$：本体感知状态；
- $o_{t+1}^{\tau_v}$：视频流匹配过程中的中间未来帧状态；
- $H(\cdot)$：从视频生成过程中提取隐状态；
- $m_t$：统一的全身运动 latent，覆盖 locomotion、躯干、高度、足部、手部。

训练目标是联合建模视频-动作分布：

$$
p_{va}(o_{t+1}, m_t \mid o_t, p_t, l)
$$

### 3.2 全身运动 Latent

MotionWAM 在 **SONIC** 控制器的基础上定义统一运动 latent：

$$
m_t = (m_t^{\text{cont}}, k_t)
$$

- **$k_t$**：SONIC motion token，使用 **Finite Scalar Quantization（FSQ）** 压缩为 2 个 token、每 token 32 级，共 64 维离散向量，概括 locomotion、躯干、高度、足部交互意图；
- **$m_t^{\text{cont}}$**：连续通道，包含 SONIC 未覆盖的末端执行器命令（左右 gripper 或灵巧手指令）。

部署时，SONIC 把组装好的 latent 解码为全身关节命令 $a_t$。

### 3.3 双 DiT 架构

MotionWAM 由 **Video DiT** 和 **Motion DiT** 组成：

| 组件 | 作用 |
|------|------|
| **Video DiT** | 初始化自 Cosmos-Predict2.5-2B，将历史帧 $o_t$ 与未来帧 $o_{t+1}$ 编码为 VAE latent，并通过流匹配预测未来 |
| **Motion DiT** | 接收 Video DiT 的隐状态、本体状态 $p_t$、embodiment 标签 $e$，预测全身运动 latent 的速度场 |

关键设计：**只跑一次 Video DiT 的前向传播**，在某个固定的流时间步 $\tau_f \approx 1$（接近纯噪声端）读取中间隐状态：

$$
h_t^{\tau_f} = H\left(v_\theta^{\text{video}}\left(z_{t+1}^{\tau_f}, \tau_f \mid z_t^0, l\right)\right), \quad z_{t+1}^{\tau_f} \sim \mathcal{N}(0, I)
$$

这里 $z_t^0$ 是历史帧的干净 VAE latent，未来帧用高斯噪声初始化。Video DiT 在 **单次想象（one-shot imagination）** 模式下运行，不进行迭代去噪，从而保证实时性。

Motion DiT 随后通过交错 self/cross-attention 处理 $h_t^{\tau_f}$、$p_t$ 和带噪运动 latent token，输出速度场，积分后得到 $m_t$。

### 3.4 三阶段训练框架

| 阶段 | 更新模块 | 数据 | 目标 |
|------|----------|------|------|
| **Stage 1** | 仅 Video DiT | ~2,136 小时 egocentric 人类/人形视频 | 让视频世界模型从第一视角学习视觉动态先验 |
| **Stage 2** | Video DiT + Motion DiT | 跨 embodiment 的 Unitree G1 异构数据 | 在世界模型隐状态与动作之间建立跨本体映射 |
| **Stage 3** | 完整网络 | 少量目标任务的全身遥操作演示 | 微调为统一全身运动 token，完成端到端 loco-manipulation |

Stage 1 的关键洞察是：**视觉动态比动作多样性更稀缺**。用大量无动作标注的视频单独预训练视频分支，可以在不受少量动作数据限制的情况下获得 scale。

Stage 2 使用 per-embodiment 输入/输出 projector 包裹共享的 Motion DiT trunk，从而在同一模型上训练多种 end-effector 和动作标注格式。

Stage 3 把动作输出切换为统一全身运动 token，其中离散 SONIC token 索引 $k_t \in \{0, \dots, K-1\}$ 被表示为连续标量 $\tilde{k}_t \in \mathbb{R}$，在同一流匹配目标下回归，推理时通过最近邻取整恢复：

$$
\hat{k}_t = \text{round}(\tilde{k}_t)
$$

$$
\tilde{m}_t = (m_t^{\text{cont}}, \tilde{k}_t) \xrightarrow{\text{flow}} \hat{m}_t = (\hat{m}_t^{\text{cont}}, \tilde{k}_t) \xrightarrow{\text{round}} (\hat{m}_t^{\text{cont}}, \hat{k}_t) \xrightarrow{\text{SONIC}} a_t
$$

### 3.5 损失函数

Video DiT 的流匹配损失：

$$
\mathcal{L}_{\text{video}} = \mathbb{E}_{\tau_v, z_{t+1}^0, \epsilon_v} \left\| v_\theta^{\text{video}}(z_{t+1}^{\tau_v}, \tau_v \mid z_t^0, l) - (\epsilon_v - z_{t+1}^0) \right\|^2
$$

其中 $z_{t+1}^{\tau_v} = (1 - \tau_v) z_{t+1}^0 + \tau_v \epsilon_v$，$\epsilon_v \sim \mathcal{N}(0, I)$。

Motion DiT 的流匹配损失：

$$
\mathcal{L}_{\text{motion}} = \mathbb{E}_{\tau_a, m_t^0, \epsilon_m} \left\| v_\phi^{\text{motion}}(m_t^{\tau_a}, \tau_a \mid h_t^{\tau_f}, p_t, e) - (\epsilon_m - m_t^0) \right\|^2
$$

Stage 2 的联合损失：

$$
\mathcal{L}_{\text{Stage 2}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}
$$

保留视频损失是为了防止 dynamics prior 被动作信号覆盖。

Stage 3 沿用同样的联合损失，在目标 embodiment 的全身遥操作数据上端到端微调。

---

## 四、实验

### 4.1 实验设置

- **机器人平台**：Unitree G1，配备双 ALOHA2 gripper，头部 Intel RealSense D435i RGB 相机；
- **策略服务器**：单张 NVIDIA RTX 4090，通过 WebSocket 与机器人闭环通信；
- **控制器**：SONIC 全身控制器，将预测的运动 latent 解码为 29-DoF 关节角；
- **Stage 3 数据**：VR 遥操作（PICO VR + 脚踝/手部追踪），通过 SMPL-24 retarget 到 G1，每个任务约 200 条 episode，50 Hz 录制。

### 4.2 九项真实世界任务

| 任务 | 语言提示 |
|------|----------|
| **PnP Bottle** | Pick the bottle and place it in the basket. |
| **Kick Soccer** | Kick the soccer into the goal net. |
| **Retrieve Item** | Put the bag on the table and then close the drawer. |
| **Load Cart** | Push the cart forward and put the clothes on the table into the cart. |
| **Toss Garbage** | Throw the garbage into the trash can. |
| **Lift Basket** | Take out the clothes basket under the table and place it on the table. |
| **Stock Shelves** | Place the drinks on the upper shelf and the vegetables on the lower shelf. |
| **Wipe Board** | Clean the whiteboard thoroughly. |
| **Do Laundry** | Throw the clothes into the washing machine. |

这些任务共同考验：waist control、height regulation、squatting locomotion、body-hand coordination、task-driven foot interaction。

### 4.3 与 SOTA VLA 基线对比

所有方法都在相同的 Stage 3 演示数据上微调，并通过相同的 SONIC 接口输出动作。MotionWAM 在 **9 项任务全部获胜**：

- **MotionWAM**：总体成功率 **76.1%**；
- **GR00T-N1.7**（最强 VLA 基线）：总体成功率 **43.9%**；
- **π0.5**：总体成功率低于 20%；
- **Qwen3DiT**（参数匹配的 VLM+DiT 消融）：在 locomotion 重的任务上接近 0%。

在需要全身协调的任务上提升尤为显著：

| 任务 | MotionWAM 相对 GR00T-N1.7 的绝对提升 |
|------|--------------------------------------|
| Kick Soccer | +40% |
| Load Cart | +40% |
| Retrieve Item | +40% |
| Wipe Board | +45% |
| Do Laundry | +30% |

这说明：把策略条件化到视频世界模型的中间去噪特征，比静态 VLM 特征更能提供闭环物理人形控制所需的动态先验。

### 4.4 三阶段训练消融

在 5 项代表性任务上评估（Lift Basket、Retrieve Item、Load Cart、Toss Garbage、Kick Soccer）：

| 变体 | Stage 1 | Stage 2 | Lift Basket | Retrieve Item | Load Cart | Toss Garbage | Kick Soccer | 平均 |
|------|---------|---------|-------------|---------------|-----------|--------------|-------------|------|
| w/o Stage 2 | ✓ | – | 65 | 45 | 30 | 30 | 40 | 42.0 |
| w/o Stage 1 | – | ✓ | 70 | 75 | 60 | 35 | 55 | 59.0 |
| **Full** | **✓** | **✓** | **80** | **90** | **75** | **45** | **60** | **70.0** |

结论：
- **Stage 1** 提供 egocentric 视觉动态先验；去掉它导致动作预测不准确（平均 59.0%）；
- **Stage 2** 提供跨 embodiment 动作 grounding；去掉它性能崩溃（平均 42.0%）；
- 两阶段互补，验证了“视频分支与动作分支依次专业化”的设计。

### 4.5 实时推理频率

在 NVIDIA A100 上测量闭环策略执行频率：

| 模型 | 可训练参数量 | 频率 |
|------|--------------|------|
| GR00T-N1.7 | 1.6B | 6.5 Hz |
| Qwen3DiT | 2.3B | 9.0 Hz |
| Cosmos Policy | 2.0B | 0.7 Hz |
| **MotionWAM** | **2.5B** | **4.9 Hz** |

MotionWAM 比同样基于世界模型的 Cosmos Policy 快约 **7 倍**（4.9 Hz vs. 0.7 Hz），因为 Cosmos Policy 需要迭代去噪未来视频，而 MotionWAM 只读取单次前向传播的中间特征。4.9 Hz 已足以支撑闭环人形平衡控制。

---

## 五、优势与局限

### 5.1 优势

1. **统一动作空间释放全身能力**：上下半身不再解耦，腿部可以主动执行踩、踢、推等任务；
2. **实时 WAM**：通过单次 Video DiT 前向传递获取条件特征，避免迭代去噪；
3. **三阶段训练可扩展**：Stage 1 用廉价无动作视频获得 scale，Stage 2/3 用少量动作数据完成 grounding；
4. **跨 embodiment 共享 trunk**：per-embodiment projector 让同一 Motion DiT 在多种机器人数据上训练；
5. **显著超越 VLA**：在真实 G1 上总体成功率比最强 VLA 基线高 32% 以上。

### 5.2 局限

1. **仅在 Unitree G1 上验证**：三阶段范式是否迁移到其他人形硬件尚未验证；
2. **未见严格 OOD 物体泛化**：训练与测试物体视觉相似，未报告全新物体上的成功率；
3. **依赖单目第一视角相机**：当被操作物体离开视野或头部相机视角漂移时，模型容易丢失视觉 grounding；
4. **Stage 3 数据量仍有限**：每个任务约 200 条 episode，更大规模微调数据可能进一步提升鲁棒性；
5. **未显式建模社交/安全约束**：当前主要关注任务成功率，未涉及与人交互的安全边界。

---

## 六、历史意义与后续影响

MotionWAM 把 WAM 从“桌面短程操作”推向了“真实世界全身人形 loco-manipulation”，是 WAM 领域的重要里程碑。它与相关工作的关系：

| 相关工作 | 关系 |
|----------|------|
| **DiT4DiT** | MotionWAM 沿用其 dual-DiT 视频-动作框架，并将其扩展到实时人形全身控制 |
| **Cosmos Predict / Cosmos Policy** | MotionWAM 的视频 DiT 初始化自 Cosmos-Predict2.5-2B，并通过单次前向特征条件化实现比 Cosmos Policy 更快的推理 |
| **SONIC** | 作为底层全身控制器，为 MotionWAM 提供了统一的连续+离散运动 latent 接口 |
| **GR00T-N1.7 / π0.5 / WholeBodyVLA** | 代表 VLA 路线；MotionWAM 证明在这些全身任务上，WAM 的视频动态先验优于静态 VLM 先验 |
| **Uni-Tac / AHA-WAM 等 2026 WAM** | 同属 WAM 浪潮；MotionWAM 率先把 WAM 部署到真实人形机器人并展示任务驱动的足部交互 |

---

## 七、总结

MotionWAM 通过**双 DiT 视频-运动架构 + 统一全身运动 latent + 三阶段渐进训练**，实现了首个闭环实时的人形 WAM 策略。它突破了传统上下半身解耦控制的限制，让腿部从“被动平衡”升级为“主动任务执行”，在 Unitree G1 的 9 项真实 loco-manipulation 任务上以 76.1% 的总体成功率大幅领先 VLA 基线。尽管跨平台泛化、OOD 物体鲁棒性和视觉 grounding 仍有提升空间，MotionWAM 为从大规模视频预训练到人形全身控制的可行路径提供了有力证据，也标志着 WAM 正从桌面走向真实世界的人形机器人。
