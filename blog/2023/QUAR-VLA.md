---
title: QUAR-VLA
date: 2023-12-22
categories: [VLA]
---

# QUAR-VLA: 面向四足机器人的视觉-语言-动作模型

[paper link](https://arxiv.org/abs/2312.14457)

这篇论文来自西湖大学 MiLAB，提出了一套专门针对**四足机器人（Quadruped Robots）**的视觉-语言-动作（VLA）范式，命名为 **QUAR-VLA**（Vision-Language-Action tasks for QUAdruped Robots），并发布了对应的数据集 **QUARD** 与模型族 **QUART**。与当时主流的机械臂 VLA（如 RT-2、OpenVLA）不同，QUAR-VLA 首次系统性地把 VLA 思想落地到了具有敏捷移动能力的腿式机器人上，覆盖了导航、复杂地形 locomotion 和全身操作等多类任务。

---

## 一、研究背景：为什么需要 QUAR-VLA？

四足机器人因其出色的地形通过性和机动性，一直是机器人学的重要方向。但传统方法通常把**感知、规划、决策**拆成独立模块：

- **Vision-Action（QUAR-VA）**：依赖目标图像或粗粒度视觉指令做导航，难以表达“先左转、再穿过门洞”这类组合式、细粒度指令。
- **Language-Action（QUAR-LA）**：用自然语言下达指令，但缺少视觉感知，机器人在真实环境中缺乏自主性。

QUAR-VLA 的核心动机是把**第一视角 RGB 图像**和**自然语言指令**同时作为输入，让模型端到端生成可执行的高层动作命令，把感知、规划、决策真正融合到一个网络里。

---

## 二、QUARD 数据集：首个大规模多任务四足机器人数据集

### 2.1 数据规模与任务类型

论文发布了 **QUAdruped Robot Dataset (QUARD)**，这是当时首个同时包含图像、语言指令和本体感知信息的大规模四足机器人数据集：

- **仿真数据**：约 348K 条 episodes，在 NVIDIA Isaac Gym 中并行采集。
- **真实数据**：约 3K 条 episodes，用于弥合 sim-to-real gap。
- **任务类型**：涵盖 7 大类、多个子技能，包括：
  - **导航**：Go to Object、Go to Object and avoid obstacle
  - **复杂地形 locomotion**：Crawl under Bar、Go through Tunnel
  - **全身操作**：Stop Object（拦截运动物体）、Unload Object（把背包里的小球倒入指定盒子）、Distinguish Letter（转向指定视觉字母）

### 2.2 数据采集平台

- **机器人平台**：WR-2 四足机器人，12 个关节，站立高度约 25 cm，体长约 40 cm。
- **传感器**：前置 RealSense D435 相机提供 RGB/深度图像。
- **控制层级**：
  - **高层控制器 5 Hz**：接收模型输出的高层命令。
  - **低层控制器 50 Hz**：由 MPC 或预训练命令跟踪策略将高层命令映射为关节力矩/位置。

这种“高层动作 + 低层跟踪”的分层设计非常关键：它既避免了直接预测高频关节动作的复杂度，又保留了四足机器人所需的灵活步态和姿态控制能力。

---

## 三、动作空间：12 维高层命令

QUART 输出的动作空间不是低层关节角度，而是 12 维高层控制命令，包含一个终止信号：

$$
[v_x, v_y, \omega_z, \theta_1, \theta_2, \theta_3, f, h_z, \varphi, s_y, h_z^f, t]
$$

各维度含义如下：

| 维度 | 含义 |
|------|------|
| $v_x, v_y$ | 机器人 base 在 x/y 方向的速度 |
| $\omega_z$ | 偏航角速度 |
| $\theta_1, \theta_2, \theta_3$ | 步态模式参数（如 trot、pace 等） |
| $f$ | 步态频率 |
| $h_z$ | 机器人身体高度 |
| $\varphi$ | 俯仰角 |
| $s_y$ | 足宽/站立宽度 |
| $h_z^f$ | 抬脚高度 |
| $t$ | 终止信号 |

每个连续维度被均匀离散化为 **256 个 bin**，从而把连续控制问题转化为语言模型熟悉的 next-token prediction 问题。

---

## 四、模型架构：QUART-1 与 QUART-2

论文提出了两个模型变体，分别面向**推理效率**和**涌现能力**。

### 4.1 QUART-1：轻量高效的端到端策略

QUART-1 约 **3000 万参数**，结构紧凑，强调 fast inference：

- **视觉编码器**：ImageNet 预训练的 EfficientNet-B3。
- **语言条件化**：使用 **FiLM（Feature-wise Linear Modulation）** 将自然语言指令注入图像特征，使网络在浅层就关注任务相关视觉信息。
- **TokenLearner**：将大量视觉 token 压缩为紧凑的固定数量 token，降低后续 Transformer 计算量。
- **Transformer Decoder**：基于压缩后的 token 自回归生成离散动作 token。

整体流程可写作：

$$
\text{QUART-1}(a_d | s, w) = p_1(a_d | t) \, \tau_1(t | z_v) \, q_v(z_v | s, w)
$$

其中 $s$ 为图像，$w$ 为语言指令，$q_v$ 为视觉-语言编码器，$\tau_1$ 为 TokenLearner，$p_1$ 为 Transformer decoder。

### 4.2 QUART-2：利用预训练 VLM 的涌现能力

QUART-2 则走另一条路线——**基于预训练视觉-语言大模型（VLM）进行符号微调（Symbol Tuning）**：

- **Tokenizer**：复用 VLM 已有的 tokenizer。
- **Action Token 映射**：因为该 VLM 的词表中 0–1000 的整数都有独立 token，QUART-2 直接把 256 个动作 bin 映射到对应整数 token 上，无需像 PaLM-E 那样覆盖低频 token。
- **自注意力跨模态对齐**：与 QUART-1 的 FiLM 不同，QUART-2 通过 Transformer 的自注意力机制直接学习图像、语言与动作 token 之间的关联。

公式上：

$$
\text{QUART-2}(a_d | s, w) = p_2(a_d | t) \, \tau_2(t | s, w)
$$

其中 $\tau_2$ 为 VLM tokenizer，$p_2$ 为 decoder-only 大语言模型。

### 4.3 Action Detokenize

推理时，模型输出的离散动作 token 需要被反离散化回连续动作：

- 对每个维度，根据 bin 索引取区间中点作为连续值。
- 终止信号 $t$ 保持离散。
- 得到的 12 维高层命令发送给低层控制器执行。

---

## 五、Sim-to-Real：混合训练弥合域鸿沟

由于真实数据采集昂贵，QUART 主要依赖仿真数据训练。为了把仿真中学到的策略零样本迁移到真实四足机器人，论文采用了 **co-training（联合训练）** 策略：

- 在训练过程中按一定比例混合仿真数据与真实数据。
- 通过控制真实数据比例，让模型在保留仿真数据多样性的同时，学习真实场景的视觉外观分布。
- 由于 QUART 输出的是高层命令而非直接关节力矩，低层控制器的域适应能力进一步缓冲了 sim-to-real gap。

这种“高层动作 + 混合训练 + 低层跟踪”的组合，是四足机器人 VLA 与机械臂 VLA 在工程落地上最显著的区别之一。

---

## 六、实验与发现

论文进行了约 **4000 次真实世界评估试验**，主要结论包括：

- **策略有效性**：QUART 在导航、复杂地形和全身操作任务上均取得了较高成功率。
- **泛化能力**：模型能够处理训练时未完全见过的物体、场景和指令组合。
- **涌现能力**：得益于 VLM 预训练知识和多模态对齐，QUART 展现出了一定的语义推理和指令跟随能力，例如理解空间关系词（“left / right”）、顺序词（“before / then”）以及常识性指令（“move fast”）。

在 QUART-1 与 QUART-2 的对比上：

- **QUART-1**：参数量小、推理快，更适合资源受限的端侧部署。
- **QUART-2**：利用预训练 VLM，在需要语义理解和泛化的任务上表现更强，但推理成本更高。

---

## 七、与 RT-2 / OpenVLA 的对比视角

把 QUAR-VLA 放到当时的 VLA 版图里看，有几个鲜明特点：

| 维度 | RT-2 (2023) | OpenVLA (2024) | QUAR-VLA (2023) |
|------|-------------|----------------|-----------------|
| 载体 | 固定基座机械臂 | 固定基座机械臂 | 四足机器人 |
| 动作层级 | 末端 6-DoF + gripper | 末端 6-DoF + gripper | base 速度 + 步态 + 姿态 |
| 数据 | 真实机器人数据为主 | Open X-Embodiment | 仿真 + 少量真实 |
| 模型规模 | 5B–55B | 7B | 30M + VLM |
| 核心挑战 | 语义泛化 | 开源与高效微调 | sim-to-real、腿式动力学 |

QUAR-VLA 的价值在于：它把 VLA 从“桌面操作”拓展到了“移动+操作+locomotion”的更复杂具身形态，证明了 VLA 范式在四足机器人上的可行性。

---

## 八、局限与思考

1. **动作抽象层级**：QUART 输出的是高层命令，仍然依赖低层控制器。这意味着模型本身不学习底层动力学，某些高度动态的技能（如跳跃、后空翻）无法直接生成。
2. **真实数据量小**：3K 真实 episodes 相对于 348K 仿真数据仍然偏少，虽然 co-training 有效，但在视觉域差异极大的户外场景可能仍有限制。
3. **评估维度**：4000 次真实试验已属庞大，但任务种类和机器人平台仍较单一，后续工作需要在更多平台、更开放环境中验证。

---

## 总结

QUAR-VLA 是较早将 VLA 范式系统性地拓展到**四足机器人**的研究工作。它通过提出 QUAR-VLA 任务范式、构建 QUARD 多任务数据集、设计 QUART-1/QUART-2 两种模型变体，并采用 co-training 实现 sim-to-real，展示了视觉-语言-动作模型在腿式机器人导航、locomotion 和全身操作中的潜力。对于关注具身智能、四足机器人以及 VLA 落地的研究者来说，这是一篇值得细读的奠基性工作。
