---
title: QUAR-VLA
date: 2023-12-22
categories: [VLA]
---

# QUAR-VLA: 面向四足机器人的视觉-语言-动作模型

[paper link](https://arxiv.org/abs/2312.14457)

这篇论文来自西湖大学 MiLAB，提出了一套专门针对**四足机器人（Quadruped Robots）**的视觉-语言-动作（VLA）范式，命名为 **QUAR-VLA**（Vision-Language-Action tasks for QUAdruped Robots），并发布了对应的数据集 **QUARD** 与模型 **QUART**。与当时主流的机械臂 VLA（如 RT-2、OpenVLA）不同，QUAR-VLA 首次系统性地把 VLA 思想落地到了具有敏捷移动能力的腿式机器人上，覆盖了导航、复杂地形 locomotion 和全身操作等多类任务。

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

- **仿真数据**：约 256K 条 episodes（论文 Table 1 统计为 259K），在 NVIDIA Isaac Gym 中并行采集。
- **真实数据**：约 3K 条 episodes，用于弥合 sim-to-real gap。
- **任务类型**：涵盖 7 大类、多个子技能，按难度分为 easy / medium / hard：
  - **感知（Easy）**：Distinguish Letter（识别字母并转向目标）
  - **基础导航（Medium）**：Go to Object（驶向目标物体）
  - **高级导航与全身操作（Hard）**：Go to Object and avoid obstacle（避障导航）、Go through Tunnel（穿越隧道）、Crawl under Bar（钻杆）、Unload Object into Receptacle（背负小球并倒入容器）

### 2.2 数据采集平台

- **机器人平台**：WR-2 四足机器人，12 个关节，站立高度约 25 cm，体长约 40 cm。
- **传感器**：前置 RealSense D435 相机提供 RGB/深度图像。
- **控制层级**：
  - **高层控制器 5 Hz**：接收模型输出的高层命令。
  - **低层控制器 50 Hz**：由预训练命令跟踪策略将高层命令映射为关节力矩/位置。

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

## 四、模型架构：基于 Fuyu-8B 的端到端 VLA

QUART 的核心设计是**直接复用并端到端微调一个 decoder-only 的多模态大语言模型（MLLM）**。论文主实验采用的是 **Fuyu-8B**，并在补充实验中对比了 **LLaVA-7B** 版本。

### 4.1 为什么选择 Fuyu-8B？

Fuyu-8B（Adept, 2023）有两个对机器人控制非常有利的特性：

1. **原生图像编码**：不像 LLaVA 那样需要额外的 CLIP/SigLIP 视觉编码器，Fuyu-8B 可以直接对原始图像 patch 进行编码，简化了视觉-语言对齐的复杂度。
2. **整数 token 完备**：0–1000 的每个整数在词表中都有独立 token，因此可以把 256 个动作 bin 直接映射到对应整数 token 上，无需像 RT-2/PaLM-E 那样覆盖低频词表。

这种把动作 bin 映射到已有整数 token 的策略，本质上是一种 **符号微调（Symbol Tuning）**。

### 4.2 输入输出流程

整体流程可写作：

$$
\text{QUART}(a_d | s, w) = p(a_d | t) \, \tau(t | s, w)
$$

其中：

- $s$：第一视角 RGB 图像；
- $w$：自然语言指令（例如 "What action should the legged robot take to go to the Red box slowly with a trotting gait?"）；
- $\tau$：Fuyu-8B 的 tokenizer，将图像和文本统一编码为 token 序列；
- $p$：decoder-only Transformer，自回归生成离散动作 token；
- $a_d$：离散化的动作 token。

### 4.3 Action Detokenize

推理时，模型输出的离散动作 token 需要被反离散化回连续动作：

- 对每个维度，根据 bin 索引取区间中点作为连续值。
- 终止信号 $t$ 保持离散。
- 得到的 12 维高层命令发送给低层控制器执行。

---

## 五、训练方法：参数全部打开的全参微调

这是论文最核心的工程细节之一。QUART 没有采用冻结视觉编码器或只训练策略头（policy head）的浅层微调，而是**把整个 8B 多模态大模型端到端打开进行全参微调**。

### 5.1 训练配置

| 配置项 | 设置 |
|--------|------|
| 基座模型 | Fuyu-8B（decoder-only MLLM） |
| 训练方式 | **全参微调（Full Fine-tuning）** |
| 视觉编码器 | 不单独冻结，随整个 MLLM 一起更新 |
| 语言模型主体 | 不冻结，随整个 MLLM 一起更新 |
| 学习目标 | Next-token prediction / Behavior Cloning Loss |
| 损失函数 | 带因果掩码的分类交叉熵（Categorical Cross-Entropy with Causal Masking） |
| 学习率 | **2e-5** |
| Batch Size | **256** |
| 训练步数 | **100K gradient steps** |
| 推理频率 | 约 **2 Hz** |

### 5.2 为什么参数全部打开？

论文的补充实验（Supplementary Table 1）专门对比了多种“VLM + 策略头”的变体：

- **Unaligned VLM + P(MLP)**：R3M 视觉特征 + MLP 策略头，缺少图文对齐，几乎无法完成复杂任务。
- **Aligned VLM + P(MLP)**：CLIP / VC-1 视觉特征 + MLP 策略头，能完成简单导航，但 crawl、unload 等需要全身协调的任务成功率为 0。
- **Aligned VLM + P(Transformer)**：类似 RT-1 的 Transformer 策略头，结果与 MLP 策略头趋势一致，说明问题不在策略头结构，而在是否真正端到端地利用 VLM。
- **VLA(Fuyu-8B)**：即 QUART，**把整个 decoder-only VLM 打开微调**，动作各个维度在生成过程中可以相互依赖、联合推理，从而完成 crawl、unload 等复杂任务。

这一对比非常关键：它说明对于四足机器人这种需要多动作维度协调的任务，**仅冻结视觉主干、只训练一个轻量策略头是远远不够的**。必须让动作 token 的梯度回流到整个 MLLM，才能学到“身体高度、俯仰角、步态、足宽”之间的隐式协调关系。

### 5.3 损失计算细节

训练时只对未来动作 token 计算交叉熵损失，与标准因果语言模型训练一致：

- 输入序列：`[图像 token] + [指令文本 token]`
- 目标序列：`[动作 token 1] + [动作 token 2] + ... + [动作 token 12]`
- 模型自回归地预测每个动作 token，损失只作用于动作 token 位置，不对输入图像/指令 token 计算损失。

这种设计让模型保留预训练 MLLM 的视觉-语言理解能力，同时通过全参微调把动作生成能力“写入”到同一个生成空间中。

### 5.4 Action Head 是什么？动作是以 Next-Token 方式生成的吗？

这也是理解 QUART 架构时非常关键的两个问题。

#### QUART 没有独立的 Action Head

在传统的 “VLM + Policy Head” 范式里，通常会用一个**独立的动作头（Action Head）**来输出动作：

- 先由冻结或微调的视觉编码器提取图像特征；
- 把语言指令单独编码成 embedding；
- 将图像特征和语言特征拼接后，送入一个 **MLP 或 Transformer 策略头**；
- 策略头一次性回归或分类出动作向量。

QUART 并没有这个独立的动作头。它直接复用 Fuyu-8B 的**语言模型输出头（LM Head）**，让同一个 decoder-only Transformer 自回归地预测动作 token。换句话说：

> **动作预测不是“在 VLM 后面接一个专门的动作网络”，而是把动作当成一种特殊的语言 token，让 VLM  itself 来生成。**

这也是为什么论文里会说：

> “Within the VLA framework, we have utilized the entire decoder-only VLM backbone... This approach allows for the implicit learning of interdependencies between different action dimensions through the use of a transformer.”

#### 动作确实是以 Next-Token Prediction 方式生成的

具体流程如下：

1. 输入序列：`[图像 token] + [指令文本 token]`。
2. 模型开始自回归生成：`[动作 token 1] → [动作 token 2] → ... → [动作 token 12]`。
3. 每个动作 token 对应一个动作维度：
   - token 1 → $v_x$
   - token 2 → $v_y$
   - token 3 → $\omega_z$
   - ...
   - token 12 → $t$（终止信号）
4. 每个 token 从词表中的 256 个整数 token 中采样，对应 256 个离散 bin。
5. 12 个 token 生成完毕后，通过 detokenize 映射回连续动作值。

#### 与独立 Action Head 的本质区别

| 特性 | VLM + Action Head（基线） | QUART（VLA） |
|------|---------------------------|--------------|
| 动作输出结构 | 独立 MLP / Transformer Head | 复用 LM Head |
| 动作维度关系 | 通常一次性并行输出，维度间无显式交互 | 自回归生成，后面 token 能看到前面 token |
| 与 VLM 的关系 | VLM 只提供特征，不参与动作生成 | VLM 直接生成动作 token |
| 训练方式 | 通常只训练 Action Head | 全参微调整个 VLM |
| 复杂任务表现 | crawl、unload 成功率为 0 | crawl、unload 可达 0.12–0.32 |

这种 next-token 生成方式的好处在于：模型在预测“足宽”时可以参考已经生成的“身体高度”和“俯仰角”，在预测“步态频率”时可以参考“速度指令”。这种维度间的隐式协调，是独立 Action Head 难以学到的。

#### 一个细节：Action Head 与 Symbol Tuning 的关系

虽然 QUART 没有独立 Action Head，但它仍然需要对“哪些 token 代表合法动作”进行约束。这正是 **Symbol Tuning** 的作用：

- Fuyu-8B 的词表中，0–1000 的整数本来就有独立 token。
- QUART 通过训练告诉模型：当看到机器人任务 prompt 时，接下来要生成的 token 必须是 0–255 这些整数。
- 推理时可以通过限制采样空间（只允许这 256 个整数 token）来保证输出合法性。

所以你可以把 QUART 理解为：**用一个预训练 MLLM 的 LM Head 作为“隐式的 Action Head”，通过 Symbol Tuning 让这个 Head 学会输出动作 token。**

### 5.5 关键问题：全参微调后为什么还能理解新指令？训练时加了 VLM 数据吗？

这是理解 QUART 训练策略时最容易混淆的两个点，需要单独说明。

#### 训练数据只包含机器人数据，没有混入 VLM 预训练数据

论文中的 **co-training** 指的是把**仿真机器人数据**和**真实机器人数据**按一定比例混合训练，用来缓解 sim-to-real 的视觉域差异。**它并没有像 RT-2 那样把机器人数据与 VQA、Captioning 等 VLM 预训练数据混合。**

这一点可以从论文的两个细节得到印证：

1. 数据集统计（Table 1）只列出了机器人任务数据，总计约 259K 仿真 episodes + 3K 真实 episodes，没有提到任何 web-scale 视觉-语言数据。
2. 论文没有使用 “co-fine-tuning”、“web data”、“VLM pretraining data” 等 RT-2 中常见的表述；训练目标被明确描述为 “next token prediction objective, which corresponds to the behavior cloning loss in robot learning”。

#### 那为什么全参微调后仍能识别未见过的指令？

既然没有额外 VLM 数据“保底”，QUART 为什么还能泛化到训练时没见过的 verbal 指令（例如把 “Distinguish Letter” 换成 “Identify Letter”）？论文没有专门做灾难性遗忘分析，但可以从以下几个角度理解：

1. **预训练 MLLM 的语言能力底子足够厚**
   Fuyu-8B 是在大规模图文数据上预训练好的通用多模态模型，其语言理解能力在预训练阶段已经成型。256K 条机器人轨迹相对于它见过的互联网数据来说仍然是一个较小的微调量，不足以完全抹去原有的语言表征。

2. **机器人指令本身具有丰富的语言多样性**
   QUARD 的语言模板不是单一固定的，而是包含大量可替换变量：目标物体（box、cube、oven、drawer）、颜色（red、blue、green、yellow）、速度（fast、normal、slow）、步态（trotting、pace 等）以及空间关系（left、right、front、back、corner）。模型在训练时必须理解这些词汇的语义才能正确输出动作，这相当于在机器人数据内部“复习”了语言理解能力。

3. **Symbol Tuning 没有破坏语言词表**
   QUART 的动作离散化直接把 256 个 bin 映射到 Fuyu 词表中已有的整数 token（0–255），而不是像 PaLM-E 版本 RT-2 那样覆盖低频词表。这意味着模型不需要重新学习“数字”的含义，语言 token 空间基本保持完整，减少了语言能力的退化。

4. **损失函数仍然接触语言 token**
   虽然交叉熵损失只计算动作 token，但输入中的指令文本 token 会参与前向传播和梯度反向传播。模型为了在动作预测上取得低损失，必须维持指令文本与视觉、动作之间的正确对齐，这间接保留了语言理解能力。

5. **高层动作任务对语言精度的要求相对宽松**
   四足机器人的高层指令（如 “go to the red box slowly”）不需要模型像聊天机器人那样生成复杂长文本，只需要把关键语义词（物体、颜色、速度、步态）映射到正确的动作模式。因此即使语言模型部分能力有所下降，剩下的语言表征也足以支撑这类“关键词-动作”映射。

**与 RT-2 的对比**：RT-2 明确采用 co-fine-tuning（机器人数据 + Web VLM 数据混合）来防止灾难性遗忘；QUAR-VLA 则通过“强大的预训练 MLLM + 多样化机器人指令 + 不破坏词表的 symbol tuning” empirically 实现了类似的新指令泛化，但没有在训练中加入额外 VLM 数据。这是一个值得注意的工程取舍：它简化了数据 pipeline，但也意味着如果机器人数据量继续增大或任务语言更加复杂，可能需要显式引入 VLM 数据来保持语言能力。

---

## 六、Sim-to-Real：混合训练弥合域鸿沟

由于真实数据采集昂贵，QUART 主要依赖仿真数据训练。为了把仿真中学到的策略零样本迁移到真实四足机器人，论文采用了 **co-training（联合训练）** 策略：

- 在训练过程中按一定比例**混合仿真机器人数据与真实机器人数据**（注意：不是混合 VLM 预训练数据与机器人数据）。
- 通过控制真实数据比例，让模型在保留仿真数据多样性的同时，学习真实场景的视觉外观分布。
- 由于 QUART 输出的是高层命令而非直接关节力矩，低层控制器的域适应能力进一步缓冲了 sim-to-real gap。

论文的 scaling 实验（Table 3）显示，在固定 3K 真实数据的前提下，随着仿真数据从 0K 增加到 256K，真实场景成功率从 3/20 提升到 13/20，证明仿真数据对真实部署具有显著增益。

这种“高层动作 + 混合训练 + 低层跟踪”的组合，是四足机器人 VLA 与机械臂 VLA 在工程落地上最显著的区别之一。

---

## 七、实验与发现

论文进行了约 **4000 次真实世界评估试验**，主要结论包括：

- **策略有效性**：QUART 在导航、复杂地形和全身操作任务上均取得了较高成功率（Table 2）。
- **泛化能力**：模型能够处理训练时未完全见过的物体、场景和指令组合。
- **涌现能力**：得益于 VLM 预训练知识和多模态对齐，QUART 展现出了一定的语义推理和指令跟随能力，例如理解空间关系词（“left / right”）、顺序词（“before / then”）以及常识性指令（“move fast”）。

补充实验还发现，**多任务联合训练显著优于单任务训练**：QUART-Multi 在几乎所有任务上都超过了 QUART-Single，说明不同任务之间的共享知识对四足机器人控制非常重要。

---

## 八、与 RT-2 / OpenVLA 的对比视角

把 QUAR-VLA 放到当时的 VLA 版图里看，有几个鲜明特点：

| 维度 | RT-2 (2023) | OpenVLA (2024) | QUAR-VLA (2023) |
|------|-------------|----------------|-----------------|
| 载体 | 固定基座机械臂 | 固定基座机械臂 | 四足机器人 |
| 动作层级 | 末端 6-DoF + gripper | 末端 6-DoF + gripper | base 速度 + 步态 + 姿态 |
| 数据 | 真实机器人数据为主 | Open X-Embodiment | 仿真 + 少量真实 |
| 模型规模 | 5B–55B | 7B | 8B（Fuyu-8B） |
| 训练方式 | 全参微调 / Co-fine-tuning | LoRA / 全参微调 | **全参微调** |
| 核心挑战 | 语义泛化 | 开源与高效微调 | sim-to-real、腿式动力学 |

QUAR-VLA 的价值在于：它把 VLA 从“桌面操作”拓展到了“移动+操作+locomotion”的更复杂具身形态，并通过消融实验证明了**端到端全参微调 VLM** 对于四足机器人多动作维度协调的必要性。

---

## 九、局限与思考

1. **动作抽象层级**：QUART 输出的是高层命令，仍然依赖低层控制器。这意味着模型本身不学习底层动力学，某些高度动态的技能（如跳跃、后空翻）无法直接生成。
2. **真实数据量小**：3K 真实 episodes 相对于 256K 仿真数据仍然偏少，虽然 co-training 有效，但在视觉域差异极大的户外场景可能仍有限制。
3. **推理频率**：2 Hz 的控制频率对大多数导航和慢速操作任务足够，但对需要高频反馈的敏捷运动（如越障、跑跳）仍然偏低。
4. **评估维度**：4000 次真实试验已属庞大，但任务种类和机器人平台仍较单一，后续工作需要在更多平台、更开放环境中验证。

---

## 总结

QUAR-VLA 是较早将 VLA 范式系统性地拓展到**四足机器人**的研究工作。它通过提出 QUAR-VLA 任务范式、构建 QUARD 多任务数据集、基于 Fuyu-8B 的端到端 VLA 模型 QUART，并采用 **全参微调 + co-training** 实现 sim-to-real，展示了视觉-语言-动作模型在腿式机器人导航、locomotion 和全身操作中的潜力。对于关注具身智能、四足机器人以及 VLA 落地的研究者来说，这是一篇值得细读的奠基性工作。
