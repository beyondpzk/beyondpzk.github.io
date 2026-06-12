---
title: Motus
date: 2025-12-15
---

# Motus

[paper link](https://arxiv.org/abs/2512.13030)

这份论文 **"Motus: A Unified Latent Action World Model"** 是一篇在具身智能（Embodied AI）领域具有里程碑意义的工作。它直击了当前具身模型中“感知、世界模型、控制”三者割裂的痛点。

以下是我解析与阅读笔记。

---

### 一、 核心贡献提炼 (Core Contributions)

1. **统一的具身多模态生成框架**：打破了以往视觉-语言-动作模型（VLA）、世界模型（WM）、逆动力学模型（IDM）和视频生成模型（VGM）孤立存在的局面，首次提出通过单一网络同时建模边缘分布、条件分布和联合分布，实现五大具身范式的无缝切换。
2. **MoT架构与UniDiffuser式流匹配**：创新性地设计了Mixture-of-Transformer (MoT) 架构与Tri-model Joint Attention机制，在不破坏预训练视觉语言模型（VLM）和视频生成模型（VGM）强大先验的前提下接入动作专家；并利用流匹配（Rectified Flow）中独立的噪声时间步长，实现了不同推断模式的动态调度。
3. **基于光流的“潜在动作”与缩放法则**：提出将光流作为像素级的“通用动作（Delta Action）”，通过自编码器压缩为与物理动作空间对齐的潜在动作，从而彻底打通了海量无动作标签的互联网视频、人类第一人称数据与多机器人数据，构建了六级数据金字塔进行大规模动作预训练。

---

### 二、 方法论深挖 (Methodology)

Motus 并非从头训练一个庞大的模型，而是站在巨人的肩膀上，将开源的强大基座（Wan 2.2 5B VGM, Qwen3-VL 2B）与一个全新初始化的动作专家（Action Expert）进行融合。

#### 1. 架构流程图 (Architecture Flow)

以下是 Motus 模型的整体架构与数据流转的流程图：

```text
[ Input Condition ]                      [ Generation Targets ]
 ├─ Current Obs (当前图像) ──────┐        ├─ Noisy Video (加噪视频) ──┐
 └─ Language Inst (语言指令) ────┤        └─ Noisy Action (加噪动作) ─┤
                                 ▼                                    ▼     ▼
                        [ Understanding Expert ]             [ Generative ] [ Action   ]
                        (VLM: Qwen3-VL)                      (VGM: Wan)     (Expert)
                                 │                                    │         │
                                 └──────────┐           ┌─────────────┘         │
                                            ▼           ▼                       ▼
                                      =========================
                                     [[ Tri-model Joint Attn ]]  <-- (三模型特征在此融合互通)
                                      =========================
                                                │       │
                                  ┌─────────────┘       └─────────────┐
                                  ▼                                   ▼
[ Output ]         Predicted Video Velocity (v_o)        Predicted Action Velocity (v_a)

------------------------------------------------------------------------------------------
[ Latent Action 提取流 (预训练用) ]：
Frame(t) & Frame(t+1)  ==>  [DPFlow提取光流]  ==>  [DC-AE压缩降维]  ==>  [14维 Latent Action]
                                                                              │
                                       (替代缺失的机器人动作数据，输入到上面的模型中) <┘

```

#### 2. 核心机制解析与公式解释

**A. UniDiffuser风格的流匹配 (Rectified Flow) 目标函数**
为了在一个模型中统一 VLA, WM, IDM 等模式，模型对视频和动作独立分配了不同的时间步（$\tau_o$ 和 $\tau_a$）：

$$l_\theta^{action} = \mathbb{E}_{\tau_a \sim U(0,T_\tau), \epsilon_a \sim \mathcal{N}(0,I)} \left\| v_\theta^a - (\epsilon_a - a_{t+1:t+k}) \right\|_2^2$$
$$l_\theta^{obs} = \mathbb{E}_{\tau_o \sim U(0,T_\tau), \epsilon_o \sim \mathcal{N}(0,I)} \left\| v_\theta^o - (\epsilon_o - o_{t+1:t+k}) \right\|_2^2$$
$$l_\theta = l_\theta^{action} + l_\theta^{obs}$$

*   **物理意义**：这是基于 Rectified Flow 的速度场匹配损失。$v_\theta^o$ 和 $v_\theta^a$ 分别是模型预测的视频和动作的速度场。
*   **必要性**：通过让 $\tau_o$ 和 $\tau_a$ 独立采样，模型学会了在任意给定单边信息（或无信息）的情况下推断另一边。推理时，若要作为 VLA，只需将视频噪声时间步设为极大值（条件），动作时间步从极大降至 $0$（生成）；若要作为世界模型 WM，则反之。这是一种极其优雅的联合概率与条件概率统一建模方案。

**B. 潜在动作对齐损失 (Latent Action Alignment)**
为了让无标注视频中的光流能够迁移到机器人的控制空间：

$$L = L_{recon} + \lambda_a \|a_{real} - a_{pred}\|_2^2 + \beta L_{KL}$$

*   **物理意义**：这是一个变分自编码器（VAE）的损失。$L_{recon}$ 负责重建光流保证视觉动态信息的保留；中间项是**弱动作监督对齐项**，利用 $10\%$ 的真实机器人或 Task-agnostic 随机探索数据，强制要求编码器输出的 Latent 表征能够映射回真实的关节/末端执行器动作空间；$L_{KL}$ 是分布正则化。
*   **必要性**：直接将高维光流输入给动作专家会导致维度灾难，且无法与微调阶段的物理动作对齐。引入少量弱监督的对齐项，是打通“视频像素动态”与“物理世界控制”的关键桥梁。

**C. 动作密集-视频稀疏预测 (Action-Dense Video-Sparse Prediction)**
这是一个非常 engineering-oriented 但极其重要的设计。由于视频 Token 数量远大于动作 Token，直接联合训练会导致模型被视频生成主导（Loss 坍塌到视觉侧）。作者通过降采样视频帧（例如视频帧率设为动作的 $1/6$），在 Attention 层强制平衡两者的计算权重。

---

### 三、 实验分析与批判 (Experimental Analysis & Critique)

#### 1. 主要结果 (Key Takeaways)
*   **仿真环境 (RoboTwin 2.0)**：在高达 50+ 个任务的 Randomized 复杂设定下，超越了当前最强基线 $\pi_0$ 绝对成功率达 $+45\%$。
*   **真实世界部署**：在 Agilex-Aloha-2 和 AC-One 两个不同构型平台上，执行折叠毛巾、煮咖啡等长程复杂任务，相比于 $\pi_0$ 有巨大提升（部分任务从几乎 $0\%$ 提升至 $90\%$ 以上）。
*   **模式通用性**：实验证明 Motus 作为 IDM 模式使用时，甚至比专门用 ResNet18/DINOv2 训练的专用 IDM 误差（MSE）还要低（$0.014$ vs $0.044$）。

#### 2. 局限性与漏洞分析 (Reviewer's Critique)
站在审稿人角度，本文虽然宏大且有效，但有几个潜在的漏洞或未交代清楚的痛点：
*   **推理延迟与计算成本 (Inference Latency & Cost)**：融合了一个 5B 的 VGM、一个 2B 的 VLM 加上动作专家，总参数量达到 8B 级别。要在真实机器人上进行实时闭环控制（Real-time Closed-loop Control），尤其是基于 Flow Matching 需要多步去噪（文中设定的是 10 步 inference steps），其控制频率（Hz）是多少？论文完全没有报告推理耗时（FPS 或 Latency），这在具身智能论文中是一个明显的缺陷。
*   **Latent Action 的跨构型泛化边界**：论文利用 Curobo 采集了 task-agnostic 数据来进行 Latent space 与 Real action 的对齐。但是，双臂机器人和四足狗的 Action Space 截然不同。光流能否真正被压缩成一个“万能”的维度，并在不同构型（甚至是非多指灵巧手与灵巧手之间）无缝切换？这一点缺乏更为极端的跨构型消融实验。
*   **幻觉与复合误差 (Hallucination & Compounding Errors)**：作为世界模型，长程多步 rollout（自回归预测多步未来）时的复合误差表现如何？文章只展示了生成的视频片段，没有像 DreamerV3 等工作一样给出长程动作预测下的累积 Reward 或准确率衰减曲线。

#### 3. 潜在改进方向 (Potential Improvements)
如果让我接手并延续这项工作：
*   **引入 Consistency Models / 蒸馏**：针对高延迟问题，将动作专家的 10 步 Rectified Flow 蒸馏为单步或两步生成模型（如 Flow Matching distillation），在保持性能的同时极大提升机器人闭环执行的频率。
*   **基于物理约束的光流对齐**：目前的对齐纯粹基于神经网络的弱监督。可以引入机器人的前向运动学（Forward Kinematics, FK）作为微调阶段的可微正则化项，确保生成的“潜在动作”不仅在视觉上合理，在运动学连杆上也不发生自碰撞。

---

### 四、 延伸思考 (Inspirations for Current Research)

如果我的研究方向同样是具身智能/世界模型，这篇论文提供了几个极其宝贵的“解题思路”：

1. **破局“无标注数据利用”：光流即动作**
   当前具身研究最大的瓶颈是具有动作标签的真实机器人轨迹（Robot Trajectories）太少。Motus 给出了一个标准答案：**用光流提取物理位移，用 VAE 进行降维，辅以极少量的随机探索（Motor babbling）数据进行锚定（Anchoring）**。我们可以直接借鉴这种方法，将 YouTube 上的烹饪视频和教学视频无损地转化为动作预训练语料。
2. **多专家协同的 MoT 机制优于全参数微调**
   很多工作（如 OpenVLA）强行将视觉和动作 token 塞进同一个大模型中，这往往会导致“灾难性遗忘”（模型忘了怎么说话或者丧失了世界常识）。Motus 的 `Tri-model Joint Attention` 提供了一种优雅的插件化思路：保持强模型（VLM/VGM）的主体冻结，只在 Attention 层做 Cross-modality 的 query/key 交互。这种范式可以直接用到我们的模型设计中，极大降低显存消耗并保持基座能力。
3. **Diffuser 调度的终极形态**
   独立采样不同的 $\tau$ 这点非常具有启发性。这告诉我们，不要再去分别为“策略网络（Policy）”和“动力学模型（Dynamics）”设计不同的结构，只需定义一个联合连续时间的 ODE 框架，通过在不同边界条件上设置掩码（极端的 $\tau$ 值），就可以实现一鱼多吃。在多模态生成（如文本+视频+音频同步生成）领域，这套数学逻辑也是完全通用的。

## latent action做法

这是一个非常核心的问题，它触及了这篇论文解决“具身智能数据荒”的最强杀招：**如何把网上的猫猫狗狗、人类做饭的无动作标签视频（Video-only Data），变成能训练机器人动作专家的有效数据？**

答案就是：**用光流（Optical Flow）提取纯粹的物理运动，再用 DC-AE 将其压缩降维，伪造出一种“通用机器语言”——Latent Action（潜在动作）。**

结合论文的第 4.2 节（Latent Actions）和 Figure 3，我们来深度拆解这个“炼金术”的具体细节：

---

### 一、 核心逻辑：为什么要用“光流 + DC-AE”？

如果直接把前后两帧图像（比如切菜）喂给模型让它猜“动作”是什么，模型很容易“作弊”：它会去记忆菜板的颜色、刀的材质（这些叫 task-irrelevant appearance information），而不是真正学习“手往下切”这个物理动作。

作者的解法非常巧妙：
1.  **光流（Optical Flow）**：计算连续两帧之间每个像素点的位移向量。光流图滤除了一切颜色、光照和纹理，只保留了纯粹的**运动趋势（Motion Pattern）**。这就是像素级的“Delta Action”。
2.  **DC-AE (Deep Compression Autoencoder)**：虽然光流很纯粹，但它是高维的（比如 $256 \times 256 \times 2$）。真实机器人的控制指令（如末端 6D 位姿 + 夹爪开合）只有十几维。如果直接把几万维的光流硬塞给动作专家，不仅维度极度不匹配，而且包含了大量背景的微小扰动噪声。因此，必须用 DC-AE 把它强力压缩到一个极低的维度，与真实的机器人动作维度对齐。

---

### 二、 Latent Action VAE 的架构细节 (对照 Figure 3)

这个压缩管道（Pipeline）分为以下几个极其清晰的步骤：

#### 1. 提取光流并转换为 RGB (DPFlow)
*   给定视频的第 $t$ 帧和第 $t+1$ 帧。
*   使用现成的强大光流估计模型 **DPFlow**，计算出极其稠密的像素位移图（Velocity Field）。
*   **工程细节**：为了方便后续的标准化卷积网络处理，光流通常被转换成 3 通道的 RGB 图像（例如：色调代表方向，饱和度代表速度大小）。

#### 2. 深度压缩自编码器 (DC-AE Encoder)
*   将刚才的光流 RGB 图送入预训练的 **DC-AE**（论文引用了 MIT Han Song 团队 2025 年初刚发表的高效扩散模型压缩器 DC-AE，这是一个极其强悍的变分自编码器）。
*   DC-AE 将高维的图像极度压缩，输出为 **四个 512 维的 Token（即 $4 \times 512$ 的特征图）**。到了这一步，整张画面的宏观物理运动已经被高度浓缩了。

#### 3. 终极降维 (Lightweight Encoder)
*   仅仅 $4 \times 512 = 2048$ 维对机器人来说还是太高了。
*   作者在 DC-AE 之后，又接了一个轻量级的编码器（Lightweight downsampling modules，通常是 MLP 或卷积层）。
*   **核心参数**：它将这 $4 \times 512$ 的特征，硬生生地投影（Project）成了一个仅仅 **14 维的向量（14-dimensional vector）**。

#### 4. 为什么要设定为 14 维？(Dimensional Correspondence)
论文明确指出：“*roughly matching the scale of typical robot action spaces*”。
14 维非常接近真实双臂机器人或灵巧手的一个控制频率周期内的自由度数量（例如：双臂各 6 个关节 + 2 个夹爪 = 14 维）。这种**硬性的维度对应（Dimensional correspondence）**，确保了从光流压缩出来的 Latent Action，在数学结构和信息瓶颈（Information Bottleneck）上，能够自然地与真实的物理控制信号完美映射。

---

### 三、 炼金术的核心：如何把“压缩的光流”变成“可执行的动作”？(Training Alignment)

如果仅仅做上面的压缩，模型学出来的 14 维向量只是一堆乱码（对机器人毫无意义）。最难的一步是：**怎么让这个 14 维向量的分布，跟真实机器人的电机控制信号分布对齐？**

作者在训练这个 Latent Action VAE 时，采用了一种极其聪明的**“混合监督（Weak Action Supervision）”**策略：

#### 1. 损失函数拆解 (Equation 2)
训练 VAE 的损失函数为：$L = L_{recon} + \lambda_a \|a_{real} - a_{pred}\|_2^2 + \beta L_{KL}$

*   **$L_{recon}$ (重建损失)**：占比 $90\%$ 的纯净互联网视频数据（无动作标签），只通过这部分 Loss 强迫 VAE 能够完美还原最初的光流图。这保证了 14 维向量没有丢失关键的物理运动信息。
*   **$\|a_{real} - a_{pred}\|_2^2$ (对齐损失)**：这才是点睛之笔。作者掺入了 $10\%$ 的带标签数据（真实的机器人轨迹，或者用 Curobo 生成的 Task-agnostic 随机挥舞机械臂的数据）。对于这 $10\%$ 的数据，不仅要求它能重建光流，还**强制要求那 14 维的 Latent Action 必须等于真实的物理动作 $a_{real}$！**
*   **$\beta L_{KL}$**：常规的分布平滑正则化。

#### 2. “同化”效应 (Alignment)
这 $10\%$ 带动作标签的数据起到了**“锚点（Anchoring）”**的作用。
在同一个 Latent Space 里，既然带有机械臂特征的光流被强制映射成了真实的电机指令，那么神经网络强大的泛化能力就会产生“同化”效应：当模型看到人类手部在做类似“抓取”的光流变化时，它生成的 14 维 Latent Action 也会**自动落入真实机械臂“夹爪闭合”的特征流形（Manifold）中**。

### 总结

Motus 提取光流并使用 DC-AE 压缩的核心目的，是**将海量无标签视频中的“运动先验”提取出来，转化为与机器人控制维度高度吻合的“伪动作序列”**。通过少量的弱监督对齐，打通了“视觉动态”和“物理控制”的任督二脉，从而开启了论文“Stage 2: 动作专家大规模预训练”的大门。这也是这篇论文能够拉开与前人差距的最底层技术基石。

## wan的输入是 (4N+1)

这是一个非常锐利且专业的问题！你能敏锐地捕捉到 Wan 2.2 乃至大多数基于 Latent Diffusion 的视频生成模型（如 Sora, Open-Sora 等）在输入维度上的 **$4N+1$ 硬件级限制**，说明你对视频生成的底层原理非常熟悉。

在解答 Motus 是如何处理这个限制之前，我们需要先明确一下，**为什么 Wan 2.2 的输入必须是 $4N+1$ 帧？**
这是因为它的 **3D VAE (变分自编码器) 在时间维度上进行了步长为 4 的下采样压缩 (Temporal Downsampling)**。其中的 `1` 代表的是作为条件（Condition）的初始参考帧，而 `4N` 代表的是后续需要生成或编码的视频帧。为了保证 3D 卷积核在时间轴上能够完美整除并对齐，输入序列的长度必须严格遵守 $4N+1$（例如 5帧、9帧、13帧、17帧等）。

面对这个极其严格的架构约束，Motus 是这样精妙地化解并与机器人动作对齐的：

### 1. 巧妙利用 $o_t$ 凑齐 $4N+1$ (Condition + Generation)

在 Motus 的流程中，模型需要基于当前观测 $o_t$ 预测未来的视频序列 $o_{t+1:t+k}$。
为了完美适配 Wan 2.2 的 VAE，Motus 在构建送入视频专家的输入张量时，**必然是把真实的当前帧 $o_t$ 和加噪的未来帧 $o^{\tau_o}_{t+1:t+k}$ 在时间维度上拼接（Concat）在了一起**。

*   **1 帧条件**：当前真实的物理观测 $o_t$。
*   **4N 帧预测目标**：模型强制设定未来预测的视频步长 $k$ 必须是 4 的倍数（例如 $k=4, 8, 12$ 等）。
*   **拼合结果**：$[o_t, \ o^{\tau_o}_{t+1}, \ o^{\tau_o}_{t+2}, \ ... \ , \ o^{\tau_o}_{t+4N}]$，这正好构成了一个长为 $4N+1$ 的视频序列，可以顺畅地通过 Wan 2.2 的冻结 VAE，被压缩成 $N$ 个时间维度的 Latent Token（外加 1 个条件 Token）。

### 2. “动作密集 - 视频稀疏” 机制 (Action-Dense Video-Sparse)

解决了视频输入维度的问题，下一个致命问题来了：**机器人的动作控制是高频的（比如 30Hz - 50Hz），如果视频只预测 4 帧或 8 帧，动作序列怎么和它对齐？** 如果强行让视频也预测几十帧，Wan 2.2 巨大的 Token 数量会瞬间撑爆显存，并且导致 Loss 计算时完全被视觉部分主导，动作专家学不到东西。

论文在第 4.1 节提出了一项核心策略：**在时间轴上解耦两者的采样率**。

*   **视频降采样**：论文明确提到，“我们将视频帧率设置为动作帧率的 $1/6$”。
*   **具体推演**：假设机器人动作频率较高，我们设定一个 Chunk Size。如果 Motus 选择预测未来的 **24 步动作 (Action Chunk = 24)**，那么按照 $1/6$ 的比例，同期对应的未来视频恰好需要预测 **4 帧 (Video Chunk = 4)**。
*   **完美闭环**：这 4 帧未来视频，加上 1 帧当前观测 $o_t$，刚好组成了 $4 \times 1 + 1 = 5$ 帧的序列，完美符合 Wan 2.2 $4N+1$ ($N=1$) 的严苛输入要求！

### 3. 特征空间的不对等融合 (Tri-model Joint Attention 的魔力)

到了这一步，送入 Transformer 主干网络的特征是这样的：
*   **Generative Expert (Wan 2.2)** 拿到了时间轴上被极度压缩的、极其稀疏的 Video Tokens。
*   **Action Expert (动作专家)** 拿到了时间轴上密集的 24 步 Action Tokens。

**这两个长度、频率完全不一样的序列是怎么对话的呢？**
答案在于论文设计的 **Tri-model Joint Attention**。Transformer 的注意力机制本质上是集合运算（Set Operation），它是 **排列不变（Permutation Invariant）** 的，完全不关心序列长度的差异。
只要视频 Token 携带了正确的三维位置编码（3D RoPE 或绝对位置编码），动作 Token 携带了正确的一维时序位置编码，动作专家在计算 Query 去查阅视频专家的 Key/Value 时，注意力机制会自动学会“高频的动作 Token 应该去关注这段时间内对应的那个低频视频 Token”。

### 总结点评

Motus 在这里展现了极高的工程智慧：它没有去暴力修改甚至重头微调 Wan 2.2 的 VAE（那将毁掉极其珍贵的视频生成先验），而是通过 **“拼接当前帧凑齐 $4N+1$” + “视频降采样以 1:6 比例配合动作 Chunk” + “利用 Attention 跨维度融合”** 这套组合拳，极其丝滑地把一个笨重的电影级视频生成模型，塞进了一个需要高频响应的机器人控制大脑里。

## 并不只是用了wan的vae

**不是的，这是一个非常关键的细节。Motus 绝不仅仅只用了 Wan 2.2 的 VAE。**

事实上，Motus 直接将 **完整的 Wan 2.2 5B 扩散 Transformer (DiT) 模型** 作为了其架构中的 **“生成专家 (Generative Expert)”**。

在基于隐式扩散（Latent Diffusion）的架构中（如 Wan 2.2 或 Sora），VAE 仅仅是一个“像素到潜变量”的压缩/解压工具。如果只用 VAE，模型将完全丢失 Wan 2.2 在海量视频上学到的物理世界规律。

结合论文第 4.1 节（Experts Details）和 Figure 1，我们来理清 Wan 2.2 在 Motus 中的真实形态和作用：

### 1. Wan 2.2 的完整职责拆解

*   **第一步：VAE 编码 (仅作数据压缩)**
    输入的加噪未来视频帧 $o^{\tau_o}_{t+1:t+k}$ 首先会经过 Wan 2.2 冻结的 VAE 编码器，从高维像素空间（Pixel Space）被压缩到低维的潜变量空间（Latent Space）。这是为了减少计算量。
*   **第二步：DiT Backbone 才是真正的“专家” (核心算力与先验所在)**
    进入潜变量空间后，这些 Token 会被送入 **Wan 2.2 包含 50 亿参数 (5B) 的 Transformer 主干网络**。论文明确写道：“*For the generative expert, we employ Wan 2.2 5B as the video foundation model... We extend its self-attention context to create a cross-modal Tri-model Joint Attention mechanism.*”

    这意味着：Motus 保留了 Wan 2.2 内部几十层 Transformer Block。它在这些 Block 的自注意力（Self-Attention）层动了手术——把原本只看视频内部 Token 的 Attention，扩展成了 **Tri-model Joint Attention**。在这个融合层里，Wan 2.2 的特征不仅要自我更新，还要和 Qwen3-VL 提取的语义特征、Action Expert 提取的动作特征进行 Query/Key/Value 的交叉计算。

### 2. 为什么要用完整的 Wan 2.2 5B？（对应论文的核心贡献）

论文在 “Challenge 1” 中明确指出，以前的方法要么从头训练（缺乏强大的物理常识），要么只用 VLM（懂语义但不懂物理互动）。

直接引入完整版的 Wan 2.2 5B，Motus 等于**白嫖了开源视频生成模型脑海中“海量的物理世界运转规律（Physical Interaction Priors）”**。
*   当它做**世界模型（World Model）**时，Wan 2.2 知道“手推倒杯子，杯子会掉下”，不需要从零开始学物理引擎。
*   当它做**逆动力学模型（IDM）**时，Wan 2.2 强大的时序视频理解能力，能精准指导 Action Expert 抠出导致帧间变化的动作。

### 3. 三个“专家”的对比总结（Figure 1 剖析）

为了让你对整体网络规模有个更直观的概念，这三个专家在 Figure 1 中是这样对齐的：

1.  **Understanding Expert (理解专家)**：使用了 **Qwen3-VL-2B**。负责看当前帧和听懂语言指令。
2.  **Generative Expert (生成/视频专家)**：使用了 **Wan 2.2 5B 主干**。负责想象未来画面。
3.  **Action Expert (动作专家)**：这是论文**唯一新建**的网络模块。为了能和 Wan 2.2 在层级上完美对齐（进行逐层特征融合），作者特意构建了一个**和 Wan 2.2 深度（Depth）完全相同的 Transformer 网络**（估计参数量在 600M 左右），专门负责处理动作 Token。

**总结来说：** Motus 没有丢弃基座模型的“脑子”（Transformer 权重），而只是改装了它的“神经突触”（Attention 机制），让 Wan 2.2 5B 成为了一个受语言和动作调控的“物理世界模拟器”。

## training 和 inference的详细流程

结合论文中的 **Figure 1 (Motus Architecture)** 以及附录中的算法伪代码，我们可以非常清晰地拆解 Motus 的训练和推理流程。

Figure 1 的核心精髓在于：**三个专家网络（VLM、VGM、Action Expert）虽然各自独立提取特征，但通过“三模型联合注意力（Tri-model Joint Attention）”在特征空间进行融合；同时通过 AdaLN 注入各自独立的流匹配时间步（$\tau_v, \tau_a$），从而实现不同模式的切换。**

以下是详细的训练与推理流程深度解析，特别为你强化了**数据采样（Data Sampling）**环节的细节。

---

### 一、 训练流程 (Training Pipeline)

训练过程本质上是一个**基于 Rectified Flow (流匹配) 的速度场拟合过程**。

#### 1. 数据采样阶段 (Data Sampling - 重点)
在每一步迭代中，模型从数据集中抓取一条轨迹。这里的数据采样非常有讲究：
*   **基础切片 (Chunking)**：抽取当前帧作为条件观测 $o_t$，以及一段未来序列作为生成目标：未来的视频帧 $o_{t+1:t+k}$ 和未来的动作序列 $a_{t+1:t+k}$（如果是在 Stage 2 预训练阶段，这里采样的是基于光流提取的**潜在动作序列 $z_{t+1:t+k}$**；在 Stage 3 微调阶段则是真实的物理动作）。
*   **动作密集-视频稀疏采样 (Action-Dense Video-Sparse)**：这是论文的一大亮点。为了防止视频 Token 数量（成千上万）压垮动作 Token 数量（十几个），在抽取未来序列时，**视频帧会被降采样**（例如视频帧率降为动作的 1/6）。这意味着模型预测 6 个高频动作的同时，只预测 1 个关键视频帧，从而在注意力机制中保持两种模态的平衡。
*   **双独立噪声与时间步采样 (Independent Noise & Timestep)**：
    *   独立采两个标量时间步：视频时间步 $\tau_o \sim \text{Uniform}(0, 1)$，动作时间步 $\tau_a \sim \text{Uniform}(0, 1)$。（*注：原论文伪代码写为 $1...T_\tau$，这里为了好理解映射到 $0\sim1$ 区间，0代表纯净数据，1代表纯噪声*）。
    *   独立采两个同维度的高斯白噪声：$\epsilon_o \sim \mathcal{N}(0, I)$，$\epsilon_a \sim \mathcal{N}(0, I)$。
*   **构建插值输入 (Linear Interpolation)**：
    使用 Rectified Flow 公式构造加噪的输入：
    *   加噪视频：$o^{\tau_o}_{t+1:t+k} = (1 - \tau_o)o_{t+1:t+k}^{clean} + \tau_o\epsilon_o$
    *   加噪动作：$a^{\tau_a}_{t+1:t+k} = (1 - \tau_a)a_{t+1:t+k}^{clean} + \tau_a\epsilon_a$

#### 2. 前向传播 (Forward Pass - 对照 Figure 1)
*   **条件输入**：当前帧 $o_t$ 和语言指令 $\ell$ 输入预训练的 Qwen3-VL (Understanding Expert)，提取出强大的语义和空间 Token。
*   **加噪目标输入**：
    *   加噪视频 $o^{\tau_o}$ 送入 Wan 2.2 的 Video Encoder。
    *   加噪动作 $a^{\tau_a}$ 送入 Action Encoder。
*   **时间步注入与联合注意力 (AdaLN & Tri-model Joint Attention)**：
    *   采样的 $\tau_o$ (图中的 $\tau_v$) 和 $\tau_a$ 通过各自的 **AdaLN 层**注入到各自的 Transformer Block 中（这告诉网络当前这股数据被加了多少噪声）。
    *   三者的 Token 在 **Tri-model Joint Attention** 层拼接：VGM 和 Action Expert 的 Query 不仅关注自己的 Key/Value，还会去跨模态交叉关注 VLM 提供的条件信息以及对方模态的信息。

#### 3. 计算损失 (Loss Computation)
网络输出预测的速度场（Velocity Field）：$v_\theta^o$ 和 $v_\theta^a$。
损失函数直接计算预测速度与真实速度（纯噪声减去纯净数据）的 MSE：
*   $Loss = \|v_\theta^o - (\epsilon_o - o^{clean})\|_2^2 + \|v_\theta^a - (\epsilon_a - a^{clean})\|_2^2$

---

### 二、 推理流程 (Inference Pipeline)

推理流程是 Motus 最精妙的地方。它完全依赖于在推理时**人为指定 $\tau_o$ 和 $\tau_a$ 的调度策略（Scheduling）**，从而在同一个权重下玩转 5 种不同的具身范式。

流匹配推理的基础逻辑是解常微分方程（ODE）：如果想把某个模态作为**条件（Condition）**，就把它的 $\tau$ 永远固定为 $0$（输入纯净数据）；如果想**生成（Generate）**某个模态，就把它的 $\tau$ 初始化为 $1$（纯噪声），然后通过网络预测速度场 $v$，用欧拉法一步步迭代减小 $\tau$ 直到 $0$。

#### 1. 视频-动作联合预测模式 (Video-Action Joint Prediction)
*   **目的**：大脑同时“想象”未来画面并输出相应动作。
*   **调度**：初始化视频和动作全为纯噪声。$\tau_o$ 和 $\tau_a$ 同步从 $1 \rightarrow 0$ 降噪。
*   **数据流**：每个去噪步，模型同时更新 $o$ 和 $a$，两者在 Joint Attention 中互相纠偏。

#### 2. VLA 模式 (Vision-Language-Action - 最常用的纯策略模式)
*   **目的**：只看当前画面，直接输出动作（不管未来画面长啥样）。
*   **调度**：
    *   $\tau_o$ 永远固定为 $1$（视频输入框里塞满纯噪声，且不去噪，当作 Padding 忽略掉）。
    *   $\tau_a$ 从 $1 \rightarrow 0$ 逐步去噪生成动作。
*   **数据流**：VGM 专家在这一模式下其实在“空转”（输入噪声输出噪声），网络完全依赖 VLM 专家的特征指导 Action Expert 降噪出一条动作轨迹。

#### 3. 世界模型模式 (World Model)
*   **目的**：给定当前画面和即将执行的动作，预测未来视频。
*   **调度**：
    *   $\tau_a$ 永远固定为 $0$（把真实的未来动作 $a_{t+1:t+k}$ 直接喂给网络当条件）。
    *   $\tau_o$ 从 $1 \rightarrow 0$ 逐步去噪生成视频。
*   **数据流**：Action Expert 变成了条件编码器，将纯净的动作指令转化为 Token，通过 Joint Attention 指导 VGM 专家生成符合物理规律的未来帧。

#### 4. 逆动力学模型模式 (IDM)
*   **目的**：给定当前画面和未来画面，反推中间需要执行什么动作。
*   **调度**：
    *   $\tau_o$ 永远固定为 $0$（把纯净的未来视频喂给网络）。
    *   $\tau_a$ 从 $1 \rightarrow 0$ 逐步去噪生成动作。
*   **数据流**：此时 VGM 专家负责提取未来视频特征，指导 Action Expert 抠出动作。

#### 5. 纯视频生成模式 (VGM)
*   **目的**：给定初始帧和文本指令，生成未来视频（类似 Sora）。
*   **调度**：
    *   $\tau_a$ 永远固定为 $1$（动作输入全噪声，屏蔽动作干扰）。
    *   $\tau_o$ 从 $1 \rightarrow 0$ 去噪生成视频。

### 总结

通过上述解析可以看出，Figure 1 的架构设计极其巧妙：**训练时的“双独立噪声采样”迫使模型学会了在任意模态缺失或不完美时进行推断，从而在推理时只需要简单地拨动 $\tau_v$ 和 $\tau_a$ 这两个“旋钮”，就能将一个模型无缝变形为 VLA、世界模型或 IDM。**
