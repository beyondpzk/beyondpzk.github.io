---
layout: post
title: DreamZero
date: 2026-02-17
categories: []
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# DreamZero

[paper link](https://arxiv.org/abs/2602.15922) 


---

# 世界动作模型与零样本机器人策略 (DreamZero)

**核心文献**：Ye et al., "World Action Models are Zero-shot Policies", NVIDIA, 2026.

---

## 第一部分：引言与背景 (Introduction & Background)

### 1.1 从 VLA 到 WAM 的范式转移
在过去的几年中，基于视觉-语言-动作（Vision-Language-Action, VLA）的模型如 RT-2、OpenVLA 等占据了主流。VLA 的核心逻辑是将机器人控制视为一个多模态序列建模问题，利用预训练的大语言模型（LLM）或视觉语言模型（VLM）的语义能力来实现指令遵循。

然而，现有的 VLA 模型面临一个核心痛点：**物理常识的缺失**。
*   **VLA 的优势**：语义泛化。通过互联网文本数据，它们知道“把可乐给泰勒·斯威夫特”意味着什么。
*   **VLA 的劣势**：物理运动泛化。如果你要求机器人执行一个训练数据中未包含的具体动作（如“解开鞋带”），即便语义理解了，VLA 往往无法生成符合物理动力学的精准运动轨迹。它们缺乏对“世界如何演变”的深层理解。 <alphaxiv-paper-citation paper="2602.15922v1" title="VLA Limitations" page="2" first="Although VLM priors" last="and motor control" />

### 1.2 世界动作模型 (WAM) 的提出
为了解决上述问题，一种新的范式：**世界动作模型 (World Action Model, WAM)**。
*   **核心思想**：利用在大规模互联网视频上预训练的视频生成模型（Video Diffusion Models）作为基座。
*   **直觉**：视频生成模型通过学习预测下一帧，实际上隐式地学习了物理世界的动力学（物体如何掉落、液体如何流动、刚体如何碰撞）。
*   **DreamZero**：这是 NVIDIA 提出的一个 14B 参数量的 WAM。它不仅仅预测动作，而是**联合预测未来视频帧和动作**。

### 1.3 核心贡献概览
在我们深入技术细节前，先概览 DreamZero 的突破性成果：
1.  **零样本泛化**：在未见过的任务和环境中，泛化能力是 SOTA VLA 模型的 2 倍以上。
2.  **数据效率**：证明了从多样化、非重复的数据中学习比从重复性演示中学习更有效。
3.  **实时控制**：通过系统级和算法级优化（DreamZero-Flash），将 14B 的扩散模型推理速度提升 38 倍，实现 7Hz 的闭环控制。
4.  **跨具身迁移**：仅用 10-20 分钟的**纯视频**（无动作标注）数据，就能实现跨机器人甚至人到机器人的技能迁移。 <alphaxiv-paper-citation paper="2602.15922v1" title="Contributions" page="2" first="This results in" last="real-robot experiments." />

---

## 第二部分：DreamZero 模型架构详解 (Model Architecture)

这部分是课程的核心。我们将详细解构 DreamZero 是如何构建的。

### 2.1 整体架构设计
DreamZero 是建立在一个预训练的视频扩散模型（Wan2.1-I2V-14B）之上的。

*   **输入模态 (Inputs)**：
    1.  **视觉上下文 (Visual Context)**：通过 VAE 编码器处理当前和过去的观测图像 $o_{0:l}$。
    2.  **语言指令 (Language)**：通过文本编码器处理指令 $c$。
    3.  **本体感知 (Proprioception)**：机器人的关节状态 $q_l$。

*   **骨干网络 (Backbone)**：
    采用 **DiT (Diffusion Transformer)** 架构。与某些工作仅使用视频模型作为特征提取器不同，DreamZero 直接微调这个 14B 的 DiT，使其成为一个联合生成器。

*   **输出模态 (Outputs)**：
    模型同时预测两个“头”：
    1.  **未来视频 (Future Video)**：$o_{l:l+H}$，即世界模型部分。
    2.  **未来动作 (Future Action)**：$a_{l:l+H}$，即策略部分。

### 2.2 联合预测与逆动力学 (Joint Prediction & Inverse Dynamics)
DreamZero 的训练目标可以被视为一个分解过程。公式 (1) 展示了其数学本质：
$$ \pi_{\theta}(o_{l:l+H}, a_{l:l+H} | o_{0:l}, c, q_l) = \pi_{\theta}(o_{l:l+H} | o_{0:l}, c, q_l) \cdot \pi_{\theta}(a_{l:l+H} | o_{0:l+H}, q_l) $$
这不仅是预测动作，而是先预测“世界将如何演变”（视频预测），再基于预测的未来推导“需要什么动作来实现这个未来”（逆动力学模型 IDM）。DreamZero 将这两步合并在一个端到端的模型中进行联合训练。 <alphaxiv-paper-citation paper="2602.15922v1" title="Formulation" page="6" first="Note that joint" last="IDM" />

### 2.3 训练目标：流匹配 (Flow Matching)
DreamZero 没有使用传统的 DDPM 目标，而是采用了**流匹配 (Flow Matching)**。这是一个更现代的生成模型训练范式，能产生更直的生成轨迹，通常推理效率更高。

*   **定义**：给定块索引 $k$ 和时间步 $t_k \in [0, 1]$。
*   **加噪过程**：线性插值。
    $$ z_{t_k}^k = t_k z_1^k + (1 - t_k) z_0^k $$
    $$ a_{t_k}^k = t_k a_1^k + (1 - t_k) a_0^k $$
    其中 $z_1, a_1$ 是清晰数据，$z_0, a_0$ 是高斯噪声。
*   **损失函数**：模型预测的是**速度向量 (Velocity)** $v_k$，即数据与噪声的差值。
    $$ \mathcal{L}(\theta) = \mathbb{E}_{z, a, \{t_k\}} \left[ \frac{1}{K} \sum_{k=1}^K w(t_k) \| u_\theta([z_{t_k}^k, a_{t_k}^k]; \mathcal{C}_k, c, q_k, t_k) - v_k \|^2 \right] $$
    这就要求模型不仅要学会去噪视频，还要学会去噪动作，且二者共享同一个 DiT 主干进行特征交互。 <alphaxiv-paper-citation paper="2602.15922v1" title="Loss Function" page="7" first="We train the" last="velocity" />

### 2.4 关键设计：自回归生成 (Autoregressive Generation)
DreamZero 采用了**自回归 (Autoregressive, AR)** 架构，而非视频生成中常见的双向 (Bidirectional) 架构。

*   **为什么选择自回归？**
    1.  **推理速度**：AR 架构允许使用 **KV Cache**。在机器人闭环控制中，历史帧的特征不需要重复计算。
    2.  **模态对齐**：双向模型通常需要固定长度的视频块，可能需要对视频进行降采样，这会破坏原始帧率（FPS），导致动作与视频的时间对齐错乱。AR 模型保持原始帧率，确保了极高精度的视频-动作同步。 <alphaxiv-paper-citation paper="2602.15922v1" title="Autoregressive Benefits" page="7" first="Autoregressive generation possesses" last="executed actions." />
    3.  **闭环纠错 (Closed-loop Correction)**：这是 WAMs 最精妙的设计之一。
        *   在推理时，模型预测了下一块的视频和动作。
        *   机器人执行动作后，环境会产生**真实的观测**。
        *   DreamZero 将**真实观测**注入到 KV Cache 中，替换掉之前预测的视频帧。
        *   这消除了纯视频生成中常见的“误差累积”问题。如果不这样做，几秒钟后视频就会崩坏，动作也会随之失效。

---

## 第三部分：从模型到系统——实时控制优化 (Real-time Execution)

一个 14B 参数的扩散模型，单步推理需要数秒，如何用于 30Hz 的机器人控制？这部分我们将探讨 DreamZero 的工程奇迹。

### 3.1 挑战：反应延迟 (Latency)
未优化的基线模型生成一个动作块（Action Chunk）需要约 5.7 秒。这对于闭环控制是不可接受的。

### 3.2 算法级优化：DreamZero-Flash
这是论文中最具创新性的优化点。
*   **问题**：标准扩散模型需要多步去噪（如 16 步）才能生成高质量视频，进而生成准确动作。如果减少步数，视频质量下降，动作也随之变差。
*   **洞察**：在推理时，我们其实不需要完美的视频，我们只需要准确的动作。
*   **方法**：**解耦噪声调度 (Decoupled Noise Schedules)**。
    *   在训练时，视频模态使用 Beta 分布采样噪声 $t_{video} \sim \text{Beta}(7, 1)$，使其倾向于高噪声状态（接近 1）。
    *   动作模态保持均匀采样 $t_{action} \sim U(0, 1)$。
    *   **效果**：这迫使模型学会从**极其模糊/高噪的视频潜变量**中直接恢复出**清晰的动作**。
    *   **结果**：推理时，只需 **1 步 (Single-step)** 去噪即可生成高精度动作，尽管此时视频生成质量很差，但这不影响控制。这使得推理速度提升了 2.33 倍。 <alphaxiv-paper-citation paper="2602.15922v1" title="Flash Concept" page="9" first="DreamZero-Flash closes this" last="remains partially noisy." />

### 3.3 系统级优化
1.  **异步闭环执行 (Asynchronous Execution)**：不等待推理完成再动。机器人执行当前的动作块，GPU 同时在后台计算下一个动作块。只要推理时间小于动作块的执行时间（约 1.6秒），就能实现流畅控制。
2.  **DiT 缓存 (DiT Caching)**：利用流匹配生成轨迹的直线性。如果前后两步的速度向量相似度高，直接复用上一步的计算结果。
3.  **量化与编译**：
    *   使用 **NVFP4** (4-bit Floating Point) 量化权重（在 Blackwell 架构上）。
    *   **CUDA Graphs** 消除 CPU 启动开销。
    *   最终实现了 **38倍** 的速度提升，延迟降至 150ms。 <alphaxiv-paper-citation paper="2602.15922v1" title="Optimization Summary" page="9" first="Collectively, these techniques" last="control at 7Hz." />

---

## 第四部分：实验结果与分析 (Experimental Analysis)

### 4.1 零样本泛化能力 (Zero-shot Generalization)
实验对比了 DreamZero 与 SOTA VLA 模型（如 GR00T, $\pi_0$）。
*   **未见过的环境**：DreamZero 在新环境（未见过的光照、背景）中的表现是 VLA 的 **2倍** 以上。
*   **未见过的任务**：对于“解鞋带”、“给人体模型摘帽子”等训练数据中从未见过的任务，VLA 基本失败（成功率接近 0%），而 DreamZero 展现出了惊人的适应能力（平均任务进度约 40%）。
*   **原因分析**：VLA 仅仅是模仿动作，而 WAM 通过视频预测理解了物体之间的几何和物理交互关系。

### 4.2 数据策略：多样性 vs 重复性
这是一个反直觉的发现。
*   传统机器人学习：倾向于在一个任务上收集大量重复数据以获得高精度。
*   DreamZero 策略：收集了 500 小时的数据，覆盖 22 个不同环境，任务极其杂乱（如“整理桌子”、“扔垃圾”）。
*   **结果**：使用多样化非重复数据训练的模型，泛化性能显著优于使用重复性数据训练的模型（50% vs 33% 任务进度）。这证明了 WAM 能够从杂乱的现实世界交互中提取通用的物理规律。 <alphaxiv-paper-citation paper="2602.15922v1" title="Data Ablation" page="18" first="As shown in" last="repetitive data lacks." />

### 4.3 跨具身迁移 (Cross-Embodiment Transfer)
这是最具前景的方向之一。
*   **实验设置**：AgiBot（双臂轮式机器人）是主训练对象。
*   **迁移来源**：YAM 机器人（另一种双臂）或 **人类第一视角视频**。
*   **方法**：仅使用这些外部视频进行**视频预测训练**（没有动作标签）。
*   **结果**：仅加入 12-20 分钟的外部视频，在未见任务上的性能提升了 **42%**。这表明，仅仅是“看”别人怎么做（视觉体验），就能显著增强机器人的物理理解，进而转化为更好的策略。 <alphaxiv-paper-citation paper="2602.15922v1" title="Cross Embodiment" page="16" first="video-only demonstrations from" last="just 10–20 minutes" />

---

## 第五部分：讨论与总结 (Discussion & Future Directions)

### 5.1 为什么 WAM 是未来？

(我觉得这是一个很重要的点!!!)
*   **Scaling Laws**：实验表明，从 5B 到 14B，模型性能有显著提升（50% vs 21%），而 VLA 模型增大参数量并未带来物理操作上的质变。WAM 似乎能更好地利用 Scaling Law。 <alphaxiv-paper-citation paper="2602.15922v1" title="Scaling" page="18" first="WAMs exhibit clearer" last="erroneous actions." />
*   **数据护城河的瓦解**：如果纯视频数据（YouTube, Ego4D）可以用来提升机器人策略，那么机器人学习的数据瓶颈将被打破。

### 5.2 局限性
*   **计算成本**：即便优化了，14B 模型仍然需要高端 GPU (H100/Blackwell) 支持，部署在边缘端仍有挑战。
*   **长程规划**：目前的 WAM 是 System 1（直觉反应），缺乏 System 2（慢思考、长程逻辑推理）。

### 5.3 思考
1.  DreamZero 的自回归架构如何避免了视频生成中的“幻觉”导致动作失效？
2.  DreamZero-Flash 的训练策略使得动作预测不再依赖高质量视频特征，这是否违背了 WAM “利用视频先验”的初衷？请讨论其中的权衡。
3.  如果将 DreamZero 的架构扩展到人形机器人（Humanoid），你认为最大的挑战是什么？

---

## Figure4 解读


这是一个非常敏锐且切中要害的问题！您触及到了 DreamZero 模型架构中关于**时序建模（Temporal Modeling）**和**训练目标（Training Objective）**的核心细节。

简单直接的回答是：**是的，Flow Matching 的 Loss 计算包含了中间每一帧（或更准确地说是每一个“块”）的视频与动作的监督。** 它的输入形式更接近您描述的第二种 `[Language, video, action, video, action, ...]` 的自回归结构，但具体实现是以**块（Chunk）**为单位进行的。

为了彻底讲清楚这个问题，我们需要结合论文中的 **Figure 4**、**Appendix C (Figure 14)** 以及 **Algorithm 1** 来详细拆解。

### 1. 输入与监督结构：基于“块”的自回归 (Chunk-wise Autoregressive)

DreamZero 并不是一次性把整个长序列扔进去只算最后的 Loss，也不是像简单的 RNN 那样逐帧计算。它是基于**块（Chunk）**的。

*   **什么是 Chunk？**
    论文中设定一个块包含 $K$ 个潜在视频帧（Latent Frames）和对应的动作序列。
    *   通常设置 $K=2$（即每个块包含 2 个视频帧及其对应的动作）。
    *   一个完整的训练样本包含 $M$ 个块（默认 $M=4$）。

*   **序列结构**
    输入序列在逻辑上是这样的：
    $$ [ \text{Conditioning} ] \rightarrow [\text{Chunk}_1] \rightarrow [\text{Chunk}_2] \rightarrow \dots \rightarrow [\text{Chunk}_M] $$
    
    其中每个 $[\text{Chunk}_k]$ 内部包含了 **联合的** Video Latents ($z_k$) 和 Action Latents ($a_k$)。

### 2. 训练时的监督机制 (Teacher Forcing)

在训练阶段（Figure 4 左图），模型采用了类似 LLM 训练中的 **Teacher Forcing** 策略。这意味着模型在预测第 $k$ 个块时，能够看到**真实的、清晰的（Clean）** 前 $k-1$ 个块的历史信息。

具体流程如下：

1.  **输入 (Input)**：
    *   **历史上下文 (Context)**：$C_k = \{(z^1_1, a^1_1), \dots, (z^{k-1}_1, a^{k-1}_1)\}$。这是前 $k-1$ 个块的**真实数据（Ground Truth）**。
    *   **当前噪声输入 (Current Noisy Input)**：$z_{t_k}^k$ (带噪声的视频) 和 $a_{t_k}^k$ (带噪声的动作)。
    *   **条件 (Condition)**：语言指令 $c$ 和当前的本体感知状态 $q_k$。

2.  **预测 (Prediction)**：
    模型接收上述输入，预测当前块的**速度向量 (Velocity)** $v_{pred}$。这个 $v_{pred}$ 同时包含了视频流场和动作流场。

3.  **监督 (Loss Calculation)**：
    **这是您问题的核心答案：** Loss 是对**所有块**进行求和的。
    根据公式 (3) 和 Algorithm 1：
    $$ \mathcal{L}(\theta) = \mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^M \| u_\theta(\dots) - v_k \|^2 \right] $$
    
    *   **中间监督**：模型必须准确预测 $\text{Chunk}_1$，然后准确预测 $\text{Chunk}_2$，以此类推。
    *   **Joint 监督**：在每一个 Chunk 内部，Loss 同时惩罚视频预测误差和动作预测误差。

### 3. 图解补充：Appendix C 的 Figure 14

为了更直观地理解，请看论文附录 C 中的 **Figure 14 (Attention Strategy)**，这张图比 Figure 4 更能解释时序结构：

*   **Training (左图)**：
    *   $C_0$ (初始条件) $\rightarrow$ 预测 $Z_1, Y_1$ (第1块的视频和动作)。
    *   $C_0, Z_1, Y_1$ (作为历史 KV Cache) $\rightarrow$ 预测 $Z_2, Y_2$ (第2块)。
    *   $C_0, Z_1, Y_1, Z_2, Y_2$ (作为历史) $\rightarrow$ 预测 $Z_3, Y_3$ (第3块)。

*   **Attention Mask**：
    这就构成了您所说的 `[Language, video, action, video, action...]` 的交替结构。第 $k$ 步的预测不仅依赖于 Language，还严格依赖于 $1 \dots k-1$ 步的所有 Video 和 Action。

### 总结

回到您的问题：
1.  **输入结构**：是交替的自回归结构，形式为 `Condition -> (Video_1, Action_1) -> (Video_2, Action_2) -> ...`。
2.  **中间监督**：**有**。Flow Matching 的 Loss 会计算每一个中间步骤（Chunk）的视频重建误差和动作重建误差。模型必须学会不仅“达到终点”，还要学会“每一步怎么走”。

这种设计确保了模型不仅能规划长程目标，还能在每一步都保持视频生成的连贯性和动作执行的准确性。 <alphaxiv-paper-citation paper="2602.15922v1" title="Training Objective" page="7" first="We train the model" last="Eq. 3" />

## 初始的 $C_0$ 包括历史多少桢


根据论文 **Appendix C (Page 21)** 以及 **Figure 14** 的详细说明，我们可以非常精确地回答这个问题。

在 Figure 14 中，$C_0$ 代表的是一个**条件块 (Conditioning Chunk)**。根据论文的设置，这个 $C_0$（以及随后的 $C_1, C_2$ 等）的具体规模如下：

### 1. 潜变量空间 (Latent Space)
*   **$C_0$ 包含 2 个潜在帧 (Latent Frames)**。
*   论文中明确提到："For DreamZero, we set each chunk as $K=2$ latent frames."（对于 DreamZero，我们将每个块设置为 $K=2$ 个潜在帧）。

### 2. 原始像素空间 (Raw Pixel Space)
*   **$C_0$ 对应约 8 个原始视频帧**。
*   **推导过程**：
    *   论文提到："For Agibot training data, the video is sampled at 5FPS ratio... video and action span 1.6 seconds per chunk."（视频采样率为 5FPS，每个块跨度 1.6 秒）。
    *   $1.6 \text{ seconds} \times 5 \text{ FPS} = 8 \text{ frames}$。
*   **验证**：论文还提到最大上下文长度是 4 个块（$M=4$），即 8 个潜在帧，这"equivalent to 33 raw frames"（相当于 33 个原始帧）。这符合视频 VAE 常见的时序压缩特性（通常是 $4 \times \text{Latent} + 1$ 左右）。

### 3. 时间跨度 (Time Span)
*   **$C_0$ 代表了 1.6 秒的历史信息**。

### 总结
在 Figure 14 的架构中，初始条件 $C_0$ 并不是仅仅指“第一帧图片”，而是一个包含了 **1.6秒历史信息（约8帧原始图像，压缩为2帧潜变量）的时空数据块**。

**注意：** 在推理的最开始（$t=0$），当机器人刚启动还没有历史视频时，算法会将当前的初始观测图像（$o_{init}$）进行处理（通常是复制或通过 VAE 编码），来填充这就绪的 $C_0$ 槽位，作为启动生成的第一个条件。 <alphaxiv-paper-citation paper="2602.15922v1" title="Context Details" page="21" first="For DreamZero, we" last="seconds per chunk." />

## Causal Mask是按chunk来算的,并不是按token级别来算的

**DreamZero 的 Causal Mask 是按“块（Chunk）”级别来算的，而不是按 Token 级别来算的。**

这意味着：在同一个 Chunk 内部的所有 Token 之间，使用的是**全注意力（Full Attention / Bidirectional）**，只有在不同的 Chunk 之间，才使用**因果掩码（Causal Mask）**。

我们需要区分 **LLM（如 GPT）** 和 **Diffusion Transformer (DiT)** 在处理序列时的根本差异。

### 1. 为什么不是 Token 级别的自回归？

如果在 DiT 中使用 Token 级别的自回归（像 GPT 那样，Token $i$ 只能看 Token $0...i-1$），那么模型在生成一张图片（或几帧视频）时，必须从左上角一个像素一个像素地生成到右下角。

但 **Diffusion/Flow Matching** 的工作原理不是这样的。
*   **Diffusion 的逻辑**：它是通过迭代去噪（Iterative Denoising）来生成数据的。
*   **并行性**：在去噪的某一步（timestep $t$），模型是**同时**看到当前 Chunk 内所有的 Noisy Tokens 的。
*   **空间建模**：为了理解图像的全局结构（比如“这里有张桌子”），左上角的 Token 需要能看到右下角的 Token。

因此，在 **同一个 Chunk 内部**，必须是**双向注意力（Bidirectional Attention）**。

### 2. DreamZero 的 Attention Mask 结构

根据论文 **Appendix C** 的 **Figure 14**，我们可以画出它的 Attention Mask 矩阵。假设我们有 3 个 Chunk ($C_0, C_1, C_2$)，每个 Chunk 包含多个 Tokens（视频 Patch Tokens + 动作 Tokens）。

Attention Mask 矩阵是一个 **块状下三角矩阵 (Block Lower-Triangular Matrix)**：

$$
\begin{bmatrix}
\text{Block}_0 & -\infty & -\infty \\
\text{Block}_{1,0} & \text{Block}_1 & -\infty \\
\text{Block}_{2,0} & \text{Block}_{2,1} & \text{Block}_2
\end{bmatrix}
$$

*   **Block 对角线 ($\text{Block}_k$)**：这是 **Chunk $k$ 内部的注意力**。这里是**全 1 矩阵（Full Attention）**。Chunk $k$ 内部的所有视频 Token 和动作 Token 都可以互相看见。这是为了让 DiT 能够学习当前块内的空间结构和视频-动作对齐。
*   **下三角区域 ($\text{Block}_{k, j}, k > j$)**：这是 **Chunk $k$ 对历史 Chunk $j$ 的注意力**。这也是**全 1**。当前块可以看所有历史块。
*   **上三角区域 ($\text{Block}_{k, j}, k < j$)**：这是 **Mask 掉的区域 ($-\infty$)**。当前块**不能**看未来的块。

### 3. 论文证据 (Appendix C, Figure 14)

看一下论文的 **Figure 14 (a)**：

> "QKV Self-Attention mask for DreamZero training... Given conditioning frames ($C_0, C_1, C_2$), we train the model to predict the velocities of next frames ($Z_1, Z_2, Z_3$) and actions ($Y_1, Y_2, Y_3$)."

*   图中显示，预测 $Z_1, Y_1$ 时，它们能看到 $C_0$。
*   预测 $Z_2, Y_2$ 时，它们能看到 $C_0, Z_1, Y_1$。
*   **关键点**：在预测 $Z_1$（Chunk 1 的视频）和 $Y_1$（Chunk 1 的动作）时，它们是**作为一个整体**被处理的。$Z_1$ 中的所有 Token 和 $Y_1$ 中的所有 Token 共享同一个 TimeStep $t_1$，并且在计算 Self-Attention 时是完全互通的。

### 总结

DreamZero 的自回归是 **Inter-Chunk Autoregressive（块间自回归）**，但 **Intra-Chunk Bidirectional（块内双向）**。

*   **宏观上**：它像 GPT，Chunk 1 -> Chunk 2 -> Chunk 3。
*   **微观上**：它像 BERT/ViT，每个 Chunk 内部是全连接的，因为它是基于扩散模型的生成，而不是基于 Next-Token Prediction 的生成。

这种设计既保留了长程时序生成的因果性（通过 Causal Mask），又保留了视频帧内和帧间极强的空间一致性（通过 Full Attention）。 <alphaxiv-paper-citation paper="2602.15922v1" title="Attention Mask" page="21" first="We visualize the attention" last="inference in Figure 14." />


## Figure14 理解

您的理解完全正确！那个结构正是 `[History, Block1, Block2, ...]`。

在论文的符号体系（特别是 **Figure 14**）中，这个“历史”被称为 **$C_0$ (Conditioning Context)**。

让我们通过 Figure 14 来精确对齐一下术语，以免混淆：

### 1. 序列的真实结构

输入到 Attention 层的序列确实是包含“历史”的。我们可以把它看作：

$$ [\mathbf{C_0}, (\mathbf{Z_1}, \mathbf{Y_1}), (\mathbf{Z_2}, \mathbf{Y_2}), \dots] $$

*   **$C_0$ (您的 "History")**：
    这是**初始条件块**。它包含了机器人当前的观测图像（以及过去几帧的观测）。在推理或训练开始时，这部分是**已知且固定（Clean/Frozen）**的，不需要去噪。
    *   它作为“Prompt”或“Context”存在。
    *   它占据了序列的最开端位置。

*   **$(Z_1, Y_1)$ (您的 "Block 0")**：
    这是模型需要生成的**第一个预测块**（未来第 1-8 帧视频 + 动作）。
    *   在计算 Attention 时，它**必须**看到 $C_0$。
    *   如果没有 $C_0$，模型就不知道当前机器人在哪里，也就无法预测下一步动作。

### 2. Attention Mask 的细节 (Figure 14)

让我们再看一次 Figure 14 的 Attention Mask 矩阵，这次我们把 $C_0$ 加进去：

| | $C_0$ (History) | $Z_1, Y_1$ (Block 1) | $Z_2, Y_2$ (Block 2) |
|---|---|---|---|
| **$C_0$** | **1** (Self) | 0 | 0 |
| **$Z_1, Y_1$** | **1** (Attend to Hist) | **1** (Self) | 0 |
| **$Z_2, Y_2$** | **1** (Attend to Hist) | **1** (Attend to Prev) | **1** (Self) |

*   **第一行 ($C_0$)**：它只看自己（其实在推理时它甚至不需要计算 Attention 输出，因为它只是 KV Cache 的一部分，供别人查询）。
*   **第二行 ($Z_1, Y_1$)**：这是**关键**。第一个生成的块，其 Attention Mask 对 $C_0$ 是 **1 (可见)**。这意味着它**完全依赖** $C_0$ 中的视觉特征来生成。

### 3. "冷启动"问题 (Cold Start)

您可能会问：“如果机器人刚开机，没有历史怎么办？”
*   在这种情况下，$C_0$ 依然存在。
*   $C_0$ 会被填充为**当前这一刻的静态帧**（可能复制多次以填满 $K=2$ 的潜在帧长度，或者 VAE 编码器会自动处理单帧输入）。
*   所以，序列永远是以一个**非空的 $C_0$** 开始的。

### 总结

是的，您的理解 `[history, block0, block1, ..]` 是完全正确的。

*   **History** = 论文中的 **$C_0$** (Conditioning Frames)。
*   **Block 0** = 论文中的 **$(Z_1, Y_1)$** (1st Generated Chunk)。
*   **Block 1** = 论文中的 **$(Z_2, Y_2)$** (2nd Generated Chunk)。

第一个生成的块 $(Z_1, Y_1)$ **绝对有**历史信息输入，这个历史信息就是 $C_0$。 <alphaxiv-paper-citation paper="2602.15922v1" title="Initial Context" page="21" first="Given conditioning frames (C0" last="predict the velocities" />

**History ($C_0$) 包含 8 帧原始图像（Raw Frames），对应时长 1.6 秒。**

以下是具体的换算细节（基于论文附录 C）：

1.  **原始图像层 (Pixel Space)**：
    *   **帧数**：**8 帧**。
    *   **时长**：**1.6 秒**。
    *   **采样率**：论文中 AgiBot 数据是以 5 FPS 进行采样的（$1.6s \times 5Hz = 8$ 帧）。

2.  **模型输入层 (Latent Space)**：
    *   这 8 帧原始图像经过 VAE 压缩后，变成 **2 个潜在帧 (Latent Frames)**。
    *   这就是为什么论文中说 "we set each chunk as $K=2$ latent frames"。

所以，在您的序列 `[history, block0, block1, ...]` 中：
*   **history ($C_0$)** = 过去的 8 帧（1.6秒）。
*   **block0 ($Z_1$)** = 预测未来的 8 帧（1.6秒）。
*   **block1 ($Z_2$)** = 再预测未来的 8 帧（1.6秒）。

以此类推。

## latent的形式,是VAE还是DinoV2这种

**在机器人控制领域，基于 DinoV2 等判别式特征的 Tokenizer 确实可能比基于重建的 VAE Tokenizer 更有优势。**

目前的学术界正处于这两种路线的激烈博弈中。为了深入探讨您的观点，我们需要从**表征的本质**、**预训练的红利**以及**任务的目标**三个维度来拆解。

### 1. 表征的本质：重建 (Reconstruction) vs. 语义 (Semantics)

这是 VAE 和 DinoV2 最根本的区别。

*   **VAE (DreamZero/Sora/Wan 路线)**：
    *   **目标**：**像素级重建**。它的 Loss 是 MSE 或 LPIPS，目的是让解码出来的图片和原图一模一样。
    *   **问题**：这导致 Latent Space 中包含了大量**对机器人决策无关的高频信息**。例如：墙纸的花纹、光照的微小抖动、地毯的纹理。
    *   **劣势**：模型不仅要学习预测“手臂怎么动”，还要花费大量参数去预测“背景的纹理怎么变”。这对于控制任务来说是**算力浪费**。

*   **DinoV2 (以及 LeCun 的 JEPA 路线)**：
    *   **目标**：**语义一致性**。它的 Loss 是对比学习或掩码预测，目的是理解“这是什么物体”、“它在哪里”、“它和谁相似”。
    *   **优势**：DinoV2 对光照、纹理不敏感，但对**几何结构、物体类别、空间关系**极其敏感。这恰恰是机器人做规划（Planning）和控制（Control）最需要的“状态（State）”。
    *   **您的观点支撑**：如果我们将世界模型看作一个“状态转移机”，DinoV2 提供的 $z_t$ 是更紧凑、更鲁棒的 State，因此基于它的 Action Prediction 应该更准、泛化性更强。

### 2. 为什么 DreamZero (以及大多数 WAM) 依然选了 VAE？

如果 DinoV2 这么好，为什么 DreamZero 还是用了 VAE？原因在论文中其实隐含了（这也是当前研究的一个无奈之处）：

**为了“蹭”视频生成模型的预训练权重（Pretraining Prior）。**

*   **DreamZero 的核心卖点**：它不是从头训练的，它是基于 **Wan2.1-14B**（一个强大的文生视频模型）微调的。
*   **绑定关系**：Wan2.1 在预训练时，是基于 VAE 的 Latent Space 训练的。所有的物理知识（重力、碰撞、流体）都存储在那个 DiT 权重里，而这个权重是和 VAE 的分布强耦合的。
*   **代价**：如果 DreamZero 想用 DinoV2，它就**无法复用** Wan2.1 的 14B 参数。它必须从头训练一个基于 DinoV2 特征的 DiT。这需要成千上万张 H100 和亿级视频数据，这对于学术界甚至普通工业界实验室都是不可接受的。

**结论**：使用 VAE 是一种**工程上的妥协**，是为了继承互联网规模的视频物理先验。

### 3. 未来的方向：Latent World Models (JEPA / V-JEPA)

您提到的“DinoV2 更有优势”，其实正是 **Yann LeCun** 一直倡导的 **JEPA (Joint Embedding Predictive Architecture)** 架构，或者最近的 **V-JEPA**。

*   **V-JEPA** 的逻辑：完全放弃像素解码。我预测下一帧的 Embedding（类似 DinoV2 特征），而不是预测像素。
*   **优势**：
    1.  **效率极高**：不需要预测高频噪声。
    2.  **泛化更强**：专注于物体运动和交互。
*   **劣势**：**不可视化**。你无法直接把预测的特征解码成视频给人类看。这在调试时很不直观（你不知道机器人到底想干嘛，你只能看到一堆向量），而且无法像 DreamZero 那样做“视频生成”的 Demo。
