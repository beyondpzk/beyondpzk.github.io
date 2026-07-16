---
title: Emu1
date: 2023-07-11
categories: [others]
---

# Emu1

takeaways:

1. Causal Transformer;
2. 统一AR,两个heads.
3. 外挂 SD用于生成.

[paper link](https://arxiv.org/abs/2307.05222)

Emu 的核心创新在于打破了以往多模态大模型“只能看图说话”的局限，提出了**多模态序列中的统一生成式预训练**（Generative Pretraining in Multimodality）。它不仅能接受图像、文本、视频交错的输入，还能原生输出连续的视觉特征，进而生成图像。

本讲义将从背景动机、模型架构设计、统一损失函数、数据工程（尤其是创新的视频-文本交错数据），以及多阶段的训练策略这五个维度，为大家进行深度拆解。

---

## 1. 背景与动机：当前多模态大模型的局限

在深入 Emu 之前，我们需要回顾一下现有的大型多模态模型（Large Multimodal Models, LMMs）的发展现状。

自从大型语言模型（LLMs）通过“预测下一个词（predict-the-next-word）”展现出惊人的理解和生成能力后，研究者开始尝试将视觉信号接入 LLM，例如 Flamingo、BLIP-2 和 LLaVA。然而，这些主流的 LMM 普遍存在两个显著的局限性：

1. **单向的生成目标**：现有的 LMM 几乎无一例外地将视觉编码器冻结，仅将其输出作为 LLM 的条件输入（Condition），而模型的训练目标依然是“预测下一个文本 Token”。这意味着模型缺乏对视觉信号的直接监督，更无法原生生成图像，极大地限制了模型的表征容量。
2. **多模态数据的利用率低**：主流模型大多依赖图文对（Image-Text Pairs）或包含静态图片的图文交错文档进行训练。它们忽视了互联网上最庞大、最自然的多模态数据源——**视频**。视频天然包含了交错的图像帧和文本字幕，且具有极强的跨模态时序相关性。

为了解决这些痛点，Emu 的作者们提出了一个“杂食性（Omnivore）”的自回归训练框架。正如论文中所述：

> "Emu is end-to-end trained with a unified objective of classifying the next text token or regressing the next visual embedding in the multimodal sequence."

---

## 2. 模型核心架构详解 (Detailed Architecture)

Emu 的架构设计是这篇论文最精妙的部分。要实现图文并茂的输入与输出，模型需要将 2D 的空间视觉信号与 1D 的离散文本信号统一到一个自回归框架中。

Emu 的整体架构由四个核心组件构成：**视觉编码器（Visual Encoder）**、**因果变换器（Causal Transformer）**、**多模态大语言模型（Multimodal Modeling LLM）** 和 **视觉解码器（Visual Decoder）**。

### 2.1 视觉编码器：EVA-CLIP
为了提取高质量的视觉特征，Emu 使用了 10 亿参数量的 **EVA-01-CLIP** 作为视觉编码器。
* **功能**：将输入的图像（或视频帧）转换为密集的 2D 视觉特征图（Dense Visual Features）。
* **作用**：提供具有强语义和细粒度视觉信息的底层表征，为后续的模态对齐打下基础。

### 2.2 因果变换器：Causal Transformer (核心创新组件)
图像在本质上是 2D 空间信号，**不具备文本那样自然的从左到右的因果依赖性（Causal Dependency）**。如果强行按照光栅扫描顺序（Raster Order，即从左上到右下逐像素扫描）进行自回归建模，效果往往不佳。

为了解决这个问题，作者引入了 **Causal Transformer**：
* **输入机制**：模块接收一组随机初始化的连续嵌入向量 $\{e_1, e_2, \dots, e_N\}$ 作为查询（Queries）。
* **注意力机制**：它采用了类似于标准 Transformer 解码器的结构。在交叉注意力层（Cross-attention Layer）中，上述查询向量会与 EVA-CLIP 提取的空间视觉特征（作为 Keys 和 Values）进行交互。
* **输出机制**：模块最终输出 $N$ 个视觉因果嵌入序列 $\{z_1, z_2, \dots, z_N\}$。

> "To better capture the characteristics of images and achieve unified modeling of different modalities, we propose a Causal Transformer module to transform 2D spatial visual signals to 1D causal sequences in a latent space Z."

通过这种方式，Emu 成功地将 2D 的空间信息压缩并重构为一个具有 $1D$ 因果依赖关系的潜在空间序列，使其能够与文本 Token 无缝拼接。如果输入的是包含 $T$ 帧的视频，那么视频将被编码为 $T \times N$ 个视觉因果嵌入。
(这种方式的好处是不管图像输入是多大,得到的vision tokens总是固定的.)

### 2.3 多模态序列拼接与大语言模型 (LLaMA)
经过处理后，图像被转换为 $N$ 个因果嵌入。为了让 LLM 识别图像的边界，Emu 在每个图像特征序列的前后分别插入了特殊的界定符 `[IMG]` 和 `[/IMG]`。

随后，这些视觉嵌入与离散的文本 Token 一起构成了多模态交错序列，输入到拥有 130 亿参数的 **LLaMA** 模型中。在这个阶段，无论是图像还是文本，都被视为序列中的一个“元素”，LLaMA 将对它们进行统一的自回归建模。

### 2.4 视觉解码器：Stable Diffusion
当模型处于推理阶段（Inference）并被要求生成图像时，LLaMA 会自回归地输出 $N$ 个视觉嵌入。这 $N$ 个嵌入不能直接展示给用户，需要一个图像生成器将其解码为像素级图像。
* Emu 使用了预训练的 **Stable Diffusion v1.5** (潜在扩散模型) 作为视觉解码器。
* 作者修改了 Stable Diffusion U-Net 中的交叉注意力线性投影层，使其条件输入不再是文本的 CLIP 嵌入，而是 Emu 生成的这 $N$ 个视觉因果嵌入。

---

## 3. 统一的训练目标与损失函数

在预训练阶段，Emu 的核心任务可以概括为：**在多模态序列中预测下一个元素**。

假设我们有一个多模态交错序列 $x = (x_1, x_2, \dots, x_n)$，其中包含了原始图像、视频帧和文本。在将其中的视觉连续信号通过 Causal Transformer 转化为因果序列后，我们得到了一个全新的由文本 Token 和视觉嵌入混合构成的序列 $u = (u_1, u_2, \dots, u_m)$。

Emu 采用了统一的自回归极大似然估计来近似原始语料的分布：

$\max_{\theta} \sum_{u \in D} \sum_{i=1}^{|u|} \log P(u_i | u_1, \dots, u_{i-1}; \theta) \approx p(x)$

> "Different from existing LMMs that compute the predict-the-next loss on text tokens only, in training Emu, all input elements including both discrete text tokens and continuous image embeddings are accounted for loss computation."

这里有一个极其关键的工程设计。由于文本是离散的，而图像嵌入是连续的，模型需要使用两套不同的损失函数机制（两个不同的 Head）：
* **分类损失（Classification Loss）**：当预测的下一个元素是离散的文本 Token 时，使用标准的**交叉熵损失（Cross-Entropy Loss）**。语言建模头会计算词表上的概率分布。
* **回归损失（Regression Loss）**：当预测的下一个元素是连续的视觉嵌入时，使用 **$\ell_2$ 范数回归损失（$\ell_2$ Regression Loss）**。回归头会直接预测多维空间中的连续向量，并与 Causal Transformer 输出的真实嵌入计算 $\ell_2$ 距离。

这种将离散分类与连续回归统一在一个自回归前向传播中的设计，赋予了 Emu 强大的生成能力。

---

## 4. 数据构建：引入交错视频-文本数据的创新

大模型的成功离不开高质量的数据。除了常规的图文对（LAION-2B, LAION-COCO）和图文交错网页数据（MMC4），Emu 团队在数据工程上的最大贡献是构建了 **YT-Storyboard-1B** 数据集。

视频是多模态特征的天然载体，但处理视频数据的计算成本通常是极其高昂的。为了高效利用视频资源，作者采取了一种极其聪明的做法：
1. **提取故事板缩略图**：不直接下载和解码笨重的原始视频，而是爬取 YouTube 视频的“故事板（Storyboard）”缩略图。这些是网站为了视频预览而生成的关键帧集合。这直接将数据存储和处理成本降低了 20 倍。
2. **结合时间戳与字幕**：将这些故事板图像与视频底部的字幕文件按照时间戳（Timestamps）进行排序对齐。

如此一来，原本复杂的视频被优雅地转化为了按照时间顺序交错排列的“图像帧-文本序列”。这不仅保留了强烈的跨模态时序相关性，还极大扩充了多模态上下文学习（In-context Learning）的训练语料。

---

## 5. 多阶段训练策略

Emu 的训练并非一蹴而就，而是分为三个缜密的阶段，这也是目前构建全能型基础模型的标准范式。

### 5.1 阶段一：多模态统一预训练 (Pretraining)
在此阶段，模型（除解码器外）进行端到端（End-to-End）训练。
* **参数初始化**：视觉编码器使用 EVA-CLIP 权重，LLM 使用 LLaMA 权重，Causal Transformer 随机初始化。
* **训练细节**：总参数量 14B，在 128 张 A100 (80GB) GPU 上训练了 10k 步，消耗了大约 1500 亿个 Token 的数据，用时约 2 天。

### 5.2 阶段二：视觉解码器微调 (Visual Decoding)
预训练完成后，LLaMA 已经学会了在给定文本或多模态上下文后输出合理的“视觉嵌入”。但我们还需要一个解码器把它们变成人眼可识别的图片。
* **过程**：冻结 Emu 的视觉编码器和 LLM 模块，仅微调 Stable Diffusion (SD) v1.5 中的 U-Net 权重。
* **数据与策略**：使用 LAION-COCO 和高美学质量的 LAION-Aesthetics 数据集进行文本到图像（Text-to-Image）的训练。为了提升生成质量，训练中还引入了无分类器引导（Classifier-free Guidance），在 10% 的概率下随机丢弃图像嵌入条件。
(我理解finetune SD时,输入是带noise的图, 条件是)

### 5.3 阶段三：多模态指令微调 (Instruction Tuning)
为了让模型遵循人类意图并进行自然对话，作者在最后阶段进行了指令微调（Instruction Tuning），最终得到了 **Emu-I** 模型。
* **技术实现**：冻结预训练好的主体参数，采用 **LoRA (Low-Rank Adaptation)** 技术，仅在 LLM 的自注意力层的线性投影上添加可训练的旁路。
* **训练数据**：结合了纯文本指令（ShareGPT, Alpaca）、图像指令（LLaVA）以及视频指令（VideoChat, Video-ChatGPT），使用统一的格式进行监督微调。

---

## 6. 模型能力评估与涌现特性

Emu 展现出了作为多模态通才（Generalist）的强大能力，主要体现在以下几个维度：

* **强大的多模态理解与推理**：在零样本（Zero-shot）和少样本（Few-shot）的图像/视频问答任务上（如 VQAv2, VizWiz, MSVDQA 等），Emu 的表现大幅超越了 Flamingo-9B 和 Kosmos-1 等现有模型。
* **上下文多模态生成 (In-Context Text-to-Image Generation)**：这是 Emu 最令人惊艳的涌现能力之一。当你给模型提供两张带有特定艺术风格（如油画）的图片作为上下文（Prompt），并输入新文本时，Emu 能够深刻理解上下文中的视觉风格，并生成与之风格完全一致的新图片。这证明了模型能够真正在多模态序列中进行逻辑和风格的迁移。
* **图像融合 (Image Blending)**：由于模型是在连续的视觉嵌入层面上进行回归，这赋予了它在潜在空间中融合多张图像概念的能力，例如将“猫”和“老虎”的视觉特征结合，生成一张高度逼真的“虎斑猫”。

---

## 7. 总结与反思

1. **统一架构是未来的趋势**：摒弃“文本专属模型”与“图像专属模型”的隔离，将所有模态映射到 1D 因果序列并进行统一自回归预测，是迈向真正意义上的通用人工智能（AGI）的重要一步。
2. **数据的形态决定模型的上限**：通过创新性地将视频转化为交错的图文序列（YT-Storyboard-1B），不仅绕开了算力瓶颈，更让模型学习到了远比静态图文对更丰富的常识与物理规律。

当然，Emu 也在论文结尾坦诚了其**局限性**。由于其基于自回归架构，生成图像或长文本的推理速度依然较慢；模型仍难以完全避免视觉和语言上的幻觉（Hallucinations）；同时，由于训练数据以英文为主，模型在多语言处理上还存在短板。这些也是在座各位同学未来在研究生或博士阶段可以深入探索的课题。

## training samples

在 Emu 的统一自回归训练框架下，无论是单张图片、图文对、长篇网页文章，还是包含时间流的视频，最终都会被展平、拼接成一个**1D 的多模态交错序列（1D Multimodal Interleaved Sequence）**。

为了让大家具象化地理解 Emu 是如何处理和学习这些数据的，我将按照论文中提到的四种主要数据类型，为大家拆解它们在输入到 LLaMA 模型前的真实底层序列形态。

为了方便展示，我们做一个符号约定：
*   我们用 `[IMG]` 和 `[/IMG]` 代表图像/视频帧的开始和结束界定符（特殊 Token）。
*   我们用 ``v_1`, `v_2`, ..., `v_N`` 代表一张图像经过视觉编码器（EVA-CLIP）和因果变换器（Causal Transformer）处理后，输出的 $N$ 个连续视觉因果嵌入向量（Continuous Visual Causal Embeddings）。对于 Emu 来说，这就是模型眼中的“一张图”。

---

### 1. 基础图文对数据 (Image-Text Pairs)
**数据来源**：LAION-2B, LAION-COCO。

这类数据最简单，就是“一张图 + 一段描述”。但在构建训练样本时，Emu 的团队做了一个非常关键的工程设计：**随机将图像放在文本的前面或后面**。

*   **形态 A（看图说话模式 / Captioning）**：图像在先，文本在后。
    > `[IMG] `v_1`, `v_2`, ..., `v_N` [/IMG] A cute puppy is sitting and resting on the lawn, surrounded by many flowers.`
    *   **训练逻辑**：当模型自回归地处理到文本部分时，语言建模头（Language Modeling Head）被激活。模型需要基于前面的视觉连续向量，计算交叉熵损失（Cross-Entropy Loss）来预测下一个离散单词。这赋予了模型图像理解和描述的能力。

*   **形态 B（文本生图模式 / Text-to-Image）**：文本在先，图像在后。
    > `A cute puppy is sitting and resting on the lawn, surrounded by many flowers. [IMG] `v_1`, `v_2`, ..., `v_N` [/IMG]`
    *   **训练逻辑**：当模型处理到 `[IMG]` 之后时，回归头（Regression Head）被激活。模型需要基于前面的文本，预测下一个连续的视觉嵌入向量 ``v_i``，并计算预测向量与真实向量的 $\ell_2$ 回归损失。这赋予了模型原生的图像生成能力。

---

### 2. 图文交错网页数据 (Interleaved Image and Text)
**数据来源**：Multimodal-C4 (MMC4)。

这类数据模拟了人类阅读长篇连载文章、博客或教科书的过程，图像和文本是深度交织的。研究人员会截取长度为 1024 个 Token，且最多包含 5 张图的子序列。

*   **序列示例（一篇旅游博客）**：
    > `We started our journey early in the morning. The sunrise over the mountain was breathtaking. [IMG] `v_1`, ..., `v_N` [/IMG] After hiking for two hours, we found a hidden waterfall. The water was crystal clear. [IMG] `v_1`, ..., `v_N` [/IMG] We decided to set up our camp near the river...`

*   **训练逻辑**：这种长程的交错序列极大地锻炼了模型的**多模态上下文学习能力（Multimodal In-context Learning）**。模型不仅要学习局部图像和局部句子的对应关系，还要学会在被长文本隔开的多个图像之间建立语义联系。这就是为什么 Emu 在论文后续的实验中，能够根据前置的几张图像风格，零样本（Zero-shot）生成相同风格新图像的原因。

---

### 3. 视频-文本交错数据 (Interleaved Video and Text)
**数据来源**：YT-Storyboard-1B（Emu 团队的核心创新数据集）。

这是本篇论文最精彩的数据工程。Emu 没有直接解码庞大的视频流，而是将 YouTube 视频的“故事板缩略图（Storyboard Images）”与底部的“字幕（Subtitles）”根据时间戳严格对齐并排序。

*   **序列示例（以论文 Figure 3 中的鸸鹋自然纪录片为例）**：
    > `[IMG_frame1] `v_1`, ..., `v_N` [/IMG] the female puts the egg in a shallow pit the male bird`
    > `[IMG_frame2] `v_1`, ..., `v_N` [/IMG] covers it with leaves for protection`
    > `[IMG_frame3] `v_1`, ..., `v_N` [/IMG] incubation of eggs is the job of the male bird in nature.`

*   **训练逻辑**：在这个序列中，文本字幕和视频关键帧像拉链一样咬合在一起。
    *   相比于静态图文，这里的图像（相邻帧）在视觉上具有高度的连贯性和细微的物理状态变化（例如鸟类覆盖树叶的动作）。
    *   通过对这种极高密度的时序交错序列进行自回归预测（同时计算文本的分类损失和图像帧的回归损失），Emu 隐式地学习到了**物理世界的因果关系（Causality）、时间流逝（Temporal Dynamics）以及动作的连贯性**。这使得 Emu 在视频问答（Video QA）任务上表现出了远超同类模型的水准。

---

### 4. 指令微调阶段的对话数据 (Instruction Tuning Samples)
**数据来源**：LLaVA, Video-ChatGPT 等指令数据集。

在预训练阶段（上述 1、2、3）结束后，模型拥有了强大的生成和理解能力，但它还不知道如何与人类“聊天”。因此，在第三阶段，所有的训练样本都被严格封装进了带有特定角色前缀的对话模板中。

*   **序列示例（图像问答任务）**：
    > `[USER]: based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase. [IMG] `v_1`, ..., `v_N` [/IMG] What is the man in the picture doing?`
    > `[ASSISTANT]: the answer is: surfing`

*   **训练逻辑**：在这个阶段，视觉编码器和模型主体是被冻结的（Frozen），模型仅通过 LoRA（低秩微调）更新 LLM 的自注意力层参数。训练的目标是让模型收敛到一种“听从人类指令、以特定格式输出”的概率分布上，最终得到我们能够交互的 **Emu-I** 模型。

---

### 课堂总结
同学们，观察这些样本形态，我们可以得出一个极其重要的结论：**Emu 架构的优雅之处，就在于它的“一视同仁”**。

无论原始数据是图像、文本还是视频，经过视觉分词器（Causal Transformer）的降维转换后，在多模态大语言模型（LLaMA）的眼中，它们都变成了完全平权的“序列元素（Tokens/Embeddings）”。模型唯一需要做的事情，就是在给定历史序列的条件下，预测下一个元素——如果是离散文本，就算分类损失；如果是连续图像向量，就算回归损失。

理解了这些数据的输入形态，你们就真正掌握了《EMU》这篇论文的核心方法论。建议大家课后仔细对比一下这种统一序列建模方法与早期的 BLIP 或 Flamingo 架构在数据处理上的本质区别。我们下节课再见。
