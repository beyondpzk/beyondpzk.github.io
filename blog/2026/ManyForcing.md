---
title: ManyForcing
date: 2026-01-01
---

# 各种Forcing 技术

> **摘要**：本文系统介绍序列生成模型训练中的"Forcing"技术。以Teacher Forcing为核心，阐释其加速收敛的机制与引发Exposure Bias的根本缺陷；以此为线索，详细分析Free Running、Scheduled Sampling、Professor Forcing、TeaForN (N-gram TF)、Diffusion Forcing、Self Forcing等多种变体与改进策略的设计哲学、数学原理及适用场景。

---

## 1. 核心矛盾：训练与推理的分布差异

在自回归（Autoregressive）序列生成中，模型需要根据历史信息预测下一个输出。这带来一个根本性挑战：训练时应当用什么作为模型下一步的输入？

| 阶段 | 输入来源 | 分布特性 |
|------|---------|---------|
| **训练** | 可从真实数据（Ground Truth）获取 | 完美、无噪声 |
| **推理** | 只能依赖模型自身先前生成的输出 | 可能包含错误、偏离训练分布 |

这种**训练-推理的输入分布差异**，是所有"Forcing"技术试图解决的核心问题。

---

## 2. 基础技术

### 2.1 Teacher Forcing

**原理**：在训练过程的每一步，无论模型上一时刻的预测是什么，都强制将**真实的目标序列（Ground Truth）**作为下一时刻的输入。

- **数学表示**：

  给定输入序列 $$x = (x_1, \ldots, x_T)$$ 和目标序列 $$y^* = (y_1^*, \ldots, y_T^*)$$，条件概率的极大似然估计（MLE）目标为：

  $$
  \mathcal{L}_{\text{TF}} = - \sum_{t=1}^{T} \log P(y_t^* \mid y_{1:t-1}^*, x)
  $$

  其中，时刻 $$t$$ 的历史 $$y_{1:t-1}^*$$ **完全来自真实数据**，与模型预测无关。

- **训练并行化**：由于每一步的输入均不依赖于模型输出，Teacher Forcing天然支持**跨时间步并行计算**。在Transformer解码器中，通过因果掩码（Causal Masking），可一次性计算整个序列所有位置的损失，极大加速训练。

- **实现方式**：在代码层面，只需将目标序列**右移一位**作为输入。例如：

  ```python
  X = data[:, :-1, :]  # 输入序列 (去除最后一个时刻)
  y = data[:, 1:, :]   # 目标序列 (右移一位)
  ```

**优缺点**：

- ✅ **优点**：
  - 训练稳定、收敛迅速，因为模型始终看到正确的历史信息。
  - 梯度传播路径清晰，优化难度低。
  - 天然支持并行训练，适用于Transformer等现代架构。
- ❌ **缺点**：
  - **暴露偏差（Exposure Bias）**：模型在训练时从未见过自身预测的错误作为输入，导致推理时一旦出现预测偏差，误差会**沿时间步累积放大**，输出质量急剧下降。
  - **缺乏长程序列生成能力**：模型仅学习单步最优预测，而非全局序列最优。
  - **过度矫正（Overcorrection）**：Teacher Forcing强制将真实词作为输入，可能干扰模型已正确生成的语义路径，导致生成不通顺的序列。

### 2.2 Free Running（又名 Student Forcing）

**原理**：训练时完全模拟推理环境，将模型**上一时刻的实际预测输出**作为下一时刻的输入。

- **数学表示**：

  令模型在时刻 $$t-1$$ 的预测输出为 $$\hat{y}_{t-1}$$，则训练目标为：
  $$
  \mathcal{L}_{\text{FR}} = - \sum_{t=1}^{T} \log P(y_t^* \mid \hat{y}_{1:t-1}, x)
  $$
  其中 $$\hat{y}_{1:t-1}$$ 是模型**自己生成的序列**，而非真实数据。

- **训练动态**：
  - 模型必须在自身预测的**不完美上下文**中学习，迫使它具备**误差恢复能力**。
  - 这是一个端到端的序列级优化问题，与推理时的行为完全一致。

- **训练困难**：
  - **梯度传播困难**：误差沿时间步反向传播（BPTT），路径极长，易发生梯度消失/爆炸。
  - **离散采样的不可微性**：对于文本等离散输出，从分布中采样（Argmax或Multinomial）操作**不可微**，无法直接端到端训练，通常需要借助强化学习（如策略梯度）来优化。
  - **收敛缓慢**：模型初期预测质量差，输入噪声大，训练初期损失极高，不易收敛。

- **本质矛盾**：Free Running（也称为Student Forcing）完美解决了训练-推理分布不一致的问题，但**梯度无法直接回传**；Teacher Forcing解决了梯度传播问题，却引入了分布偏差。后续所有改进技术，本质上都是在这对矛盾中寻找**新的平衡点**。

---

## 3. 改进策略：从“完全依赖教师”到“逐步自立”

### 3.1 Scheduled Sampling（计划采样）

**原理**：在训练过程中，以一定概率 $$p$$ 选择使用真实数据，以概率 $$1-p$$ 选择使用模型自身的预测作为输入。这个概率 $$p$$ **随训练进行逐渐衰减**，使模型从完全依赖教师，平滑过渡到完全自立。

- **数学表示**：

  对于时刻 $$t$$ 的输入 $$\tilde{y}_{t-1}$$，采用伯努利采样：
  $$
  \tilde{y}_{t-1} =
  \begin{cases}
  y_{t-1}^*, & \text{以概率 } p \text{ （Teacher Forcing）}\\
  \hat{y}_{t-1}, & \text{以概率 } 1-p \text{ （Free Running）}
  \end{cases}
  $$

  概率 $$p$$ 的衰减策略包括**线性衰减**、**指数衰减**或**逆Sigmoid衰减**等。

- **课程学习视角**：Scheduled Sampling可视为一种**课程学习（Curriculum Learning）**策略——先学习简单任务（正确历史输入），再逐步增加难度（不完美历史输入），从而平稳过渡到真实推理环境。

- **Transformer中的实现挑战**：
  - 由于Transformer的解码器在训练时通过**因果掩码一次性并行计算**所有位置，这种并行性与Scheduled Sampling要求的**顺序依赖**（每一步的输入依赖于上一步的预测）天然冲突。
  - 一种解决方案是**两遍解码策略（Two-Pass Decoding）**：第一遍使用Teacher Forcing并行生成所有隐藏状态；第二遍按顺序逐位置决定是否替换输入，虽能缓解问题，但会牺牲部分并行效率。

- **效果与局限**：
  - 能一定程度缓解Exposure Bias，生成质量优于纯Teacher Forcing。
  - 但近年研究表明，Scheduled Sampling可能引入**有偏的优化目标**，在某些任务上优化不稳定，甚至不如纯Teacher Forcing。

### 3.2 Professor Forcing

**原理**：借鉴生成对抗网络（GAN）的思想，不直接修改输入策略，而是通过**对抗训练**使模型在Free Running模式下的**隐藏状态分布**，与Teacher Forcing模式下的隐藏状态分布**尽可能一致**。

- **架构设计**：
  - **生成器（Generator）**：即序列生成模型本身，在两种模式下分别产生隐藏状态序列。
  - **判别器（Discriminator）**：一个辅助网络，负责区分输入隐藏状态序列是来自Teacher Forcing模式（真）还是Free Running模式（假）。
  - **对抗目标**：生成器努力使Free Running产生的隐藏状态分布逼近Teacher Forcing的分布，从而骗过判别器。

- **数学表示**（对抗训练形式）：

  令 $$h_t^{\text{TF}}$$ 为Teacher Forcing模式下的隐藏状态，$$h_t^{\text{FR}}$$ 为Free Running模式下的隐藏状态，判别器 $$D$$ 的输出为真伪概率。生成器的对抗损失为：
  $$
  \mathcal{L}_{\text{adv}} = - \mathbb{E} \left[ \log D(h^{\text{FR}}) \right]
  $$
  总训练目标为序列生成损失与对抗损失的加权和：
  $$
  \mathcal{L} = \mathcal{L}_{\text{TF}} + \lambda \cdot \mathcal{L}_{\text{adv}}
  $$

- **意义**：Professor Forcing不要求模型输出与真实值完全一致，而是从**表征层面**对齐训练与推理的动力学行为，是一种更深层次的分布匹配。

### 3.3 TeaForN (Teacher-Forcing with N-grams)

**原理**：Google提出的TeaForN不直接改变输入策略，而是**扩展模型的前瞻视野**——在每一步训练时，不仅预测下一个token，还预测未来的N个token。

- **架构设计**：
  - 使用**堆叠的N个解码器**，沿辅助时间轴依次解码未来的N个token。
  - 模型参数更新基于N步预测结果，而非单步，使模型具备更长的规划视野。

- **核心价值**：
  - 模型被强制学习当前决策对**未来N步**的影响，从而内隐地习得误差恢复策略——即使当前步骤使用了不完美的历史输入，仍能规划出合理的未来路径。
  - 相较于Scheduled Sampling，TeaForN**保持了Teacher Forcing的训练稳定性和并行性**，同时增强了对Exposure Bias的鲁棒性。

---

## 4. 前沿新范式

### 4.1 Diffusion Forcing（扩散强制）

**原理**：由MIT CSAIL提出，将扩散模型的**变分噪声**与Teacher Forcing的**因果序列生成**融合。核心创新是：**每个token拥有独立的噪声水平**，训练时模型学习对不同噪声程度的token进行“去噪”。

- **独立噪声水平**：

  令序列中第 $$t$$ 个token的噪声水平为 $$k_t$$（可为0到K之间的整数），训练时每个token被独立地施加对应强度的噪声。模型学习预测原始干净token：
  $$
  \mathcal{L}_{\text{DF}} = \sum_{t=1}^{T} \left\| \text{Denoise}(x_t^{\text{noisy}}, k_t) - x_t^{\text{clean}} \right\|^2
  $$

  其中 $$x_t^{\text{noisy}} = \alpha_{k_t} x_t^{\text{clean}} + \sigma_{k_t} \epsilon$$，$$\alpha, \sigma$$ 为噪声调度参数。

- **因果掩码与变长序列生成**：
  - 训练时使用**因果掩码**，确保第 $$t$$ 个token的预测仅依赖当前及之前的token。
  - 推理时，可通过调整不同位置token的噪声水平，实现**灵活的长序列外推**——对过去token设置较低噪声（保持稳定），对未来token设置较高噪声（允许规划与探索）。

- **融合的价值**：
  - 继承了Teacher Forcing的因果性和扩散模型的全序列规划能力。
  - 在机器人操控、长程决策任务中表现突出，能稳定生成远超训练长度的连续序列。

### 4.2 Self Forcing（自我强制）

**原理**：由Adobe Research提出，专为**自回归视频扩散模型**设计。核心是：训练时对完整视频执行**自回归展开（Autoregressive Rollout）**，使每一帧都基于**模型先前已生成的帧**进行去噪，而非真实帧。

- **训练流程**：
  1. **自回归展开**：在训练中完整执行自回归推理过程，每生成一帧后，将其存入KV Cache供后续帧生成时使用。
  2. **全局损失监督**：由于训练中可获取完整生成视频，可以在**整个视频级别**施加全局分布匹配损失（如SiD、DMD），直接优化序列质量。
  3. **效率优化**：采用**少步扩散模型**与**随机梯度截断策略**，平衡序列级训练的计算成本。

- **数学表示**：

  令 $$\hat{v}_1, \ldots, \hat{v}_T$$ 为模型通过自回归展开生成的视频帧序列。Self Forcing的训练损失可分解为：
  $$
  \mathcal{L}_{\text{SF}} = \underbrace{\sum_{t=1}^{T} \mathcal{L}_{\text{denoise}}(\hat{v}_t)}_{\text{逐帧去噪损失}} + \underbrace{\mathcal{L}_{\text{global}}(\hat{v}_{1:T})}_{\text{全局视频级损失}}
  $$

  其中全局损失直接衡量生成视频整体与真实视频的分布差异。

- **核心突破**：
  - 训练与推理完全一致——都是基于模型自身生成内容进行自回归扩展，**从根本上消除Exposure Bias**。
  - 单GPU即可实现**亚秒级延迟的实时视频流生成**，质量超越许多非因果扩散模型。
  - 引入的滚动KV Cache机制，支持高效的任意长度视频外推。

---

## 5. 总结与展望

### 5.1 方法对比一览

| 方法 | 训练输入来源 | 核心思想 | 并行性 | 适用场景 | 提出时间/机构 |
|------|------------|---------|--------|---------|------------|
| **Teacher Forcing** | 100% 真实数据 | 用正确答案引导每一步 | ✅ 高 | 标准序列生成训练 | 1989, Williams & Zipser |
| **Free Running** | 100% 模型预测 | 完全模拟推理环境 | ❌ 低 | 端到端序列训练（需RL辅助） | 早期RNN训练 |
| **Scheduled Sampling** | 混合，概率衰减 | 从教师强制平滑过渡到自立 | ⚠️ 中等 | 序列到序列生成 | 2015, Bengio et al. |
| **Professor Forcing** | 对抗对齐隐藏状态 | GAN式分布匹配 | ⚠️ 中等 | 文本/语音生成 | 2016, Lamb et al. |
| **TeaForN** | 100% 真实数据 | N步前瞻预测 | ✅ 高 | 机器翻译、文本生成 | 2020, Google |
| **Diffusion Forcing** | 独立噪声token | 扩散模型+因果掩码 | ✅ 高 | 视频生成、机器人规划 | 2024, MIT CSAIL |
| **Self Forcing** | 自回归展开+缓存 | 视频级全局监督 | ⚠️ 中等 | 自回归视频扩散 | 2025, Adobe |

### 5.2 未来趋势

"Forcing"技术的演进清晰反映了三个趋势：

1. **从“单点最优”到“序列最优”**：早期方法优化单步预测，Diffusion Forcing和Self Forcing等新范式转向**端到端的序列级优化**，直接评估整个生成序列的质量。

2. **从“文本专属”到“跨模态泛化”**：Teacher Forcing起源于文本生成，而Self Forcing（视频）、Diffusion Forcing（机器人规划）表明，"Forcing"的设计思想已成功迁移到连续模态和具身智能领域。

3. **从“训练-推理割裂”到“行为统一”**：Self Forcing通过训练中的自回归展开，实现了训练与推理行为的**完全统一**，Exposure Bias问题正在被根本性地解决，而非仅仅缓解。

---

## 参考文献

[1] Williams, R. J., & Zipser, D. (1989). A learning algorithm for continually running fully recurrent neural networks. *Neural Computation*.

[2] Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *NIPS*.

[3] Lamb, A. M., Goyal, A. G. A. P., Zhang, Y., Zhang, S., Courville, A. C., & Bengio, Y. (2016). Professor forcing: A new algorithm for training recurrent networks. *NIPS*.

[4] Chen, B., et al. (2024). Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. *arXiv:2407.01392*.

[5] Huang, X., Li, Z., He, G., Zhou, M., & Shechtman, E. (2025). Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. *NeurIPS 2025 Spotlight*. arXiv:2506.08009.

[6] Mihaylova, T., & Martins, A. F. T. (2019). Scheduled Sampling for Transformers. *ACL*.

[7] Xu, Y., et al. (2019). Rethinking Exposure Bias in Adversarial Language Modeling. *arXiv:1910.11235*.

[8] Google Research. (2020). TeaForN: Teacher-Forcing with N-grams. *arXiv*.

[9] Graves, A. (2018). NIPS 2018 Keynote: "The Limitations of Teacher Forcing".
