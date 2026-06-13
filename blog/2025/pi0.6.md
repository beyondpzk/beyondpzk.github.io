---
title: pi0.6
date: 2025-11-18
categories: [others]
---

# $\pi^*_{0.6}$

[paper link](https://arxiv.org/abs/2511.14759)

# 🤖 具身智能与机器人高级研讨

**文献**：*$\pi^*_{0.6}$: a VLA That Learns From Experience* (Physical Intelligence, 2025)

---

## 导言与课程概述

我们深入探讨了视觉-语言-动作（Vision-Language-Action, VLA）模型在机器人控制中的应用。大家都知道，目前主流的 VLA 模型（如 RT-2, $\pi_{0}$ 等）高度依赖于**模仿学习（Imitation Learning）**。然而，模仿学习存在一个致命的理论缺陷：**协变量偏移（Covariate Shift）与误差累积（Compounding Errors）**。当机器人在实际部署中遇到未见过的状态，或者自身产生微小执行误差时，由于这些状态不在专家的示范数据分布中，策略往往会崩溃。

> "Practice makes perfect: while people are remarkably flexible in acquiring new skills, mastery invariably requires learning from repeated attempts."

正如这篇研究所指出的，人类掌握技能需要通过反复的“练习”与“试错”。为了让 VLA 模型突破模仿学习的性能天花板，获得更高的鲁棒性和执行吞吐量（Throughput），我们必须引入**强化学习（Reinforcement Learning, RL）**。

我们将以 Physical Intelligence 团队最新提出的 **RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)** 算法及其训练出的 **$\pi^*_{0.6}$** 模型为例，进行一次深度的解剖。我们将从理论基础出发，详细拆解其独特的多模态模型架构，并分析其在诸如叠衣服、制作意式浓缩咖啡、组装纸箱等复杂长视野（Long-horizon）真实世界任务中的表现。

---

## 第一部分：理论基础与核心挑战 (Theory & Challenges)

### 1.1 为什么在真实世界中对 VLA 进行强化学习极其困难？

在纯虚拟环境中（如 Atari 游戏或 Mujoco 物理仿真），我们拥有完美的重置机制（Reset）和高频密集的奖励信号（Dense Rewards）。但在真实物理世界中对一个拥有数十亿参数的 VLA 模型进行 RL 微调，面临着三座大山：
1. **真实世界反馈的稀疏性**：倒咖啡成功与否只有在最后一步才能判定，奖励极其稀疏。
2. **异构数据的融合**：我们有完美的人类专家示范（Demonstrations）、机器人自主探索的次优数据（Autonomous rollouts），还有人类在机器人运行中途介入的纠正数据（Corrections/Interventions）。如何在一个统一的框架下利用这些好坏参半的数据？
3. **大模型架构与传统 RL 算法的冲突**：VLA 模型通常采用自回归（Autoregressive）或流匹配/扩散（Flow Matching / Diffusion）架构生成动作。传统的策略梯度方法（如 PPO）对于连续时间的流匹配模型而言，难以直接计算精确的对数似然（Log-likelihood），从而导致梯度更新极不稳定且计算成本高昂。

### 1.2 预备数学定义

在标准的马尔可夫决策过程（MDP）中，策略定义为 $\pi(a_t|o_t)$，其中 $o_t \in \mathcal{O}$ 是观测， $a_t \in \mathcal{A}$ 是动作。轨迹 $\tau = (o_0, a_0, \dots, o_T)$ 的累积回报（Return）定义为 $R(\tau) = \sum_{t=0}^T r_t$。

状态价值函数（Value Function）定义为：
$$V^\pi(o_t) = \mathbb{E}_{\tau_{t+1:T}} \left[ \sum_{t'=t}^T r_{t'} \right]$$

动作优势函数（Advantage Function）衡量了在状态 $o_t$ 下采取动作 $a_t$ 相比于平均策略期望有多大的提升：
$$A^\pi(o_t, a_t) = \mathbb{E}_{\rho^\pi(\tau)} \left[ \sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(o_{t+N}) \right] - V^\pi(o_t)$$

---

## 第二部分：RECAP 算法框架 (The RECAP Algorithm)

为了解决上述挑战，作者提出了 **RECAP** 算法。这是一种基于“优势条件化（Advantage-conditioning）”的离线/在线迭代强化学习框架。其核心思想是：**我们不直接去通过 PPO 优化大模型的梯度，而是训练一个价值网络，然后将“这个动作是好是坏（优势）”作为一种语言 prompt（提示词）输入给 VLA 模型，让模型学会根据“优势指标”来生成动作。**

RECAP 的工作流包含三个循环往复的步骤：

### 2.1 步骤一：数据收集 (Data Collection)
对于目标任务 $\ell$，首先使用在示范数据上经过监督微调（SFT）的初始策略 $\pi^0_\ell$ 进行实机部署。
在部署期间，收集两类数据：
*   **完全自主探索的轨迹**（Autonomous Rollouts）。
*   **专家介入纠正的轨迹**（Human Interventions/Gated DAgger）。当机器人即将发生灾难性失败时，人类专家接管并展示如何纠正错误。
这些数据无论成功与否，全部加入数据集 $\mathcal{D}_\ell$ 中。

### 2.2 步骤二：分布式价值函数训练 (Distributional Value Function Training)
为了评估异构数据集 $\mathcal{D}_\ell$ 中动作的好坏，我们需要一个强大的 Critic。RECAP 并没有训练一个简单的标量价值网络，而是训练了一个**多任务分布式价值函数（Multi-task Distributional Value Function）**。

设定一个稀疏的步进奖励函数，用以衡量距离任务成功的剩余步数：
$$r_t = \begin{cases} 0 & \text{if } t = T \text{ and success} \\ -C_{\text{fail}} & \text{if } t = T \text{ and failure} \\ -1 & \text{otherwise} \end{cases}$$
这里 $C_{\text{fail}}$ 是一个极大的惩罚常数。

价值网络的目标是预测连续的累积回报 $R_t(\tau)$。为了增加大模型预测的稳定性，算法将回报离散化为 $B=201$ 个区间（Bins），记为 $R^B_t$。价值网络 $p_\phi(V | o_t, \ell) \in \Delta^B$ 输出在这 201 个区间上的分类概率分布。训练目标是最小化交叉熵损失：
$$\min_\phi \mathbb{E}_{\tau \in \mathcal{D}} \left[ \sum_{o_t \in \tau} H(R^B_t(\tau), p_\phi(V | o_t, \ell)) \right]$$

通过对分布求期望，我们可以得到连续的价值估计：
$$V^{\pi_{\text{ref}}}(o_t, \ell) = \sum_{b \in [0, B]} p_\phi(V=b | o_t) v(b)$$

### 2.3 步骤三：基于优势条件化的策略提取 (Advantage-Conditioned Policy Extraction)
有了价值函数，我们就能计算数据集中每个状态-动作对的优势 $A^{\pi_{\text{ref}}}(o_t, a_t)$。传统的 AWR (Advantage Weighted Regression) 算法会抛弃或严重抑制低优势数据。但 RECAP 采用了**优势条件化（Advantage Conditioning）**。

RECAP 设定了一个任务特定的改进阈值 $\epsilon_\ell$（通常设为该任务所有估计价值的 30% 分位数）。如果一个动作的优势大于该阈值，我们将其标记为一个改进指标 $I_t = \text{True}$；否则 $I_t = \text{False}$。

在数学上，带正则化的策略优化可以等价于最大化以下目标：
$$\hat{\pi}(a|o) \propto \pi_{\text{ref}}(a|o) p(I | A^{\pi_{\text{ref}}}(o, a))^\beta$$

在实际操作中，RECAP 将这个改进指标 $I_t$ 转化为自然语言（例如 `"Advantage: positive"` 或 `"Advantage: negative"`），直接拼接进 VLA 模型的 Prompt 中。这样，模型在训练时能够“看到”哪些数据是好的，哪些是坏的。在推断（部署）时，我们强行给模型输入 `"Advantage: positive"`（即强制 $I_t = \text{True}$），模型就会自动按照产生高优势动作的分布进行动作采样！

---

## 第三部分：$\pi^*_{0.6}$ 模型架构深度解析 (Detailed Model Architecture)

这一部分是本次研讨课的重中之重。$\pi^*_{0.6}$ 是如何承载上述复杂的 VLM 推理与连续控制动作生成的？

### 3.1 基础骨架：从 Gemma 3 到 VLA
$\pi^*_{0.6}$ 模型是 $\pi_{0.5}$ 的演进版本。它的基础视觉-语言模型（VLM）骨架采用了 **Gemma 3 (4B 参数)**。模型接收的观测输入 $o_t$ 包含了多视角的相机图像特征 $X_t^1, \dots, X_t^n$ 和机器人的本体感受态 $q_t$（关节角度等）。语言输入 $\ell$ 包含高层任务提示（如 `"make me an espresso"`）以及其他元数据。

### 3.2 预测级联结构 (Cascade Prediction Structure)
为了实现高层逻辑推理与底层物理控制的解耦，模型在时间步 $t$ 的输出严格遵循以下顺序生成：
1.  **高级子任务预测 $\hat{\ell}$**：首先，VLM 自回归地输出一段文本 token，表示当前正在执行的子任务（例如 `"pick up the coffee cup"`）。这为后续动作生成提供了细粒度的语义指导。
2.  **优势指标条件注入 $I_t$**：对于 $\pi^*_{0.6}$，在这里插入文本 `"Advantage: positive"` 或 `"Advantage: negative"`。这仅影响后续动作生成的似然度。
3.  **动作预测 (Action Chunking)**：随后，模型生成一个动作块（Action chunk） $a_{t:t+H}$，通常以 50Hz 的频率包含未来一段水平线 $H$ 步内的关节位置和夹爪指令。

### 3.3 动作生成的双轨机制与 KI 架构 (Knowledge Insulation)
这是该架构最精妙的设计。传统 VLA 把动作离散化后当成文本 token 一起训练，但这会破坏预训练 VLM 原有的语言空间。$\pi^*_{0.6}$ 采用了 **知识隔离 (Knowledge Insulation, KI)** 策略，将动作生成分为“离散表示”和“连续表示”双轨进行：

*   **离散动作空间 (FAST Tokenizer)**：模型仍然会自回归地预测离散化后的动作 token $a^\ell_{t:t+H}$。这部分直接通过 VLM 的标准 next-token prediction 交叉熵损失进行训练。
*   **连续动作空间 (Flow Matching Action Expert)**：这部分是真正的物理执行层。模型外挂了一个专属的 **动作专家 (Action Expert)** 模块（大小为 860M 参数）。这个模块通过交叉注意力（Cross-attention）机制读取主干 VLM 产生的隐藏状态（Hidden states）。**关键在于，从动作专家到 VLM 主干的梯度被切断了（Stop Gradient）。** 这样，动作专家的流匹配训练完全不会干扰 Gemma 3 骨干的内部权重分布，实现了“知识隔离”。

### 3.4 流匹配损失函数 (Flow Matching Loss)
动作专家采用流匹配（Flow Matching）生成连续动作 $a_{t:t+H}$。流匹配可以视为扩散模型（Diffusion）的一种推广。连续部分的对数似然无法精确计算，但流匹配损失构成了该似然的变分下界。

整体动作的联合对数似然被拆解并下界化为：
$$\log \pi_\theta(a_{t:t+H}, a^\ell_{t:t+H} | I_t, o_t, \ell, \hat{\ell}) \geq \mathbb{E}_{\eta, \omega} \left[ \log p_\theta(a^\ell_{t:t+H} | I_t, o_t, \dots) - \alpha_\eta \| \omega - a_{t:t+H} - f_\theta(a^{\eta, \omega}_{t:t+H}, I_t, \dots) \|^2 \right]$$

其中：
*   $f_\theta$ 是拥有 860M 参数的动作专家网络。
*   $\eta \in [0,1]$ 是时间索引。
*   $\omega \sim \mathcal{N}(0, I)$ 是标准高斯噪声。
*   $a^{\eta, \omega}_{t:t+H} = \eta a_{t:t+H} + (1 - \eta) \omega$ 是加噪后的动作。
*   $\alpha_\eta$ 是损失权重。

### 3.5 价值网络架构 (Value Function Architecture)
回想第二部分，我们需要一个价值网络来提供优势 $A^{\pi_{\text{ref}}}(o_t, a_t)$。为了保证在线训练的效率（在 VLA 训练的同时计算优势），价值网络 $V^{\pi_{\text{ref}}}$ 采用了和 VLA 极其相似的架构，但规模缩减为 **670M 参数的 Gemma 3**。它的输入包含同样的图像 $X_t$、本体态 $q_t$ 和文本 $\ell$，输出头则改为预测那 201 个离散区间的概率分布。为了防止在有限任务数据上过拟合，价值网络在训练时还混入了一定比例的多模大多模态网络预训练数据（Web data）进行共训练（Co-training）。

---

## 第四部分：实验设置与结果分析 (Experiments & Empirical Analysis)

为了验证 RECAP 框架，Physical Intelligence 团队使用了极具挑战性的多阶段、长视野双臂操控任务。机器人的硬件配置为两个 6 自由度机械臂（带平行夹爪），配备 3 个摄像头（1 个基座，2 个腕部）。

### 4.1 评估任务 (Evaluation Tasks)
1.  **衣物折叠 (Laundry)**：包括基础的 T 恤/短裤折叠，以及极具挑战性的 **多样化衣物折叠 (Diverse items)**（11 种衣物，如毛巾、衬衫、毛衣、牛仔裤等）。对于多样化衬衫折叠，限时高达 500 秒。
2.  **意式浓缩咖啡 (Cafe - Double Shot Espresso)**：极其复杂的长序列任务，包括拿取手柄、磨豆、压粉、安装到咖啡机、放杯子、提取浓缩咖啡。必须在 200 秒内完成，涉及液体和精细的力控。
3.  **纸箱组装 (Box Assembly)**：真正的工厂级部署任务。需要从压扁的纸板开始折叠成箱子、贴标签、放置在板条箱中，限时高达 600 秒。

### 4.2 核心对比基线 (Baselines)
*   **Pre-trained $\pi_{0.5}$ / $\pi_{0.6}$**：纯粹基于模仿学习的基础模型。
*   **Offline RL + SFT**：仅使用专家示范进行 RECAP 预训练和 SFT（无自主收集数据）。
*   **AWR (Advantage Weighted Regression)**：使用价值网络，但在策略更新时使用 AWR 取代优势条件化。
*   **PPO**：采用 DPPO/FPO 变体，将单步流匹配对数似然代入 PPO 截断目标中。

### 4.3 实验结果深度探讨
我们主要关注两个核心指标：**成功率 (Success Rate)** 和 **吞吐量 (Throughput, 即每小时成功完成的任务数)**。吞吐量是一个工业界极度看重的综合指标，因为它同时惩罚了任务失败和动作拖沓。

1.  **RL 大幅提升上限与吞吐量**：在最具挑战性的“多样化衣物折叠”和“浓缩咖啡”任务中，加入了实机部署数据和 RECAP 训练的最终 $\pi^*_{0.6}$ 模型，其吞吐量相较于只做 SFT 的版本 **翻了一倍以上 (More than doubles)**！且失败率降低了约 2 倍。
2.  **迭代提升 (Iterative Improvement)**：图表表明，随着迭代次数增加（收集数据 -> 训练价值网络 -> 训练策略 -> 再次收集数据），模型性能稳步上升。在长视野的纸箱组装任务中，第二轮迭代直接让吞吐量提升了 2 倍。
3.  **算法层面对比**：对比 PPO 和 AWR，RECAP 中的优势条件化（Advantage Conditioning）碾压了两者。PPO 在这种离线/近离线的设置下，针对流匹配模型极难稳定训练（作者甚至必须使用极小的信任域约束 $\eta=0.01$），导致表现平庸。AWR 虽然能保证一定的成功率，但它实际上抛弃了大量次优数据，导致学到的策略速度极慢，吞吐量大幅下降。
4.  **消除特定失效模式 (Failure Mode Removal)**：研究人员专门设计了一个极其严苛的 T恤折叠测试（要求领口必须居中朝上，且以对抗性初始姿态摆放）。仅经过两次自主运行的 RL 迭代（没有任何新专家介入），$\pi^*_{0.6}$ 就自行学会了纠正“将领口朝下放置”的错误，成功率飙升至 97%。这证明了强化学习真正赋予了模型从自身错误中学习（Learning from experience）的能力。

---

## 第五部分：无分类器引导的部署优化 (Classifier-Free Guidance at Test Time)

在研讨课的最后，我想补充一个极为有趣的技术细节。

在方程 $\hat{\pi}(a|o) \propto \pi_{\text{ref}}(a|o) p(I | A^{\pi_{\text{ref}}}(o, a))^\beta$ 中，参数 $\beta$ 控制了我们有多“渴望”追求高优势动作。在训练阶段，算法会随机丢弃 $I_t$（类似于 dropout），从而同时训练出条件模型 $\pi_\theta(a_{t:t+H} | I_t, o_t, \ell)$ 和无条件模型 $\pi_\theta(a_{t:t+H} | o_t, \ell)$。

在部署（测试）时，如果我们希望 $\beta > 1$（即极其贪婪地追求完美动作），我们不需要重新训练模型！通过无分类器引导（Classifier-Free Guidance, CFG）技术，流匹配推断中的向量场更新可以被修改为：
$$\nabla_a \log \pi_\theta(\cdot | o_t, \ell) + \beta \Big( \nabla_a \log \pi_\theta(\cdot | I_t, o_t, \ell) - \nabla_a \log \pi_\theta(\cdot | o_t, \ell) \Big)$$
这使得我们可以在推理时动态调节机器人策略的“激进程度”，这也是扩散/流匹配模型引入 RL 后带来的一项独特福利。

---

## 第六部分：总结与未来展望 (Conclusion & Future Directions)

今天我们深度拆解了 $\pi^*_{0.6}$ 和 RECAP 算法。我们可以得出以下核心结论：

1.  **VLA 模型的下一个范式是 RL**：仅仅依靠人类示范已经无法应对真实世界的复杂多样性。能够利用真实世界成功/失败经验进行自适应学习的模型，才能真正走向商用。
2.  **优势条件化是目前大规模连续 VLA 模型的最佳解法**：它避开了 PPO 复杂的对数似然与截断计算，用类似监督学习（SFT）的稳定方式，巧妙地通过自然语言 Prompt 实现了策略提取，完美适配了大语言模型的预训练框架。
3.  **知识隔离 (KI)**：在多模态联合训练中，通过切断底层动作生成器到高级认知大模型之间的梯度，既保护了模型的常识推理能力，又允许底层控制逻辑的高效适应。

**未来的开放性问题（留给同学们的思考题）：**
*   **自主探索能力**：RECAP 的探索目前依然偏向贪心（Greedy）策略，主要依赖策略自身的随机性与人类介入。如何为 VLA 引入更为结构化、内在动机驱动（Intrinsic Motivation）的探索算法？
*   **奖励与重置自动化**：目前系统的成功判定和重置依然需要大量人力。能否利用多模态 VLM 自身的视觉评判能力（VLM-as-a-Judge）实现全自动化的在线 RL 循环？

今天的研讨课就到这里。讲义内容极度密集，请大家课后重点消化流匹配动作专家与优势条件化的结合部分。下课！

## $\pi$ 的演进历史

### $\pi$ 系列模型的演进史：从模仿学习到强化学习的跨越

Physical Intelligence 团队推出的 $\pi$ 系列模型，是目前具身智能（Embodied AI）领域最具代表性的视觉-语言-动作（VLA）大模型之一。它的演进完美展示了机器人控制从“实验室环境的模仿”走向“真实世界的泛化”，再到“基于物理反馈的自主进化”的完整技术脉络。

以下是 $\pi$ 系列从 $\pi_0$ 到 $\pi^*_{0.6}$ 的核心技术演进史：

#### 1. $\pi_0$：奠定连续流匹配 VLA 架构
*   **核心痛点**：早期的 VLA 模型（如 RT-2）通常强制将机器人的连续关节运动“离散化”为文本 token。这虽然方便了语言模型训练，但极大地损失了物理控制的平滑性和精度。
*   **技术突破**：在 [$\pi_0$](https://alphaxiv.org/abs/2410.24164) 中，研究团队首次在预训练的视觉语言模型（VLM）之上，外挂了一个**流匹配（Flow Matching）架构**。
*   **演进意义**：流匹配类似于扩散模型，它允许模型在保留互联网级语义常识的同时，直接生成高频率、平滑的连续动作块（Action Chunks）。这奠定了 $\pi$ 系列双轨制（离散语义 + 连续动作）的基石。

#### 2. $\pi_{0.5}$：迈向真实世界的开放泛化 (Open-World Generalization)
*   **核心痛点**：$\pi_0$ 虽然在实验室任务上表现优异，但一旦进入未见过的真实家庭环境，就会因为数据分布的变化（协变量偏移）而频繁失败。
*   **技术突破**：[$\pi_{0.5}$](https://alphaxiv.org/abs/2504.16054) 大幅提升了模型的泛化能力。它引入了异构数据共训练（Co-training），将不同形态的机器人数据、互联网图像以及物体检测数据混合训练。
*   **级联预测机制**：$\pi_{0.5}$ 引入了非常重要的层次化推理机制——在生成底层物理动作之前，模型必须先用自然语言预测当前的**高级子任务**（例如先输出文本 `"pick up the coffee cup"`）。这种语义级的自我提示大大增强了长序列任务的稳定性。

#### 3. $\pi_{0.6}$：引入知识隔离 (Knowledge Insulation) 与 Gemma 3 基座
*   **核心痛点**：当我们在 VLM 上同时训练文本语义预测和复杂的物理动作时，底层动作的梯度反向传播很容易“污染”并破坏 VLM 原本强大的语言推理与世界常识（即灾难性遗忘）。
*   **技术突破**：$\pi_{0.6}$ 将主干网络升级为了更强大的 **Gemma 3 (4B)** 模型，并将流匹配动作专家（Action Expert）的参数量扩展到了 860M。
*   **知识隔离 (KI)**：这是 $\pi_{0.6}$ 最关键的架构改动。动作专家模块通过交叉注意力（Cross-attention）读取 VLM 的隐藏状态，但**彻底切断了回传给 VLM 的梯度（Stop Gradient）**。这意味着动作控制的训练完全不会改变 VLM 的内部权重，实现了“高级认知”与“底层小脑”的完美解耦。

#### 4. $\pi^*_{0.6}$：通过强化学习实现自我纠错与提效
*   **核心痛点**：前面的 $\pi_0$ 到 $\pi_{0.6}$ 本质上依然属于**模仿学习**。模型只能学到“专家是怎么做的”，但不知道动作背后的好坏标准，一旦遇到意外很容易崩溃；且动作执行速度往往受限于人类示教的速度。
*   **技术突破**：$\pi^*_{0.6}$ 跨越了模仿学习的瓶颈，引入了 **RECAP（优势条件化策略）** 强化学习算法。
*   **工作原理**：
    *   团队单独训练了一个多任务价值函数（Value Function），用来预测机器人在当前状态下距离任务成功的剩余步数。
    *   利用这个价值网络，计算出每一次物理尝试中具体动作的“优势（Advantage）”。
    *   将优势评估转化为自然语言提示（如 `"Advantage: positive"` 或 `"Advantage: negative"`），直接作为特征注入到 $\pi_{0.6}$ 的上下文中进行微调。
*   **最终形态**：在实机部署时，只要我们强制输入 `"Advantage: positive"`，$\pi^*_{0.6}$ 就会自动按照能够最大化成功率的分布去生成动作。它不仅能完成煮咖啡、叠复杂衣物等长视野任务，而且执行的吞吐量（速度与成功率的综合指标）相较于单纯的模仿学习翻了一倍以上。

---

**总结：**
$\pi$ 系列的演进有着非常清晰的技术主线：**$\pi_0$ 解决动作生成域的映射问题 $\rightarrow$ $\pi_{0.5}$ 解决跨场景的泛化问题 $\rightarrow$ $\pi_{0.6}$ 解决大模型多模态微调的知识遗忘问题 $\rightarrow$ $\pi^*_{0.6}$ 最终引入强化学习，赋予了机器人从自身试错经验中持续进化的能力。**
