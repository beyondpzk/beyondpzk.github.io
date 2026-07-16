---
title: LiteVLA
date: 2026-07-05
categories: [VLA]
---

# LiteVLA 系列：从 CPU 极限可行到 Orin 实时闭环的端侧 VLA 演进

> **核心论文**：
> - *Lite VLA: Efficient Vision-Language-Action Control on CPU-Bound Edge Robots* ([arXiv:2511.05642](https://arxiv.org/abs/2511.05642))
> - *LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics* ([arXiv:2603.03380](https://arxiv.org/abs/2603.03380))
> - *LiteVLA-H: Dual-Rate Vision-Language-Action Inference for Onboard Aerial Guidance and Semantic Perception* ([arXiv:2605.00884](https://arxiv.org/abs/2605.00884))

---

## 执行摘要

LiteVLA 系列是 Clark Atlanta University 与 Siemens 合作推进的**端侧 Vision-Language-Action（VLA）**研究工作，目标是在资源受限的边缘硬件上实现语言条件的视觉-运动闭环控制。该系列沿着一条清晰的工程递进路线展开：

| 工作 | 硬件平台 | 核心指标 | 关键突破 |
|------|----------|----------|----------|
| **Lite VLA** (2025.11) | Raspberry Pi 4 (CPU) | ~11 s / query (0.09 Hz) | 首次在纯 CPU 教育机器人上跑通 GGUF 量化 VLA |
| **LiteVLA-Edge** (2026.03) | Jetson AGX Orin | **150.5 ms (6.6 Hz)** | 4-bit GGUF + GPU offload，进入实时闭环控制 regime |
| **LiteVLA-H** (2026.05) | Jetson AGX Orin | 动作 50.65 ms (19.74 Hz)，语义 149–164 ms (6–7 Hz) | 双速率调度，区分"反应式动作"与"解释性语义" |

三篇论文共享同一个技术信念：**用尽可能小的多模态骨干（SmolVLM-256M）+ 激进的量化 + 与 ROS 2 紧密集成的部署流水线**，把 VLA 从云端/桌面 GPU 搬到真实机器人边缘设备上。

---

## 1. 为什么需要 LiteVLA？

### 1.1 大模型 VLA 的部署困境

RT-2、OpenVLA、Qwen-RobotNav 等模型展示了 VLA 在语义理解和任务泛化上的强大能力，但它们通常需要：

- 7B+ 参数骨干；
- 桌面级 GPU（RTX 4090）或云端推理；
- 数百瓦功耗；
- 网络连接（非离线）。

对于灾难响应、国防、地下设施、GPS 拒止环境或电池供电的移动机器人而言，这种部署形态不可接受。LiteVLA 系列追问一个更工程化的问题：**一个 256M 参数的多模态模型，能否在边缘设备上产生足够快、足够稳定的动作？**

### 1.2 设计哲学：骨干固定，优化部署

LiteVLA 三阶段工作的共同选择：

- **骨干**：SmolVLM-256M（或同族紧凑多模态模型），参数量固定；
- **微调**：LoRA / 全参微调，将视觉-语言映射转换为图像-动作映射；
- **量化**：4-bit NF4 / GGUF Q4K_M，压缩模型体积与内存带宽；
- **运行时**：llama.cpp，利用其高度优化的 CPU/GPU 量化内核；
- **集成**：ROS 2 节点，直接发布 `geometry_msgs/Twist` 等标准控制消息。

> 与 SmolVLA、EdgeVLA、EfficientVLA 等工作的区别：LiteVLA 不追求新架构或新训练目标，而是**把"如何在真实边缘硬件上跑起来"作为首要研究问题**。

---

## 2. Lite VLA：CPU 极限可行（2025.11）

### 2.1 系统定位

Lite VLA 是系列的开篇之作，验证了一个极端命题：**能否在 Raspberry Pi 4 这样的纯 CPU 平台上实现 VLA 推理？**

硬件配置：
- Raspberry Pi 4，4GB RAM
- 四核 ARM Cortex-A72 @ 1.5 GHz
- 无 GPU/NPU 加速
- 载体：TurtleBot 4

### 2.2 数据与训练

- 通过遥操作收集 **15,083 帧图像-动作对**；
- 动作空间：线性速度 + 角速度，映射为语义动作字符串，如 `forward 0.2 3.0s`、`turn left 0.1 2.5s`；
- 数据划分：85% 训练 / 15% 验证；
- 图像预处理：resize 到 224×224，归一化，随机水平翻转；
- LoRA 微调：rank r=8，scaling α=8，dropout=0.1；
- 骨干：SmolVLM-256M。

### 2.3 关键工程决策：混合精度量化

Lite VLA 尝试了三种量化配置：

| 配置 | 推理延迟 | 输出稳定性 | 说明 |
|------|----------|------------|------|
| FP32 全模型 | ~18 分钟/次 | 稳定 | 完全不可部署 |
| NF4 backbone + FP32 projection head | **~2 分钟/次** | **稳定** | 9× 加速，最佳平衡点 |
| 全 NF4 | ~1.5 分钟/次 | 不稳定/幻觉 | projection head 量化导致动作漂移 |

**核心发现**： projection head（负责把语言模型输出映射为动作）对数值精度极其敏感，必须保留 FP32；而 backbone 可以大胆压缩到 4-bit。这一洞察被后续 LiteVLA-Edge 继承。

### 2.4 推理延迟与 Action Chunking

- 纯 CPU 推理延迟：**~11.1 s / query（0.09 Hz）**；
- 采用 **Action Chunking**：VLA 以 0.09 Hz 输出一段动作序列，ROS 2 低层控制器以更高频率执行该序列，从而实现"思考慢、执行不慢"的异步控制。

这个延迟显然无法支撑真正的闭环反应控制，但它证明了：**即使没有 GPU，一个量化后的 256M VLA 也能在机器人上本地运行**。这为后续 LiteVLA-Edge 的 GPU 加速版本奠定了方法基础。

### 2.5 EDGE-VLA-ROADMAP

论文还提出了一个六阶段演进路线图：

1. **地面单 agent**：Raspberry Pi + ROS 2 + NF4 量化；
2. **空中无人机**：引入 IMU-高度计融合、三维机动；
3. **多 agent 协同**：TaskGraph + ROS 2 DDS 共享状态；
4. **多模态 grounding**：RGB + 深度 + 热成像融合；
5. **持续/强化学习**：端侧经验回放、在线 LoRA、安全 PPO；
6. **协作边缘推理**：P2P 知识共享、联邦安全协议。

后续 LiteVLA-Edge 和 LiteVLA-H 可以看作是该路线图中第 1→2 阶段的推进。

---

## 3. LiteVLA-Edge：进入实时闭环（2026.03）

### 3.1 从"能跑"到"够快"

LiteVLA-Edge 把平台从 Raspberry Pi 4 升级到 **NVIDIA Jetson AGX Orin（64GB）**，通过 GPU offload 把端到端延迟从 11 秒降到 **150.5 ms（6.6 Hz）**。这不仅是量变，更是质变：

> 从"开环预测-执行"切换到**实时闭环视觉-运动控制**。

在 150 ms 量级，机器人可以在运动过程中根据最新视觉反馈修正轨迹，而不是每走一步都停下来"思考"。

### 3.2 训练与量化流程

| 步骤 | 配置 | 目的 |
|------|------|------|
| 监督微调 | FP32，LoRA r=8, α=8 | 保持动作映射的高精度 |
| 量化格式 | **GGUF Q4K_M** | 4-bit 权重，减小内存占用 |
| 运行时 | **llama.cpp CUDA backend** | GPU 加速量化推理 |
| 上下文窗口 | n_ctx = 512 | 限制 KV cache |
| 最大输出 token | 12 | 减少解码开销 |
| GPU offload | 全部 42 层 transformer | 最小化 CPU-GPU 数据传输 |

### 3.3 系统架构

```
RGB 相机 → 视觉编码器 → SmolVLM-256M 多模态 Transformer → 动作 token
                                                        ↓
                                                 ROS 2 bridge
                                                        ↓
                                              geometry_msgs/Twist
                                                        ↓
                                                   控制器/执行器
```

关键设计：**感知-推理-执行解耦**。VLA 只负责高层速度指令，低层 100 Hz 控制心跳由 ROS 2 独立维护。这样既保留了端到端语义理解能力，又允许安全覆盖和确定性调试。

### 3.4 延迟与稳定性

- **平均端到端延迟**：150.5 ms
- **标准差**：0.125 ms（极低抖动）
- **推理频率**：~6.6 Hz
- **运行模式**：完全离线，不依赖云端

低抖动对于控制非常重要：如果 VLA 推理时间大幅波动，ROS 2 的低层控制器将难以维持稳定的控制频率。LiteVLA-Edge 的 σ=0.125 ms 说明 llama.cpp 的量化 CUDA 内核在 Jetson 上具有很高的时间确定性。

### 3.5 与同期工作的定位

| 模型 | 参数量 | 硬件 | 频率 | 特点 |
|------|--------|------|------|------|
| OpenVLA | 7B | RTX 4090 | 低 | 通用性强，端侧不可行 |
| EdgeVLA | — | Jetson | 10–15 Hz | 分层架构，牺牲部分推理深度 |
| EfficientVLA | — | 高端 edge GPU | — | 知识蒸馏 + action chunking，依赖 TensorRT |
| **LiteVLA-Edge** | **256M** | **Jetson AGX Orin** | **6.6 Hz** | **GGUF 量化，跨平台灵活，完全本地** |

LiteVLA-Edge 的论点不是"比别人快"，而是"在 40W 级边缘模块上，用通用量化格式和开源运行时，实现可用的闭环控制"。

---

## 4. LiteVLA-H：双速率调度与空中场景（2026.05）

### 4.1 核心问题：同一模型如何同时"反应快"和"说得清"？

LiteVLA-H 把场景扩展到**无人机空中导航**。无人机既需要：

- **快**：对视觉变化快速反应（避免碰撞、跟踪目标）；
- **慢**：进行场景理解、危险描述、操作员 narration。

如果每次推理都要生成完整句子，系统将被限制在语义模式的 6–7 Hz；如果只做动作，又失去了可解释性和高层感知。LiteVLA-H 的解决方案是**双速率调度（dual-rate scheduling）**。

### 4.2 关键观察：预填充主导（Pre-fill Dominance）

LiteVLA-H 对延迟做了精细分解：

$$
L(n) = P(I_t, x_t, m_t) + \sum_{i=1}^{n} D_i
$$

- $P$：多模态预填充成本（图像编码 + prompt 融合）；
- $D_i$：第 $i$ 个 token 的解码成本。

实测发现：**$P \gg D_i$**。在 Jetson AGX Orin 上：

- 预填充 $P \approx 47.8$ ms；
- 每个额外 token 的边际成本仅 1–2 ms；
- 动作分支总延迟 50.65 ms，其中 **94.4% 来自预填充**。

这意味着："缩短输出长度"对降低首 token 延迟帮助有限，真正决定反应速度的是**图像-文本融合阶段**。

### 4.3 双速率调度器

基于上述观察，LiteVLA-H 设计了两个查询周期：

| 分支 | 周期 | 延迟 | 频率 | 用途 |
|------|------|------|------|------|
| **动作分支** | $\Delta_a$ | **50.65 ms** | **19.74 Hz** | 外环制导（速度/航向/航点） |
| **语义分支** | $\Delta_s = K \cdot \Delta_a$ | 149.90–164.57 ms | 6.08–6.67 Hz | 场景描述、危险报告、操作员 narration |

取 $K=3$ 时，三个动作周期约 151.95 ms，刚好与单句语义延迟 149.90 ms 对齐。因此语义刷新可以自然地安排在每第 3 个动作周期，或通过事件触发（检测到危险、置信度下降、任务状态转换）。

调度策略：**动作查询立即执行，语义查询 opportunistic**。这保证了动作回路的实时性不被语义生成阻塞。

### 4.4 训练目标：知识保留微调

为了防止动作 specialization 导致模型失去通用视觉-语言能力，LiteVLA-H 采用混合损失：

$$
L = \lambda_a L_{act} + \lambda_s L_{sem} + \lambda_g L_{gen} + \lambda_{kp} L_{kp}
$$

- $L_{act}$：动作损失；
- $L_{sem}$：空中语义损失；
- $L_{gen}$：通用 caption/VQA 损失；
- $L_{kp}$：知识保留正则项（可选，如对预训练骨干的 KL 蒸馏）。

消融实验显示：

| 训练配方 | 动作成功率 | 保留 Caption CIDEr | 空中语义 F1 |
|----------|------------|---------------------|-------------|
| 仅动作数据 | 最高 | 0.31 | 0.42 |
| + 空中语义 | 略降 | 0.45 | 0.81 |
| + 通用 VL 复习 | 接近最高 | 0.76 | — |
| **完整方法** | 与动作-only 差 1.1% | **0.82** | 高 |

这说明：通过混合训练，可以在不严重牺牲动作能力的前提下保留模型的语义描述能力。

### 4.5 控制与安全设计

LiteVLA-H 明确把 VLA 放在**外环（outer-loop）**：

- VLA 输出速度、航向或模式级指令；
- 传统飞控负责内环姿态稳定（高频）；
- 下游设置命令包络、过期 token 拒绝、紧急悬停/返航等安全层。

这种分层让 VLA 的抖动或偶发错误 token 不会直接传导到电机，是空中部署的必要安全设计。

---

## 5. 系列对比与演化脉络

| 维度 | Lite VLA | LiteVLA-Edge | LiteVLA-H |
|------|----------|--------------|-----------|
| **发表时间** | 2025.11 | 2026.03 | 2026.05 |
| **硬件** | Raspberry Pi 4 | Jetson AGX Orin | Jetson AGX Orin |
| **计算后端** | CPU (llama.cpp) | GPU (llama.cpp CUDA) | GPU (FP16) |
| **骨干** | SmolVLM-256M | SmolVLM-256M | SmolVLM-256M |
| **量化** | NF4 backbone + FP32 head | GGUF Q4K_M | FP16 |
| **动作延迟** | ~11 s (0.09 Hz) | **150.5 ms (6.6 Hz)** | **50.65 ms (19.74 Hz)** |
| **语义延迟** | 无 | 无 | 149–164 ms (6–7 Hz) |
| **应用场景** | 地面机器人概念验证 | 地面机器人实时闭环 | 无人机外环制导 + 语义感知 |
| **核心贡献** | CPU 可行 | 实时闭环 | 双速率调度、预填充主导分析 |

可以清晰看到一条演进线：

> **可行性（Lite VLA）→ 实时性（LiteVLA-Edge）→ 多时间尺度调度（LiteVLA-H）**

每一步都没有改变骨干模型，而是**持续优化部署形态和任务调度策略**。

---

## 6. 技术启示与工程经验

### 6.1 小骨干 + 强部署，是端侧 VLA 的现实路径

LiteVLA 系列证明：在 256M 参数级别，通过合适的量化、运行时和调度，已经可以实现：

- 地面机器人 6.6 Hz 闭环控制；
- 无人机 19.74 Hz 外环制导；
- 同时保留 6–7 Hz 语义感知。

这与 7B 级 VLA 形成互补：大模型负责云端通用推理和复杂任务分解，小模型负责端侧高频执行。

### 6.2 量化不是简单压缩，而是精度-动作的 trade-off

Lite VLA 的消融表明，projection head 必须保留 FP32；LiteVLA-Edge 的 GGUF Q4K_M 则在全模型量化和动作稳定性之间取得了平衡。这提示我们：

> 在 VLA 中，**动作 token 的数值稳定性比文本 token 更敏感**，因为微小的量化误差可能对应真实的物理运动偏差。

### 6.3 预填充是端侧 VLA 的主要瓶颈

LiteVLA-H 最重要的理论贡献是指出：在小模型、短输出场景下，**首动作时间（TTFA）由图像-文本预填充主导**，而非解码长度。因此优化方向应该是：

- 减少视觉 token 数量（如 LightVLA 的 token pruning）；
- 缓存可复用的 prompt 结构；
- 重叠图像预处理与上一控制周期；
- 避免不必要的语义请求。

### 6.4 动作与语义应该分速率运行

对于需要同时"反应"和"解释"的机器人，强制单速率运行会浪费模型的能力。LiteVLA-H 的双速率调度提供了一个实用模板：

- 动作：硬实时、高频、短输出；
- 语义：软实时、低频、长输出；
- 语义作为监督/日志/人机交互服务，而非控制信号。

---

## 7. 局限与未来方向

### 7.1 当前局限

1. **任务级评估不足**：三篇论文的主要证据都是延迟/频率，真实的任务成功率、泛化能力、长时间稳定性仍需更多评测。
2. **仿真为主**：LiteVLA-Edge/H 的真实世界飞行/移动实验报道有限。
3. **数据集规模小**：Lite VLA 仅 15K 样本，远小于 Qwen-RobotNav（15.6M）或 NavFoM（8M）。
4. **动作空间简单**：主要面向差速移动机器人速度指令或无人机外环航点，复杂操作任务未涉及。
5. **碰撞率未报告**：与 VLX-Go、TrackVLA 等工作相比，缺少 CR 等安全指标。

### 7.2 未来方向

- **预填充优化**：视觉 token 剪枝、更轻量的 projector、图像预处理流水线重叠；
- **多任务扩展**：从跟踪/导航扩展到 ObjNav、VLN、机械臂操作；
- **真实世界验证**：在真实 TurtleBot、无人机平台上做闭环飞行/移动实验；
- **记忆与预测**：在需要时长时记忆（如 ReMem-VLA）和预测模块（如 FutureVLA）应被选择性激活，而非常驻；
- **混合架构**：LiteVLA 作为端侧执行器 + 云端大模型作为高层规划器。

---

## 8. 总结

LiteVLA 系列代表了一条**以部署为导向的端侧 VLA 研究路线**。它没有提出新的多模态架构或新的策略学习目标，而是系统性地回答了："如何把一个小型多模态模型真正部署到边缘机器人上并产生足够快的动作？"

从 Raspberry Pi 4 上的 0.09 Hz 概念验证，到 Jetson Orin 上的 6.6 Hz 地面闭环，再到 19.74 Hz 的无人机双速率外环制导，LiteVLA 展示了端侧 VLA 的渐进可行性。其工程经验——特别是混合精度量化、llama.cpp 部署、ROS 2 集成、预填充主导分析、动作-语义分速率调度——对当前正致力于把 VLA 从实验室搬到机器人上的研究者和工程师具有直接参考价值。

---

**相关链接**：
- Lite VLA (arXiv): https://arxiv.org/abs/2511.05642
- LiteVLA-Edge (arXiv): https://arxiv.org/abs/2603.03380
- LiteVLA-H (arXiv): https://arxiv.org/abs/2605.00884
- VLX-Go GitHub: https://github.com/om-ai-lab/VLX-Go
- SmolVLM: https://arxiv.org/abs/2504.05299
- llama.cpp: https://github.com/ggerganov/llama.cpp
