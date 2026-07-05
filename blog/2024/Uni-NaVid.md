---
title: Uni-NaVid
date: 2024-12-09
categories: [VLA]
---

# Uni-NaVid：一个统一四种具身导航任务的视频 VLA 模型

> **论文**：*Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks*  
> **作者**：Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, He Wang  
> **机构**：北京大学计算机学院 CFCS、Galbot、北京智源人工智能研究院  
> **arXiv**：[2412.06224](https://arxiv.org/abs/2412.06224)（v1 2024-12-09）  
> **项目主页**：[https://pku-epic.github.io/Uni-NaVid](https://pku-epic.github.io/Uni-NaVid)

---

## 1. 为什么关注这项？

2024 年是 VLA（Vision-Language-Action）在机器人领域爆发的一年。在操作任务里，RT-2、OpenVLA、Octo 等模型已经展示了「看图像 + 听指令 + 输出动作」的潜力。但在**移动与导航**这个更古老的领域，大多数方法仍然是一个任务一个模型：VLN 用一套架构、ObjectNav 用另一套、EQA 再用另一套。

Uni-NaVid 的核心命题很简单，但很重要：

> **能不能用一个视频 VLA 模型，同时处理四种主流导航任务？**

这四种任务分别是：

- **VLN**（Vision-and-Language Navigation）：按自然语言指令走到终点；
- **ObjectNav**：按目标物体类别搜索并接近；
- **EQA**（Embodied Question Answering）：先导航到相关区域，再回答问题；
- **Human Following**：按语言描述在人群中跟随特定行人。

更难得的是，Uni-NaVid 不是只在仿真器里跑分，还直接部署到了 Unitree GO2 四足机器人上，只用 RGB 和语言指令就能完成真实场景导航。

---

## 2. 一句话总结

Uni-NaVid 是一个**统一的视频视觉-语言-动作（VLA）导航模型**。它以 egocentric RGB 视频流和自然语言指令为输入，在单一模型内同时学习四种导航任务，通过在线视觉 token 合并机制把 LLM 推理时间压到约 0.2 秒，支持 5 Hz 级别的实时控制。

---

## 3. 核心结果速览

| 任务 / 基准 | 关键指标 | Uni-NaVid | 对比 / 说明 |
|---|---:|---:|:---|
| VLN-CE R2R Val-Unseen | SR / SPL | **47.0% / 42.7%** | 较 NaVid 提升 +25.7% SR，仅 RGB |
| VLN-CE RxR Val-Unseen | SR / SPL | **48.7% / 40.9%** | 长程复杂指令场景，大幅超过 NaVid |
| HM3D ObjectNav | SR / SPL | **73.7% / 37.1%** | 仅 RGB，超过使用深度/里程计的 PIRLNav-IL-RL |
| HM3D-OVON zero-shot | SR / SPL | **43.9% / 21.8** | 超过 GroundingDINO + VLFM |
| MP3D-EQA（CE） | GT ACC | **54.4%** | 优于 NaviLLM，且用连续环境而非离散 landmark |
| Human Following 自建基准 | SR / FR / CR | **61.21% / 71.93% / 2.07%** | 超过 PoliFormer 与 IBVS（含 GT bbox 版本） |
| ScanQA | BLEU-1 / CIDEr | **46.85 / 94.72** | 视频理解能力也未丢失 |
| 真实世界 VLN | 简单 / 复杂 | **92% / 84%** | NaVid 为 80% / 20% |
| 推理速度 | 单次 | **≈ 0.2 s** | 约 5 Hz，支持非阻塞执行 |

一句话：**把四个任务塞进一个模型，不仅没有互相拖累，多数任务反而比之前专门设计的单任务模型更强。**

---

## 4. 技术架构：视频 + 指令 → 4 个动作

### 4.1 统一的问题形式化

Uni-NaVid 把所有任务都写成同一个形式：

> 给定自然语言指令 $I$ 和到时刻 $T$ 的 egocentric RGB 视频 $O_T = \{x_1, \dots, x_T\}$，预测未来 4 个低级动作：
> $$\{A_T, A_{T+1}, A_{T+2}, A_{T+3}\}$$

动作空间只有 4 个离散动作：

- `FORWARD`：前进约 25 cm
- `TURN-LEFT`：左转约 30°
- `TURN-RIGHT`：右转约 30°
- `STOP`：停止

这个动作空间和 Habitat VLN-CE / ObjectNav / MP3D-EQA 一致，所以既能用于连续环境，也便于直接部署到真实机器人。

### 4.2 模型 pipeline

```
egocentric RGB 视频帧 {x_1, ..., x_T}
    ↓
[EVA-CLIP 视觉编码器] → 每帧 256 个视觉 token
    ↓
[在线视觉 token 合并] → 当前 64 tokens + 短期 4×B tokens + 长期稀疏 tokens
    ↓
[两层 MLP 投影器]
    ↓
拼接 {长期 token}{短期 token}{当前 token}<NAV>{指令}
    ↓
[Vicuna-7B LLM]
    ↓
输出 <Action0><Action1><Action2><Action3>
```

三个关键组件：

1. **EVA-CLIP** 做视觉编码；
2. **在线视觉 token 合并模块** 压缩历史视频；
3. **Vicuna-7B** 做语言理解和动作生成。

### 4.3 在线视觉 token 合并：把长视频「剪」进固定长度

这是 Uni-NaVid 最让人眼前一亮的设计。

如果直接把每帧 256 个 token 送给 LLM，视频长度一增加，LLM 的输入 token 数就线性爆炸，推理延迟会到 1–2 秒，根本无法实时控制。

Uni-NaVid 的做法是借鉴 Atkinson-Shiffrin 的记忆模型，把历史帧分成三级：

| 记忆类型 | 池化尺度 | 每帧 token 数 | 作用 |
|:---|:---:|:---:|:---|
| 当前（Current） | $\alpha=2$ | 64 | 保留细粒度空间信息，即时避障 |
| 短期（Short-term） | $\alpha=8$ | 4 | 保留近期拓扑与运动上下文 |
| 长期（Long-term） | $\alpha=16$ | 1 | 稀疏语义/路标记忆，防止 token 线性增长 |

- 当前帧保留高分辨率；
- 近期 $B=64$ 帧用中等分辨率；
- 更早的历史每帧只留 1 个 token，并按**余弦相似度做增量合并**（阈值 $\tau=0.95$）。相似的新帧就平均进已有长期 token，不相似才新建一个。

这样 LLM 看到的视觉 token 数几乎不随导航步数增长，推理时间稳定在 **0.2 秒左右**。

### 4.4 一个小而关键的 trick：`<NAV>` token

输入序列里有一个特殊的 `<NAV>` token：

```
{Long term tokens}{Short term tokens}{Current tokens}<NAV>{Instruction}
```

它的作用是告诉模型「现在该输出动作」。

对于 EQA 任务，导航阶段保留 `<NAV>`，让模型继续走；停止后移除 `<NAV>`，模型就转为回答问题。消融实验显示，去掉 `<NAV>` 后 EQA 准确率从 47.3% 掉到 20.4%。

这个设计非常 VLA：用特殊 token 区分不同行为模式，而不是用多个头或多个模型。

### 4.5 为什么一次预测 4 个动作？

Uni-NaVid 不是每步只预测下一个动作，而是一次输出未来 4 步。

好处有两个：

1. **长程规划**：模型必须考虑接下来一小段轨迹，减少短视行为；
2. **非阻塞部署**：机器人执行当前 4 个动作时，模型可以异步处理下一帧并生成新指令，把 LLM 推理等待时间「藏」在执行里，实现约 5 Hz 的有效控制频率。

---

## 5. 数据与训练：5.9M 样本的多任务混合

Uni-NaVid 用了 **5.9M** 训练样本：

| 数据类型 | 样本数 | 说明 |
|:---|---:|:---|
| 多任务导航数据 | 3.6M | Habitat 系列仿真器 |
| 开放世界视频 VQA / 视频描述 | 2.3M | LLaMA-VID、Panda-70M 等真实世界数据 |

四种导航数据的构成：

| 任务 | 数据集 | 样本数 |
|:---|:---|---:|
| VLN | VLN-CE R2R + RxR | 2.4M |
| ObjectNav | HM3D ObjectNav | 483k |
| EQA | MP3D-EQA | 250k |
| Human Following | 自建 Habitat 3.0 基准 | 544k |

训练分两阶段：

1. **模态对齐预训练**：只训练投影器 $P_V$；
2. **端到端联合微调**：同时训练投影器和 Vicuna-7B，使用全部 3.6M 导航数据 + 2.3M 开放世界视频数据。

这里 2.3M 视频 VQA/caption 数据非常关键。消融显示，去掉这些数据后 EQA 准确率几乎归零（1.19%）。这说明**纯导航训练会让 LLM 遗忘开放世界视觉-语言能力**，真实世界 VQA 数据是 VLA 模型的「稳定剂」。

---

## 6. 实验亮点与洞察

### 6.1 VLN：RGB-only 就能接近用全景/里程计/深度的方法

在 VLN-CE R2R Val-Unseen 上，Uni-NaVid 仅用单目 RGB 就达到 **SR 47.0% / SPL 42.7%**，超过使用全景、里程计、深度的 ETPNav 在 SPL 上的表现（49.0%）。

在更难的 RxR 上优势更大：**SR 48.7%**，比同为 RGB-only VLA 的 NaVid（23.8%）翻倍还多。

### 6.2 ObjectNav：开放词汇 zero-shot 也很强

HM3D ObjectNav 上，Uni-NaVid **SR 73.7%**，超过使用里程计+深度的 PIRLNav-IL-RL（70.4%）。

在开放词汇 HM3D-OVON 上，Uni-NaVid zero-shot 就超过使用 GroundingDINO 的 VLFM 和微调后的 DAgRL+OD，说明它的语言 grounding 能力确实是从统一训练里学到的，而不是靠外部检测器。

### 6.3 Human Following：端到端打败了需要 GT bbox 的模块方法

Uni-NaVid 只凭语言描述（如 "man in blue t-shirt and gray trousers"）就能在拥挤场景中跟随目标行人，不需要上游人体检测器，也不需要真值 bbox。

在自建基准上，Uni-NaVid 的 SR（61.21%）比使用 GT bbox 的 IBVS†（50.58%）还高 10 个点，这说明「描述理解 → 目标识别 → 运动规划」的端到端学习比模块化流水线更鲁棒。

### 6.4 真实世界部署：Unitree GO2

论文在 Unitree GO2 四足机器人上做了 zero-shot 真实部署：

- 传感器：头部 RealSense D455 RGB；
- 计算：远程 A100，推理约 0.2 s，网络传输约 0.3 s；
- 控制：接收到 4 个动作后顺序执行，执行期间异步上传新帧。

25 条简单指令 + 25 条复杂指令的结果：

| 方法 | 简单 | 复杂 |
|:---|:---:|:---:|
| NaVid | 80% | 20% |
| **Uni-NaVid** | **92%** | **84%** |

复杂指令如「前进 5 步，左转 4 步，再右转 5 步」，Uni-NaVid 的优势非常明显。

---

## 7. 消融实验告诉我们的三件事

### 7.1 `<NAV>` token 很重要

去掉后 EQA 暴跌。因为模型分不清什么时候该继续导航、什么时候该直接回答。

### 7.2 三级记忆缺一不可

| 记忆配置 | VLN SR | ObjNav SR | EQA ACC | Follow SR |
|:---|:---:|:---:|:---:|:---:|
| 仅当前帧 | 9.61% | 44.3% | 32.5% | 56.3% |
| 当前 + 短期 | 39.7% | 67.8% | 44.1% | 59.7% |
| 当前 + 短期 + 长期 | **48.7%** | **73.7%** | **47.3%** | **61.2%** |

VLN 对长期记忆最敏感——没有历史上下文，语言指令根本对不齐。

### 7.3 多任务训练确实有协同

相比单任务训练，多任务联合训练在四个任务上都有提升。VLN、ObjectNav、EQA 提升更明显，Human Following 提升较小，可能是因为跟随更依赖最近几帧，对长程记忆需求低。

---

## 8. 局限与思考

Uni-NaVid 很强，但它也勾勒出了当前 VLA 导航的几个边界：

1. **任务范围仍有限**：四种任务之外，demand-driven navigation、multi-agent navigation、social navigation 等还没覆盖。

2. **机器人 embodiment 假设偏窄**：数据收集时假设机器人高度 0.88–1.25 m、半径 0.1–0.6 m。如果想泛化到不同尺寸或形态（无人机、机械臂移动平台），需要显式引入 embodiment 先验。

3. **动作输出偏简单**：只输出离散低级动作，没有连续平滑轨迹。对真实机器人来说，这意味着底层还要做运动控制平滑，不能直接用于高精度操作或高速移动。

4. **数据多样性受仿真器限制**：数据量从 3M 增加到 6M 时边际收益递减，说明场景和指令多样性仍是瓶颈。这也是为什么真实世界数据（如那 2.3M VQA）如此重要。

---

## 9. 对 VLA 领域的启发

Uni-NaVid 给我的三个核心启发：

1. **统一格式 + 多任务训练能产生协同**。把不同任务写成同一输入/输出格式，让模型共享场景理解、语言 grounding、目标搜索等通用能力，这是通向通用导航基础模型的一条可行路径。

2. **长视频 VLA 必须做在线 token 压缩**。LLM 上下文不能无限增长，Uni-NaVid 的三级记忆 + 增量合并是一个工程上很优雅、效果也很扎实的解法。

3. **VQA 数据是 VLA 的「稳定剂」**。纯机器人数据训练会让 VLM  backbone 遗忘开放世界能力，混合适量真实世界视频问答/描述数据能显著缓解这个问题。

---

## 10. 总结

Uni-NaVid 是 2024 年底导航 VLA 领域的一个重要节点。它证明了：

> **一个统一的视频 VLA 模型，可以在单一架构内处理多种导航任务，并且通过合理的记忆压缩和前瞻预测，实现仿真到真实的 zero-shot 部署。**

它不一定解决了所有问题，但它为「通用导航基础模型」提供了一个清晰的起点。接下来值得关注的方向包括：扩展到更多任务、引入机器人 embodiment 先验、生成连续动作轨迹、以及构建更大规模的真实世界导航数据集。

---

*参考：Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks (arXiv:2412.06224v1, 2024-12-09)*
