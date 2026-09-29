---
title: "导航基础模型对比：DualVLN、ABot、Qwen-RobotNav 与 LightNav-0"
date: 2026-09-29
categories: [VLN]
description: "按任务比较导航模型的观测配置、评测指标和训练数据，并说明横向比较的边界。"
---

# 导航基础模型对比：DualVLN、ABot、Qwen-RobotNav 与 LightNav-0

导航模型的数字只有放回任务、传感器和控制协议中才有意义。本文先列出模型配置，再按评测任务拆分指标，最后比较训练数据。

## 参考资料

- [DualVLN 论文笔记](/blog/2025/2025-12-09-DualVLN)
- [ABot-N0 论文笔记](/blog/2026/2026-02-12-abot-n0)
- [ABot-N1 论文笔记](/blog/2026/2026-07-11-ABot-N1)
- [Qwen-RobotNav 论文笔记](/blog/2026/2026-06-17-qwenrobotnav)
- [LightNav-0 论文笔记](/blog/2026/2026-09-01-lightnav-0)

## 评测结果：按任务阅读

> 主表列代表性 KPI；`—` 表示该方法未报告该项，或该评测集不是该方法的直接评测对象。不同方法的观测配置、控制器协议存在差异，横向比较需结合脚注。

### 模型配置

| 方法 | 参数量与观测配置 |
| --- | --- |
| DualVLN | System 2: Qwen-VL-2.5 7B + System 1: 轻量 DiT；单目 RGB |
| ABot-N0 | LLM backbone: Qwen3-4B；全景 RGB / EVT 单目 |
| ABot-N1 | Slow: Qwen3.5-4B + Fast: Qwen3.5-2B；三视角 RGB |
| Qwen-RobotNav-4B | VLN 全景 RGB；OVON/EVT 单目 |
| Qwen-RobotNav-8B | VLN 全景 RGB；OVON/EVT 单目 |
| LightNav-0 | Qwen3-VL-4B-Instruct；单目 RGB，无 depth/odo/pano |

### R2R-CE Val-Unseen

| 方法 | NE↓ | OS↑ | SR↑ | SPL↑ |
| --- | ---: | ---: | ---: | ---: |
| DualVLN | 4.05 | 70.7 | 64.3 | 58.5 |
| ABot-N0 | 3.78 | 70.8 | 66.4 | 63.9 |
| ABot-N1 | 3.32 | 75.2 | 70.9 | 67.5 |
| Qwen-RobotNav-4B | 3.80 | 77.2 | 69.5 | 63.6 |
| Qwen-RobotNav-8B | 3.53 | 78.5 | 72.1 | 66.6 |
| LightNav-0 | — | — | 68.5 | 62.8 |

### RxR-CE Val-Unseen

| 方法 | NE↓ | SR↑ | SPL↑ |
| --- | ---: | ---: | ---: |
| DualVLN | 4.58 | 61.4 | 51.8 |
| ABot-N0 | 3.83 | 69.3 | 60.0 |
| ABot-N1 | 3.13 | 73.9 | 63.9 |
| Qwen-RobotNav-4B | 3.80 | 75.2 | 65.0 |
| Qwen-RobotNav-8B | 3.58 | 76.5 | 65.7 |
| LightNav-0 | — | 73.6 | 64.5 |

### VLN-PE R2R Val-Unseen

| 方法 | SR↑ | SPL↑ |
| --- | ---: | ---: |
| DualVLN | 51.60 | 42.49 |
| Qwen-RobotNav-4B | 60.28 | 55.24 |
| Qwen-RobotNav-8B | 65.50 | 61.19 |

### HM3D-OVON Unseen

| 方法 | SR↑ | SPL↑ |
| --- | ---: | ---: |
| ABot-N0 | 54.0 | 30.5 |
| Qwen-RobotNav-4B | 53.1 | 20.9 |
| Qwen-RobotNav-8B | 51.2 | 24.0 |
| LightNav-0 | 47.0 | — |

### EVT-Bench STT

| 方法 | SR↑ | TR↑ | CR↓ |
| --- | ---: | ---: | ---: |
| ABot-N0 | 86.9 | 87.6 | 8.54 |
| ABot-N1 | 90.1 | 89.8 | 4.27 |
| Qwen-RobotNav-4B | 77.4 | 90.0 | 6.40 |
| Qwen-RobotNav-8B | 78.6 | 89.7 | 5.70 |
| LightNav-0 | 91.7 | 87.7 | — |

### Social-VLN 与 Short-Horizon OVON

| 方法 | Social-VLN SR↑ | Short-Horizon OVON SR↑ |
| --- | ---: | ---: |
| DualVLN | 37.2 | — |
| ABot-N1 | — | 84.9 |

### ABotN 导航评测

| 方法 | ABotN-PointBench Outdoor SR↑ | ABotN-PointBench Indoor SR↑ | ABotN-POIBench SR↑ |
| --- | ---: | ---: | ---: |
| ABot-N0 | 76.9 | 89.6 | 20.9 |
| ABot-N1 | 92.9 | 95.4 | 77.3 |

### VLNVerse Fine

| 方法 | SR↑ | SPL↑ |
| --- | ---: | ---: |
| Qwen-RobotNav-4B | 62.61 | 56.22 |
| Qwen-RobotNav-8B | 63.75 | 57.93 |

### ObjectNav

| 方法 | MP3D SR↑ | MP3D SPL↑ | HM3D v2 SR↑ | HM3D v2 SPL↑ |
| --- | ---: | ---: | ---: | ---: |
| Qwen-RobotNav-4B | 52.2 | 16.0 | 75.6 | 30.6 |
| Qwen-RobotNav-8B | 48.8 | 17.7 | 71.2 | 33.0 |
| LightNav-0 | 53.3 | — | 79.5 | — |

### INSIGHT-Bench

| 方法 | SR↑ | SPL↑ | NE↓ |
| --- | ---: | ---: | ---: |
| LightNav-0 | 43.7 | 41.5 | 3.88 |

### 指标口径与比较边界

NE 以米计，越低越好；SR、SPL、OS、TR、CR 按百分比列出，方向见表头。`—` 仅表示本次整理未收录该项，不能据此推断论文没有报告。

- DualVLN 的 VLN-PE 结果来自“物理控制器 + zero-shot transfer”设置；Qwen-RobotNav 的 VLN-PE 结果来自“flash controller”设置，两者协议不同，不宜直接比较。
- ABot-N1 的 `Short-Horizon OVON` 是对标准 OVON 重新锚定起点的短程评测，不等同于 ABot-N0/Qwen 使用的标准 `HM3D-OVON` 三 split 评测。
- ABot-N0 在 `ABotN-PointBench`、`ABotN-POIBench` 上的数值，来自 ABot-N1 论文中作为基线的评测结果。
- Qwen-RobotNav-8B 的 EVT-Bench 仅报告 STT（Single Target）split；ABot-N0/N1 还报告了 DT、AT。
- Qwen-RobotNav 在 VLN-CE 另有单目成绩：R2R SR 65.7 / SPL 59.6，RxR SR 73.4 / SPL 63.5。
- Qwen-RobotNav 的 VLN-CE 单目成绩同样包含 4B/8B 两个版本；由于观测配置不同，上表 以论文中的全景（Pano）成绩为准。
- ABot-N0：单 LLM backbone，`Qwen3-4B`。
- ABot-N1：双系统，`Slow Qwen3.5-4B + Fast Qwen3.5-2B`；边缘部署版将 Slow 压缩为 `Qwen3.5-2B`，Fast 使用 `DiT 306M + DINOv2-Base`，但论文未单独公开该部署版的 benchmark KPI 表。
- LightNav-0 的公开 README 只摘要了 SR/SPL 等主要指标；论文完整表包含 NE / nDTW / CR 等，上表 中对应未摘要项标为 `—`。
- LightNav-0 的 ObjectNav 结果是单目 RGB、无 depth/odometry；OVON 的 SPL 未在 README 摘要中列出。
- LightNav-0 ObjectNav 其余摘要结果：MP3D SR 53.3；HM3D v1 SR 74.5；HM3D v2 SR 79.5；OVON Seen SR 55.3、Synonyms SR 53.3、Unseen SR 47.0。
- LightNav-0 EVT-Bench 的 DT split 摘要结果：SR 82.6、TR 80.1。
- LightNav-0 的 scaling 结论：模型规模 2B → 4B 在 R2R SR/SPL 提升 8.6/7.4，8B 结果混合；环境规模从 1/8 扩展到全量，在 R2R/RxR 上收益更明显。

## 训练数据规模与组成

| 方法 | 训练数据总量 | 训练数据分类 | 各类数据量 / 说明 |
|---|---|---|---|
| DualVLN | 论文未统一披露主训练集总样本量；明确公开的是 Social-VLN 训练数据 | System 2：StreamVLN 数据配方；System 1：由 VLN-CE 轨迹投影生成的 pixel-goal grounding 样本；Social-VLN：动态避障导航数据 | System 2：样本数未公开，QwenVL-2.5 微调 1 epoch；System 1：独立样本数未公开；Social-VLN：763K episodes / 60 个 MP3D 场景 |
| ABot-N0 | 约 21.9M = 16.9M 轨迹 + 5.0M 推理 | 轨迹数据：5 类导航任务；推理数据：6 类认知/对齐任务 | 轨迹：Point-Goal 4.0M；Instruction-Following 2.8M（R2R/RxR 1.5M、Door-Traversal 0.3M、Language-Guided Person Search 0.2M、Short-Horizon 0.8M）；Object-Goal 3.6M（HM3D+OVON 1.8M、OVON-sub 0.2M、InteriorGS 1.6M）；POI-Goal 2.5M；Person-Following 4.0M。推理：Navigable Areas 1.2M；Social CoT 0.8M；Instruction Reasoning 1.3M；Object Reasoning 0.1M；POI Grounding 0.5M；General VQA 1.1M |
| ABot-N1 | 约 30M 预训练样本 + 0.5M GRPO 后训练 episodes | 预训练慢系统；预训练快系统；后训练 GRPO | 慢系统 13.3M：Point-Goal 2.36M、Instruction-Following 5.11M、Object-Goal 0.11M、POI-Goal 3.0M、Person-Following 2.7M。快系统 16.4M：Point-Goal 6.20M、Instruction-Following 4.67M、Object-Goal 2.10M、POI-Goal 0、Person-Following 3.4M。后训练：Point-Goal 0.5M episodes，按 Safe/Critical/Danger 分层并剔除不稳定样本 |
| Qwen-RobotNav | 约 15.6M = 轨迹规划约 85% + 视觉语言/推理约 15% | 轨迹规划：5 类任务；视觉语言/推理：3 类 | 轨迹规划：Instruction-Following 5.63M（R2R 1.491M、RxR 4.140M）；Point-Goal 0.984M；Object-Goal 2.0M；Target Tracking 1.486M；Autonomous Driving 3.2M；T2V 自动生成 40K。视觉语言/推理：General VL 1.0M；Navigation-specific reasoning 873K；Discrete multi-round navigation 362K |
| LightNav-0 | 公开说明未给统一“样本数”，明确公开为 2000+ 真实室内外场景 + 4000+ 小时 VLA 对齐数据 | Real2Sim2Real 数据引擎；三阶段后训练：Embodied Reasoning（ER）、SFT、RL | ER：大规模图像/视频空间理解数据，未公开精确样本量；SFT：基于 2000+ 真实场景合成的 4000+ 小时视觉-语言-动作对齐数据；RL：仿真 rollout 中通过成功/失败奖励优化，未公开精确样本量 |
