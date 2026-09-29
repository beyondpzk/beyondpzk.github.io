---
title: "室内轻地图导航：Teach-and-Repeat 与语义记忆"
date: 2026-09-29
categories: [VLN]
description: "用关键帧拓扑图、局部几何与语义记忆组织跨房间导航，说明组件分工与验证路径。"
topic: navigation
type: 工程实践
summary: "用关键帧拓扑图、局部几何与语义记忆组织跨房间导航，说明组件分工与验证路径。"
---

# 室内轻地图导航：Teach-and-Repeat 与语义记忆

> **问题定义**：在轻地图或无预建图条件下实现室内跨房间导航（例如"从客厅走到厨房"）；允许先领着机器人转一圈（teach），之后让它自主导航到去过的地方（repeat）。
>
> **依据**：结合视觉示教复走、语义导航和空间记忆的公开论文，比较各组件的适用边界。
>
> **核心思路**：Teach 阶段建立关键帧拓扑图、局部占据表示与语义场景图；Repeat 阶段检索语义目标、搜索拓扑路径，再由局部策略逐段执行。各组件已有研究基础，但组合系统的可靠性仍需单独验证。

---

## 一、问题拆解

"客厅走到厨房"的本质是：跨房间、10~30m 长程、目标在起点不可见、需要空间记忆。按两种设定分开：

| 模式 | 含义 | 现状 |
|---|---|---|
| **M1 无图冷启动** | 第一次进门直接说"去厨房" | ObjectNav/语义 VLN 问题，现有最强方案 SR 仅 60~80% |
| **M2 先转一圈再导航（Teach-and-Repeat）** | 先示教，再在地图有效期间重复导航 | 可复用示教路线与跨任务记忆，已有成熟研究可参考 |

本方案主推 M2，M1 作为 repeat 阶段的兜底（目标未命中时主动探索补全记忆）。

---

## 二、四条技术路线

| 路线 | 代表工作 | 地图重量 | 记住去过的地方 | 室内跨房间能力 |
|---|---|---|---|---|
| **A. 拓扑记忆 + 图像目标** | ViNG → GNM → ViNT → NoMaD → LM-Nav → GuideNav | 轻（关键帧图，GuideNav 1km 仅 24MB） | ✅ teach-and-repeat 正统 | 强，但目标只能是图像/路线，无语义 |
| **B. 在线轻量建图 + 语义记忆** | VL-Nav、TravExplorer、SGImagineNav、Uni-LaViRA | 轻（占据栅格 + 场景图在线增量构建） | ✅ 场景图/航点历史持久保存 | 强（VL-Nav 室内 SR 83.4%；TravExplorer 真机 64%，含双层别墅/跨楼层） |
| **C. 无图端到端 VLA + 隐式记忆** | PoliFormer、NaVid、NaVILA、StreamVLN、NavFoM、Qwen-RobotNav、DualVLN、ABot-N1 | 零 | ❌ 记忆死在单 episode 上下文里 | 中（R2R SR 50~70%），找目标靠学到的探索偏置 |
| **D. 云边 agentic 系统** | ABot-N0（Planner + Topo-Memory）、Qwen-RobotNav（证据笔记本） | 轻（在线积累的拓扑/文本记忆） | ✅ 最接近产品形态 | 强，但依赖云端大模型 |

### 关键事实

- **路线 C 的天花板**：纯无图端到端方案的记忆只活在单次 episode 的上下文窗口里，"去过的房间"无法跨任务复用。NavFoM 在 300 米级长程探索上"仍显著落后"；LongNav-R1 用 RL 解决了"一直走"但 wrong-goal 比例 74.5%——**能走不代表能到**。这条路线单独解决不了本问题。
- **teach-and-repeat 的正统是路线 A**：GuideNav 是目前最纯粹的视觉 VT&R——Teach 阶段牵着走一圈建关键帧拓扑图（DINOv3 描述符 + 自适应关键帧选择），Repeat 阶段用 CosPlace 做视觉位置识别 + Reloc3r 回归相对 SE(2) 位姿，4940m 导航转向成功率 46/46 = 100%，仅 6 次人工干预。AutoInspect 是工业版（49 天无中断，MTBF 78h），但依赖 LiDAR SLAM 预建图。
- **路线 B 补上语义层**：VL-Nav 在线构建 3D 场景图（物体节点带质心/置信度/最佳视角位姿 + 房间节点）；TravExplorer 在线建可通行体素图 + 概率实例地图 + 空间价值地图——去过的地方不再重复探索（负证据衰减、前沿移除）。
- **组件集成问题**：本文梳理的工作各有侧重。将示教接口、持久语义记忆和局部执行组织成可维护的室内系统，需要额外的接口与恢复机制；这不等于相关领域不存在完整方案。

---

## 三、推荐方案：Teach-Repeat + 语义记忆

### 3.1 Teach 阶段（示教并建立初始记忆）

边走边做三件轻量的事：

1. **关键帧拓扑图**（GuideNav 方案）
   - 节点：DINOv3 图像描述符 + 相对位姿链；自适应关键帧选择（转动/场景变化大时加密）；
   - 边：相邻关键帧的可通行连接；
   - 体量：整个家几 MB（GuideNav 实测 1km 约 24MB）。
2. **在线占据栅格**（VL-Nav / TravExplorer 方案）
   - 深度或 LiDAR 反投影，标记可通行区域与 frontier；
   - teach 没走到的角落留给 repeat 阶段补探索，不要求一次走完。
3. **语义标注层**（拼图中最关键的一块，需要与示教数据对齐）
   - 每个关键帧/区域过一遍 VLM 打标签（"厨房""冰箱""客厅沙发"）；
   - 写入场景图，等价于 ABot-N0 Topo-Memory 的 Function Layer + Object/POI Layer，或 LM-Nav 的 CLIP landmark 落地，但换成在线 VLM 打标；
   - 标签带最佳视角位姿与置信度（借 VL-Nav 物体节点设计）。

**定位选型**：有 LiDAR 就用 AutoInspect 式 ICP 配准到先验图（重复定位精度极高）；纯视觉用 GuideNav 验证过的 CosPlace（视觉位置识别 + Markov 信念更新）+ Reloc3r（相对位姿回归）组合。

### 3.2 Repeat 阶段（检索语义目标并逐段执行）

```
"去厨房"
  ↓
① 语义检索：VLM/CLIP 在场景图查"厨房" → 命中关键帧节点 + 最佳视角位姿
  ↓
② 全局规划：拓扑图 Dijkstra/A* → 途经关键帧序列（客厅 → 走廊 → 厨房门口）
  ↓
③ 局部执行：逐段跟踪途经点，二选一——
   · 稳妥派：图像目标策略（OmniVLA 图像目标模式 SR 1.00；NoMaD 90%）
   · 能力派：VLA 策略（ABot-N1 pixel-goal / Qwen-RobotNav PointNav，
            目标 = 下一路段的相对位姿，模型只消费局部坐标）
  ↓
④ 失败恢复：Uni-LaViRA Second Chance Backtrack（回岔口换方向）
   + ABot-N0 式记忆写回（"厨房移门关了" → 边标记 Blocked，下次自动绕路）
```

### 3.3 两个加分项

- **teach 不完备没关系**：repeat 时语义检索未命中 → 切 TravExplorer 式 frontier 探索主动找，找到后写回场景图。记忆持续生长（ABot-N0 的"成功失败都写回"哲学）。
- **日常走动即数据飞轮**：每次 repeat 的观测持续更新关键帧描述符与语义标签，环境变化（家具挪动、门开关）需经过变化检测和一致性检查；必要时局部重新示教。

### 3.4 为什么不直接上无图端到端 VLA？

无图端到端策略依赖当前观测、历史上下文与训练先验。显式示教记忆则保存目标场地的路线和语义关联，使重复任务不必每次重新搜索。两者适合分层组合。

不同论文中的转向成功率、任务成功率和图像目标到达率不能直接换算为这套方案的成功率。验证时应固定场景与硬件，对比有无示教记忆两种配置，并记录失败类型、人工干预和地图维护成本。

---

## 四、各组件的出处对照

| 组件 | 借鉴来源 | 对应博客 |
|---|---|---|
| 关键帧拓扑图 + 视觉 VT&R | GuideNav（DINOv3 + CosPlace + Reloc3r） | [GuideNav：纯视觉公里级导盲导航的 VT&R 技术解析](/blog/2025/2025-12-05-guidenav) |
| 拓扑图原型 + 图搜索 + 途经点分层 | ViNG / ViNT / NoMaD | [ViNG：用视觉目标学习开放世界导航](/blog/2020/2020-12-17-ving) 等 |
| 语言目标 → 图节点落地 | LM-Nav（GPT-3 + CLIP + 拓扑图） | [LM-Nav：用预训练大模型组合实现机器人语言导航](/blog/2022/2022-07-10-LM-Nav) |
| 在线占据栅格 + frontier 探索 | VL-Nav / TravExplorer | [VL-Nav：神经符号推理式视觉语言导航](/blog/2025/2025-02-02-vlnav)、[TravExplorer：基于可通行感知 3D 规划的跨楼层具身探索](/blog/2026/2026-07-21-TravExplorer) |
| 3D 场景图（物体/房间节点） | VL-Nav / SGImagineNav | 同上、[SGImagineNav：基于场景图想象世界模型的具身导航](/blog/2025/2025-08-09-SGImagineNav) |
| 四层 Topo-Memory 与记忆写回 | ABot-N0（Excluded/Confirmed 机制） | [ABot-N0：面向通用具身导航的 VLA 基础模型](/blog/2026/2026-02-12-abot-n0) |
| 图像目标局部策略 | OmniVLA / NoMaD / GNM | [OmniVLA：面向机器人导航的全模态视觉-语言-动作模型](/blog/2025/2025-09-23-OmniVLA) 等 |
| pixel-goal VLA 局部策略 | ABot-N1 / DualVLN | [ABot-N1：面向通用视觉-语言导航的慢-快解耦基础模型](/blog/2026/2026-07-11-ABot-N1)、[DualVLN：慢思考、快执行——迈向通用视觉-语言导航的双系统基础模型](/blog/2025/2025-12-09-DualVLN) |
| 局部坐标 PointNav 接口 | Qwen-RobotNav | [Qwen-RobotNav: 面向 Agentic 系统的可扩展导航基础模型](/blog/2026/2026-06-17-qwenrobotnav) |
| 航点历史 + Backtrack | Uni-LaViRA（TDM + SCB） | [Uni-LaViRA：以语言-视觉-机器人动作翻译统一具身导航](/blog/2026/2026-05-26-Uni-LaViRA) |
| 工业级 T&R 可靠性参考 | AutoInspect | [AutoInspect：面向长期自主工业巡检的足式机器人系统](/blog/2024/2024-04-19-autoinspect) |

---

## 五、最小可行实现路径

1. **最小原型**：GuideNav 式纯视觉 VT&R 打底（关键帧图 + CosPlace 定位 + 相对位姿跟踪）；teach 后离线用 VLM 给关键帧批量打语义标签；repeat 时"去厨房" = 文本检索标签 → 图搜索 → 复走。本质是把 GuideNav + LM-Nav 拼起来，两者都有成熟参考实现。
2. **V2**：加在线占据栅格 + frontier 补探索（借 TravExplorer）；加 backtrack 与记忆写回（借 Uni-LaViRA / ABot-N0），处理"门关了""椅子挪了"。
3. **V3**：局部策略换成 VLA（ABot-N1 pixel-goal 或 Qwen-RobotNav PointNav），为复合指令提供导航子任务能力；取放物体仍需独立操作系统；底层 T&R 记忆架构不变。

### 待决策的开放问题

- 定位走纯视觉（CosPlace + Reloc3r，低成本）还是 LiDAR ICP（AutoInspect 式，更稳）——取决于硬件平台；
- 语义打标用云端 VLM（准，需联网）还是边缘小 VLM + CLIP 检索（可离线）；
- 场景图的持久化格式与更新策略（多久重打标、冲突如何合并）。
