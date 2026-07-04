---
title: Qwen-RobotNav
date: 2026-06-17
categories: [VLA]
---

# Qwen-RobotNav: 面向 Agentic 系统的可扩展导航基础模型

> **论文**: *Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System*  
> **作者**: Qwen Team  
> **发布时间**: 2026-06-17  
> **项目主页**: https://qwen.ai/blog?id=qwen-robotnav  
> **开源代码**: https://github.com/QwenLM/Qwen-RobotNav

---

## 一、概述与核心贡献

### 1.1 一句话总结

Qwen-RobotNav 是阿里巴巴通义千问团队提出的**统一导航基础模型**，它并没有为不同导航任务设计不同的网络结构，而是把"多任务导航"的核心挑战重新定义为**观察上下文建模（observation context modeling）**问题。通过在推理时外部可调的任务模式与观察参数，同一个 Qwen3-VL 骨干网络即可支撑从室内 VLN、目标搜索、目标跟踪到自动驾驶的多种导航形态，并自然成为上层 Agent 的导航原语。

### 1.2 核心指标（SOTA 亮点）

| 任务/基准 | 结果 | 对比 |
|---|---|---|
| VLN-CE RxR Val-Unseen (全景) | **76.5% SR** | 超越 NavFoM +12.1% SR |
| VLN-CE R2R Val-Unseen (全景) | **72.1% SR / 66.6% SPL** | 超越 NavFoM +10.4% SR |
| EVT-Bench 单目标跟踪 | **90.0% TR** | 超越 ABot-N0 +2.4%，超越 TrackVLA++ +9.0% |
| HM3D v2 ObjectNav | **75.6% SR**（仅 RGB，单目前向） | 新 SOTA |
| NAVSIM navtest | **91.4 PDMS** | 超越 ReflectDrive +0.3 |
| HM-EQA / MT-EQA / EXPRESS-Bench | **76.7 / 54.4 / 79.27** | 相对 FAST-EQA 提升 7.5 / 3.9 / 10.57；**导航步数减少 77%** |

### 1.3 三大核心贡献

1. **参数化观察接口（Parameterised Observation Interface）**：提出可在推理时动态配置的观察编码策略，包括 token 预算 `B`、时间衰减 `γ`、相机权重 `wc`、帧采样模式 `m` 等，实现"同一模型、不同任务、不同上下文策略"。
2. **面向 Agent 的导航原语设计**：Qwen-RobotNav 被封装为上层 LLM Planner（如 Qwen3.6-Plus）可调用的工具，支持任务模式切换与观察配置切换，支撑长程记忆与上下文压缩。
3. **大规模异构数据联合训练**：15.6M 训练样本覆盖 5 大导航任务族，并混入 15% 视觉-语言推理数据，防止纯轨迹训练导致的"反应式动作映射坍塌"。

---

## 二、研究背景与问题定义

### 2.1 为什么需要统一导航模型？

具身导航任务家族极其多样：
- **指令跟随（VLN）**：依赖长程语言-视觉对齐，需要保留全局历史以反复对照远处路标；
- **点目标导航（PointNav）**：输入简洁坐标，强调几何路径规划与避障；
- **物体目标导航（ObjNav）**：需要在探索与趋近之间切换，从全局搜索转入局部精细操作；
- **目标跟踪（Tracking）**：高度依赖最近几帧，旧历史多为噪声；
- **自动驾驶（Driving）**：高速、多智能体、安全约束强。

近期工作（NavFoM、ABot-N0、Uni-NaVid 等）已证明单一架构可以处理多任务，但它们通常采用**固定的观察策略**：均匀采样或固定滑动窗口。这导致无法在长程记忆与短程反应之间按需切换，更难被上层 Agent 动态调用。

### 2.2 关键洞察

Qwen-RobotNav 的核心论点是：

> 不同导航任务共享同一个感知-规划骨干，但对"如何消费视觉流"有根本不同的需求。因此，**关键不是设计更多任务头，而是把观察上下文作为一等公民、外部可控变量**。

### 2.3 统一任务形式化

所有任务都被统一为**航点轨迹预测**：

```
输入：文本指令 L + 多相机多步观测 I_{1:N}^{1:T} ∈ R^{H×W×3}
输出：未来 K=8 个航点 W = {(x_k, y_k, θ_k)}_{k=1}^{K}
```

其中 `(x, y)` 为地面坐标，`θ` 为朝向。**关键约束如下：**

- **坐标系**：以当前时刻机器人为原点的**局部坐标系**（非全局地图坐标）。
- **归一化**：训练时将航点归一化到 `[-1, 1]`，使用每个数据集坐标的 **99th percentile** 作为尺度因子。
- **上下文压缩**：视觉 token 数量随 `T·N` 线性增长，因此必须有一套可压缩、可配置的上下文编码机制。

> **为什么必须是局部坐标？** 论文中 waypoint 输出始终定义在局部坐标系下。这意味着推理时输入的 goal 也必须是相对坐标——模型不消费全局地图或拓扑信息，全局路径规划由上层 Planner 负责拆分和转换。

---

## 三、模型架构详解

### 3.1 整体结构

Qwen-RobotNav 直接继承 **Qwen3-VL**，并只做最小改动：

```
多视角 RGB 图像
    ↓
[任务自适应观察编码]  ← 外部配置 Φ=(B, γ, wc, m, b_min, b_max)
    ↓
带自然语言时间/视角标签的视觉 token 序列
    ↓
Qwen3-VL（Vision Encoder + LLM Backbone）
    ↓
轻量 4 层 MLP Action Head
    ↓
24-dim 航点输出 (x, y, θ) × 8
```

三大组件：
1. **视觉编码器**：基于 SigLIP-2 ViT，支持动态分辨率（2D-RoPE）和 DeepStack 多层级视觉注入；
2. **语言骨干**：Qwen3-VL 的 LLM，直接处理拼接后的视觉+语言 token；
3. **动作头**：4 层 MLP（hidden dim 512，GELU），将最终隐状态映射为 8 个航点。

### 3.2 任务自适应观察编码（核心创新）

这是整篇论文的灵魂。给定配置 `Φ = (B, γ, {wc}, m, b_min, b_max)`：

#### 3.2.1 帧采样模式 `m`

- `random`：从完整历史中均匀采样，保留全局上下文；
- `latest`：只取最近窗口，强调短程反应。

#### 3.2.2 时间权重 `ωt`

对保留的 `T'` 帧，按指数衰减计算每帧重要性：

```
ωt = exp(γ · t / (T' - 1)),  t = 0, ..., T' - 1
```

- `γ = 0`：均匀权重；
- `γ` 越大：越偏向最新帧。例如 `γ = 2` 时，最新帧权重约为最旧帧的 **7.4 倍**；`γ = 3` 时约 **20.1 倍**。

#### 3.2.3 联合权重矩阵 `W[t, c]`

```
W[t, c] = ωt · wc
```

`wc` 是每相机重要性。例如四相机默认权重可为 `[2.0, 1.0, 0.5, 1.0]`（前、右、后、左），因为前向相机包含最丰富的可执行线索。

#### 3.2.4 约束分配算法

给定总 token 预算 `B`，对 `T'×N` 个 (time, camera) 单元：

1. **保底分配**：每个单元先分 `b_min` 个 token；
2. **比例分配**：剩余预算 `B - T'·N·b_min` 按联合权重 `W[t,c]` 比例分配；
3. **封顶重分**：超过 `b_max` 的单元释放余额，迭代重分直到稳定。

可行性检查：`T'·N·b_min ≤ B ≤ T'·N·b_max`。若训练采样超出此范围，`B` 被 clip 到可行区间。

该分配决定每帧每相机的像素分辨率（保持长宽比缩放），最终进入 ViT 编码。

> **关键设计**：所有这些参数在训练时**每样本随机采样**，因此模型不会过拟合到某一种固定配置，推理时任意配置都能 zero-shot 工作。  
> **作者备注**：该分配是经验性启发式，用于暴露统一接口。作者认为未来可探索更原则性的 token 分配方法。

### 3.3 视角与时间标识

视觉 token 进入 LLM 后本身不携带"这是哪个相机、哪一时刻"的信息。Qwen-RobotNav 通过**自然语言标签**解决：

```
Time step 0
  Front View <image>
  Right View <image>
  Back View <image>
  Left View <image>
Time step 1
  Front View <image>
  ...
```

- 不引入新的位置嵌入或结构修改；
- "Front / Left / Back" 等词本身携带空间语义，被预训练 LLM 自然理解；
- 实验发现描述性名称略优于数字方位角（如 "right 90 degrees"）。

### 3.4 具身感知提示设计

不同平台（室内机器人 vs. 自动驾驶汽车）通过 system prompt 前缀区分：

- 室内机器人："Imagine you are a robot programmed for navigation tasks"
- 自动驾驶："Imagine you are a car programmed for autonomous driving"

这种文本化的 embodiment prior 让模型调用不同的预训练世界知识（室内布局 vs. 交通规则）。新增平台只需新增 prompt 模板，零参数改动。

### 3.5 动作规划头

- 4 层 MLP；
- 训练时将航点归一化到 `[-1, 1]`，使用每个数据集坐标的 99 百分位作为缩放因子；
- 损失为预测航点与真实航点之间的 MSE；
- 推理时反归一化。

#### 3.5.1 航点归一化与反归一化（代码实现）

论文中的归一化策略可复现为以下代码：

```python
import numpy as np

class WaypointNormalizer:
    """
    论文方案：在完整训练集上统计每个坐标轴的 99th percentile 作为 scale factor。
    训练前 fit，训练/推理时复用同一组因子。
    """
    def __init__(self):
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_theta = 1.0

    def fit(self, waypoints_dataset):
        """
        waypoints_dataset: ndarray of shape (N, K, 3)
            N = 训练样本数，K = 8 waypoints，3 = (x, y, θ)
        """
        # 拉平所有样本、所有 waypoints 的同一坐标
        all_x = waypoints_dataset[..., 0].flatten()
        all_y = waypoints_dataset[..., 1].flatten()
        all_theta = waypoints_dataset[..., 2].flatten()

        # 取绝对值的 99th percentile 作为对称缩放因子
        self.scale_x     = np.percentile(np.abs(all_x),     99)
        self.scale_y     = np.percentile(np.abs(all_y),     99)
        self.scale_theta = np.percentile(np.abs(all_theta), 99)

        # 防止除零（理论上不会发生，但防御性编程）
        self.scale_x     = max(self.scale_x,     1e-6)
        self.scale_y     = max(self.scale_y,     1e-6)
        self.scale_theta = max(self.scale_theta, 1e-6)

    def normalize(self, waypoints):
        """输入 shape (K, 3) 或 (N, K, 3)，输出归一化到 [-1, 1] 附近"""
        norm = waypoints.copy().astype(np.float32)
        norm[..., 0] = waypoints[..., 0] / self.scale_x
        norm[..., 1] = waypoints[..., 1] / self.scale_y
        norm[..., 2] = waypoints[..., 2] / self.scale_theta
        return norm

    def denormalize(self, waypoints_norm):
        """推理时反归一化，还原为实际物理单位 (m, m, rad)"""
        raw = waypoints_norm.copy().astype(np.float32)
        raw[..., 0] = waypoints_norm[..., 0] * self.scale_x
        raw[..., 1] = waypoints_norm[..., 1] * self.scale_y
        raw[..., 2] = waypoints_norm[..., 2] * self.scale_theta
        return raw
```

**关键说明**：
- `np.percentile(np.abs(...), 99)` 确保正负方向对称缩放，使约 99% 的数据点绝对值 ≤ 1；
- 三轴**独立统计**（x、y、θ 的 scale factor 不同），因为它们的物理量纲和数值范围差异巨大；
- `fit()` 必须在**整个训练集**上一次性完成，不能按 batch 动态计算，否则会导致分布偏移。

### 3.6 训练策略

#### 3.6.1 训练目标

```
L = L_traj + λ · L_VL
```

- `L_traj`：航点回归 MSE（仅导航轨迹样本激活）；
- `L_VL`：标准 next-token prediction loss（视觉-语言推理样本）；
- `λ = 1.0`。

#### 3.6.2 配置随机化

每样本独立随机采样：
- `γ ~ U[1, 3]`
- `B ~ U[2048, 4096]`
- 每相机 `wc` 从各自区间采样（如前相机 `U[1.5, 2.5]`）
- `b_min ~ U_Z[1, 8]`，`b_max ~ U_Z[128, 256]`
- 帧采样模式 random / latest 各占 50%

#### 3.6.3 联合训练

- 85% 导航轨迹规划数据；
- 15% 导航相关视觉-语言数据（含通用 VQA、图像描述、视觉定位、导航专用推理、离散多轮 VLN 等）；
- 按数据集采样率进行 batch 级平衡，确保每轮训练覆盖所有任务族。

#### 3.6.4 优化细节

- 从预训练 Qwen3-VL 初始化，端到端微调；
- AdamW，β1=0.9，β2=0.95，weight decay=1e-2；
- 余弦学习率，前 3% 步 warmup；
- 视觉编码器/LLM：`2e-5`；动作头：`1e-4`；
- 梯度裁剪 1.0；
- 8B 模型全局 batch size 256，总计 **2,816 H100 GPU 小时**。

### 3.7 推理时的定位需求与坐标转换

Qwen-RobotNav 的 waypoint 输出始终定义在**以当前机器人位姿为原点的局部坐标系**中。因此，在 Agentic 系统中调用 PointNav 模式时，上层 Planner 必须完成以下坐标转换：

```
global_goal（地图坐标） → 定位模块获取 current_pose → relative_goal = global_goal - current_pose
```

**关键要点：**
- **训练时**的 PointNav 标签已经是相对坐标（以当前帧为原点）；
- **推理时**输入的 goal 也必须是相对坐标，模型不消费全局地图或拓扑信息；
- 如果 A→B 距离较远（如跨房间），上层 Planner 必须先将全局路径拆分为多个 sub-goals（每段 5~10m），每段独立做一次坐标转换后调用 Qwen-RobotNav。

这与论文中 Agentic 系统"77% 更少导航步数"的结果一致：上层 Planner 负责高效分解，Qwen-RobotNav 负责局部视觉导航执行。

---

## 四、数据工程：15.6M 样本如何构建

### 4.1 公开数据集获取指南

| 数据集 | 许可/门槛 | 推荐用途 |
|---|---|---|
| **VLN-CE R2R/RxR** | 需 Matterport3D 数据（学术申请，约 2-3 天批准） | 指令跟随 + PointNav 样本构造 |
| **HM3D** | 需申请（学术用途） | PointNav / ObjNav |
| **HM3D-OVON** | 公开 | 开放词汇 ObjNav |
| **EVT-Bench** | 公开下载 | 目标跟踪 |
| **nuScenes** | 公开下载 | 自动驾驶 |
| **OpenScene** | 公开下载 | 驾驶场景底图 |

**实操建议**：如果仅想复现 PointNav 能力，最低配置是 **Matterport3D + VLN-CE**，从中二次构造 PointNav 样本即可，无需额外仿真采集。

### 4.2 导航轨迹规划数据（85%，约 13.3M）

| 任务族 | 数据集/来源 | 样本数 | 关键说明 |
|---|---|---|---|
| 指令跟随 | VLN-CE R2R | 1.49M | 教师强制展开，单/多相机配置 |
| 指令跟随 | VLN-CE RxR | 4.14M | 更长路径、多语言、密集路标 |
| 点目标导航 | Habitat (MP3D/HM3D) | 984K | 直接接近 348K、短程 174K、长程 400K、命令式 62K |
| 物体目标导航 | MP3D + HM3D-OVON | 2.00M | 基于骨架图的探索轨迹 + VLM 开放式目标标注 |
| 目标跟踪 | EVT-Bench | 1.49M | 单目标跟踪（STT），拥挤室内场景 |
| 自动驾驶 | nuScenes + OpenScene | 3.2M | 多视图 + 可选指令/自车状态/历史轨迹先验 |

#### 4.2.1 点目标导航（PointNav）的样本构造

Qwen-RobotNav 的 PointNav 数据**并非在仿真器中重新采集**，而是**从已有的 VLN-CE R2R/RxR 回放轨迹中二次构造**，具体流程：

1. **轨迹回放**：使用 Habitat 中的 teacher forcing 重播 ground-truth 轨迹，记录每帧多相机 RGB 和全局位姿；
2. **随机选点**：从当前帧 `t` 随机采样一个未来时刻 `t+k`（通常 `k ∈ [5, 30]` 步），该 future pose 作为 Point Goal；
3. **Waypoint 插值与 Heading 计算**：从当前位姿到目标位姿的底层轨迹上，等弧长插值出 **8 个 waypoints**，每个含 `(x, y, θ)`。其中偏航角 `θ` 有两种获取方式：

   - **方案 A（首选）**：如果轨迹带有仿真器/SLAM 位姿（含 yaw 角），直接读取并同步插值；
   - **方案 B（Fallback）**：仅有坐标时，用相邻点的切线方向近似 `θ = atan2(dy, dx)`，最后一个点复制前一个的 heading。

   **Heading 计算代码**：

   ```python
   import numpy as np

   def compute_heading_from_poses(poses):
       """
       方案 A：从位姿中获取真实 heading (yaw)。
       poses: ndarray (N, 4) 四元数格式 [x, y, z, qw, qx, qy, qz]
              或 (N, 3) 欧拉角格式 [x, y, yaw]
       返回: (N,) heading θ，范围 [-π, π]
       """
       if poses.shape[1] == 3:
           # 已有 yaw 角，直接 wrap
           yaw = poses[:, 2]
       else:
           # 四元数 → euler yaw (Z-up 坐标系)
           qw, qx, qy, qz = poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6]
           yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz))
       return np.arctan2(np.sin(yaw), np.cos(yaw))  # wrap to [-π, π]

   def compute_heading_from_trajectory(waypoints_xy):
       """
       方案 B：从轨迹坐标计算切线方向 heading。
       waypoints_xy: ndarray (K, 2) — K 个 (x, y) 点
       返回: (K,) heading θ，第 k 个点指向下一个点的方向
       """
       K = len(waypoints_xy)
       headings = np.zeros(K)
       for k in range(K - 1):
           dx = waypoints_xy[k + 1, 0] - waypoints_xy[k, 0]
           dy = waypoints_xy[k + 1, 1] - waypoints_xy[k, 1]
           headings[k] = np.arctan2(dy, dx)
       # 最后一个点：继承前一个 heading（或继续延伸切线）
       headings[-1] = headings[-2] if K > 1 else 0.0
       return np.arctan2(np.sin(headings), np.cos(headings))
   ```

4. **局部坐标转换**：将 8 个 waypoints 从全局坐标系转换到以当前位姿为原点的局部坐标系（与训练标签格式对齐）；
5. **动作分布再平衡**：前进步骤以 45% 概率采样（防止直走 dominate），转向和停止动作 100% 保留；
6. **减速轨迹**：当目标距离 ≤ 1.5m 时，生成步长线性递减的减速轨迹，教导平滑停止。

**核心结论**：如果你有任意的漫游/回放轨迹，不需要重新跑仿真采集，只需要从轨迹中采样 future goal 并插值 waypoints，即可生成 PointNav 训练样本。

#### 4.2.2 物体目标导航的骨架探索轨迹

为避免"直奔目标"的短视轨迹，Qwen-RobotNav 采用：
1. 从俯视占用图提取最大连通区域；
2. 形态学骨架化得到中轴图；
3. 随机采样目标点，沿骨架随机分支探索，死胡同回溯；
4. 三次样条插值平滑为 0.25m 步长的物理可行轨迹；
5. 终点处用 VLM 标注可见物体，形成开放词汇目标。

#### 4.2.3 自动驾驶数据

将驾驶统一为航点预测。同一底层轨迹可实例化为多种条件变体：
- 仅多视图相机；
- 加导航指令；
- 加自车状态；
- 加历史真值轨迹。

这让模型学会在不同先验信息下保持稳定规划。

### 4.3 自动生成视频数据（40K）

为弥补仿真器与真实世界的视觉域差距，论文提出 T2V 自动生成流水线：

1. **Prompt/指令生成**：LLM 生成第一人称导航场景描述与对应语言指令；
2. **文本生成视频**：T2V 模型渲染约 5 秒 egocentric 视频；
3. **VLM 质量过滤**：评估场景一致性、导航正确性、目标到达、停止行为、运动连续性、碰撞避免等；
4. **轨迹提取**：单目深度+姿态估计恢复相机位姿，转成地面 2D 轨迹 `[x, y, yaw]`；
5. **运动学过滤**：剔除位移过小、抖动过大、瞬移、异常加速度、高频噪声的样本。

该流程无需 3D 资产或物理仿真，可直接生成多样化真实场景视频。

### 4.4 为什么需要 15% 视觉-语言数据："反应式动作映射坍塌"详解

这是理解 Qwen-RobotNav 数据设计的核心。**反应式动作映射坍塌**（Collapse into Reactive Action-Sequence Mappers）是指：当模型仅在导航轨迹数据上训练时，它逐渐放弃深层的空间推理、语义理解与指令解析，退化为一个浅层的"视觉模式 → 动作"条件反射映射器——看到某种视觉输入就输出固定动作序列，而不真正理解环境的几何结构、任务目标或语言指令。

论文原话：

> *"models trained on navigation trajectory data alone tend to collapse toward reactive action sequences and lose general-purpose spatial reasoning"*

#### 4.4.1 为什么会发生？（数据层面的根本原因）

导航轨迹数据有一个致命特性：**相邻帧高度相似，动作变化极小**。

| 数据特性 | 后果 |
|---------|------|
| 帧间冗余高（egocentric 视频，前后帧 90% 像素重叠） | 模型发现不必做深层推理，只需基于局部视觉纹理预测下一步即可 |
| 动作分布极端不平衡（直行 dominate，转弯/停止稀少） | 模型趋向输出"安全"的默认动作（如直走） |
| 轨迹是"可预测的"（走廊一般直走，到路口才转） | 模型学到的是时序惯性，而非空间规划 |
| 缺乏语言/语义监督信号 | 模型没有动力维持预训练 VLM 的开放世界理解能力 |

**本质上**：轨迹数据提供了太多**统计捷径**（statistical shortcuts），模型作为优化器，自然会走最容易的路——把 VLM backbone "冻结"成视觉特征提取器，然后在上面套一个浅层的动作反射。

#### 4.4.2 坍塌后的典型症状

| 症状 | 表现 |
|------|------|
| **语言指令当耳旁风** | 给 "turn left at the kitchen" 和 "turn right at the kitchen" 输出几乎相同的轨迹 |
| **无法处理 unseen 环境** | 在训练过的建筑布局中表现尚可，换到新建筑就撞墙或原地打转 |
| **丧失物体语义理解** | 找不到 "sofa" 因为它只在训练数据中见过特定纹理的沙发 |
| **长程记忆归零** | 走了 20 步后完全忘记起点在哪，也无法对照远处的路标 |
| **动作平滑但无脑** | 输出看起来很"流畅"的轨迹，但实际上是惯性滑行，遇到障碍才被动反应 |

论文中把这种状态称为 **"reactive"**（反应式）而非 **"deliberative"**（推理式）：模型像一个被训练过的昆虫，靠局部刺激驱动，没有认知地图。

#### 4.4.3 解决方案：VL Co-training 如何"救活"模型

Qwen-RobotNav 混入 15% 视觉-语言推理数据（VQA、场景描述、空间推理问答等），与轨迹数据联合训练。

| VL 数据的强制作用 | 效果 |
|-----------------|------|
| **Next-token prediction loss** 要求模型维持语言输出能力 | backbone 不能退化为纯视觉编码器，必须保持语言空间活跃 |
| **空间推理 QA**（如 "what's on your left?"） | 强制模型建立相机视角与空间方位的显式关联 |
| **指令进度评估**（"have you passed the door?"） | 强制模型维持时序记忆与里程碑跟踪 |
| **开放世界 VQA** | 保留预训练 VLM 的物体识别、属性理解、常识推理 |

**通俗类比**：
- **纯轨迹训练** = 只让一个人死记硬背开车路线（左转→直行→右转），不给他地图也不教交规。换条路就懵。
- **加入 VL 数据** = 同时让他做地理习题、读路标、解释交通规则。他必须保持对空间的语义理解，不能仅靠肌肉记忆开车。

> **结论**：这 15% 的 VL 数据不是"锦上添花"，而是**防止模型脑死亡的必要疫苗**。

### 4.5 视觉-语言数据（15%，约 2.37M）

| 类型 | 数量 | 作用 |
|---|---|---|
| 通用 VQA | ~669K | 维持开放世界视觉理解 |
| 视觉定位 | ~178K | RefCOCO/COCO/Objects365 |
| 图像描述 | ~6K | 基础视觉-语言对齐 |
| 多图像推理/比较 | ~38K | 跨帧空间推理 |
| 导航专用推理 | 873K | 自由式 QA + 结构化多视角推理 |
| 离散多轮 VLN | 362K | CVDN/SOON/REVERIE/SRDF 等图结构轨迹 |

#### 4.5.1 结构化多视角推理

从 VLN 轨迹中构造四组件推理链：
1. **History reasoning**：基于历史视角总结已走路径；
2. **Scene analysis**：描述当前四视角可见内容；
3. **Instruction progress**：评估已完成/剩余子目标；
4. **Action reasoning**：导出下一步动作与置信度。

每个样本被拆成 4 个独立 QA 对，强制模型在动作前进行系统性语言化推理。

### 4.6 数据增强

- **指令改写**：每条约指令生成 3 个同义变体，保留空间方向与相对路标；
- **图像质量增强**：使用 Qwen-Image-Edit 进行 prompt 引导的风格迁移，将仿真渲染转为照片级真实感；
- **相机/观察增强**：相机高度 `U[0.5, 1.5]m`、水平视场 `U[90°, 120°]`、长宽比 2:1~4:3；
- **速度增强**：低速变体、随机子步长、PointNav 多运动尺度；
- **PointNav 专用**：前视-only 变体，匹配仅前向相机的部署场景。

### 4.7 典型训练样本示例

为了更直观地理解 Qwen-RobotNav 的训练数据形态，下面给出 5 大导航任务族以及 VL 推理数据的**伪样本（pseudo-sample）**。所有样本共享统一的输出格式：未来 8 个航点 `W = [(x, y, θ)] × 8`，区别主要在于输入指令 `L`、观察配置 `Φ`、视觉历史 `I` 以及 embodiment 提示前缀。

---

#### 示例 1：指令跟随（VLN-CE R2R）

**Embodiment 前缀**：
```
Imagine you are a robot programmed for navigation tasks.
```

**任务模式**：`VLN`

**指令 L**：
```
Enter the open space, then go through the door on the right. Wait just past the black doormat.
```

**观察配置 Φ**：
```json
{
  "B": 3072,
  "gamma": 2.0,
  "camera_weights": {"Front": 2.0, "Right": 1.0, "Back": 0.5, "Left": 1.0},
  "sample_mode": "random",
  "b_min": 4,
  "b_max": 256
}
```

**视觉历史 I**（4 相机 × 8 步，随机采样自 episode）：
```
Time step 0
  Front View <image_480x360>
  Right View <image_320x240>
  Back View <image_224x168>
  Left View <image_320x240>
Time step 3
  Front View <image_512x384>
  ...
Time step 7
  Front View <image_512x384>
  Left View <image_320x240>
```

**真值航点 W**（归一化前，单位 m / rad）：
```json
[
  [0.25, 0.00, 0.00],
  [0.50, 0.05, 0.05],
  [0.75, 0.15, 0.10],
  [1.00, 0.30, 0.18],
  [1.15, 0.45, 0.25],
  [1.25, 0.55, 0.20],
  [1.30, 0.60, 0.10],
  [1.32, 0.62, 0.00]
]
```

**说明**：随机采样保留从餐厅到客厅再到门口的全局历史，帮助模型在关键转弯处重新对照指令中的 "door on the right"。

---

#### 示例 2：点目标导航（PointNav，长程）

**Embodiment 前缀**：
```
Imagine you are a robot programmed for navigation tasks.
```

**任务模式**：`PointNav`

**指令 L**（相对坐标式，与论文一致）：
```
Target: (2.4, -1.8) in your local frame. Distance: 3.0 m. Bearing: -37°.
```

**观察配置 Φ**：
```json
{
  "B": 2048,
  "gamma": 1.5,
  "camera_weights": {"Front": 2.0, "Right": 1.0, "Back": 0.5, "Left": 1.0},
  "sample_mode": "latest",
  "b_min": 4,
  "b_max": 256
}
```

**视觉历史 I**（4 相机 × 4 步，取最近窗口）：
```
Time step 4
  Front View <image_384x288>
  Right View <image_256x192>
  Back View <image_192x144>
  Left View <image_256x192>
Time step 7
  Front View <image_384x288>
  ...
```

**真值航点 W**：
```json
[
  [0.30, -0.10, -0.12],
  [0.60, -0.25, -0.15],
  [0.90, -0.45, -0.18],
  [1.10, -0.70, -0.20],
  [1.25, -1.00, -0.22],
  [1.35, -1.30, -0.20],
  [1.40, -1.50, -0.12],
  [1.42, -1.58, -0.05]
]
```

**说明**：长程目标通常没有直接视线，因此航点呈现绕过障碍物的弯曲路径。`latest` 采样强调局部避障，同时 `γ=1.5` 仍保留一定早期上下文用于方向保持。

---

#### 示例 3：物体目标导航（ObjNav）

**Embodiment 前缀**：
```
Imagine you are a robot programmed for navigation tasks.
```

**任务模式**：`ObjNav`

**指令 L**（开放式目标）：
```
Navigate to a four-tier bookshelf filled with books.
```

**观察配置 Φ**：
```json
{
  "B": 4096,
  "gamma": 0.8,
  "camera_weights": {"Front": 2.0, "Right": 1.0, "Back": 0.5, "Left": 1.0},
  "sample_mode": "random",
  "b_min": 4,
  "b_max": 256
}
```

**视觉历史 I**（4 相机 × 10 步，随机采样覆盖骨架探索轨迹）：
```
Time step 0
  Front View <image_512x384>   // 走廊起点
  Right View <image_256x192>
  Back View <image_192x144>
  Left View <image_256x192>
Time step 5
  Front View <image_384x288>   // 进入房间分支
  ...
Time step 12
  Front View <image_512x384>   // 书架出现在视野中
  ...
```

**真值航点 W**：
```json
[
  [0.25, 0.00, 0.00],
  [0.50, 0.05, 0.05],
  [0.70, 0.15, 0.10],
  [0.85, 0.25, 0.08],
  [0.95, 0.30, 0.05],
  [1.00, 0.32, 0.02],
  [1.02, 0.33, 0.00],
  [1.02, 0.33, 0.00]
]
```

**说明**：`B=4096` 和 `γ=0.8` 提供丰富全局历史，帮助模型记住已探索的走廊和房间，避免重复搜索；最后几步航点减速并停止在书架前。

---

#### 示例 4：目标跟踪（Tracking）

**Embodiment 前缀**：
```
Imagine you are a robot programmed for navigation tasks.
```

**任务模式**：`Tracking`

**指令 L**：
```
Follow the woman in the light green top as she moves from the window to the bookshelf, then into the adjacent study room; maintain tracking when she exits and proceeds down the hallway.
```

**观察配置 Φ**：
```json
{
  "B": 2048,
  "gamma": 3.0,
  "camera_weights": {"Front": 2.5, "Right": 1.0, "Back": 0.3, "Left": 1.0},
  "sample_mode": "latest",
  "b_min": 4,
  "b_max": 256
}
```

**视觉历史 I**（4 相机 × 3 步，仅最近窗口，前相机高分辨率）：
```
Time step 5
  Front View <image_512x384>   // 目标在视野中央
  Right View <image_256x192>
  Back View <image_128x96>
  Left View <image_256x192>
Time step 6
  Front View <image_512x384>   // 目标向右移动
  ...
Time step 7
  Front View <image_512x384>   // 目标即将离开视野
  Right View <image_384x288>   // 右侧相机开始捕获目标
  ...
```

**真值航点 W**：
```json
[
  [0.20, 0.10, 0.15],
  [0.40, 0.25, 0.22],
  [0.55, 0.42, 0.28],
  [0.65, 0.60, 0.30],
  [0.70, 0.75, 0.25],
  [0.72, 0.85, 0.15],
  [0.73, 0.90, 0.08],
  [0.73, 0.92, 0.03]
]
```

**说明**：高 `γ=3.0` 和 `latest` 模式将绝大部分 token 投给最近帧，前相机权重提高以锁定目标。航点呈现向右前方追赶的轨迹。

---

#### 示例 5：自动驾驶（NAVSIM / nuScenes）

**Embodiment 前缀**：
```
Imagine you are a car programmed for autonomous driving.
```

**任务模式**：`Driving`

**指令 L**：
```
Step 1: Turn left. Ego state: speed 4.2 m/s, heading 0.12 rad. History trajectories (last 3 frames): [...].
```

**观察配置 Φ**：
```json
{
  "B": 4096,
  "gamma": 1.2,
  "camera_weights": {"Front": 2.0, "Front-Left": 1.5, "Front-Right": 1.5, "Back": 0.3, "Back-Left": 0.5, "Back-Right": 0.5},
  "sample_mode": "latest",
  "b_min": 4,
  "b_max": 256
}
```

**视觉历史 I**（6 相机 × 2 步，环视）：
```
Time step 6
  Front View <image_512x384>       // 环形路口入口
  Front-Left View <image_384x288>
  Front-Right View <image_384x288>
  Back View <image_128x96>
  Back-Left View <image_192x144>
  Back-Right View <image_192x144>
Time step 7
  Front View <image_512x384>       // 开始左转
  ...
```

**真值航点 W**（自车坐标系，单位 m / rad）：
```json
[
  [1.20, 0.10, 0.05],
  [2.30, 0.35, 0.10],
  [3.40, 0.85, 0.18],
  [4.30, 1.50, 0.25],
  [5.00, 2.30, 0.28],
  [5.50, 3.15, 0.22],
  [5.85, 4.00, 0.15],
  [6.05, 4.80, 0.08]
]
```

**说明**：驾驶场景需要同时看前方和侧前方以判断转弯时机与对向来车；历史自车轨迹作为先验输入，帮助模型保持曲率一致的左转路径。

---

#### 示例 6：视觉-语言推理（结构化多视角推理）

**输入对话**：
```
User: Given the instruction "Enter the open space, then go through the door on the right. Wait just past the black doormat.", analyze the current scene and decide the next action.

History: Just started in a modern dining area with a long wooden table and white chairs.

Current Views:
- Front: dining room with large window
- Right: bright open space with light wood flooring
- Back: hallway with round mirror
- Left: starburst artwork

What is the most appropriate next action?
```

**目标输出**（next-token prediction）：
```
The agent is facing a wall/window and must reorient toward the main living area. The correct action is to turn right significantly.
Answer: Turn Right
```

**对应真值动作标签**：`Turn Right`

**说明**：该样本不直接回归航点，而是训练模型用自然语言进行空间推理，强化 VLM-to-VLA 迁移所需的结构化思维能力。

---

#### 示例 7：训练时配置随机化的影响

同一底层 VLN 轨迹，在训练时可能被随机化成多个不同配置样本：

| 样本 | B | γ | sample_mode | 前相机分辨率 | 历史覆盖 |
|---|---|---|---|---|---|
| Variant A | 3072 | 2.0 | random | 512×384 | 全局 8 步 |
| Variant B | 2048 | 3.0 | latest | 512×384 | 最近 3 步 |
| Variant C | 4096 | 0.5 | random | 512×384 | 全局 12 步 |
| Variant D | 2560 | 1.5 | latest | 384×288 | 最近 4 步 |

**说明**：这种组合爆炸式的随机化让模型在推理时面对任意 `(B, γ, m)` 配置都能稳定工作，是 Qwen-RobotNav 可作为 Agent 可调原语的关键。

---

## 五、面向 Agentic 导航系统的部署

### 5.1 双层架构

```
上层规划器（Qwen3.6-Plus）
    ├── 长程推理、目标分解
    ├── 选择任务模式 τ_i
    ├── 选择观察配置 Φ_i
    ├── 调用辅助视觉工具（检测、场景理解、语义定位）
    └── 维护证据笔记本（Evidence Notebook）
                ↓
下层执行器：Qwen-RobotNav
    ├── 接收 Li, τ_i, Φ_i
    ├── 预测航点 Wi
    └── 执行 rollout
                ↓
    轨迹证据接口 → 关键帧索引 + 文本摘要 → 回到规划器
```

### 5.2 任务模式 `τ`

| 模式 | 行为 |
|---|---|
| VLN | 跟随自然语言路径指令 |
| PointNav | 向指定坐标/航点移动 |
| ObjNav | 搜索并接近目标物体 |
| Tracking | 保持对动态目标的跟踪 |

这些模式不是独立策略，而是**同一模型的不同接口**。

### 5.3 观察配置 `Φ` 的 Planner 用法

- **长程探索/ObjNav**：增大 `B`，使用 random 采样、较小 `γ`；
- **局部趋近/Tracking**：减小 `B`，增大 `γ`，使用 latest 采样；
- 实践中 `wc, b_min, b_max` 通常保持平台默认值，`B, γ, m` 是主要控制杠杆。

### 5.4 轨迹证据与上下文压缩

每次导航调用后，harness 将密集 rollout 转换为紧凑证据记录：

```json
{
  "subgoal": "Search the kitchen area for a mug",
  "task_mode": "ObjNav",
  "config": "B=3072, γ=2.0, m=random",
  "progress": "entered kitchen, checked countertop and dining table",
  "salient": ["sink", "countertop", "round table", "no mug observed"],
  "outcome": "target not found",
  "key_frames": [18, 31]
}
```

规划器默认基于这些文本摘要推理；需要时再检索关键帧图像。系统维护两级记忆：
1. **单段 rollout 记忆**：紧凑轨迹摘要；
2. **跨 episode 证据笔记本**：已搜索区域、候选物体位置、被拒绝假设等持久结论。

这种设计使长程任务能在 Planner 上下文有限的情况下持续进行。

---

## 六、实验评估

### 6.1 部署性能

Qwen-RobotNav-4B 在 Unitree Go2 上测试两种部署：

| 部署方式 | 平均端到端延迟 | 频率 | 特点 |
|---|---|---|---|
| 远程服务器 | 196 ms | 5.1 Hz | 平均更快，但受网络波动影响 |
| 边缘设备（Jetson Thor，FP8 + TensorRT） | 204 ms | 4.9 Hz | 更稳定，适合延迟敏感任务 |

### 6.2 视觉-语言导航（VLN-CE）

**全景设置**：

| Method | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR nDTW↑ |
|---|---|---|---|---|
| NavFoM | 61.7 | 55.3 | 64.4 | 65.8 |
| ABot-N0 | 66.4 | 63.9 | 69.3 | – |
| OmniNav | 69.5 | 66.1 | 73.6 | – |
| **Qwen-RobotNav-8B** | **72.1** | **66.6** | **76.5** | **72.5** |

**单目设置**：
- Qwen-RobotNav-4B 在 R2R 上达到 66.9% SR / 60.5% SPL，超越最强单目基线 DualVLN；
- 在 RxR 上 8B 达到 73.4% SR / 63.5% SPL，超越 DualVLN 达 12.0% SR / 11.7% SPL，体现长程指令优势。

### 6.3 VLNVerse（物理仿真器）

| Method | Fine-grained SR↑ | Fine-grained SPL↑ | Coarse-grained SR↑ | Coarse-grained SPL↑ |
|---|---|---|---|---|
| NavFoM | 51.59 | 32.40 | 38.02 | 23.15 |
| **Qwen-RobotNav-8B** | **63.75** | **57.93** | **46.59** | **41.54** |

SPL 提升远大于 SR，说明路径效率显著更优。

### 6.4 VLN-PE（Flash 控制器）

| Method | TL | NE↓ | FR↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|---|---|
| InternVLA-N1 | 10.11 | 4.13 | 0.45 | 67.63 | 60.36 | 54.93 |
| **Qwen-RobotNav-8B** | **9.17** | **3.73** | 4.05 | **72.99** | **65.50** | **61.19** |

最低导航误差、最高 Oracle 成功率。

### 6.5 物体目标导航

**MP3D & HM3D（闭词汇）**：

| Method | MP3D SR↑ | MP3D SPL↑ | HM3D SR↑ | HM3D SPL↑ |
|---|---|---|---|---|
| CogNav† | 46.6 | 16.1 | 72.5* | 26.2* |
| Uni-NaVid‡ | – | – | 73.7* | 37.1* |
| **Qwen-RobotNav-4B** | **52.2** | 16.0 | **75.6** | 30.6 |
| Qwen-RobotNav-8B | 48.8 | **17.7** | 71.2 | 33.0 |

注：† 使用深度/里程计；* 在 HM3D v1 上报告；Qwen-RobotNav 在更难的 HM3D v2 上评估，且仅 RGB。

**HM3D-OVON（开放词汇）**：

| Method | Seen SR↑ | Synonyms SR↑ | Unseen SR↑ |
|---|---|---|---|
| ABot-N0 | 55.3 | 55.4 | **54.0** |
| **Qwen-RobotNav-4B** | **57.7** | **60.1** | 53.1 |
| Qwen-RobotNav-8B | 56.1 | 57.8 | 51.2 |

Qwen-RobotNav 仅用**单目前向相机**，ABot-N0 使用全景多视图，仍能在 Seen/Synonyms 上胜出。

### 6.6 主动视觉跟踪（EVT-Bench STT）

| Method | TR↑ | CR↓ | SR↑ |
|---|---|---|---|
| TrackVLA++ | 81.0 | 2.10 | 86.0 |
| ABot-N0 | 87.6 | 8.54 | 86.9 |
| **Qwen-RobotNav-4B** | **90.0** | 6.40 | 77.4 |
| Qwen-RobotNav-8B | 89.7 | **5.70** | 78.6 |

跟踪率（TR）最高，但成功率（SR）低于专用跟踪器。作者解释：多任务训练使模型跟踪更紧、更保守地声明成功。

### 6.7 具身问答（EQA）

| Method | HM-EQA Acc↑ | MT-EQA Acc↑ | EXPRESS LLM Score↑ |
|---|---|---|---|
| FAST-EQA | 69.2 | 50.5 | 68.7 |
| **Qwen3.6-Plus + QwenNav-8B** | **76.7** | **54.4** | **79.27** |

相比 FAST-EQA 绝对提升 7.5 / 3.9 / 10.57，且**导航步数减少 77%**。这说明上层 Planner 配合 Qwen-RobotNav 的调用效率远高于传统探索-搜索方法。

### 6.8 自动驾驶

**NAVSIM navtest**：

| Method | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|---|---|---|---|---|---|---|
| ReflectDrive† | 97.7 | **99.3** | 93.5 | 100 | 86.9 | 91.1 |
| **Qwen-RobotNav-4B** | **99.8** | 97.5 | **98.5** | 99.9 | 84.4 | **91.4** |
| Qwen-RobotNav-8B | **99.8** | 96.9 | 98.2 | 99.9 | 84.2 | 90.9 |

注：无历史自车先验时 PDMS 仅 79.5，加入 3 帧历史后提升 **11+ 点**，说明短期轨迹历史对闭环驾驶至关重要。

**AlpaSim（zero-shot）**：

| Method | Close Encounter Rate↓ | Off-Road Rate↓ | AlpaSim Score↑ |
|---|---|---|---|
| Alpamayo-R1-10B | **4.0** | **16.0** | **0.72** |
| Qwen-RobotNav-4B | 22.0 | 34.0 | 0.15 |
| Qwen-RobotNav-8B | 22.0 | 27.0 | 0.17 |

作为通用导航模型 zero-shot 迁移到 AlpaSim，仍有显著表现；扩大模型规模可改善 off-road 率和综合得分。

### 6.9 模型规模扩展（2B → 8B）

| 模型 | R2R SR | RxR SR | 长程增益 |
|---|---|---|---|
| Qwen-RobotNav-2B | 62.5 | 68.3 | 基准 |
| Qwen-RobotNav-4B | 69.5 | 75.2 | +6.9% |
| Qwen-RobotNav-8B | 72.1 | 76.5 | +8.2% |

**关键发现**：规模扩展对**长程推理任务**（RxR）的收益尤为显著（+8.2%），而对短程反应任务（如 EVT-Bench 跟踪）收益较小。这说明更大的模型主要增强了长程上下文整合与指令理解能力，而非底层视觉-运动反射。

---

## 七、消融研究

### 7.1 数据规模效应

从 12.5% 到 100% 训练数据：
- **长程任务收益最大**：VLN-CE RxR 从 52.6% 提升至 75.2% SR；
- **短程任务较早饱和**：EVT-Bench 跟踪在 25%-50% 数据后波动；
- **驾驶任务**也呈现明显随数据增长趋势。

### 7.2 Token 预算 `B` 与时间衰减 `γ`

在 500 条 VLN-CE R2R Val-Unseen 上：

- **Token Budget 扫描（γ=2.0）**：
  - B=2048：SR 70.8%，OSR 78.9%；
  - B=3584：OSR 峰值 82.7%；
  - B=4608：SR 74.6%，OSR 82.7%。
  - 结论：更多 token 通常更好，但超过一定阈值后收益递减。

- **Gamma 扫描（B=3072）**：
  - γ=0.5：OSR 78.8%；
  - γ=3.5：OSR 82.6%；
  - γ=3.0：SR 峰值 72.5%。
  - 结论：适当增加 recency bias 有益，但过高会损害严格成功率和路径质量。

---

## 八、真实世界部署

### 8.1 展览厅长程 VLN

在未见过的展览厅中，机器狗仅靠纯语言指令从客厅区域导航 21.78m 至医疗室：
- 利用家具、门框、标识等视觉路标进行空间决策；
- 收到"后退"指令后，能沿原路精确返回起点，展示反向运动原语能力。

### 8.2 室内公寓精细语言控制

- "停在床左侧床头柜前"——准确停在指定侧边；
- "出客厅前转身"——完成绕行而非直接离开。

### 8.3 Agentic 长程任务

用户提出开放式请求：*"检查 Cotti Coffee 是否有一把绿色雨伞，并汇报沿途显著观察。"*

系统循环：
1. 解析请求，观察初始场景定位；
2. 更新记忆，识别路标（如 Alibaba logo）；
3. 沿走廊向目标区域导航；
4. 到达后检查场景，发现绿色雨伞；
5. 生成基于证据的最终回答。

---

## 九、关键洞察与技术启示

1. **观察上下文比架构更重要**：Qwen-RobotNav 把多任务导航的瓶颈从"设计专用头"转向"动态管理视觉上下文"，这是构建通用导航基础模型的新范式。
2. **训练时随机化 = 推理时可控**：通过对观察参数在训练时随机采样，模型自然获得对任意配置组合的泛化能力，无需 task-specific fine-tuning。
3. **自然语言作为结构载体**：相机身份、时间顺序、具身身份全部通过普通词汇 token 表达，不修改 backbone，保持开放世界语言 grounding。
4. **VLA 共训练防止能力坍塌**：纯轨迹训练容易使模型退化为反应式动作序列映射器；混入 VL 推理数据保留了感知与空间推理基底。
5. **导航模型应成为 Agent 的原语**：通过参数化接口与轨迹证据接口，Qwen-RobotNav 可被上层 LLM 动态调用，支撑长程、多步、记忆驱动的 Agentic 导航。
6. **局部坐标与上层 Planner 的分离**：模型始终工作于局部坐标系，全局路径规划与坐标转换由上层 Planner 负责——这使得模型足够"薄"而"可控"。

---

## 十、局限与未来方向

### 10.1 当前局限

- **跟踪任务的 SR 保守**：虽然 TR 最高，但成功率低于专用跟踪器，可能是多任务训练的权衡；
- **AlpaSim zero-shot 仍有差距**：相比专门优化的驾驶模型（Alpamayo-R1）还有较大提升空间；
- **token 分配是启发式**：作者也指出当前分配算法是经验性的，未来可探索更原则性的方法；
- **真实世界评估仍以定性/少量案例为主**：大规模真实场景量化评估仍是开放问题。

### 10.2 避障的边界说明

Qwen-RobotNav **没有显式碰撞检测模块**（如占据栅格、DWA），其避障行为是**隐式的**——基于视觉观测生成绕行的 waypoint 轨迹。

**实际部署建议：**
- Qwen-RobotNav 负责输出**高层路径意图**（语义可行的绕行动作）；
- 必须配合**底层安全控制器**（如 LiDAR/深度相机的局部碰撞检测、紧急停止）；
- 对于透明/反射表面（玻璃墙）、突然出现的动态障碍，不应完全依赖模型隐式避障。

这与论文中 Unitree Go2 的真实部署一致：Qwen-RobotNav 输出 waypoints 后，由底层控制栈执行并做安全监督。

### 10.3 未来方向

- 将观察上下文参数的学习也纳入模型优化，而非仅靠训练时随机化；
- 结合在线适应（online adaptation）与强化学习，提升闭环驾驶安全性；
- 扩展到更多 embodiment（无人机、机械臂移动平台）；
- 与三维场景记忆、语义地图更深度结合，进一步释放长程 Agentic 能力。

---

## 十一、总结

Qwen-RobotNav 代表了**视觉-语言-动作（VLA）导航模型**的一次重要演进。它没有追求更复杂的网络结构，而是敏锐地捕捉到：**导航任务的多样性本质上是对"观察历史如何使用"的多样性**。通过参数化观察编码接口、自然语言结构标识、大规模异构数据联合训练，以及面向 Agent 的工具化封装，Qwen-RobotNav 在多项基准上取得 SOTA，并展示了从仿真到真实机器人、从短程反应到长程推理的强大泛化能力。

对于希望构建通用具身 Agent 的研究者与工程师而言，Qwen-RobotNav 提供了一个极具参考价值的设计范式：**把导航模型做得足够"薄"而"可控"，让上层智能体决定何时、何地、以何种方式使用它**。
