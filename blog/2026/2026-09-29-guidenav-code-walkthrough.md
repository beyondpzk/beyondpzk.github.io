---
title: "GuideNav 源码解读：从示教关键帧到视觉复走"
date: 2026-09-29
categories: [VLN]
description: "沿着 Teach 与 Repeat 两条链路，解析关键帧选择、视觉定位、相对位姿估计与控制器。"
topic: navigation
type: 工程实践
summary: "沿着 Teach 与 Repeat 两条链路，解析关键帧选择、视觉定位、相对位姿估计与控制器。"
---

# GuideNav 源码解读：从示教关键帧到视觉复走

> **对象**：GuideNav（首个纯视觉 VT&R 导盲导航系统，HRI 2026 Systems Track Honorable Mention）
> **论文**：https://arxiv.org/abs/2512.06147
> **代码**：https://github.com/guidedogrobot-navigation/GuideNav （本文基于 main 分支 `c94abf7` 逐文件阅读整理）
> **平台**：Unitree Go2 + Intel RealSense D435i（仅用 RGB）+ Jetson AGX Orin，ROS2 Humble

---

## 〇、推理（Repeat）阶段 High-Level 流程

先给全貌，细节见后文。Teach 阶段只需牵着机器人走一圈，得到一串编号关键帧（`0.png ~ N.png`）+ 每帧的 CosPlace 检索指纹——**推理时唯一的目标就是"走到最后一帧 N"**，每一帧相机图像到来时执行：

```
当前相机帧 (RGB)
     │
     ▼
① CosPlace 检索：当前帧 → 512 维描述符，与地图全部关键帧比对
     │         "我现在最像地图里的哪一帧？"
     ▼
② Markov 信念更新：结合上一帧的位置信念（每帧最多前进 0~1 帧的运动先验）
     │              滤波出当前节点 closest_node_idx（防相似场景跳帧）
     ▼
③ 子目标选择：subgoal_idx = closest_node_idx + lookahead（默认往前看 1 帧）
     │         "下一步走到哪一帧的画面？"
     ▼
④ Reloc3r 相对位姿估计：（当前帧, 子目标帧）→ 相对 SE(2) 位姿 (x, y, yaw)
     │                   若子目标已在身后（x<0）→ 子目标 +1 重估
     ▼
⑤ (ρ,α,β) 非线性控制器：相对位姿 → 线速度 v / 角速度 w
     │                   （远处快、近处慢、大转角先转后走）
     ▼
⑥ 发布 /cmd_vel → Go2 执行
     │
     ▼
closest_node_idx 到达最后一帧？ → 是：停车，Goal reached！
                                否：等下一帧，回到 ①
```

一句话概括：**定位靠"画面检索 + 时序滤波"，导航靠"逐帧复现示教时的画面"，控制靠"当前画面与子目标画面的相对位姿"——全程无几何地图、无坐标、无避障规划。**

---

## 一、仓库结构与组件地图

```
GuideNav/
├── sensor/extract_data_two.py        # Teach：ROS2 记录图像流
├── topogen/gen_dinov3.py             # Teach：DINOv3 自适应关键帧选择（论文方法）
├── sensor/build_topomap.py           # Teach：里程计选帧的"朴素版"（非论文方法）
├── guidenav/navigate.py              # Repeat：主导航节点，唯一入口（~37KB）
├── guidenav/place_recognition/       # CosPlace 位置识别 + Bayesian 信念滤波
│   ├── extract_database.py           #   预计算地图描述符 → .h5
│   ├── feature_extractor.py          #   在线查询描述符
│   ├── gallery_db.py                 #   h5 描述符库读取
│   └── bayesian_querier.py           #   Markov 信念更新（默认）
│       sliding_window_querier.py     #   滑窗滤波器（备选）
├── guidenav/match_to_control/
│   ├── feature_match.py              #   Reloc3r 相对位姿估计封装
│   └── control.py                    #   (ρ,α,β) 非线性控制器
├── guidenav/models/pr_models/        # CosPlace 网络定义（EfficientNet + GeM）
├── config/{robots.yaml, models.yaml}
└── navigate.sh                       # 启动脚本模板
```

三个外部模型的引入方式（均非 submodule）：

- **CosPlace**：网络代码直接复制进仓库（HLoc 风格接口：EfficientNet_B0 → GeM(p=3) → Linear 1280→512，输出 512 维 L2 归一化描述符）；训练权重 `efficientnet_85x85.pth` 需从 Google Drive 手动下载放入 `model_weights/`；
- **Reloc3r**：手动 clone 到 `guidenav/match_to_control/methods/reloc3r`（含 croco 子模块），并应用 `third_party/reloc3r_load_images_in_memory.patch`（新增 `load_images_reloc3r()`，支持从内存传 numpy/PIL 图像而非读文件路径）；权重首次运行自动从 HuggingFace 下载（约 1.4GB）；
- **DINOv3**：必须本地 clone repo + 自带权重（`--dinov3-repo` + `--weights`）；不传则回退 DINOv2 `dinov2_vitl14`，但选出的关键帧与论文不同。

---

## 二、Teach 阶段：牵着走一圈 → 关键帧图

### 第 1 步：记录数据（`sensor/extract_data_two.py`）

ROS2 节点订阅 RealSense 的 RGB topic（深度与里程计可选，论文未使用），各 topic 独立回调、无时间同步，每帧以 9 位小数时间戳命名存 PNG：

```bash
python sensor/extract_data_two.py --output-dir ./data/teaching_run
# 输出: teaching_run/d435_color/*.png (+ d435i_color/ + depth/ + odom.csv 可选)
```

### 第 2 步：DINOv3 自适应关键帧选择（`topogen/gen_dinov3.py`，论文方法）

```bash
python topogen/gen_dinov3.py --input ./data/teaching_run/d435_color \
    --output ./data/topomap_raw --dinov3-repo /path/to/dinov3 --weights dinov3_vitl16.pth
```

每帧过 `dinov3_vitl16`（Resize 256 BICUBIC → CenterCrop 224 → ImageNet 归一化），特征 L2 归一化。论文的"三约束"对应 `AdaptiveKeyframeSelector` 的默认 `multi_criteria` 方法：

| 论文约束 | 代码实现 |
|---|---|
| 最小时间间隔 | 距上一关键帧 ≥ **5 帧**才可能入选 |
| 外观多样性 | 与上一关键帧余弦相似度低于阈值，**或**与最近 5 个关键帧的最小相似度低于阈值 |
| 轨迹进度 | 距上一关键帧超过 **50 帧强制选入**（兜底防漏） |

阈值随地图规模分三段自适应：

| 关键帧数量 | sim_threshold | diversity_threshold |
|---|---|---|
| < 20 | 0.95（最挑剔） | 0.92 |
| 20 ~ 100 | 0.93 | 0.90 |
| ≥ 100 | 0.91（保覆盖） | 0.88 |

备选选帧策略：`adaptive_threshold`（按最近 10 帧相似度标准差动态调整，clamp 到 [0.85, 0.98]）、`diversity_buffer`（对最近 20 个特征 buffer 最小相似度 < 0.90）、`temporal_spacing`（≥10 帧 + 相似度 < 0.92）。

### 第 3 步：地图格式

选出的帧重命名为 `0.png, 1.png, ... N.png`——**编号即路线顺序，这就是全部"拓扑"**，没有显式的边文件：

```
data/topomap/
├── 0.png  1.png  ...  N.png          # 单链关键帧，文件名解析为整数序号
└── global-feats-cosplace.h5          # 预计算的 CosPlace 描述符库（HLoc 格式）
```

`.h5` 由 `extract_database.py` 生成（借自 HLoc）：每张图一个 group，内含 512 维 float32 描述符和 `image_size`。预计算命令：

```bash
python -m guidenav.place_recognition.extract_database --topomap-dir ./data/topomap
```

`--img-size` 必须与导航时一致（默认 `85 64`）；导航首次运行时若缺该文件会现场提取。`gallery_db.py` 读取时按整数文件名排序，要求图是"非环非分叉的单链"。

**地图的全部内容只有三样**——顺序、特征、图片本身：

- **顺序**：编码在文件名里（`0.png → 1.png → ...`），没有边文件、没有邻接表，拓扑是隐式单链，"下一节点"就是 `idx + 1`；
- **特征**：每张图一个 512 维 CosPlace 描述符，用于回答"我现在最像哪一帧"；
- **图片本身**：用于 Reloc3r 计算"当前画面与子目标画面的相对位姿"。

**没有的东西**同样标定了能力边界：没有坐标（任何一帧都不知道自己在哪米）、没有尺度（Reloc3r 平移归一化）、没有几何（不存占据/深度）、没有语义（不知道哪帧是厨房）、没有分叉（单链，不支持环路和岔路）。

这个极简设计带来几个实际性质：

1. **地图可手工编辑**——删掉一张 jpg 就是删掉一段路，两段路线的图片目录拼起来重编号就是拼接路线，人类可理解可维护（SLAM 点云地图做不到）；
2. **迁移分享成本极低**——一个文件夹 + 一个 h5，几十 MB，拷给另一台机器人即可走同一条路线；
3. **所有"智能"都在运行时，地图本身是哑的**——语义标签、分叉边、占据栅格等扩展都只是给哑地图外挂新字段，不动核心结构：

```
topomap_plus/
├── 0.png  1.png  ...  N.png
├── global-feats-cosplace.h5
├── edges.json        ← 新增：分叉/环路（打破单链）
├── labels.json       ← 新增：每帧 VLM 语义标签（"厨房"）
└── occupancy.pgm     ← 可选：沿线占据栅格（frontier 补探索）
```

一句话：**它本质上是"一条路线的相册"，不是"一个环境的模型"**——单链相册支持"复走"，不支持"规划一条没走过的新路"；后者需要引入真正的图结构和语义层（见 [GuideNav 与 VLA 融合：记忆、目标接口与局部执行](/blog/2026/2026-09-29-guidenav-vla-integration) 模式 1）。

**Teach 阶段的本质**：不做任何几何重建、不算位姿、不建占据图——只是把一条路线压缩成一串有代表性的画面 + 每帧一个 512 维检索指纹。1 公里约 24 MB。

### 关键帧密度与重叠要求（示教操作规范）

**帧率不是固定值，是"帧数 + 外观变化"双约束**（选帧器全部以帧数为单位，不看时间也不看里程计）：

| 约束 | 代码值 | 含义 |
|---|---|---|
| 最小间隔 | ≥ 5 帧 | 两个关键帧最密也只能隔 5 帧 |
| 强制间隔 | > 50 帧必选 | 再单调的走廊，最多隔 50 帧也要存一帧 |
| 外观触发 | 相似度 < 0.91~0.95 | 画面变化大（转弯、进门）就立即加帧 |

按 30fps 记录、人牵机器人约 0.5~1 m/s 行走换算：

- **直线走廊**：外观不变，靠 50 帧强制采样兜底 → 约每 0.8~1.7 米一个关键帧；
- **转弯/过门**：相似度阈值触发，贴近 5 帧最小间隔 → 每 3~8 厘米一个关键帧，转弯处非常密集；
- 佐证：论文"1km 地图 24MB"≈ 240~480 个关键帧，即平均每 2~4 米一个。

**重叠是硬性要求**，两个环节依赖程度不同：

1. **Reloc3r 相对位姿估计——强依赖**（真正的约束来源）。它是 CroCo/DUSt3r 系的对应点回归网络，靠两张图的共视区域建立匹配。lookahead=1 意味着当前帧永远要和"前一帧 +1 位置"的关键帧保持共视；转角处帧间距必须小到视线还没完全断开，否则估计失败（返回 None → 发零速）。
2. **CosPlace 检索——弱依赖**。全局描述符对视角变化有容忍度，但至少要是"同一个地方"；Markov 先验（窗口 [-1, 2]，每帧最多前进 0~1 个节点）也隐含假设始终处于相邻关键帧的重叠视场内。

**示教三条经验法则**：

1. 转弯要慢、原地转着拍——让外观触发机制在转角自然加密关键帧，保证任意时刻当前帧与前后关键帧都有共视；
2. 避免运动模糊和快速甩头——模糊帧描述符质量差，还会被误判为"外观变化"而入选；
3. 避免在光照极端变化时示教——重叠假设不包括"同位置完全不同外观"。

**存储量完全不是问题**（对"不是要存好多图片"的回答）：

- 论文实测 **1 公里 ≈ 24 MB**（240~480 张 JPEG）；一个家庭户型的示教路径通常只有 50~200 米 → **几十到几百张图，几 MB 到几十 MB**；
- 描述符库更小：每帧 512 维 float32 = 2KB，一千个关键帧也才 2MB；
- 如需进一步压缩，关键帧可按 Reloc3r 输入尺寸（512）降采样存储，CosPlace 端只需 85×64——原始分辨率在推理时没有任何用处。

### 选帧是全自动的（离线后处理，零人工判断）

示教时人只管牵着走，选帧是走完之后的后处理步骤，不需要任何人工挑帧。Teach 阶段人的操作只有三步：

```bash
# 1. 牵着走一圈，程序原样记录所有帧（30fps 全存，此时还不选）
python sensor/extract_data_two.py --output-dir ./data/teaching_run

# 2. 走完后跑选帧脚本，自动挑出关键帧（全自动，无人工干预）
python topogen/gen_dinov3.py --input ./data/teaching_run/d435_color --output ./data/topomap_raw ...

# 3. 一个 shell 循环把 keyframe_000000.jpg 重命名成 0.jpg, 1.jpg, ...
```

第 2 步里 `multi_criteria` 选择器自动完成所有判断：5 帧最小间隔、相似度阈值触发、50 帧强制兜底、阈值随地图规模三段自适应——全部预设好，输出 `keyframe_*.jpg` + `selection_log.txt`（每帧为什么选/没选的记录，可复查）。首次启动导航时 CosPlace 描述符库也会自动提取（缺 `.h5` 就现场算）。

即：**示教过程零操作负担（走就完了），选帧零人工判断（脚本一键跑完）**。

两个细节：

- **是后处理而非在线选帧**：走的时候 30fps 全量落盘，选帧发生在走完之后。若要"边走边出图"（示教完立刻可用），需把选择器改成在线模式——逻辑本身可流式跑（只依赖当前帧特征 + 最近几个关键帧特征），只是代码没这么组织；
- **阈值是定死的默认值**：0.91~0.95 那套阈值对论文的室外路线调过，换到室内（纹理少、走廊相似）可能选出偏稀或偏密的图，可换备选策略（`adaptive_threshold` 按最近 10 帧相似度波动自动调阈）或微调参数——这是室内化改造时大概率要动的地方。

---

## 三、Repeat 阶段：每个控制周期做什么

### 启动

```bash
python guidenav/navigate.py --robot go2 --topomap-base-dir ./data -d topomap \
    --model-weight-dir model_weights --robot-config-path config/robots.yaml
# 离线回放：加 --offline-images --img-dir ... --offline-fps 3
```

初始化（`GuideNavNode.__init__`）：加载 topomap 全部图片（按整数文件名排序），goal = 最后一帧；`init_reloc3r()` 加载 Reloc3r-512；构建 CosPlace `FeatureExtractor` + Bayesian `PlaceRecognitionTopologicalFilter`；创建 `/cmd_vel` publisher。**循环由相机帧驱动**（无独立定时器，`robots.yaml` 里 Go2 配 4Hz），每帧执行一次 `navigate_one_step`，共 6 步：

### 第 1 步：位置识别 —— "我现在最像地图里的哪一帧"

当前帧预处理（`ToTensor → Resize((64,85)) → ImageNet normalize`）后过 CosPlace 得 512 维描述符 `q`，与库中全部描述符算余弦距离 `d = √(clip(2 − 2·qᵀgᵢ, 0))`。

### 第 2 步：Markov 信念更新（`bayesian_querier.py`）

时间一致性滤波，防止感知混叠（两个相似路口跳错帧）：

- **初始化**：`belief ∝ exp(−λ₁·d)`，其中 `λ₁ = ln(δ) / (Q97.5(d) − Q2.5(d))`，δ=10（`--filter-delta`）——按距离分布的 2.5%/97.5% 分位数自适应缩放，保证不同地图下判别力一致；
- **预测步**：用长度 3 的全 1 核（窗口 `[-1, 2]`）对对称 padding 后的信念做 valid 卷积——编码"机器人每帧最多前进 0~1 个节点"的运动先验；
- **测量步**：`belief *= exp(−λ₁·d(q, gᵢ))`，归一化；
- 取 `argmax(belief)` 为当前节点 `closest_node_idx`。

单次错误匹配会被后续观测稀释，因此系统不需要显式的"丢失定位恢复"模块。备选 `sliding_window` 模式：只在 `[closest−2, closest+3)` 窗口内取最近者。

### 第 3 步：子目标选择

```python
subgoal_idx = min(closest_node_idx + lookahead, goal)   # --lookahead 默认 1
```

永远朝"当前位置再往前一帧"的画面走；goal 默认地图最后一帧（`--goal-node-idx -1`）。

### 第 4 步：Reloc3r 相对位姿估计（`match_to_control/feature_match.py`）

Reloc3r-512 输入（当前帧, 子目标帧）图像对（经补丁函数 `load_images_reloc3r(size=512)` 预处理），输出相对 SE(3) `pose2to1`，提取 SE(2)：

```python
x_rel = t[2]                      # 相机 z 轴前向
y_rel = -t[0]
yaw   = atan2(-R[0,2], R[2,2])    # 度
```

**关键细节：平移被归一化为单位长度**（`t /= ‖t‖`）——单目图像对无法恢复尺度，控制器拿到的 ρ 不是真实米制距离，只是"方向 + 伪距离"。

若 `x < 0`（子目标已在身后）→ `subgoal_idx += 1` 重新估计，直到目标在前方或超出地图（超出则发零速）；估计失败（返回 None）也发零速并计数。

### 第 5 步：非线性控制器（`match_to_control/control.py` 的 `vtr_controller`）

论文的 "Lyapunov 稳定控制" 对应经典 (ρ, α, β) 极坐标调节律（默认 `k_rho=0.8, k_alpha=0.8, k_beta=−0.4, position_tol=0.08, heading_tol=5°`），叠加启发式整形：

```
ρ = √(x²+y²),  α = wrap(atan2(y,x)),  β = wrap(θ − α)

ρ < 0.08 且 |θ| < 5°  → 到达子目标：v = w = 0
ρ < 0.08              → 原地旋转对正朝向：v=0, w = clip(1.5·k_β·θ, ±0.3·w_max)
否则：
  v_raw = v_max·(1 − e^(−k_ρ·ρ))       # 指数整形：远处快、近处自动减速
  w_raw = k_α·α + k_β·β                # Lyapunov 航向律
  ρ < 0.5    → v_raw *= max(0.4, ρ/0.5)        # 末端减速
  |α| > 30°  → v_raw *= max(0.5, 30°/|α|)      # 大转角先转后走
  ρ < 0.25   → w_raw = k_α·α + 1.3·k_β·β       # 增强末端姿态修正
  最后按 min(1, v_max/|v_raw|, w_max/|w_raw|, 0.9) 统一缩放
```

速度上限（`config/robots.yaml`）：Go2 `max_v = 0.15 m/s, max_w = 1.0 rad/s`（导盲场景刻意很慢）；发 `Twist(linear.x=v, angular.z=w)` 到 `/cmd_vel`。

### 第 6 步：终止与异常

- `closest_node_idx >= goal_node_idx` → `Goal reached!`，发零速，置 `navigation_active=False`；
- 每步 try/except，任何估计失败发零速；
- **没有专用局部规划器/避障**——VT&R 的哲学是"沿记忆的路径走"，安全依赖示教路线本身 + 人握刚性挽具在环内。

---

## 四、关键参数速查

| 参数 | 默认值 | 位置 |
|---|---|---|
| CosPlace 输入尺寸 | 85×64（宽×高） | `--img-size` |
| DINOv3 输入 | 224×224（CenterCrop） | `gen_dinov3.py` |
| Reloc3r 输入 | 512 | `feature_match.py` |
| 描述符维度 | 512（L2 归一化） | `models.yaml` |
| Bayesian δ | 10 | `--filter-delta` |
| 信念转移窗口 | [-1, 2]（长度 3 均匀核） | CLI 默认 |
| lookahead | 1 帧 | `--lookahead` |
| 到达阈值 | ρ<0.08 且 \|θ\|<5° | `vtr_controller` |
| 控制增益 | k_ρ=0.8, k_α=0.8, k_β=−0.4 | `vtr_controller` |
| Go2 速度上限 | 0.15 m/s, 1.0 rad/s | `robots.yaml` |
| 上下文队列 | 5 帧攒齐才开始输出 | `context_size=4`（ViNT 遗产，实际未使用） |

---

## 五、代码与论文表述的出入（读代码才发现的）

1. **平移无尺度**：Reloc3r 输出被归一化成单位向量，控制器的 ρ 不是米制距离——靠指数整形让"伪距离"也能稳定工作；
2. **"5Hz 闭环"没有定时器**，就是相机帧驱动，Go2 配置里实际写的是 4Hz；
3. **"轨迹进度"约束名不副实**：`gen_dinov3.py` 里只有"50 帧强制采样"，没有里程计意义上的进度约束（按 0.5m/15° 选帧的版本在未使用的 `build_topomap.py` 朴素构图器里，它需要 depth + `odom.csv`）；
4. **死代码遗产**：GNM/ViNT 相关的 waypoint 模型参数（`--wp-model`、`--subgoal-mode`）已移除但参数残留；`--use-smoothing` 未暴露为 CLI（对应代码是死路径）；Go2 SDK 不在仓库内，系统只发 `/cmd_vel`，由外部驱动执行；
5. **不提供**示例 topomap、rosbag 与 DINOv3 权重（需自行申请）；CosPlace 权重在 Google Drive。

---

## 六、对室内 teach-and-repeat 改造的启示

- **可直接复用**：整个 Repeat 管线（CosPlace + Bayesian 滤波 + Reloc3r + 控制器）与室内外无关，搬到室内零改动；Teach 的 DINOv3 选帧同理；
- **需要调的参数**：室内关键帧更密时 `lookahead=1` 可加大；到达阈值（0.08 / 5°）与速度上限按机器人平台调整；
- **需要补的层**：地图目前是"无语义的单链"，"去厨房"这类语义目标需在关键帧上叠加 VLM 打标 + 检索层；分叉/回路需把单链升级为真正的图（加边文件 + 图搜索），替代现在的"序号 +1 即下一路段"；无避障能力，室内动态障碍需要额外的安全层。
