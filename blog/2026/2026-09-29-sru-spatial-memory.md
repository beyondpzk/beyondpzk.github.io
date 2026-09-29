---
title: "SRU：面向长程无地图导航的空间循环记忆"
date: 2026-09-29
categories: [VLN]
description: "解析空间变换门、循环记忆训练方法，以及隐式空间记忆与显式拓扑记忆的互补关系。"
---

# SRU：面向长程无地图导航的空间循环记忆

> **论文**：*Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning*
> **作者**：Fan Yang（通讯）、Per Frivik、David Hoeller、Chen Wang、Cesar Cadena、Marco Hutter
> **机构**：ETH Zurich Robotic Systems Lab (RSL) + University at Buffalo Spatial AI & Robotics Lab
> **发表**：IJRR 2025（arXiv v1 2025-06-06，v2 2025-09-04 即期刊版）
> **链接**：[arXiv:2506.05997](https://arxiv.org/abs/2506.05997) ｜ [项目主页](https://michaelfyang.github.io/sru-project-website/) ｜ [代码](https://github.com/leggedrobotics/sru-pytorch-spatial-learning)（MIT）
>
> **一句话总结**：论文发现标准 RNN（LSTM/GRU/S4/Mamba）在论文的受控实验中更擅长时间任务，而空间坐标回忆表现较弱。SRU 只给循环单元加了一个"空间变换门"，就让端到端 RL 导航在 IsaacLab 四类环境平均成功率 +23.5%（楼梯场景翻倍），并零样本迁移到 Unitree B2W 真机完成百米级无地图导航。

---

## 一、研究问题与动机

### 1.1 问题：长程无地图导航的记忆瓶颈

端到端 RL 导航（mapless navigation）通常用 RNN 把历史观测隐式压缩进隐状态，让网络同时学会"隐式建图 + 规划"。这种方式没有显式建图管线的延迟和启发式规则，但论文指出了一个被长期忽视的根本缺陷：

> **经典建图的本质操作是 SE(3) 齐次变换**——把不同时刻、不同视角下的自我中心观测注册到统一参考系：
>
> $$\begin{bmatrix}\mathbf{p}'\\1\end{bmatrix}=\begin{bmatrix}R & \mathbf{t}\\ \mathbf{0}^\top & 1\end{bmatrix}\begin{bmatrix}\mathbf{p}\\1\end{bmatrix}$$
>
> 而标准 RNN 的门控结构是为"记住事件顺序"设计的，**缺少针对空间配准的显式结构归纳偏置**。

### 1.2 受控实验证据（论文附录 A，即开源仓库复现的实验）

论文设计了一个干净的探针实验：输入 landmark 坐标（当前机体系）+ 类别标签 + 自我运动矩阵 $M_t^{t-1}$，T 步后要求网络①按顺序回忆标签（时间任务）②回归所有 landmark 在**最终坐标系**下的坐标（空间任务）。结果：

- LSTM、GRU、S4、Mamba-SSM **全部**能完美完成时间任务；
- **全部**在空间任务上失败（MSE 很高）——S4/Mamba 这类线性递归结构甚至更差。

这直接解释了为什么纯 RNN 导航在需要回溯、绕行、空间推理的环境（楼梯、死胡同）里表现崩溃。

### 1.3 现有路线的不足

| 路线 | 代表 | 问题 |
|---|---|---|
| 显式建图 + 历史路径 | EMHP（Lee et al. 2024） | 管线延迟、固定上下文窗口等启发式，行程 >40m 成功率骤降 |
| 堆叠历史帧 + Transformer | GTRL（Huang et al. 2023） | 帧数是拍脑袋选的，计算量随历史长度平方增长 |
| 经典搜索/采样规划 | — | 依赖预建地图，难应对未知/动态环境 |

---

## 二、核心方法：SRU（Spatial Recurrent Unit）

### 2.1 空间变换门

SRU 对 LSTM/GRU 的改动极小：增加一个**只依赖当前输入**的空间变换门 $s_t$，以逐元素乘（"star operation"，灵感来自齐次变换的乘性形式）调制候选状态：

**SRU-LSTM**：

$$s_t = W_{xs}x_t + b_s, \qquad g_t = \tanh\big(s_t \odot (W_{xg}x_t + W_{hg}h_{t-1} + b_g)\big)$$

**SRU-GRU**：

$$s_t = W_{xs}x_t + b_s, \qquad \tilde{h}_t = \tanh\big(s_t \odot (W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h)\big)$$

**SRU-Ours**（最终版，再加 refine gate 缓解门饱和）：

$$r_t = i_t \odot \big(1-(1-f_t)^2\big) + (1-i_t)\odot f_t^2, \qquad c_t = r_t \odot c_{t-1} + (1-r_t)\odot g_t$$

直觉：$s_t$ 从当前输入（含本体状态，即自我运动信息）中学出"如何把历史隐状态旋转/平移到当前坐标系"，让记忆在空间上对齐后再写入。

### 2.2 完整导航架构

```
深度图 → RegNet+FPN 编码器（TartanAir 合成数据 VAE 预训练）
       → 自注意力（全局上下文）
       → 交叉注意力（query = 本体状态 + 相对目标，压缩为 C×1 特征）
       → SRU（融合特征 + 本体状态 + 隐状态）
       → MLP + TC-Dropout → 线/角速度指令（5Hz）
                                     ↓
                        下层学习运动控制器（50Hz）
```

- **本体状态**：线速度、角速度、投影重力、上一动作——它同时扮演了"自我运动 $M_t^{t-1}$"的角色；
- **深度编码器**是关键组件：真实图与 RL 图隐特征的 Mahalanobis 距离中位数，仅用 RL 图训练为 1.15 → 大规模合成预训练 0.82 → 再加深度噪声 0.69。

### 2.3 训练：端到端 RL（无蒸馏、无课程）

- IsaacLab 仿真，**Asymmetric Actor-Critic + PPO**（critic 可见 360° 高度扫描和干净观测，actor 只用加噪深度图）；
- **稀疏时间奖励**：episode 60s，奖励窗口 2s，加小概率随机检查鼓励提前到达：$r^{task}_t = \frac{\mathbf{1}(\cdot)}{1+\|p_t/\sigma\|_2}$；
- **两个关键正则化**（消融证明是释放 SRU 潜力的关键）：
  - **Deep Mutual Learning (DML)**：两个策略并行 PPO + KL 互蒸馏，防止 PPO 过早收敛到"只用易学的时间特征"的次优解；
  - **TC-Dropout**：rollout 与训练跨时间步共享同一 dropout mask，稳定循环记忆学习；
- **Sim-to-real**：可并行化深度噪声模型（边缘噪声/填充噪声/量化噪声）+ 本体观测加噪 + 目标表示为单位方向 + log 距离（泛化到远距离）。

---

## 三、实验结果

### 3.1 仿真（IsaacLab，4800 episodes × 120 个未见环境）

**不同循环单元对比（Table 1，成功率 %）**：

| 模型 | Maze | Pillar | Stair | Pit | Overall |
|---|---|---|---|---|---|
| GRU | 68.1 | 73.6 | 35.7 | 66.7 | 61.0 |
| LSTM | 70.3 | 78.2 | 33.1 | 72.7 | 63.5 |
| SRU-GRU | 73.1 | 78.8 | 74.1 | 74.8 | 75.2 |
| SRU-LSTM | 75.9 | 76.7 | 79.3 | 74.1 | 76.5 |
| **SRU-Ours** | 76.0 | 81.0 | 82.8 | 75.6 | **78.9** |

仅 SRU 改动平均 +21.8%，SRU-Ours 总 +23.5%；**楼梯环境成功率翻倍以上**（最需要 3D 空间记忆的场景）。

**对比基线（Table 2，Overall SR%）**：GTRL 38.2 → EMHP 60.4 → **Ours 78.3**（vs EMHP +29.6%）。把 GTRL 的记忆换成 SRU（GTRL*）从 38.2 → 66.3，证明增益来自记忆单元而非架构其他部分。

**长距离泛化**：训练最大起止距离仅 30m，但 SRU 在 50m 内 SR>80%、**120m 内 >70%**；EMHP 超过 40m 后骤降（episode 时间加倍也救不回）。

**关键消融**：

- 注意力（Table 3）：无注意力 50.5 → GoT 68.4 → 本文空间注意力 78.9；
- **DML（Table 4）**：SRU-Ours 无 DML 65.7 → 有 DML **78.9**（相对 +24.3%）——正则化让网络真正用上空间记忆而非走时间捷径；
- TC-Dropout（Table 5）：77.2 → 78.9。

### 3.2 真机（零样本 sim-to-real，无真实数据微调）

- **平台**：Unitree B2W 轮腿机器人 + ZEDX 双目深度 + Jetson AGX Orin，LiDAR 惯性状态估计提供本体状态与相对目标，运动控制策略来自 RIVR；
- **场景**：办公室（含动态封堵死胡同测试：**SRU 成功回溯改路，LSTM 在死胡同间循环**）、校园主厅、户外露台、森林；
- **里程**：单目标 >70m、单次任务 >100m（训练时最大 30m）。

---

## 四、代码仓库解析

### 4.1 仓库定位与结构

[leggedrobotics/sru-pytorch-spatial-learning](https://github.com/leggedrobotics/sru-pytorch-spatial-learning)（MIT）**只含附录 A 受控实验与 SRU 单元实现**，README 明确声明不含端到端导航系统、感知管线与真机部署代码：

```
network/lstm_sru.py        # SRU-LSTM
network/lstm_sru_gate.py   # SRU-Ours（含 refine gate）
network/gru_sru.py         # SRU-GRU
network/vanilla_mamab.py   # Mamba 基线（文件名拼写如此）
network/s4_utils/          # S4 基线
dataloader/points_dataset.py   # 随机轨迹点云数据集
dataloader/spiral_dataset.py   # 螺旋轨迹数据集（Fig.2 实验）
run_pointcloud.py          # 训练/评估入口（--train / --wandb）
params/pointcloud.yaml     # hidden 512, seq_len 16, batch 128, 1000 epoch
```

代码要点：SRU 为手写 cell（合并门线性层 + 独立 `transform_gate = nn.Linear`，正交初始化，遗忘门 bias 初始化为 1+噪声），支持多层堆叠；损失 = MSE（坐标）+ 10×BCE（标签）；NAdam 优化器。**未提供预训练权重和数据集**（在 .gitignore 中）。

### 4.2 完整系统的关联组件

| 仓库 | 内容 | 论文组件 |
|---|---|---|
| `sru-pytorch-spatial-learning`（本仓库） | SRU 单元 + 受控实验 | SRU 单元 ✅ |
| `sru-navigation-sim` | IsaacLab 任务扩展、深度观测、稀疏奖励 | 仿真环境 ✅ |
| `sru-navigation-learning` | RL 训练框架（rsl_rl、DML） | PPO/DML/TC-Dropout ✅ |
| `sru-depth-pretraining` | 深度编码器预训练（号称含预训练模型） | RegNet 编码器 ✅ |
| `sru-robot-deployment` | Unitree B2W + ZEDX 部署（ROS2） | 真机系统 ✅ |

上表列出调研笔记中的关联仓库名称；组件是否公开、权重是否可用及版本兼容性，应以项目主页与相应仓库说明为准，不能由单元实验仓库推断完整系统可以直接复现。

### 4.3 安装与复现

```bash
pip install -e .   # 包名 sru-pytorch，需 Python≥3.8、PyTorch≥2.0
# 依赖：pypose mamba-ssm causal-conv1d matplotlib scipy scikit-learn wandb
python run_pointcloud.py --train   # 复现附录 A 实验
```

---

## 五、局限性（论文自述 + 客观观察）

- RNN 本质存在指数记忆衰减，论文的 "long-range" 指**超出局部感知半径（>10m）的局部无地图导航**（百米量级），公里/小时级全局导航仍需全局地图——作者自己明确承认；
- SRU 隐式记忆"到底记住了什么"不可解释（XAI 问题）；
- 真机依赖外部 LiDAR 状态估计提供相对目标与本体状态——**并非完全纯视觉自主**；
- 提升集中在需要 3D 空间记忆的场景（楼梯翻倍），平地环境提升温和；
- GRU 训练不稳定（论文只统计成功 run）。

---

## 六、对导航系统设计的启示（与 teach-and-repeat 路线的关系）

SRU 与 GuideNav VT&R 是**同一问题的两种记忆哲学**，正好互补：

| | SRU（隐式空间记忆） | GuideNav（显式视觉记忆） |
|---|---|---|
| 记忆形式 | RNN 隐状态，学到什么不可控 | 关键帧相册，可解释可编辑 |
| 有效范围 | 百米内、单 episode | 公里级、跨 episode 持久 |
| 应对未探索区域 | ✅ 端到端泛化、死胡同会回溯 | ❌ 只能复走示教路线 |
| 部署成本 | 一个前向推理 | 检索 + 位姿估计管线 |

值得借鉴到 [室内巡逻执行流程：从示教建图到局部绕障](/blog/2026/2026-09-29-indoor-patrol-execution) 方案的三点：

1. **局部避障/回溯执行器可以换成 SRU 策略**：它比 NoMaD 这类短时上下文策略多了百米级空间记忆，"绕障后回路线""死胡同掉头"这类行为是内生能力而非外部记忆管理；
2. **空间对齐问题也值得在 VLA 中单独研究**：ABot-N0/Qwen-RobotNav 用 Transformer 上下文装历史帧，其历史融合也涉及空间对齐，但不能直接沿用 RNN 探针实验的结论；SRU 的空间变换门思想（或显式位姿条件化）可作为 VLA 记忆增强的参考；
3. **DML + TC-Dropout 的训练配方**：任何用 RL/循环记忆训练导航策略的项目都可直接借用，消融显示这是释放空间记忆潜力的关键（+24.3%）。
