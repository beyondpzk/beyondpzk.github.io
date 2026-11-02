---
title: 高速视觉避障：安全屏蔽强化学习技术报告
date: 2026-11-02
paperDate: 2026-02-09
topic: navigation
type: 论文精读
tags: [无人机, PPO, CBF, HOCBF, Sim-to-Real]
summary: 训练阶段用 Dijkstra 距离场和 CBF 奖励学习绕障，部署阶段用 HOCBF 安全过滤器修正控制。实飞最高速度达 7.5 m/s，但高速仿真成功率仍明显下降，安全保证依赖模型、感知与约束可行性。
draft: true
---

# 高速视觉避障：安全屏蔽强化学习技术报告

## 核心结论

论文 *High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning* 提出“学习策略 + 模型安全约束”的无人机高速避障框架：**Dijkstra 负责训练中的全局导航引导，CBF 奖励鼓励安全行为，HOCBF 在部署时修正危险动作。** 实飞最高速度达 **7.5 m/s**。

阅读范围：论文 arXiv v1 正文及原始理解笔记；未核对作者代码、未复现实验。

## 方法与证据

### 1. 输入、网络与控制输出

训练采用 **PPO + 非对称 Actor–Critic**。Actor 使用带噪观测，Critic 使用训练期无噪声真值，辅助价值估计；部署只需要 Actor 和安全过滤器。

| 模块 | 内容 |
| --- | --- |
| 视觉输入 | 100 × 60 深度图，经 CNN 提取特征 |
| 状态输入 | 机体系速度、姿态旋转矩阵、上一时刻动作、归一化水平目标位移、当前高度与目标高度 |
| 时序融合 | 视觉特征与状态拼接 → GRU → MLP，输出连续动作的高斯分布参数 |
| 策略动作 | 质量归一化总推力、期望横滚角、俯仰角和偏航角 |
| 执行链路 | 策略指令 → HOCBF 修正 → PX4 跟踪 → 电机控制 |

控制指令更新频率为 **100 Hz**。（§III-A、III-B，Fig. 2）

### 2. 训练：全局引导与安全奖励

**Dijkstra 导航奖励。** 在训练地图上预计算到目标的最短可行路径距离场，用三线性插值得到当前位置的代价 $V_t$：

$$
r_{\mathrm{nav},t}=\lambda\,\operatorname{clip}(V_{t-1}-V_t,-C,C).
$$

沿可行路径接近目标即可获得奖励，因此绕过 U 形障碍、暂时远离目标的行为也能得到正确引导。距离场用于训练奖励，部署无需在线运行 Dijkstra。（§III-C，式 4）

**CBF 安全奖励。** 由训练地图的 ESDF 得到最近障碍距离 $d(x)$，定义安全余量：

$$
h(x)=d(x)-d_{\mathrm{safe}},\qquad
\dot h(x)=\nabla d(x)^{\top}v,
$$

$$
r_{\mathrm{safety}}=
\operatorname{clip}(\dot h+\gamma h,-2,0).
$$

这里的 $\gamma$ 是 CBF 系数。奖励同时考虑距离和接近障碍的速度：即使尚未碰撞，过快逼近障碍也会受罚。它是软惩罚，不能强制动作安全。（§III-C，式 5）

总奖励还包含动作幅度与变化惩罚、碰撞惩罚、到达奖励，以及速度和高度约束。PPO 是策略优化工具，论文的关键增量在奖励设计与部署安全过滤。

### 3. 部署：HOCBF 安全过滤

实时深度图重建局部点云，提取最近障碍点。以相对位置 $r$ 定义：

$$
h(r)=\|r\|^2-r_{\mathrm{safe}}^2.
$$

考虑位置受加速度影响的二阶动力学，要求：

$$
\ddot h+\alpha_1\dot h+\alpha_0h\geq0.
$$

过滤器求解二次规划，在满足约束的前提下尽量保留策略原始指令：

$$
u^*=\arg\min_u\frac12\|u-u_{\mathrm{raw}}\|^2
\quad\text{s.t. HOCBF 安全约束成立}.
$$

约束满足时保留原动作，违反时修正动作。这里用 $u$ 统称优化变量；论文式 8 按加速度推导线性约束，而策略输出是推力与姿态角，两者的具体转换需要进一步核对实现。（§III-D，式 6–8）

### 4. 训练与实机配置

- **仿真训练**：Isaac Lab，1,000 架无人机并行，16 个难度递增的程序生成场景；进入目标半径 5 m 内判定成功。
- **迁移处理**：训练加入深度像素丢失、噪声与延迟随机化；部署采用 Navier–Stokes 图像修补处理深度缺失。
- **实机系统**：RealSense D435i、Orin NX、PX4；VIO 与 IMU 经 EKF 融合提供状态估计，深度截断到 7 m。

### 5. 关键实验

**消融实验：** 8 个未参与训练的场景，每场景 50 次试验。下表为不同目标速度下的导航成功率。（Table I）

| 配置 | 3 m/s | 5 m/s | 7 m/s | 9 m/s |
| --- | ---: | ---: | ---: | ---: |
| 欧氏距离奖励 | 51.75% | 35.00% | 21.25% | 0% |
| Dijkstra | 62.25% | 67.75% | 56.25% | 43.75% |
| Dijkstra + CBF 奖励 | 84.50% | 94.00% | 52.25% | 37.25% |
| 完整方法，含 HOCBF | **88.75%** | **100%** | **75.00%** | **47.50%** |

Dijkstra 明显改善可达性；CBF 奖励在中低速有效，但在 7、9 m/s 下低于仅用 Dijkstra 的结果。加入 HOCBF 后，7 m/s 成功率由 52.25% 升至 75%，提高 **22.75 个百分点**。

**基线比较：** 另一组 4 个测试环境中，目标速度 7 m/s 时，完整方法、DiffPhys、EGO-Planner 的成功率分别为 **42.5%、22.5%、0%**。这是不同测试集，不能与消融表混用。（Table II）

**实飞：** 室内进行 15 m 障碍通道往返飞行；室外森林飞行距离超过 35 m，平均速度约 **7 m/s**，论文报告最高速度 **7.5 m/s**。室外策略调整了训练障碍密度与目标速度上限。（§IV-B，Fig. 6–7）

## 局限与个人判断

- **安全保证有条件。** HOCBF 的前向不变性依赖初始安全条件、模型与状态估计准确性、约束可行性及执行跟踪。仅凭形式约束不能推出真实系统绝不碰撞；9 m/s 下完整方法的消融成功率仍只有 47.5%。
- **实飞证据有限。** 正文未给出完整实飞成功率统计与置信区间；7.5 m/s 表明已展示的速度能力，不代表该速度下任意障碍环境都可靠。
- **复现仍有缺口。** 正文未完整披露 PPO 超参数和奖励权重；安全过滤器的动作映射、执行器限制及 QP 不可行时的处理需要核对。动态障碍与细小障碍也是作者列出的后续方向。

**对原笔记的修正：** 原笔记中的 Python 程序是教学基线，使用球形障碍、简化动力学、欧氏距离奖励以及“加速度 + 偏航角速度”动作，未实现 Dijkstra 距离场和 HOCBF。它可以解释 PPO 训练流程，不能作为本文完整方法的复现。

**个人判断：** 最值得借用的是“全局信息辅助训练、局部观测驱动策略、显式约束修正控制”的分工。验证时应同时记录成功率、碰撞率、实际速度和过滤器介入情况，判断性能提升来自策略学习还是在线修正。

## 参考资料

- Jiarui Zhang 等，*High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning*，arXiv:2602.08653v1，2026-02-09。[正文](https://arxiv.org/html/2602.08653v1) · [alphaXiv](https://www.alphaxiv.org/abs/2602.08653)


## 原始笔记：完整公式、训练代码与动作说明

以下保留原笔记的全部实质内容。代码属于教学基线，与论文方法的差异见前文。

下面给你一份**完整的单文件训练代码**，注释中的 `[公式 X]` 与代码前的公式编号一一对应。代码包含程序生成的简化飞行环境、深度图 CNN、带 GRU 的 Actor、非对称 Critic、奖励计算和 PPO 更新，可直接运行。

先区分一下：这份代码是**可运行的 PPO 教学基线，不是论文作者的原始代码**。论文使用 Isaac Lab、Dijkstra 代价图和部署时的 HOCBF 安全过滤器；论文没有公布完整 PPO 超参数和所有奖励权重。为了让示例无需下载仿真器或数据集也能运行，代码用简化的 3D 球形障碍环境，并用目标距离进展近似导航奖励。[Training Setup](https://www.alphaxiv.org/abs/2602.08653?page=5)[Navigation Reward](https://www.alphaxiv.org/abs/2602.08653?page=4)

保存为 `train_flight_ppo.py`，安装依赖并运行：

```bash
pip install numpy torch
python train_flight_ppo.py
```

### 公式与代码注释对应表

**公式（1）：Actor 的观测。**深度图与本体状态一起输入 Actor：

$$
o_t = \left(D_t,\;p_t\right),
\qquad
p_t =
\left[
v_t,\,
\sin\psi_t,\,
\cos\psi_t,\,
a_{t-1},\,
\frac{g_t-x_t}{20},\,
\frac{z_t}{4},\,
\frac{z_{\text{goal}}}{4}
\right].
$$

代码中的 `make_actor_observation` 及 `ActorCritic.actor_step` 实现这个输入接口。

**公式（2）：视觉编码与时序融合。**

$$
f_t^{\text{CNN}} = \operatorname{CNN}(D_t),
\qquad
h_t = \operatorname{GRU}
\left(
\operatorname{MLP}\left([f_t^{\text{CNN}},p_t]\right),
h_{t-1}
\right).
$$

代码中的 `self.cnn`、`self.fusion` 和 `self.gru` 对应此式。

**公式（3）：连续动作分布。**策略先产生高斯分布，再用 $\tanh$ 将动作限制在 $[-1,1]$：

$$
u_t \sim \mathcal{N}(\mu_t,\sigma_t),
\qquad
a_t=\tanh(u_t),
$$

$$
\log\pi(a_t\mid o_t)
=
\sum_i\log\mathcal{N}(u_{t,i};\mu_{t,i},\sigma_{t,i})
-\sum_i\log(1-a_{t,i}^2+\varepsilon).
$$

代码中的 `sample_squashed_gaussian` 实现动作采样及修正后的 log-probability。

**公式（4）：示例环境的状态更新。**这只是为自包含运行而设计的简化动力学：

$$
v_{t+1}=\operatorname{clip}_{v_{\max}}
\left(0.995v_t+a_t^{\text{world}}\Delta t\right),
\qquad
x_{t+1}=x_t+v_{t+1}\Delta t.
$$

代码中的 `ProceduralFlightEnv.step` 实现该动力学。这里策略动作被解释为加速度和偏航角速度，不等同于论文真实系统中的姿态／推力控制接口。

**公式（5）：导航进展、安全 shaping 与平滑惩罚。**示例用目标距离代替论文的 Dijkstra 距离场：

$$
r_{\text{nav},t}
=
\|g-x_{t-1}\|_2-\|g-x_t\|_2.
$$

安全 shaping 用离散近似构造：

$$
b_t=d_t-d_{\text{safe}},
\qquad
\dot b_t\approx\frac{d_t-d_{t-1}}{\Delta t},
\qquad
r_{\text{safety},t}
=
\operatorname{clip}(\dot b_t+\gamma b_t,-2,0).
$$

动作平滑项为：

$$
r_{\text{smooth},t}=-\|a_t-a_{t-1}\|_2^2.
$$

代码中的 `step` 使用这些量计算示例奖励。论文的 CBF 奖励使用 ESDF；本示例以最近障碍表面距离近似。[Safety Reward](https://www.alphaxiv.org/abs/2602.08653?page=4)

**公式（6）：总奖励。**

$$
r_t =
w_{\text{nav}}r_{\text{nav},t}
+w_{\text{safety}}r_{\text{safety},t}
+w_{\text{smooth}}r_{\text{smooth},t}
+r_{\text{terminal},t}
+r_{\text{aux},t}.
$$

**公式（7）：GAE 优势与回报目标。**

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

$$
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},
\qquad
\hat R_t=\hat A_t+V(s_t).
$$

代码中的 `compute_gae` 实现这一步。

**公式（8）：PPO 概率比。**

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid o_t)}.
$$

**公式（9）：截断策略损失。**

$$
L_{\text{policy}}=
-\mathbb{E}_t
\left[
\min\left(
\rho_t\hat A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

**公式（10）：Critic 价值损失。**

$$
L_{\text{value}}=
\mathbb{E}_t\left[(V_\psi(s_t)-\hat R_t)^2\right].
$$

**公式（11）：总 PPO loss。**

$$
L_{\text{total}}=
L_{\text{policy}}+
c_VL_{\text{value}}-
c_H\mathcal{H}(\pi_\theta).
$$

代码用未经过 $\tanh$ 变换的高斯分布熵作为熵正则近似；具体实现并非论文披露的配置。

```python
# train_flight_ppo.py
#
# 自包含、可运行的 PPO 训练示例：
#   程序生成 3D 球形障碍环境
#   深度图 CNN + 本体状态 + GRU -> Actor -> 连续动作
#   privileged state -> Critic -> 状态价值
#
# 安装：
#   pip install numpy torch
#
# 运行：
#   python train_flight_ppo.py
#
# 可选参数：
#   python train_flight_ppo.py --total-steps 100000 --num-envs 8
#   python train_flight_ppo.py --device cuda
#
# 重要说明：
#   1. 本代码是简化 PPO 示例，不是论文作者的原始实现。
#   2. 环境是轻量级程序生成仿真，不是 Isaac Lab。
#   3. 代码中的导航奖励用目标距离进展近似，未实现论文的 Dijkstra 代价图。
#   4. 本代码不含部署时的 HOCBF-QP 安全过滤器。
#   5. 公式编号与下方代码注释中的 [公式 X] 对应。

import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# 1. 训练配置
# ============================================================

class Config:
    def __init__(self, args):
        self.seed = args.seed
        self.device = args.device
        self.num_envs = args.num_envs
        self.total_steps = args.total_steps
        self.horizon = args.horizon

        # PPO 参数
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.update_epochs = 4
        self.max_grad_norm = 0.5

        # 示例环境参数
        self.max_episode_steps = 400
        self.dt = 0.05

        # [公式 6] 示例奖励权重。
        # 它们是本代码的示例值，不是论文披露的超参数。
        self.navigation_coef = 1.0
        self.safety_coef = 0.15
        self.smooth_coef = 0.01
        self.collision_reward = -20.0
        self.success_reward = 20.0


# ============================================================
# 2. 轻量级程序生成环境
# ============================================================

class ProceduralFlightEnv:
    """
    简化的 3D 无人机点质量环境。

    环境每一步的输入：
        action: shape (4,)
          action[0] = 前向加速度指令，范围 [-1, 1]
          action[1] = 横向加速度指令，范围 [-1, 1]
          action[2] = 垂直加速度指令，范围 [-1, 1]
          action[3] = 偏航角速度指令，范围 [-1, 1]

    环境每一步的输出：
        observation:
          depth:       (1, 60, 100) 深度图
          proprio:     (14,)       Actor 的本体状态输入
          privileged:  (15,)       Critic 的训练期特权状态输入

        reward: float              每步奖励
        done: bool                 episode 是否结束
        info: dict                 成功、碰撞、距离等诊断信息

    这是一种教学用的简化动力学和深度图生成方式。
    """

    def __init__(self, seed=0, max_episode_steps=400, dt=0.05):
        self.rng = np.random.default_rng(seed)
        self.max_episode_steps = max_episode_steps
        self.dt = dt

        self.image_h = 60
        self.image_w = 100
        self.camera_far = 10.0

        self.reset()

    def reset(self):
        """随机生成起点、目标和障碍物，返回初始观测。"""
        self.position = np.array(
            [1.0, self.rng.uniform(-1.0, 1.0), 1.5],
            dtype=np.float32,
        )
        self.velocity = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0

        self.goal = np.array(
            [19.0, self.rng.uniform(-1.0, 1.0), 1.5],
            dtype=np.float32,
        )

        self.previous_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0

        # 障碍物：[中心 x, 中心 y, 中心 z, 半径]
        obstacles = []
        for _ in range(18):
            x = self.rng.uniform(3.0, 17.0)
            y = self.rng.uniform(-3.7, 3.7)
            z = self.rng.uniform(0.6, 3.0)
            radius = self.rng.uniform(0.30, 0.75)

            obstacles.append(
                np.array([x, y, z, radius], dtype=np.float32)
            )

        self.obstacles = np.stack(obstacles, axis=0)

        # 用于 [公式 5] 中估算障碍距离的时间差分项。
        self.previous_clearance = self._nearest_clearance()[0]

        return self.get_observation()

    def _nearest_clearance(self):
        """
        返回最近障碍表面距离及最近障碍中心。

        clearance > 0：无人机在障碍球外。
        clearance = 0：无人机位于障碍球表面。
        clearance < 0：无人机与障碍球相交。
        """
        centers = self.obstacles[:, :3]
        radii = self.obstacles[:, 3]

        center_distances = np.linalg.norm(
            centers - self.position[None, :],
            axis=1,
        )
        surface_distances = center_distances - radii

        index = int(np.argmin(surface_distances))
        return float(surface_distances[index]), centers[index]

    def _render_depth_image(self):
        """
        生成简化深度图。

        这里用低分辨率射线与球体求交，再复制放大到 100x60。
        这让示例无需外部渲染器即可运行；它并非真实相机渲染。
        """
        rows, cols = 12, 20

        azimuth = np.linspace(-0.75, 0.75, cols, dtype=np.float32)
        elevation = np.linspace(-0.35, 0.35, rows, dtype=np.float32)
        az_grid, el_grid = np.meshgrid(azimuth, elevation)

        # 相机坐标系中的单位射线。
        local_dirs = np.stack(
            [
                np.cos(el_grid) * np.cos(az_grid),
                np.cos(el_grid) * np.sin(az_grid),
                np.sin(el_grid),
            ],
            axis=-1,
        )

        # 按 yaw 将射线转到世界坐标系。
        cosine = math.cos(float(self.yaw))
        sine = math.sin(float(self.yaw))

        world_dirs = local_dirs.copy()
        world_dirs[..., 0] = (
            cosine * local_dirs[..., 0] - sine * local_dirs[..., 1]
        )
        world_dirs[..., 1] = (
            sine * local_dirs[..., 0] + cosine * local_dirs[..., 1]
        )

        ray_dirs = world_dirs.reshape(-1, 3)
        depth = np.full(
            (ray_dirs.shape[0],),
            self.camera_far,
            dtype=np.float32,
        )

        centers = self.obstacles[:, :3]
        radii = self.obstacles[:, 3]
        relative_centers = centers - self.position[None, :]

        # 对每一条射线、每一个球体计算射线-球体求交。
        # 球心相对射线起点为 c，单位射线方向为 d：
        # t = d·c - sqrt((d·c)^2 - (c·c - r^2)).
        dot = ray_dirs @ relative_centers.T
        c_term = (
            np.sum(relative_centers * relative_centers, axis=1)[None, :]
            - radii[None, :] ** 2
        )
        discriminant = dot * dot - c_term

        has_intersection = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant, 0.0))
        t = dot - root

        valid = has_intersection & (t > 0.0)
        t = np.where(valid, t, self.camera_far)

        depth = np.minimum(depth, np.min(t, axis=1))
        depth = np.clip(depth / self.camera_far, 0.0, 1.0)
        depth = depth.reshape(rows, cols)

        # 12*5 = 60，20*5 = 100。
        depth = np.repeat(np.repeat(depth, 5, axis=0), 5, axis=1)

        # 输出形状 (1, 60, 100)，作为 CNN 的单通道输入。
        return depth[None, :, :].astype(np.float32)

    def _proprioception(self):
        """
        [公式 1] 构造 Actor 的 14 维本体状态输入。

        维度依次是：
          3  速度
          2  yaw 的 sin/cos
          4  上一时刻动作
          3  目标相对位置
          1  当前高度
          1  目标高度
        """
        goal_relative_position = self.goal - self.position

        proprio = np.concatenate(
            [
                self.velocity / 5.0,
                np.array(
                    [math.sin(self.yaw), math.cos(self.yaw)],
                    dtype=np.float32,
                ),
                self.previous_action,
                goal_relative_position / 20.0,
                np.array(
                    [self.position[2] / 4.0, self.goal[2] / 4.0],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        return proprio.astype(np.float32)

    def _privileged_state(self):
        """
        构造仅供 Critic 使用的特权状态。

        Actor 看不到这个向量；它包含无人机位置和最近障碍物信息。
        真实系统中需要由仿真器状态或训练期地图提供这类量。
        """
        clearance, nearest_center = self._nearest_clearance()

        relative_obstacle_position = nearest_center - self.position

        nearest_index = int(
            np.argmin(
                np.linalg.norm(
                    self.obstacles[:, :3] - self.position[None, :],
                    axis=1,
                )
                - self.obstacles[:, 3]
            )
        )
        nearest_radius = float(self.obstacles[nearest_index, 3])

        state = np.concatenate(
            [
                self.position / 20.0,                         # 3
                self.velocity / 5.0,                          # 3
                (self.goal - self.position) / 20.0,            # 3
                relative_obstacle_position / 10.0,             # 3
                np.array([nearest_radius], dtype=np.float32),  # 1
                np.array(
                    [math.sin(self.yaw), math.cos(self.yaw)],
                    dtype=np.float32,
                ),         [118;1:3u                                    # 2
            ],
            axis=0,
        )

        assert state.shape == (15,)
        return state.astype(np.float32)

    def get_observation(self):
        """返回 Actor 输入和 Critic 输入。"""
        return {
            "depth": self._render_depth_image(),
            "proprio": self._proprioception(),
            "privileged": self._privileged_state(),
        }

    def step(self, action):
        """
        执行动作，推进简化动力学，返回下一观测和奖励。

        [公式 4] 描述本函数中的速度、位置更新。
        [公式 5]、[公式 6] 描述本函数中的奖励项。
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_goal_distance = float(
            np.linalg.norm(self.goal - self.position)
        )

        # 将机体系前向和横向加速度指令转换到世界坐标系。
        forward_acceleration = 4.0 * float(action[0])
        lateral_acceleration = 4.0 * float(action[1])
        vertical_acceleration = 2.0 * float(action[2])
        yaw_rate = 1.5 * float(action[3])

        cosine = math.cos(float(self.yaw))
        sine = math.sin(float(self.yaw))

        world_acceleration = np.array(
            [
                cosine * forward_acceleration
                - sine * lateral_acceleration,
                sine * forward_acceleration
                + cosine * lateral_acceleration,
                vertical_acceleration,
            ],
            dtype=np.float32,
        )

        # [公式 4] 简化动力学：加速度更新速度，速度更新位置。
        self.velocity += world_acceleration * self.dt
        self.velocity *= 0.995

        speed = float(np.linalg.norm(self.velocity))
        max_speed = 5.0
        if speed > max_speed:
            self.velocity *= max_speed / speed

        self.position += self.velocity * self.dt
        self.yaw += yaw_rate * self.dt
        self.step_count += 1

        new_goal_distance = float(
            np.linalg.norm(self.goal - self.position)
        )

        # [公式 5] 用目标欧氏距离的下降量作为简化导航进展奖励。
        navigation_reward = old_goal_distance - new_goal_distance

        # [公式 5] 离散近似安全 shaping。
        clearance, _ = self._nearest_clearance()
        safe_distance = 0.35

        barrier_value = clearance - safe_distance
        barrier_derivative = (
            clearance - self.previous_clearance
        ) / self.dt

        safety_reward = float(
            np.clip(
                barrier_derivative + 1.0 * barrier_value,
                -2.0,
                0.0,
            )
        )
        self.previous_clearance = clearance

        # [公式 5] 惩罚相邻动作的大幅变化。
        smooth_reward = -float(
            np.sum((action - self.previous_action) ** 2)
        )

        # 简单的速度和高度辅助项。
        current_speed = float(np.linalg.norm(self.velocity))
        speed_penalty = -0.01 * max(0.0, current_speed - 4.0) ** 2
        height_penalty = -0.05 * (float(self.position[2]) - 1.5) ** 2

        # 碰撞检测。
        center_distances = np.linalg.norm(
            self.obstacles[:, :3] - self.position[None, :],
            axis=1,
        )
        collision = bool(
            np.any(center_distances <= self.obstacles[:, 3] + 0.20)
        )

        out_of_bounds = bool(
            self.position[0] < 0.0
            or self.position[0] > 20.0
            or abs(float(self.position[1])) > 5.0
            or self.position[2] < 0.3
            or self.position[2] > 4.0
        )

        success = new_goal_distance < 0.8

        terminal_reward = 0.0
        done = False

        if collision or out_of_bounds:
            terminal_reward = -20.0
            done = True
        elif success:
            terminal_reward = 20.0
            done = True
        elif self.step_count >= self.max_episode_steps:
            done = True

        # [公式 6] 总奖励是各项加权求和。
        reward = (
            1.0 * navigation_reward
            + 0.15 * safety_reward
            + 0.01 * smooth_reward
            + speed_penalty
            + height_penalty
            + terminal_reward
        )

        self.previous_action = action.copy()

        info = {
            "success": success,
            "collision": collision or out_of_bounds,
            "distance_to_goal": new_goal_distance,
            "clearance": clearance,
        }

        return self.get_observation(), float(reward), done, info


# ============================================================
# 3. Actor-Critic 网络
# ============================================================

class ActorCritic(nn.Module):
    """
    Actor 输入：
      depth:   (B, 1, 60, 100)
      proprio: (B, 14)
      hidden:  (1, B, 128)

    Actor 输出：
      mean:     (B, 4)，高斯动作分布均值
      new_hidden: GRU 的新隐藏状态

    Critic 输入：
      privileged: (B, 15)

    Critic 输出：
      value: (B,)，状态价值估计
    """

    def __init__(self):
        super().__init__()

        # [公式 2] 深度图 CNN。
        # 输入 (B, 1, 60, 100)，输出 (B, 128) 视觉特征。
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 6, 128),
            nn.ReLU(),
        )

        # [公式 2] 拼接视觉特征和 14 维本体状态。
        self.fusion = nn.Sequential(
            nn.Linear(128 + 14, 128),
            nn.ReLU(),
        )

        # [公式 2] GRU 处理历史观测，输出时序特征。
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )

        # Actor 输出高斯分布均值。
        self.actor_mean = nn.Linear(128, 4)

        # 可学习的高斯 log standard deviation。
        self.actor_log_std = nn.Parameter(
            torch.full((4,), -0.5)
        )

        # 非对称 Critic：仅使用 privileged state。
        self.critic = nn.Sequential(
            nn.Linear(15, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def actor_step(self, depth, proprio, hidden, episode_mask):
        """
        处理一个时刻的 Actor 输入。

        episode_mask:
          1 表示 GRU 隐藏状态保留；
          0 表示 episode 刚开始，需要清零隐藏状态。
        """
        # 在 episode 边界清除 GRU 记忆。
        hidden = hidden * episode_mask.view(1, -1, 1)

        # [公式 2] CNN 编码深度图。
        visual_feature = self.cnn(depth)

        # [公式 2] 视觉特征与本体状态拼接、融合。
        fused_feature = self.fusion(
            torch.cat([visual_feature, proprio], dim=-1)
        )

        # GRU 输入形状为 (B, sequence_length=1, feature_dim)。
        gru_output, new_hidden = self.gru(
            fused_feature.unsqueeze(1),
            hidden,
        )

        mean = self.actor_mean(gru_output[:, 0, :])
        return mean, new_hidden

    def value(self, privileged_state):
        """Critic 输出状态价值 V(s)。"""
        return self.critic(privileged_state).squeeze(-1)


# ============================================================
# 4. 动作采样及其 log-probability
# ============================================================

def sample_squashed_gaussian(mean, log_std, raw_action=None):
    """
    [公式 3] 高斯动作采样，再用 tanh 限定动作在 [-1, 1]。

    返回：
      action:    tanh 后的动作，环境实际接收这个值
      raw_action: tanh 前的动作，用于 PPO 更新时重算概率
      log_prob:  tanh 变量变换修正后的 log π(a|o)
      entropy:   未 squashed 高斯分布熵的近似
    """
    std = log_std.exp().expand_as(mean)
    distribution = Normal(mean, std)

    if raw_action is None:
        raw_action = distribution.sample()

    action = torch.tanh(raw_action)

    # 高斯变量 u 的 log probability。
    log_prob = distribution.log_prob(raw_action).sum(dim=-1)

    # tanh 变量变换的 Jacobian 修正。
    log_prob -= torch.log(
        1.0 - action.pow(2) + 1e-6
    ).sum(dim=-1)

    # 常用近似：使用 squashing 前高斯分布的熵。
    entropy = distribution.entropy().sum(dim=-1)

    return action, raw_action, log_prob, entropy


def observations_to_tensors(observations, device):
    """将多个环境的 observation 合并成 batch tensor。"""
    depth = torch.as_tensor(
        np.stack([obs["depth"] for obs in observations]),
        dtype=torch.float32,
        device=device,
    )
    proprio = torch.as_tensor(
        np.stack([obs["proprio"] for obs in observations]),
        dtype=torch.float32,
        device=device,
    )
    privileged = torch.as_tensor(
        np.stack([obs["privileged"] for obs in observations]),
        dtype=torch.float32,
        device=device,
    )
    return depth, proprio, privileged


# ============================================================
# 5. GAE 优势估计
# ============================================================

def compute_gae(rewards, dones, values, bootstrap_value, gamma, gae_lambda):
    """
    [公式 7] 从 rollout 的奖励、done 标记和 Critic 价值计算：
      advantages: 优势估计 A_t
      returns:    Critic 的监督目标 R_t = A_t + V_t
    """
    horizon, num_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)

    last_advantage = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = bootstrap_value
        else:
            next_value = values[t + 1]

        nonterminal = 1.0 - dones[t]

        # [公式 7] TD residual δ_t。
        delta = (
            rewards[t]
            + gamma * next_value * nonterminal
            - values[t]
        )

        # [公式 7] GAE 递推。
        last_advantage = (
            delta
            + gamma * gae_lambda * nonterminal * last_advantage
        )
        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


# ============================================================
# 6. PPO 训练循环
# ============================================================

def train(config):
    # 固定随机种子，方便重复实验。
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(config.device)

    print(f"Training device: {device}")

    # 创建并行环境。
    environments = [
        ProceduralFlightEnv(
            seed=config.seed + env_id,
            max_episode_steps=config.max_episode_steps,
            dt=config.dt,
        )
        for env_id in range(config.num_envs)
    ]
    observations = [env.reset() for env in environments]

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    # GRU 隐藏状态，形状 (1, num_envs, 128)。
    hidden = torch.zeros(
        1,
        config.num_envs,
        128,
        dtype=torch.float32,
        device=device,
    )
    previous_done = np.zeros(config.num_envs, dtype=np.float32)

    episode_returns = np.zeros(config.num_envs, dtype=np.float32)
    completed_returns = []
    completed_successes = []

    steps_per_update = config.num_envs * config.horizon
    num_updates = max(1, config.total_steps // steps_per_update)

    for update in range(1, num_updates + 1):

        # 每个 rollout 从零隐藏状态开始。
        # 这样 PPO 重放该 rollout 更新时，也能从同一初始隐藏状态重算。
        hidden = torch.zeros(
            1,
            config.num_envs,
            128,
            dtype=torch.float32,
            device=device,
        )

        # rollout 缓冲区，第一维是时间，第二维是并行环境。
        depth_buffer = np.zeros(
            (config.horizon, config.num_envs, 1, 60, 100),
            dtype=np.float32,
        )
        proprio_buffer = np.zeros(
            (config.horizon, config.num_envs, 14),
            dtype=np.float32,
        )
        privileged_buffer = np.zeros(
            (config.horizon, config.num_envs, 15),
            dtype=np.float32,
        )
        raw_action_buffer = np.zeros(
            (config.horizon, config.num_envs, 4),
            dtype=np.float32,
        )
        old_log_prob_buffer = np.zeros(
            (config.horizon, config.num_envs),
            dtype=np.float32,
        )
        reward_buffer = np.zeros(
            (config.horizon, config.num_envs),
            dtype=np.float32,
        )
        done_buffer = np.zeros(
            (config.horizon, config.num_envs),
            dtype=np.float32,
        )
        value_buffer = np.zeros(
            (config.horizon, config.num_envs),
            dtype=np.float32,
        )
        mask_buffer = np.zeros(
            (config.horizon, config.num_envs),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # A. 收集 rollout：观测 -> Actor 动作 -> 环境响应
        # ----------------------------------------------------
        for t in range(config.horizon):
            depth_t, proprio_t, privileged_t = observations_to_tensors(
                observations,
                device,
            )

            # episode 刚重置时 previous_done=1，
            # mask=0 会在 actor_step 里清除 GRU hidden state。
            episode_mask = torch.as_tensor(
                1.0 - previous_done,
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                mean, hidden = model.actor_step(
                    depth_t,
                    proprio_t,
                    hidden,
                    episode_mask,
                )

                # [公式 3] 采样动作并计算旧策略的 log probability。
                action, raw_action, log_prob, _ = sample_squashed_gaussian(
                    mean,
                    model.actor_log_std,
                )

                # Critic 使用 privileged state 估计 V(s_t)。
                value = model.value(privileged_t)

            action_numpy = action.cpu().numpy()
            raw_action_numpy = raw_action.cpu().numpy()

            # 保存 PPO 更新需要的数据。
            depth_buffer[t] = np.stack(
                [obs["depth"] for obs in observations]
            )
            proprio_buffer[t] = np.stack(
                [obs["proprio"] for obs in observations]
            )
            privileged_buffer[t] = np.stack(
                [obs["privileged"] for obs in observations]
            )
            raw_action_buffer[t] = raw_action_numpy
            old_log_prob_buffer[t] = log_prob.cpu().numpy()
            value_buffer[t] = value.cpu().numpy()
            mask_buffer[t] = 1.0 - previous_done

            next_observations = []
            done_this_step = np.zeros(
                config.num_envs,
                dtype=np.float32,
            )

            # 每个环境分别执行动作，得到下一观测和奖励。
            for env_id, environment in enumerate(environments):
                next_obs, reward, done, info = environment.step(
                    action_numpy[env_id]
                )

                reward_buffer[t, env_id] = reward
                done_buffer[t, env_id] = float(done)
                episode_returns[env_id] += reward

                if done:
                    completed_returns.append(
                        float(episode_returns[env_id])
                    )
                    completed_successes.append(
                        float(info["success"])
                    )
                    episode_returns[env_id] = 0.0

                    # episode 结束后重置环境，并标记 GRU 要重置。
                    next_obs = environment.reset()
                    done_this_step[env_id] = 1.0

                next_observations.append(next_obs)

            observations = next_observations
            previous_done = done_this_step

        # ----------------------------------------------------
        # B. 用 rollout 末状态的 Critic 价值做 bootstrap
        # ----------------------------------------------------
        _, _, privileged_last = observations_to_tensors(
            observations,
            device,
        )

        with torch.no_grad():
            bootstrap_value = model.value(privileged_last).cpu().numpy()

        # ----------------------------------------------------
        # C. [公式 7] 计算 GAE 优势和 Critic 回报目标
        # ----------------------------------------------------
        advantages, returns = compute_gae(
            rewards=reward_buffer,
            dones=done_buffer,
            values=value_buffer,
            bootstrap_value=bootstrap_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # 转成 PyTorch tensor。
        depth_tensor = torch.as_tensor(
            depth_buffer,
            dtype=torch.float32,
            device=device,
        )
        proprio_tensor = torch.as_tensor(
            proprio_buffer,
            dtype=torch.float32,
            device=device,
        )
        privileged_tensor = torch.as_tensor(
            privileged_buffer,
            dtype=torch.float32,
            device=device,
        )
        raw_action_tensor = torch.as_tensor(
            raw_action_buffer,
            dtype=torch.float32,
            device=device,
        )
        old_log_prob_tensor = torch.as_tensor(
            old_log_prob_buffer,
            dtype=torch.float32,
            device=device,
        )
        advantage_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=device,
        )
        return_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=device,
        )
        mask_tensor = torch.as_tensor(
            mask_buffer,
            dtype=torch.float32,
            device=device,
        )

        # 优势标准化，改善优化数值稳定性。
        advantage_tensor = (
            advantage_tensor - advantage_tensor.mean()
        ) / (advantage_tensor.std() + 1e-8)

        # ----------------------------------------------------
        # D. PPO 多轮更新
        # ----------------------------------------------------
        for _epoch in range(config.update_epochs):
            # 重放完整时间序列，以保留 GRU 的时序依赖。
            hidden_update = torch.zeros(
                1,
                config.num_envs,
                128,
                dtype=torch.float32,
                device=device,
            )

            new_log_prob_list = []
            entropy_list = []
            value_prediction_list = []

            for t in range(config.horizon):
                mean_t, hidden_update = model.actor_step(
                    depth_tensor[t],
                    proprio_tensor[t],
                    hidden_update,
                    mask_tensor[t],
                )

                # 使用 rollout 时采样的 raw action，
                # 重新计算当前策略下的 log π(a_t | o_t)。
                _, _, new_log_prob_t, entropy_t = (
                    sample_squashed_gaussian(
                        mean_t,
                        model.actor_log_std,
                        raw_action=raw_action_tensor[t],
                    )
                )

                value_t = model.value(privileged_tensor[t])

                new_log_prob_list.append(new_log_prob_t)
                entropy_list.append(entropy_t)
                value_prediction_list.append(value_t)

            new_log_probs = torch.stack(new_log_prob_list)
            entropies = torch.stack(entropy_list)
            value_predictions = torch.stack(value_prediction_list)

            # [公式 8] PPO 概率比。
            probability_ratio = torch.exp(
                new_log_probs - old_log_prob_tensor
            )

            # [公式 9] clipped PPO policy loss。
            unclipped_objective = (
                probability_ratio * advantage_tensor
            )
            clipped_objective = (
                torch.clamp(
                    probability_ratio,
                    1.0 - config.ppo_clip,
                    1.0 + config.ppo_clip,
                )
                * advantage_tensor
            )
            policy_loss = -torch.minimum(
                unclipped_objective,
                clipped_objective,
            ).mean()

            # [公式 10] Critic 均方误差损失。
            value_loss = F.mse_loss(
                value_predictions,
                return_tensor,
            )

            # 高斯熵近似；见 sample_squashed_gaussian 的说明。
            entropy = entropies.mean()

            # [公式 11] 总 PPO loss。
            total_loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # 梯度裁剪，防止更新过大。
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            optimizer.step()

        # ----------------------------------------------------
        # E. 训练日志
        # ----------------------------------------------------
        if update == 1 or update % 10 == 0:
            recent_returns = completed_returns[-50:]
            recent_successes = completed_successes[-50:]

            mean_return = (
                float(np.mean(recent_returns))
                if recent_returns
                else float("nan")
            )
            success_rate = (
                float(np.mean(recent_successes))
                if recent_successes
                else float("nan")
            )

            steps_so_far = update * steps_per_update
            print(
                f"update={update:4d}/{num_updates} "
                f"steps={steps_so_far:8d} "
                f"loss={total_loss.item():8.4f} "
                f"policy={policy_loss.item():8.4f} "
                f"value={value_loss.item():8.4f} "
                f"entropy={entropy.item():7.3f} "
                f"return_last50={mean_return:8.3f} "
                f"success_last50={success_rate:6.3f}"
            )

    # 保存模型参数及配置。
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": vars(config),
    }
    torch.save(checkpoint, "flight_ppo_checkpoint.pt")

    print("训练完成。")
    print("模型保存至：flight_ppo_checkpoint.pt")


# ============================================================
# 7. 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small vision-based flight policy with PPO."
    )

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--horizon", type=int, default=128)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    train(config)
```

运行后，终端会定期打印策略损失、价值损失、熵、最近 episode 的平均回报和成功率；训练完成后会生成 `flight_ppo_checkpoint.pt`。

这里的输入输出含义是：**Actor 输入**为 `depth` 和 `proprio`；**Actor 输出**为 4 维动作；环境接收动作并输出下一观测、奖励和结束标记；**Critic 输入**为训练期 `privileged` 状态，输出价值估计。要接入实际飞行日志或 Isaac Lab，需要替换 `ProceduralFlightEnv`，使新环境遵循同样的 `reset()` / `step(action)` 接口；单独的离线深度图数据并不能直接提供 PPO 所需的动作—状态转移。


在我给的**示例代码**里，Actor 输出的 4 维动作经过 `tanh`，每一维都在 $[-1,1]$：

| 维度 | 代码变量 | 含义 | 环境中的换算 |
|---|---|---|---|
| 0 | `action[0]` | 前向加速度指令 | 乘以 $4$，即最大约 $\pm4\,\mathrm{m/s^2}$ |
| 1 | `action[1]` | 横向加速度指令 | 乘以 $4$，即最大约 $\pm4\,\mathrm{m/s^2}$ |
| 2 | `action[2]` | 垂直加速度指令 | 乘以 $2$，即最大约 $\pm2\,\mathrm{m/s^2}$ |
| 3 | `action[3]` | 偏航角速度指令 | 乘以 $1.5$，即最大约 $\pm1.5\,\mathrm{rad/s}$ |

代码中用前向和横向加速度，以及当前偏航角，把这两个机体系指令转换成世界坐标系加速度；垂直加速度直接作用于高度方向。随后仿真器用加速度更新速度，再用速度更新位置。**所以这里的动作不是姿态角或电机转速，而是简化环境中的加速度／偏航角速度指令。**

要特别区分：这只是示例代码为方便运行而定义的动作接口。论文中的四维动作是**质量归一化总推力**和**期望横滚、俯仰、偏航角**，再由 PX4 底层飞控转换为电机控制；和示例代码的四维动作定义不同。
