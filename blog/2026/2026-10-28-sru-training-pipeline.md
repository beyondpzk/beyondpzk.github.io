---
title: "SRU 训练流程拆解：从深度 VAE 到循环 PPO"
date: 2026-10-28
topic: navigation
type: 技术分析
tags: [SRU, VAE, PPO, 强化学习, 空间记忆]
summary: "围绕 SRU 无地图导航论文，拆解深度 VAE 预训练、循环状态重放、GAE 回报目标与 PPO 损失，用教学代码串联训练流程，并说明与完整导航复现的差距。"
---

# SRU 训练流程拆解：从深度 VAE 到循环 PPO

## 核心结论

论文 *Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning* 用空间增强循环记忆支持长程无地图导航。理解其训练流程，需要分清两类学习信号：深度 VAE 从重建目标学习感知表征，循环 PPO 从环境奖励与价值估计学习动作策略。

本文保留公式与 Python 示例，重点解释数据、状态和损失如何对应。论文贡献与实验结果见已有的[SRU 论文精读](/blog/2026/2026-09-29-sru-spatial-memory)。

- **阅读与核对范围**：本笔记中的公式、示例代码及仓库内已有论文笔记；本次整理未重新核对论文全文或官方代码版本，未运行模型训练。
- **内容定位**：训练机制的技术分析。代码是教学示例，不代表论文官方实现或已完成的复现。
- **关键术语**：这里是非对称（asymmetric）Actor–Critic，指 Actor 与 Critic 使用不同观测；示例采用同步采样与更新，不是异步训练。

## 方法与证据

### 先看两阶段的数据流

| 阶段 | 输入 | 学习目标 | 示例覆盖范围 |
| --- | --- | --- | --- |
| 深度 VAE 预训练 | 深度图与有效像素掩码 | 重建损失与 KL 正则 | 卷积 VAE、随机深度数据 |
| 循环 PPO | Actor 观测、Critic 特权观测与历史状态 | 策略目标、价值回归与熵正则 | SRU 单元、简化环境、GAE 与序列重放 |
| 感知与策略集成 | 深度空间特征、本体状态与相对目标 | 导航策略训练 | 仅说明接口，尚未接通 |

下文公式编号为本笔记内部编号，不对应论文原文。网络尺寸、损失权重与环境参数均按示例理解。

### 阶段一：深度图编码器的 VAE 自重建预训练

#### 数据、重建目标与损失函数

深度编码器预训练为导航策略提供几何表征的初始化。下面用卷积 VAE 解释自监督重建的输入、目标与损失；代码使用随机生成的深度张量演示训练流程，没有加载 TartanAir，也没有实现论文的 RegNet + FPN 感知架构。

##### 1. 数据与 Ground Truth (GT)
- **输入样本 ($x$)**：经传感器噪声增强后的单通道深度图，张量形状为 $[B, 1, H, W]$，单位为米。在有效感知量程内（例如 $0 < d \le 10\text{ m}$）归一化至 $[0, 1]$；超出量程、负值或无效点（NaN/Inf）使用有效掩码 $M \in \{0, 1\}^{B \times 1 \times H \times W}$ 标记。
- **Ground Truth 目标 ($y$)**：自重建任务中的真实目标深度图 $D^{\text{GT}}$。在标准的自重建设置下，GT 即为目标深度图本身（若进行去噪训练，则输入为带噪图，GT 为干净深度图）。

##### 2. 编码与潜变量采样 (Reparameterization Trick)
编码网络将输入图像映射为潜空间的均值 $\mu$ 与对数方差 $\log\sigma^2$：
$$
q_\phi(z \mid x) = \mathcal{N}\left(\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x))\right) \tag{VAE-1}
$$
训练阶段通过重参数化采样以允许梯度反向传播：
$$
\epsilon \sim \mathcal{N}(0, I), \qquad z = \mu + \exp\left(\frac{1}{2}\log\sigma^2\right) \odot \epsilon \tag{VAE-2}
$$

##### 3. 损失函数 (Loss Function)
VAE 总损失由**有效像素重建误差**和**潜分布 KL 散度正则**组成：
$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} \tag{VAE-3}
$$
其中，重建损失采用有效像素加权的均方误差（MSE）：
$$
\mathcal{L}_{\text{recon}} = \frac{\sum_{b,c,i,j} M_{b,c,i,j} \left(\hat{D}_{b,c,i,j} - D^{\text{GT}}_{b,c,i,j}\right)^2}{\sum_{b,c,i,j} M_{b,c,i,j} + \epsilon_{\text{eps}}} \tag{VAE-4}
$$
KL 散度约束潜变量后验分布逼近标准正态先验 $p(z) = \mathcal{N}(0, I)$：
$$
\mathcal{L}_{\text{KL}} = \frac{1}{B} \sum_{b=1}^B \left( \frac{1}{K} \sum_{k=1}^K \frac{1}{2}\left( \mu_{b,k}^2 + \sigma_{b,k}^2 - \log\sigma_{b,k}^2 - 1 \right) \right) \tag{VAE-5}
$$

---

#### VAE 训练示例：`train_depth_vae.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SyntheticDepthDataset(Dataset):
    """
    深度图数据集：支持自重建 (GT = 输入) 或 去噪重建 (输入带噪，GT 为干净图)。
    输入格式: [N, H, W] 或 [N, 1, H, W]，单位：米。
    """
    def __init__(self, depths, targets=None, max_depth=10.0):
        depths = torch.as_tensor(depths, dtype=torch.float32)
        if depths.ndim == 3:
            depths = depths.unsqueeze(1)
        self.depths = depths

        if targets is not None:
            targets = torch.as_tensor(targets, dtype=torch.float32)
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            self.targets = targets
        else:
            self.targets = None

        self.max_depth = float(max_depth)

    def __len__(self):
        return self.depths.shape[0]

    def _process(self, d):
        # 产生有效像素掩码 M: Eq. (VAE-4)
        valid = torch.isfinite(d) & (d > 0.0) & (d <= self.max_depth)
        d_clean = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d_norm = torch.clamp(d_clean, 0.0, self.max_depth) / self.max_depth
        d_norm = torch.where(valid, d_norm, torch.zeros_like(d_norm))
        return d_norm, valid.float()

    def __getitem__(self, idx):
        x, _ = self._process(self.depths[idx])
        # 若未指定 targets，则 GT = 输入本身 (自重建)
        target_raw = self.depths[idx] if self.targets is None else self.targets[idx]
        y_gt, valid_mask = self._process(target_raw)
        return x, y_gt, valid_mask


class SpatialDepthVAE(nn.Module):
    """
    保留空间结构的卷积 VAE 架构。
    编码器输出下采样特征图作为潜分布参数，便于与后续空间注意力网络对接。
    """
    def __init__(self, latent_channels=64):
        super().__init__()
        # 四次跨步卷积下采样 (空间分辨率缩减 16 倍)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Eq. (VAE-1): 输出潜分布均值 mu 与对数方差 logvar
        self.to_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(128, latent_channels, kernel_size=1)

        # 解码网络：反卷积上采样复原分辨率
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # 约束预测深度至 [0, 1]
        )

    def encode(self, x):
        feat = self.encoder(x)
        mu = self.to_mu(feat)
        logvar = self.to_logvar(feat).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Eq. (VAE-2): 重参数化技巧
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss_function(recon, target_gt, valid_mask, mu, logvar, beta=1e-3):
    """
    对应公式 Eq. (VAE-3), (VAE-4), (VAE-5)
    """
    # Eq. (VAE-4): 有效像素 MSE 重建损失
    diff_sq = (recon - target_gt).square() * valid_mask
    valid_elements = valid_mask.sum().clamp_min(1.0)
    recon_loss = diff_sq.sum() / valid_elements

    # Eq. (VAE-5): 解析 KL 散度
    kl_element = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
    kl_loss = kl_element.flatten(start_dim=1).mean(dim=1).mean()

    # Eq. (VAE-3): 总损失
    total_loss = recon_loss + beta * kl_loss

    return total_loss, {
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
    }


def train_vae_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on: {device}")

    # 构造仿真合成深度图虚拟数据 (512 张 128x160 深度图)
    dummy_depths = torch.clamp(torch.randn(512, 128, 160) * 2.0 + 4.0, 0.1, 10.0)
    dataset = SyntheticDepthDataset(dummy_depths, targets=None, max_depth=10.0)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SpatialDepthVAE(latent_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        total_l, rec_l, kl_l = 0.0, 0.0, 0.0

        for x, y_gt, mask in loader:
            x, y_gt, mask = x.to(device), y_gt.to(device), mask.to(device)

            recon, mu, logvar = model(x)
            loss, logs = vae_loss_function(recon, y_gt, mask, mu, logvar, beta=1e-3)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_l += loss.item()
            rec_l += logs["recon_loss"].item()
            kl_l += logs["kl_loss"].item()

        n = len(loader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {total_l/n:.4f} (Recon: {rec_l/n:.4f}, KL: {kl_l/n:.4f})")

    torch.save(model.state_dict(), "depth_vae_pretrained.pt")
    print("Pretraining completed. Model saved to depth_vae_pretrained.pt")


if __name__ == "__main__":
    train_vae_main()
```

---

### 阶段二：SRU-Ours 与强化学习 (PPO) 训练流程

#### 循环状态、优势估计与 PPO 损失

在强化学习中，不存在手工标注的“正确动作”。策略根据与环境交互的优势（Advantage）自我提升；价值网络拟合由奖励与旧价值估计构造的回报目标。下文沿用示例中的 GT 称呼，但它是带有自举估计的训练目标，并非精确真值。

```
        +-------------------------------------------------------+
        |                 Rollout 数据交互收集                  |
        |  采样 (s_t, a_t, r_t, done_t, log_pi_old, V_old)      |
        +---------------------------+---------------------------+
                                    |
                                    v
        +-------------------------------------------------------+
        |            GAE 优势估计与回报目标计算 (GT)            |
        |   TD 残差 delta_t   优势 A_hat_t   GT R_hat_t |
        |               [公式 Eq. (6), (7), (8)]                |
        +---------------------------+---------------------------+
                                    |
                                    v
        +-------------------------------------------------------+
        |              PPO 循环序列 Forward 重放                |
        | 沿时序展开计算新策略: log_pi_new(a_t), V_new(s_t)     |
        |               [公式 Eq. (1), (2), (5)]                |
        +---------------------------+---------------------------+
                                    |
                                    v
        +-------------------------------------------------------+
        |           PPO 目标更新 (反向传播与梯度更新)           |
        |    L_policy(剪切) + L_value(GT回归) - L_entropy       |
        |              [公式 Eq. (9), (10), (12), (13)]         |
        +-------------------------------------------------------+
```

##### 1. 前向循环计算与 SRU-Ours
输入特征与隐状态通过空间变换因子 $s_t$ 与 Refined Gating 机制更新：
$$
\begin{aligned}
x_t^{\pi} &= f_{\pi}(o_t^{\text{actor}}), \quad (h_t^{\pi}, c_t^{\pi}) = \operatorname{SRU}_{\pi}(x_t^{\pi}, h_{t-1}^{\pi}, c_{t-1}^{\pi}) \\
s_t &= W_{xs} x_t + b_s, \quad g_t = \tanh\left(s_t \odot \left(W_{xg} x_t + W_{hg} h_{t-1} + b_g\right)\right) \\
r_t^{\text{gate}} &= i_t \odot \left(1 - (1 - f_t)^2\right) + (1 - i_t) \odot f_t^2 \\
c_t &= r_t^{\text{gate}} \odot c_{t-1} + (1 - r_t^{\text{gate}}) \odot g_t, \quad h_t = o_t \odot \tanh(c_t)
\end{aligned}
\tag{1}
$$

对 Critic 特权网络同理更新：
$$
x_t^V = f_V(o_t^{\text{critic}}), \quad (h_t^V, c_t^V) = \operatorname{SRU}_V(x_t^V, h_{t-1}^V, c_{t-1}^V) \tag{2}
$$

动作由高斯分布经 $\tanh$ 激活映射至有界控制空间：
$$
u_t \sim \mathcal{N}\left(\mu_\theta(h_t^{\pi}), \operatorname{diag}(\sigma_\theta^2)\right), \qquad a_t = \tanh(u_t) \tag{3}
$$
状态价值估计：
$$
V_\phi(o_t^{\text{critic}}) = v_t \tag{4}
$$
针对 $\tanh$ 动作映射，真实对数概率包含变换的 Jacobian 行列式修正：
$$
\log \pi_\theta(a_t \mid o_t) = \sum_{j=1}^{\text{dim}} \left[ \log \mathcal{N}\left(u_{t,j}; \mu_{t,j}, \sigma_{t,j}^2\right) - \log\left(1 - a_{t,j}^2 + \epsilon_{\text{eps}}\right) \right] \tag{5}
$$

##### 2. TD 误差与 GAE 优势计算 (Ground Truth 定义)
单步广义时序差分（TD Residual）：
$$
\delta_t = r_t + \gamma (1 - d_t) V_{\text{old}}(o_{t+1}^{\text{critic}}) - V_{\text{old}}(o_t^{\text{critic}}) \tag{6}
$$
其中 $d_t \in \{0, 1\}$ 为终止标志符（done flag）。

GAE 优势函数从时序末端逆向递推：
$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1}^{\text{GAE}(\gamma, \lambda)} \tag{7}
$$

**Critic 学习的 Ground Truth 目标回报定义为：**
$$
\hat{R}_t = \hat{A}_t + V_{\text{old}}(o_t^{\text{critic}}) \tag{8}
$$
$\hat{R}_t$ 即代码中的 `returns`。

##### 3. PPO 损失函数定义
概率比率 $\rho_t(\theta)$：
$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)} = \exp\left( \log \pi_\theta(a_t \mid o_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid o_t) \right) \tag{9}
$$

Actor Clipped Surrogate 损失：
$$
\mathcal{L}_{\text{policy}} = -\mathbb{E}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \, \operatorname{clip}\left(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t \right) \right] \tag{10}
$$

Critic 预测裁剪范围：
$$
V_t^{\text{clip}} = V_{\text{old},t} + \operatorname{clip}\left(V_{\text{new},t} - V_{\text{old},t}, -\epsilon_v, \epsilon_v\right) \tag{11}
$$

Critic Clipped Value 损失（以 $\hat{R}_t$ 为 GT 标签）：
$$
\mathcal{L}_{\text{value}} = \frac{1}{2} \mathbb{E}_t \left[ \max\left( (V_{\text{new},t} - \hat{R}_t)^2, \, (V_t^{\text{clip}} - \hat{R}_t)^2 \right) \right] \tag{12}
$$

联合优化总损失：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + c_v \cdot \mathcal{L}_{\text{value}} - c_e \cdot \mathcal{H}(\pi_\theta) \tag{13}
$$

---

#### 策略熵正则：高斯动作与有界动作的区别

最后一项是**策略熵正则**，用于鼓励 Actor 保持一定探索性。它不是有人工 GT 标签的监督损失。

在这个代码里，Actor 先从高斯分布采样未限幅动作 $u_t$，再通过 $\tanh$ 得到有界动作 $a_t=\tanh(u_t)$。因此要区分高斯动作 $u_t$ 的熵，和实际动作 $a_t$ 的熵。以下分布均以当前观测和循环状态为条件，为简洁省略部分条件记号。

##### 不经过 tanh 时：高斯策略的熵

若策略是对角高斯分布：

$$
u_t\sim\mathcal{N}\bigl(\mu_t,\operatorname{diag}(\sigma_t^2)\bigr),
$$

它的微分熵有解析形式：

$$
\mathcal{H}(u_t)
= \frac{1}{2}\sum_j
\log\left(2\pi e\,\sigma_{t,j}^2\right).
\tag{14}
$$

标准差越大，熵越大，策略分布越分散，探索通常也越强。这个熵不依赖均值 $\mu_t$。

##### 经过 tanh 后：动作熵要考虑变换

代码把高斯采样通过 $\tanh$ 限制到 $(-1,1)$。由于这个变换会改变概率密度：

$$
\log \pi(a_t\mid o_t)
= \sum_j\log\mathcal{N}(u_{t,j};\mu_{t,j},\sigma_{t,j}^2)
-
\sum_j\log(1-a_{t,j}^2),
\qquad
a_t=\tanh(u_t).
\tag{15}
$$

因此，变换后策略的熵是：

$$
\mathcal{H}(a_t)
= \mathcal{H}(u_t)
+
\mathbb{E}_{u_t\sim\mathcal{N}(\mu_t,\operatorname{diag}(\sigma_t^2))}
\left[
\sum_j \log\left(1-\tanh^2(u_{t,j})\right)
\right].
\tag{16}
$$

因为 $\log(1-\tanh^2(u))\leq 0$，在这里的动作坐标下，经过 $\tanh$ 后的微分熵不高于未变换高斯的熵。这个期望通常用 Monte Carlo 采样近似：

$$
\widehat{\mathcal{H}}(a_t)
= -\log\pi(a_t\mid o_t),
\qquad a_t\sim\pi.
\tag{17}
$$

式 (17) 是单样本估计，训练时再对批次取平均。微分熵可以为负，不能用其正负判断实现是否正确。

##### 它怎样进入总损失？

代码通过**最小化**总损失来训练：

$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{policy}}
+c_v\mathcal{L}_{\mathrm{value}}
-c_e\mathcal{H}(\pi_\theta),
\qquad c_e\geq0.
\tag{18}
$$

这与式 (13) 是同一个目标。前面有负号，是因为优化器在最小化损失；最小化 $-c_e\mathcal{H}$ 等价于**鼓励策略熵变大**。$c_e$ 越大，探索激励越强；但太大可能让策略长期过于随机。

##### 与论文训练设计的关系

**这是 PPO 实现中常见的熵正则项，不是这篇论文独有的设计。**严格说，熵项不是 PPO clipped policy objective 必不可少的一部分；它是常加进去、用来鼓励探索的可选项。式 (18) 展示的是这种常见的组合目标，不同 PPO 实现可能使用、调整或省略其中的熵项。

**就当前笔记所依据的材料，尚不能确认作者在 PPO 损失中用了显式熵奖励。**已有论文笔记记录了 PPO、任务/平滑/安全相关奖励，以及 DML 和时间一致 Dropout，但没有给出可核实的熵系数或包含 $-c_e\mathcal{H}$ 的作者配置。因此，不能仅凭本文教学示例推断论文采用了这一项。进一步核对可参照[论文奖励设计（第 8 页）](https://www.alphaxiv.org/abs/2506.05997v2?page=8)与[训练正则（第 9 页）](https://www.alphaxiv.org/abs/2506.05997v2?page=9)；这两个页码入口为补充材料，本次整理未重新核验页面内容。

还要区分两个容易混淆的“正则”：

- $-c_e\mathcal{H}(\pi_\theta)$：**策略分布的熵正则**，影响 Actor 的探索程度。
- 论文的 $r_{\mathrm{reg}}$：**环境奖励中的动作平滑与关节加速度等惩罚**，鼓励机器人动作平稳，参见[动作平滑奖励](https://www.alphaxiv.org/abs/2506.05997v2?page=8)。本文简化环境使用的动作幅值惩罚也不等同于论文的完整奖励设计。

因此，示例代码里的最后一项可以保留为**标准 PPO 工程中常用的可选项**，但不应称为“本文提出的损失”。严格复现前，应确认作者实现或完整配置是否包含熵系数；若确认未使用，可设 $c_e=0$。若暂时无法确认，则应明确记录这一不确定性，将是否启用熵正则作为单独的消融设置，而不是把“未确认”当作“未使用”。

##### 修正示例：用当前策略的新样本估计熵

原稿的实现是：

```python
entropy = (
    dist.entropy()
    + torch.log(1.0 - action.square() + 1e-6)
).sum(dim=-1)
```

如果 `action` 是**从当前策略分布新采样**的，这可以作为式 (16) 的 Monte Carlo 数值估计；若还要通过采样路径计算熵正则梯度，则应使用可重参数化采样。PPO 更新时，`action_for_eval` 是 rollout 阶段保存的旧动作；经过多个 PPO epoch 后，这些动作不再是当前策略新采样的样本。因此，用它们估算当前策略熵并不合适。

更清楚的写法是：计算 PPO 概率时仍评估 rollout 里保存的动作；计算熵时另外从**当前策略**重新采样：

```python
# PPO 更新时：评估 rollout 保存的 action_for_eval
# 反变换和 Jacobian 修正使用同一个限幅后的动作
a_eval = action_for_eval.clamp(-0.999999, 0.999999)
u_eval = torch.atanh(a_eval)
logp_eval = (
    dist.log_prob(u_eval)
    - torch.log(1.0 - a_eval.square() + 1e-6)
).sum(dim=-1)

# 熵项：从当前策略另行采样，用式 (17) 做 Monte Carlo 估计
u_entropy = dist.rsample()       # 可重参数化采样，保留梯度路径
a_entropy = torch.tanh(u_entropy)

logp_entropy = (
    dist.log_prob(u_entropy)
    - torch.log(1.0 - a_entropy.square() + 1e-6)
).sum(dim=-1)

entropy = -logp_entropy
```

然后总损失写成：

```python
total_loss = (
    policy_loss
    + value_coef * value_loss
    - entropy_coef * entropy.mean()
)
```

这项熵正则的目的只是帮助策略探索；它没有像 Critic 的 `returns` 那样的回归目标。下方完整示例已同步采用当前策略的新样本估计熵。代码中的 `1e-6` 是数值稳定近似；动作接近饱和时，正式实现还应考虑从未限幅动作直接计算稳定的 log-Jacobian。

---

#### PPO 训练示例：`train_ppo_sru.py`

```python
import math
import torch
import torch.nn as nn
from torch.distributions import Normal


# ==============================================================================
# 1. 模拟交互环境 (生成状态、深度测距与特权高度扫描)
# ==============================================================================
class VectorNavEnv:
    """
    轻量级仿真环境，产生连续观测与奖励流。
    - Actor 观测: [4维前向测距, 线速度 v, 角速度 w, 相对目标 x, y] -> 8 维
    - Critic 特权观测: [Actor 观测 (8维), 8维全向高度扫描] -> 16 维
    - 动作输出: [线速度 v_cmd, 角速度 w_cmd] -> 2 维连续控制量
    """
    def __init__(self, num_envs=8, max_steps=128, device="cpu"):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.actor_obs_dim = 8
        self.critic_obs_dim = 16
        self.action_dim = 2
        self.dt = 0.2

        self.pos = torch.zeros(num_envs, 2, device=self.device)
        self.heading = torch.zeros(num_envs, device=self.device)
        self.vel = torch.zeros(num_envs, device=self.device)
        self.omega = torch.zeros(num_envs, device=self.device)
        self.goal = torch.zeros(num_envs, 2, device=self.device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def _get_obs(self):
        rel_world = self.goal - self.pos
        cos_h = torch.cos(self.heading)
        sin_h = torch.sin(self.heading)
        rel_x = cos_h * rel_world[:, 0] + sin_h * rel_world[:, 1]
        rel_y = -sin_h * rel_world[:, 0] + cos_h * rel_world[:, 1]

        front_scans = torch.clamp(1.5 + 0.3 * torch.randn(self.num_envs, 4, device=self.device), 0.1, 5.0)
        omni_scans = torch.clamp(2.0 + 0.2 * torch.randn(self.num_envs, 8, device=self.device), 0.1, 5.0)

        actor_obs = torch.cat([
            front_scans,
            self.vel.unsqueeze(-1),
            self.omega.unsqueeze(-1),
            rel_x.unsqueeze(-1),
            rel_y.unsqueeze(-1),
        ], dim=-1)

        critic_obs = torch.cat([actor_obs, omni_scans], dim=-1)
        return actor_obs, critic_obs

    def reset(self):
        self.pos.zero_()
        self.heading = torch.rand(self.num_envs, device=self.device) * 2 * math.pi - math.pi
        self.vel.zero_()
        self.omega.zero_()
        self.step_count.zero_()

        angles = torch.rand(self.num_envs, device=self.device) * 2 * math.pi
        radii = 5.0 + 3.0 * torch.rand(self.num_envs, device=self.device)
        self.goal[:, 0] = radii * torch.cos(angles)
        self.goal[:, 1] = radii * torch.sin(angles)
        return self._get_obs()

    def step(self, action):
        target_v = torch.clamp(action[:, 0], -1.0, 1.0) * 0.6 + 0.6
        target_w = torch.clamp(action[:, 1], -1.0, 1.0) * 1.0

        self.vel = 0.7 * self.vel + 0.3 * target_v
        self.omega = 0.7 * self.omega + 0.3 * target_w
        self.heading += self.omega * self.dt
        self.pos[:, 0] += self.vel * torch.cos(self.heading) * self.dt
        self.pos[:, 1] += self.vel * torch.sin(self.heading) * self.dt

        self.step_count += 1
        dist = torch.norm(self.goal - self.pos, dim=-1)

        # 稠密距离奖励 + 动作幅值惩罚
        r_task = 1.0 / (1.0 + (dist / 1.5).square())
        r_reg = 0.05 * (target_v.abs() + target_w.abs())
        rewards = r_task - r_reg

        success = dist < 0.4
        timeout = self.step_count >= self.max_steps
        dones = success | timeout

        if dones.any():
            reset_ids = dones.nonzero(as_tuple=True)[0]
            self.pos[reset_ids] = 0.0
            self.step_count[reset_ids] = 0
            self.vel[reset_ids] = 0.0
            self.omega[reset_ids] = 0.0
            angles = torch.rand(len(reset_ids), device=self.device) * 2 * math.pi
            radii = 5.0 + 3.0 * torch.rand(len(reset_ids), device=self.device)
            self.goal[reset_ids, 0] = radii * torch.cos(angles)
            self.goal[reset_ids, 1] = radii * torch.sin(angles)

        next_actor_obs, next_critic_obs = self._get_obs()
        return next_actor_obs, next_critic_obs, rewards, dones


# ==============================================================================
# 2. 网络架构：SRU-Ours 与非对称 Actor-Critic
# ==============================================================================
class SRUOursCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.xh_to_gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.x_to_spatial = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, state, episode_start):
        # 对应 Eq. (1): SRU 循环前向与环境重置隐状态清零
        h, c = state
        keep_mask = (1.0 - episode_start.float()).unsqueeze(-1)
        h = h * keep_mask
        c = c * keep_mask

        gates = self.xh_to_gates(torch.cat([x, h], dim=-1))
        i_pre, f_pre, o_pre, g_pre = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_pre)
        f = torch.sigmoid(f_pre)
        o = torch.sigmoid(o_pre)

        # 核心：空间调制项 s_t 逐元素相乘
        s_t = self.x_to_spatial(x)
        g = torch.tanh(s_t * g_pre)

        # Refined Gating 机制
        r = i * (1.0 - (1.0 - f).square()) + (1.0 - i) * f.square()
        c_next = r * c + (1.0 - r) * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class AsymmetricActorCritic(nn.Module):
    def __init__(self, actor_obs_dim, critic_obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Actor 分支: Eq. (1)
        self.actor_enc = nn.Sequential(nn.Linear(actor_obs_dim, hidden_dim), nn.Tanh())
        self.actor_sru = SRUOursCell(hidden_dim, hidden_dim)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic 特权评估分支: Eq. (2)
        self.critic_enc = nn.Sequential(nn.Linear(critic_obs_dim, hidden_dim), nn.Tanh())
        self.critic_sru = SRUOursCell(hidden_dim, hidden_dim)
        self.critic_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def init_hidden(self, batch_size, device):
        h_a = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_a = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_c = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (h_a, c_a), (h_c, c_c)

    def forward_actor_step(self, actor_obs, actor_state, episode_start, action_for_eval=None):
        # Eq. (1)
        xa = self.actor_enc(actor_obs)
        ha, next_actor_state = self.actor_sru(xa, actor_state, episode_start)

        # Eq. (3)
        mean = self.actor_mean(ha)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action_for_eval is None:
            pre_tanh = dist.sample()
            action = torch.tanh(pre_tanh)
        else:
            action = action_for_eval.clamp(-0.999999, 0.999999)
            pre_tanh = torch.atanh(action)

        # Eq. (5): 对数概率修正
        log_prob = (dist.log_prob(pre_tanh) - torch.log(1.0 - action.square() + 1e-6)).sum(dim=-1)
        # Eq. (17): 熵使用当前策略的新样本，不复用 rollout 保存的动作
        u_entropy = dist.rsample()
        a_entropy = torch.tanh(u_entropy)
        logp_entropy = (
            dist.log_prob(u_entropy)
            - torch.log(1.0 - a_entropy.square() + 1e-6)
        ).sum(dim=-1)
        entropy = -logp_entropy
        return action, log_prob, entropy, next_actor_state

    def forward_critic_step(self, critic_obs, critic_state, episode_start):
        # Eq. (2) & Eq. (4)
        xc = self.critic_enc(critic_obs)
        hc, next_critic_state = self.critic_sru(xc, critic_state, episode_start)
        val = self.critic_value(hc).squeeze(-1)
        return val, next_critic_state


# ==============================================================================
# 3. GAE 优势计算与 PPO 损失函数定义
# ==============================================================================
@torch.no_grad()
def compute_gae(rewards, values, dones, last_val, gamma=0.99, gae_lambda=0.95):
    """
    对应公式 Eq. (6), (7), (8)
    输入:
      rewards:  [T, N]
      values:   [T, N] (由 V_old 预测)
      dones:    [T, N]
      last_val: [N]
    输出:
      advantages: Eq. (7)
      returns:    Eq. (8) (Critic 拟合的 GT 标签)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        next_v = last_val if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t].float()
        # Eq. (6): delta_t = r_t + gamma * (1-d_t) * V_{t+1} - V_t
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        # Eq. (7): A_hat_t = delta_t + gamma * lambda * (1-d_t) * A_hat_{t+1}
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    # Eq. (8): returns = advantages + values (此项作为 Critic 的 Ground Truth)
    returns = advantages + values
    return advantages, returns


def compute_ppo_losses(
    new_log_probs,
    entropies,
    new_values,
    old_log_probs,
    old_values,
    advantages,
    returns,
    clip_eps=0.2,
    val_clip_eps=0.2,
    c_val=0.5,
    c_ent=0.01,
):
    """
    对应公式 Eq. (9), (10), (11), (12), (13)
    """
    # 优势标准化
    norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Eq. (9): rho_t = exp(log_pi_new - log_pi_old)
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Eq. (10): Actor 裁剪代理损失
    surr1 = ratio * norm_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
    policy_loss = -torch.minimum(surr1, surr2).mean()

    # Eq. (11) & Eq. (12): Critic 裁剪均方误差 (以 returns 为 GT)
    val_clipped = old_values + torch.clamp(new_values - old_values, -val_clip_eps, val_clip_eps)
    vf1 = (new_values - returns).square()
    vf2 = (val_clipped - returns).square()
    value_loss = 0.5 * torch.maximum(vf1, vf2).mean()

    # 策略探索熵奖励
    entropy_loss = -entropies.mean()

    # Eq. (13): 标量优化总目标
    total_loss = policy_loss + c_val * value_loss + c_ent * entropy_loss
    return total_loss, policy_loss, value_loss, -entropy_loss


# ==============================================================================
# 4. 完整训练前向、反向与参数更新流水线 (Training Step Loop)
# ==============================================================================
def train_ppo_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PPO policy on: {device}")

    num_envs = 16
    rollout_len = 64
    total_iterations = 30
    ppo_epochs = 4
    lr = 3e-4

    # 仿真环境
    env = VectorNavEnv(num_envs=num_envs, max_steps=128, device=device)
    model = AsymmetricActorCritic(
        actor_obs_dim=env.actor_obs_dim,
        critic_obs_dim=env.critic_obs_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 环境初始化.
    curr_actor_obs, curr_critic_obs = env.reset()
    actor_state, critic_state = model.init_hidden(num_envs, device)
    curr_starts = torch.ones(num_envs, dtype=torch.bool, device=device)
    # curr_starts 是一个长度为并行环境数 N 的布尔向量，标记当前这批观测是不是各自 episode 的第一步。

    for iteration in range(1, total_iterations + 1):
        # ----------------------------------------------------------------------
        # A. 交互与轨迹采集 (Rollout Phase)
        # ----------------------------------------------------------------------
        buf_actor_obs, buf_critic_obs = [], []
        buf_actions, buf_old_logp, buf_old_val = [], [], []
        buf_rewards, buf_dones, buf_starts = [], [], []

        # 记录本次 Rollout 初始状态，供训练更新反向重放
        # 在代码中，actor_state 是一个包含两个张量的元组 (h, c)，它对应于 LSTM 类循环神经网络的两个核心状态：
        # actor_state[0] 是隐藏状态（Hidden State，记作 $h_t$）
        # actor_state[1] 是细胞记忆状态（Cell State，记作 $c_t$）
        init_actor_state = (actor_state[0].clone(), actor_state[1].clone())
        init_critic_state = (critic_state[0].clone(), critic_state[1].clone())
        #init_critic_state[0]：Critic 的循环隐藏状态 $h_t^{V}$
        # init_critic_state[1]：Critic 的循环细胞状态 $c_t^{V}$

        model.eval()
        with torch.no_grad():
            for t_step in range(rollout_len):
                # 前向推理采样动作与旧值: Eq. (1), (3), (4), (5)
                # 根据当前 Actor 观测和循环记忆，计算策略分布，并产生或评估一个动作。
                action, log_prob, _, actor_state = model.forward_actor_step(
                    curr_actor_obs, actor_state, curr_starts
                )
                # 其中，next_actor_state 要保留给下一时刻继续用；log_prob 用于 PPO 更新；action 则用于环境交互
                # 这个val对应的是公式(4)
                val, critic_state = model.forward_critic_step(
                    curr_critic_obs, critic_state, curr_starts
                )
                # Critic 像一位信息更充分的评估者，给当前局面估计基准价值；Actor 动作的好坏，再由动作后的实际结果相对这个基准有多好来体现。

                # 对，在非对称 Actor–Critic 设置里，env.step(action) 通常会返回同一个环境状态对应的两种观测：
                next_actor_obs, next_critic_obs, rewards, dones = env.step(action)

                buf_actor_obs.append(curr_actor_obs)
                buf_critic_obs.append(curr_critic_obs)
                buf_actions.append(action)
                buf_old_logp.append(log_prob)
                buf_old_val.append(val)
                buf_rewards.append(rewards)
                buf_dones.append(dones)
                buf_starts.append(curr_starts)

                curr_actor_obs, curr_critic_obs = next_actor_obs, next_critic_obs
                curr_starts = dones

            # 在一次 rollout_len 的采样周期内，Actor 和 Critic 的网络参数都不更新。这段时间它们只做前向计算：Actor 产生动作，Critic 估计价值；
            # 获取末尾 Bootstrap Value: Eq. (6)
            last_val, _ = model.forward_critic_step(curr_critic_obs, critic_state, curr_starts)

        # 整理序列张量: [T, num_envs, ...]
        t_actor_obs = torch.stack(buf_actor_obs)
        t_critic_obs = torch.stack(buf_critic_obs)
        t_actions = torch.stack(buf_actions)
        t_old_logp = torch.stack(buf_old_logp)
        t_old_val = torch.stack(buf_old_val)
        t_rewards = torch.stack(buf_rewards)
        t_dones = torch.stack(buf_dones)
        t_starts = torch.stack(buf_starts)

        # ----------------------------------------------------------------------
        # B. 优势函数与 Ground Truth 回报计算: Eq. (6), (7), (8)
        # ----------------------------------------------------------------------
        advantages, returns = compute_gae(t_rewards, t_old_val, t_dones, last_val)

        # ----------------------------------------------------------------------
        # C. 参数反向传播与更新 (Training Step Phase)
        # ----------------------------------------------------------------------
        # 一个直观类比假设机器人按当前规则走了 64 步并把经历录下来。
        # rollout_len 是实际走这 64 步并录像。
        # ppo_epochs 是把同一段录像回看几遍，逐次调整策略。
        # 更新阶段再次调用 forward，是让“新版策略”对录像中的同一时刻、同一个动作重新打分。
        # 它不是重新在环境里走，也不是重新采样一段动作。
        model.train()
        iter_loss, iter_pi_loss, iter_v_loss = 0.0, 0.0, 0.0

        for epoch in range(ppo_epochs):
            new_log_probs, new_entropies, new_values = [], [], []

            train_a_state = (init_actor_state[0].clone(), init_actor_state[1].clone())
            train_c_state = (init_critic_state[0].clone(), init_critic_state[1].clone())

            # 沿时序 T 完整重放 SRU 循环前向 (保持时间依赖性链条)
            # Actor 和 Critic 都有 SRU 记忆。第 $t$ 步的隐藏状态取决于前一步状态，因此必须按时间顺序重放
            for t in range(rollout_len):
                # 重新计算该动作在新参数下的对数概率: Eq. (1), (5)
                _, logp, ent, train_a_state = model.forward_actor_step(
                    actor_obs=t_actor_obs[t],
                    actor_state=train_a_state,
                    episode_start=t_starts[t],   # 每到 episode_start=True 的位置，SRU 会清空对应环境的记忆，避免把上一回合的状态带进新回合。
                    action_for_eval=t_actions[t],  # 不要重新采样动作，而是计算采集阶段实际执行过的那个动作，在当前 Actor 下有多大概率。这样才能够计算PPO的概率比.
                )
                # 重新计算状态价值: Eq. (2), (4)
                # Critic 也要重新前向，因为参数更新后，它对同一观测的价值预测可能改变：
                v, train_c_state = model.forward_critic_step(
                    critic_obs=t_critic_obs[t],
                    critic_state=train_c_state,
                    episode_start=t_starts[t],   # 每到 episode_start=True 的位置，SRU 会清空对应环境的记忆，避免把上一回合的状态带进新回合。
                )
                new_log_probs.append(logp)
                new_entropies.append(ent)
                new_values.append(v)   # 新的 new_value 会与 rollout 后已经算好的 returns 比较，形成 Critic 的价值损失。

            tensor_logp = torch.stack(new_log_probs)
            tensor_ent = torch.stack(new_entropies)
            tensor_val = torch.stack(new_values)

            # 计算 PPO 联合目标: Eq. (9), (10), (11), (12), (13)
            loss, p_l, v_l, _ = compute_ppo_losses(
                new_log_probs=tensor_logp,
                entropies=tensor_ent,
                new_values=tensor_val,
                old_log_probs=t_old_logp,
                old_values=t_old_val,
                advantages=advantages,
                returns=returns,  # GT 标签
            )

            # 梯度清零、反向传播与参数优化
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            iter_loss += loss.item()
            iter_pi_loss += p_l.item()
            iter_v_loss += v_l.item()

        avg_reward = t_rewards.mean().item()
        print(f"Iter [{iteration:02d}/{total_iterations:02d}] | "
              f"Reward: {avg_reward:.4f} | "
              f"Total Loss: {iter_loss/ppo_epochs:.4f} | "
              f"Policy Loss: {iter_pi_loss/ppo_epochs:.4f} | "
              f"Value Loss: {iter_v_loss/ppo_epochs:.4f}")

    torch.save(model.state_dict(), "final_mapless_nav_policy.pt")
    print("Policy successfully trained and saved to final_mapless_nav_policy.pt")


if __name__ == "__main__":
    train_ppo_main()
```

---

### 两阶段如何衔接到导航系统

以下是需要补齐的集成路径；上面的 PPO 示例没有加载 VAE 权重，也没有实现空间注意力模块，两个脚本目前各自独立。

1. **部署时需要的权重**：
   - 提取自 `depth_vae_pretrained.pt` 中的卷积编码器部分（丢弃解码器）。
   - 提取自 `final_mapless_nav_policy.pt` 中的 `actor_enc`、`actor_sru` 与 `actor_mean`。
   - 部署策略推理时不需要 Critic 和 VAE 解码器；PPO 训练仍需要与环境交互采样。
2. **状态流对接**：
   - 将在线双目深度相机图像输入预训练编码器获得二维特征图；
   - 经过空间自注意力与根据本体感觉 $o_t^{\text{prop}}$ 引导的交叉注意力后，压缩为一维向量，拼入 `actor_obs`；
   - 连续传入 `forward_actor_step`，输出高层线速度与角速度指令，传递给机器人的低层运动控制器（如轮足/四足 Locomotion 策略）。

## 局限与个人判断

### 示例能说明什么

公式和代码展示了重建目标、循环记忆、回报估计与参数更新之间的关系。PPO 更新时需要对采样阶段实际执行的动作重新计算概率，并沿时间顺序重放隐状态；回合起点则重置对应环境的记忆。这些机制是理解循环策略训练的重点。

### 距离论文复现还缺什么

- **感知与环境**：VAE 使用随机深度数据；PPO 环境的测距与高度扫描也是随机量，没有真实障碍几何，不能据此验证空间记忆或避障能力。
- **训练配方**：示例未实现论文中的空间注意力、DML 和 TC-Dropout，奖励也采用简化的稠密距离奖励；无法据此归因论文报告的性能提升。
- **终止处理**：示例将到达目标和时间上限统一视为 `done` 并停止自举。若时间上限只是采样截断，应区分 `terminated` 与 `truncated`，保留截断前最终观测计算价值，同时阻断跨回合的 GAE 递推。
- **循环状态与动作分布**：跨 rollout 沿用的隐状态来自更新前参数，需要评估状态陈旧问题；有界动作的概率与熵估计也应在正式训练中核对数值稳定性和实现一致性。
- **验证范围**：本文未提供训练曲线、导航成功率或真机测试结果，代码可读性与公式对应关系不能替代实验验证。

**个人判断**：这份示例适合用来理解循环 PPO 的数据流。最小验证应先换成具有真实障碍与遮挡的环境，固定感知输入、奖励及训练预算，只比较 LSTM 与 SRU，记录成功率、回溯行为和多随机种子波动，再判断空间记忆是否带来收益。

## 参考资料

- 原论文：[Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning](https://arxiv.org/abs/2506.05997)。本文以已有笔记为背景，未单独核验论文版本。
- 相关本地笔记：[SRU：面向长程无地图导航的空间循环记忆](/blog/2026/2026-09-29-sru-spatial-memory)。论文实验与项目入口见该文；本文示例不作官方代码版本分析。

> **Actor 和 Critic 的参数是否共享**，与 **Actor 和 Critic 看到的观测是否相同**，是两件不同的事。
当然。关于 **Asymmetric Actor–Critic（非对称 Actor–Critic）**，最重要的是区分两个常被混在一起的问题：


## 1. “非对称”主要指观测信息不对称

Actor 负责输出动作，最终会部署到机器人上；Critic 负责估计状态价值，帮助 PPO 训练。

如果两者看到相同的信息，例如：

```text
Actor：当前深度、机器人状态、目标
Critic：当前深度、机器人状态、目标
```

这通常称为**观测对称**。Actor 和 Critic 仍然可以使用完全独立的网络参数。

如果 Critic 还看到 Actor 没有的信息，例如：

```text
Actor：前向相机深度、机器人状态、目标
Critic：以上信息 + 仿真器给出的全局高度图
```

这就是**观测非对称**，也就是通常所说的 Asymmetric Actor–Critic。Critic 获得的额外信息常称为特权信息，因为它在仿真训练时可用，但真机部署时未必能获得。论文中，Critic 除了深度输入，还使用额外的 360° 高度扫描；Actor 则基于部署时可用的观测做决策。[Actor与Critic输入](https://www.alphaxiv.org/abs/2506.05997v2?page=22)

## 2. 参数是否共享是另一个独立选择

这两种结构都可能是 Asymmetric Actor–Critic：

**参数独立，观测不对称：**

```text
Actor 的输入 → Actor 网络 → 动作
Critic 的特权输入 → 独立 Critic 网络 → 价值
```

**部分共享参数，观测不对称：**

```text
Actor 输入 → 共享编码器 ─→ Actor 头 → 动作
Critic 特权输入 → 特权编码器 ─→ Critic 头 → 价值
```

因此，“Actor 和 Critic 参数不同”本身并不意味着非对称；反过来，是否共享一部分编码器，也不能单独决定它是否非对称。判断标准主要是：**Critic 是否使用了 Actor 不可获得的额外观测信息。**

## 3. Critic 为什么可以看更多信息？

Actor 的任务是学会在有限传感器信息下作决策；Critic 的任务是评估当前处境和动作结果，以便给 PPO 提供较好的价值基线。

在仿真里，环境可能提供全局地形、精确状态或全向扫描。将这些信息给 Critic，可以帮助它更准确地预测未来回报；这些信息并不一定需要进入部署时的 Actor。

可以把它想成：**机器人只能用自己的相机看路；训练时，评估者还可以查看仿真器里的地形数据，帮助判断这一步行动的长期价值。**Critic 的价值估计会通过优势函数影响 Actor 更新，但 Critic 不会直接替 Actor 输出动作。

