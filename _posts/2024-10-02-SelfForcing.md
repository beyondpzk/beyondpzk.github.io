---
layout: post
title: SelfForcing
date: 2024-10-02
categories: [Understandings]
toc:
    sidebar: left
    max_level: 4
---

[TOC]

# SelfForcing

在生成模型（特别是 Diffusion 和 Flow Matching）的语境下， **"Self-Forcing"** 通常指的是 **Self-Conditioning（自条件化）** 技术。

这项技术最早在 *Bit Diffusion* 和 *Analog Bits* 论文中被发扬光大，后来被广泛用于提高 DiT 和 Video Diffusion 的生成质量。

简单来说：**Self-Forcing 就是让模型在去噪的每一步，都能“看到”自己上一步对最终结果的猜测。**

---

### 1. 什么是 Self-Forcing (Self-Conditioning)?

#### 核心直觉
在标准的 Diffusion/Flow Matching 中，模型 $f(x_t, t)$ 只能看到当前的噪声图 $x_t$ 和时间 $t$。它不仅要推断“噪声是什么”，还要推断“原本的图长什么样”。

**Self-Forcing 的思想是：**
如果在推理过程中，模型在 $t=0.9$ 时已经猜到了“这可能是一辆红色的车”，为什么不在 $t=0.8$ 时把这个猜测告诉模型呢？这样模型就不需要重新猜测“车是什么颜色”，而是专注于细化“车的轮廓”。

#### 数学形式
*   **普通 DiT:** 输入是 `cat(x_t)`。
*   **Self-Forcing DiT:** 输入是 `cat(x_t, x_pred)`。
    *   $x_t$: 当前时刻的噪声潜变量。
    *   $x_{pred}$: 模型**上一步**预测出来的 $x_0$（原始数据）或 $x_1$（目标数据）。

---

### 2. 为什么要用它？（特别是对自动驾驶视频）

对于自动驾驶视频生成，Self-Forcing 有两个巨大的优势：

1.  **增强一致性 (Consistency):** 视频生成最怕闪烁（上一帧车是白的，下一帧变灰了）。Self-Forcing 强迫模型在整个去噪过程中保持对物体语义的“记忆”。
2.  **加速收敛:** 模型不需要每次都从零开始猜，它是在“修正”之前的草稿。这通常能显著降低 FID/FVD。

---

### 3. 如何实现 Self-Forcing (代码级详解)

我们需要修改两个地方：**模型输入的通道数** 和 **训练/推理的逻辑**。

#### 第一步：修改模型结构 (DiT)

你需要将 DiT 的输入通道数翻倍（或者增加 Latent 的维度），以便接收 $x_{pred}$。

```python
class DiT_FlowMatching_SelfCond(nn.Module):
    def __init__(self, in_channels=4, ...):
        super().__init__()
        # 修改点：输入通道数变为 2倍 (x_t + x_pred)
        # x_t 是 4通道，x_pred 也是 4通道
        self.x_embedder = nn.Conv2d(in_channels * 2, hidden_size, ...) 
        
        # ... 其他保持不变 ...

    def forward(self, x, t, clip_features, x_self_cond=None):
        """
        x_self_cond: 模型对自己预测结果的猜测 [B, C, H, W]
        """
        # 如果没有提供 self_cond (比如推理的第一步)，用全0填充
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x)
            
        # 在通道维度拼接
        x_in = torch.cat([x, x_self_cond], dim=1)
        
        # 送入 Patch Embedding
        x = self.x_embedder(x_in)
        
        # ... 后续 DiT 逻辑不变 ...
        return v_pred
```

#### 第二步：修改训练逻辑 (Training Loop)

这是最关键的一步。在训练时，我们怎么获得“模型的预测”呢？
通常采用 **50% 概率策略**：
*   50% 的情况：把 `x_self_cond` 设为全 0（让模型学会冷启动）。
*   50% 的情况：先用模型跑一次前向（不传梯度），得到预测值，把这个预测值作为 `x_self_cond` 再跑一次前向（传梯度）。

```python
def train_step(model, x_1, clip_emb):
    # x_1: 真实 Latent (Target)
    # x_0: 噪声 (Source)
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B)
    x_t = (1 - t) * x_0 + t * x_1  # Flow Matching 插值
    
    # === Self-Forcing 逻辑 ===
    if random.random() < 0.5:
        # 情况 A: 也就是 Cold Start，不使用 Self-Conditioning
        x_self_cond = torch.zeros_like(x_1)
    else:
        # 情况 B: 模拟推理过程
        with torch.no_grad():
            # 1. 先用全0作为条件跑一次，得到初步预测
            # 注意：这里为了省事，通常第一次前向用 zeros
            v_pred_draft = model(x_t, t, clip_emb, x_self_cond=torch.zeros_like(x_1))
            
            # 2. 根据 Flow Matching 公式反推 x_1 的预测值
            # v = x_1 - x_0  =>  x_1 = x_0 + v
            # 但我们在 x_t 处，公式略有不同，简单来说：
            # x_1_pred = x_t + (1 - t) * v_pred_draft
            x_1_pred = x_t + (1 - t.view(-1,1,1,1)) * v_pred_draft
            
            # 3. 把这个预测值 detach 掉，作为条件
            x_self_cond = x_1_pred.detach()

    # === 正式训练 ===
    # 这次前向传播才计算梯度
    v_pred = model(x_t, t, clip_emb, x_self_cond=x_self_cond)
    
    loss = mse_loss(v_pred, x_1 - x_0)
    loss.backward()
```

*注意：这会增加约 25%-50% 的训练时间（因为有一半的概率要跑两次模型），但通常能换来更好的收敛效果。*

#### 第三步：修改推理逻辑 (Sampling)

在推理时（Euler Solver），我们需要把上一步计算出的 $x_1$ 预测值缓存下来，传给下一步。

```python
@torch.no_grad()
def sample(model, steps=25):
    x = torch.randn(...) # x_0
    x_self_cond = torch.zeros_like(x) # 初始条件为 0
    
    dt = 1.0 / steps
    
    for i in range(steps):
        t = i / steps
        
        # 输入 x 和 上一步预测的 x_self_cond
        v_pred = model(x, t, clip_emb, x_self_cond=x_self_cond)
        
        # 更新 x (Euler step)
        x_next = x + v_pred * dt
        
        # === 关键：更新 Self-Conditioning ===
        # 计算当前时刻模型预测的最终 x_1
        # x_1 = x_t + (1-t) * v
        x_1_pred = x + (1 - t) * v_pred
        
        # 将这个预测值作为下一步的条件
        x_self_cond = x_1_pred
        
        x = x_next
        
    return x
```

---

### 4. 另一种含义：Autoregressive Self-Forcing (针对长视频)

针对自动驾驶视频，还有一种情况叫 **Autoregressive Self-Forcing**。

如果你是基于前 4 帧生成后 4 帧：
*   **Teacher Forcing (训练时):** 输入是**真实的**前 4 帧，预测后 4 帧。
*   **Self Forcing (微调/推理时):** 输入是**模型自己生成的**前 4 帧，预测后 4 帧。

**如何使用：**
在训练后期，不要总是给模型真实的 $z_{t-1}$ 作为条件。而是先让模型生成 $z_{t-1}$，然后把这个生成的（带有误差的）Latent 作为条件输入进去预测 $z_t$。
这能让模型学会**鲁棒性**：即使上一帧生成得有点瑕疵，下一帧也能救回来，而不是误差累积导致崩坏。

### 总结

对于你的 **DiT + Flow Matching** 验证任务：

1.  **优先实现 Self-Conditioning (通道翻倍法):** 这是提升生成质量的“大杀器”，实现简单，性价比极高。
2.  **训练代价:** 虽然训练变慢了一点，但通常能让你用更少的 Steps (比如 10步) 就能生成高质量图片，推理总时间反而是赚的。
3.  **Latent 空间:** 确保你的 VAE Latent 已经 Normalize 到了标准正态分布附近，否则 Self-Conditioning 可能会因为数值范围波动而不稳定。


## Flow Matching 与Self Forcing

**Flow Matching (FM) + Self-Conditioning (SC)** 是目前生成模型中非常强力的组合（例如 Stable Diffusion 3 和许多最新的 Video Gen 模型都采用了类似的策略）。

在 Flow Matching 的框架下使用 Self-Conditioning，能够显著**拉直**生成轨迹（Straightening the Flow），这意味着你可以用**更少的步数**（比如 10-20 步）生成高质量的视频，这对于计算量巨大的视频生成任务至关重要。

---

### 1. 核心原理：为什么 FM 需要 SC？

在 Flow Matching (Optimal Transport) 中：
*   **状态:** $x_t = (1-t)x_0 + t x_1$
*   **目标:** 预测速度场 $v_t = x_1 - x_0$

模型在 $t$ 时刻看到 $x_t$，试图预测 $v_t$。
如果我们能根据当前的预测 $v_t$，反推出终点 $x_1$ 的猜测值（我们称之为 $\hat{x}_1$），并把这个 $\hat{x}_1$ 喂回给模型，模型就会想：“噢，原来我刚才猜终点长这样，那我根据这个终点修正一下我的速度方向。”

这使得 ODE 的积分路径变得非常直，减少了离散化误差。

---

### 2. 代码实现：Flow Matching + Self-Conditioning

我们需要修改三个部分：**模型输入**、**训练逻辑**、**推理逻辑**。

#### A. 修改模型结构 (DiT)

输入通道数翻倍，因为我们要把当前噪声 $x_t$ 和 猜测的干净图 $\hat{x}_1$ 拼在一起。

```python
class DiT_FM_SC(nn.Module):
    def __init__(self, in_channels=4, hidden_size=384, ...):
        super().__init__()
        # 修改点：输入通道变为 2倍 (x_t + x_self_cond)
        # x_self_cond 通常是模型对 x_1 (Clean Data) 的估计
        self.x_embedder = nn.Conv2d(in_channels * 2, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # ... 其他部分 (TimeEmbed, CLIP Projector, Blocks) 保持不变 ...

    def forward(self, x, t, clip_features, x_self_cond=None):
        """
        x: [B, C, H, W] 当前噪声图
        t: [B] 时间
        clip_features: [B, D] 文本/图像条件
        x_self_cond: [B, C, H, W] 上一步预测的 x_1
        """
        # 1. 处理 Self-Conditioning 输入
        if x_self_cond is None:
            # 如果是推理的第一步，或者训练时的 Cold Start，用全0填充
            x_self_cond = torch.zeros_like(x)
            
        # 2. 拼接输入 (Channel Concatenation)
        x_in = torch.cat([x, x_self_cond], dim=1)
        
        # 3. 正常的 DiT 流程
        x = self.x_embedder(x_in)
        # ... add pos embed ...
        # ... add time & clip condition ...
        # ... transformer blocks ...
        # ... final layer ...
        
        return v_pred  # 输出预测的速度 v
```

#### B. 修改训练逻辑 (Training Loop)

这是最关键的技巧。我们需要以一定概率（通常 50%）执行“两步走”：先猜一次，把猜的结果当条件再算一次梯度。

```python
def train_step_fm_sc(model, x_1, clip_emb, optimizer):
    """
    x_1: 真实数据 (Target)
    clip_emb: 条件
    """
    B = x_1.shape[0]
    
    # 1. 构造 Flow Matching 数据
    x_0 = torch.randn_like(x_1)          # Source (Noise)
    t = torch.rand(B, device=x_1.device) # Time [0, 1]
    
    # 插值 x_t = (1-t)x_0 + t*x_1
    t_expand = t.view(B, 1, 1, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    
    # 目标速度 v_target = x_1 - x_0
    v_target = x_1 - x_0
    
    # 2. Self-Conditioning 逻辑
    # 50% 概率使用 Null Condition (模拟推理第一步)
    # 50% 概率使用 Predicted Condition (模拟推理后续步骤)
    
    if random.random() < 0.5:
        # Case A: Cold Start (不使用 SC)
        x_self_cond = torch.zeros_like(x_1)
    else:
        # Case B: Self-Conditioning
        with torch.no_grad():
            # 先跑一次前向，不传梯度
            # 这里的 x_self_cond 设为 0，因为我们要模拟“从零开始猜”
            v_pred_draft = model(x_t, t, clip_emb, x_self_cond=torch.zeros_like(x_1))
            
            # 根据 Flow Matching 公式反推 x_1
            # 公式推导: x_t = x_0 + t * v  =>  x_1 = x_0 + v
            # 但我们不知道 x_0，我们只知道 x_t 和 v_pred
            # x_t = x_1 - (1-t)v  =>  x_1 = x_t + (1-t)v
            x_1_est = x_t + (1 - t_expand) * v_pred_draft
            
            # Detach 掉，不让梯度传回去
            x_self_cond = x_1_est.detach()

    # 3. 正式训练 (计算梯度)
    optimizer.zero_grad()
    v_pred = model(x_t, t, clip_emb, x_self_cond=x_self_cond)
    
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

#### C. 修改推理逻辑 (Sampling / Solver)

在 Euler 积分过程中，我们需要维护一个 `current_x1_estimate` 变量。

```python
@torch.no_grad()
def sample_fm_sc(model, clip_emb, latent_shape, steps=20):
    B = clip_emb.shape[0]
    device = clip_emb.device
    
    # 1. 初始化
    x = torch.randn(B, *latent_shape).to(device) # x_0
    x_self_cond = torch.zeros_like(x).to(device) # 初始猜测为 0
    
    dt = 1.0 / steps
    
    # 2. 积分循环
    for i in range(steps):
        t_val = i / steps
        t = torch.ones(B, device=device) * t_val
        
        # 前向预测，带入上一步猜测的 x_self_cond
        v_pred = model(x, t, clip_emb, x_self_cond=x_self_cond)
        
        # === 更新 Self-Conditioning ===
        # 计算当前时刻对 x_1 的最佳估计
        # x_1 = x_t + (1 - t) * v
        x_1_est = x + (1 - t_val) * v_pred
        
        # 更新条件，供下一步使用
        x_self_cond = x_1_est
        
        # === Euler Step 更新状态 ===
        # x_{t+1} = x_t + v * dt
        x = x + v_pred * dt
        
    return x # 此时 x 应该非常接近 x_1
```

---
