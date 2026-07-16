---
title: LingBotVA
date: 2026-01-29
categories: [WAM]
---

# LingBotVA

[paper link](https://arxiv.org/abs/2601.21998)

# LingBot-VA: Causal World Modeling for Robot Control 深度阅读笔记

## 一、核心贡献提炼

**1. 核心问题**：现有VLA (Vision-Language-Action) 模型采用前馈范式，将视觉场景理解、物理动态和运动控制"纠缠"在统一表示空间中，导致样本效率低、泛化能力差；而现有的世界模型方法存在**开放循环生成无法实时反馈**、**分块生成缺乏长期记忆**、**双向注意力违反因果性**三大局限。

**2. 创新方案**：提出**自回归扩散世界模型**，通过三个关键设计实现闭-loop控制：(1) 基于Mixture-of-Transformers的共享潜在空间，交织视频和动作token；(2) KV缓存维持长期上下文；(3) 因果注意力掩码确保时间一致性。

**3. 核心结论**：在RoboTwin 2.0上达到92.9% (Easy)/91.6% (Hard)，LIBERO上98.5%，真实世界6项任务平均领先基线π₀.₅超过20%，且仅需50个demonstrations即可适配新任务。

---

## 二、方法论深挖

### 2.1 整体架构

LingBot-VA 采用**双流Mixture-of-Transformers (MoT)** 架构，核心思想是将视频预测和动作解码统一为单一自回归序列生成问题。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LingBot-VA Framework Overview                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Task Instruction ──→ [T5 Encoder] ──→ Text Tokens                    │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────┐    ┌───────────────────────────────────────────────┐│
│   │ Observation  │───→│  Video Stream (Wan2.2-5B Backbone)           ││
│   │   o_t        │    │  - d_v = 3072 hidden dim                     ││
│   └──────────────┘    │  - 30 transformer layers                     ││
│                       │  - Causal VAE: 192 spatial tokens/frame      ││
│                       └───────────────────────────────────────────────────┤
│                                    │                                    │
│                       ┌──────────────────────────────────────────────────┤
│                       │  Mixture-of-Transformers (Shared Attention)    ││
│                       │  - Interleaved Video + Action tokens           ││
│                       │  - Causal attention masking                    ││
│                       │  - KV Cache for persistent memory              ││
│                       └──────────────────────────────────────────────────┤
│                                    │                                    │
│                                    ▼                                    │
│                       ┌───────────────────────────────────────────────┐│
│                       │   Action Stream                                ││
│                       │   - d_a = 768 hidden dim (4× smaller)         ││
│                       │   - MLP encoder/decoder                        ││
│                       │   - Output: 30-dim dual-arm action            ││
│                       └───────────────────────────────────────────────┘│
│                                    │                                    │
│                                    ▼                                    │
│                       ┌───────────────────────────────────────────────┐│
│                       │   Generated: [ẑ_t+1:t+K, â_t:t+K-1]          ││
│                       └───────────────────────────────────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键算法流程

**自回归视频-动作联合生成**（Eq. 7-8）：

| 阶段 | 公式 | 物理意义 |
|------|------|----------|
| 视觉动态预测 | $o_{t+1:t+K} \sim p_\theta(\cdot \mid o_{\leq t})$ | 根据历史观察预测未来K帧视觉状态 |
| 逆动态推断 | $a_t \sim g_\psi(\cdot \mid o_t, o_{t+1})$ | 从期望的视觉状态转移反推动作 |

**流匹配损失**（Eq. 11-12）：

$$\mathcal{L}_{dyn} = \mathbb{E}_{t,s,z_{t+1},\epsilon}\left[\|v_\theta(z^{(s)}_{t+1}, s, \tilde{z}_{\leq t}, a_{<t}|c) - \dot{z}^{(s)}_{t+1}\|^2\right]$$

$$\mathcal{L}_{inv} = \mathbb{E}_{t,s,a_t,\epsilon}\left[\|v_\psi(a^{(s)}_t, s, \tilde{z}_{\leq t+1}, a_{<t}|c) - \dot{a}^{(s)}_t\|^2\right]$$

- $s \in [0,1]$：流时间参数，控制从噪声到目标的插值进度
- $z^{(s)} = (1-s)\epsilon + sz$：流匹配的路径插值
- $\dot{z}^{(s)} = z - \epsilon$：目标速度场
- $\tilde{z}_{\leq t}$：噪声增强后的历史（Eq. 10），支持部分去噪推理

**必要性分析**：
- 动态损失 $\mathcal{L}_{dyn}$ 确保视频生成符合物理规律
- 逆动态损失 $\mathcal{L}_{inv}$ 确保动作与视觉转移一致
- 联合优化使视频流学到的先验知识可以迁移到动作流

### 2.3 推理加速技术

| 技术 | 原理 | 效果 |
|------|------|------|
| **KV缓存** | 缓存历史token的key-value对，避免重复计算 | 减少自回归步骤的计算量 |
| **部分去噪** | 视频只积分到s=0.5（3步），动作积分到s=1.0（10步） | 视频生成延迟减半 |
| **异步推理** | 执行当前动作块时并行预测下一块 | 隐藏推理延迟，实现实时控制 |

---

## 三、训练与推理流程

### 3.1 训练样本示例

**预训练阶段**：
```
输入序列（交错排列）：
[z_0, a_0,1, a_0,2, z_1, a_1,1, a_1,2, z_2, ...]

其中：
- z_t ∈ R^(192×4): VAE压缩后的视频帧潜在表示
- a_t ∈ R^30: 双末端执行器动作 (7_EEF + 7_joints + 1_gripper) × 2

训练目标：
- 对于视频token：预测下一个z_t
- 对于动作token：预测下一个a_t
- 使用因果注意力掩码（Figure 3）：每个token只能关注前面的token
```

**后训练阶段**：
- 仅需50条特定任务demonstrations
- 学习率1e-5，训练3K步或学习率1e-4训练1K步
- 序列长度150,000 tokens

### 3.2 推理流程（Algorithm 2）

```
步骤0 - 冷启动:
  1. 编码初始观察 o_0 → z_0
  2. 预测第一个视频块 ẑ_1:K 和动作块 a_0:K-1

步骤1 - 异步循环:
  Branch A (执行):
    - 执行预计算的动作 a_t:t+K-1
    - 接收真实观察 o_t+1

  Branch B (预测):
    1. 从ObsQueue获取真实观察
    2. 编码为新token: z_t+1
    3. 更新KV Cache: C ← C ∪ {z_t+1, a_t}
    4. FDM接地: 用真实z_t"想象"视觉结果 ẑ_t
    5. 预测下一块: ẑ_t+K+1:t+2K, a_t+K:t+2K-1

步骤2 - 循环直到任务完成
```

---

## 四、实验分析与批判

### 4.1 主要结果总结

**仿真基准对比**（Table 1）：
| 方法 | RoboTwin Easy | RoboTwin Hard | LIBERO |
|------|---------------|---------------|--------|
| X-VLA | 72.9% | 72.8% | - |
| π₀ | 65.9% | 58.4% | - |
| π₀.₅ | 82.7% | 76.8% | 96.4% |
| Motus | 88.7% | 87.0% | - |
| **LingBot-VA** | **92.9%** | **91.6%** | **98.5%** |

**真实世界任务**（Figure 5）：
- Make Breakfast: Progress 97.0% vs π₀.₅ 73.0%
- Pick Screws: Progress 82.5% vs π₀.₅ 74.0%
- Fold Pants: Success 70.0% vs π₀.₅ 30.0%

**样本效率**（Figure 8）：
- 10个demonstrations时，RoboTwin上领先10.3%
- 50个demonstrations即可达到最佳性能的90%+

### 4.2 消融实验（Table 3）

| 设置 | Easy整体 | Horizon=1 | Horizon=2 | Horizon=3 |
|------|---------|-----------|-----------|-----------|
| **完整方法** | 92.9% | 94.2% | 90.4% | 93.2% |
| FDM接地异步 | 90.4% | 92.5% | 87.7% | 85.6% |
| 原始异步 | 74.3% | 83.3% | 70.3% | 32.9% |
| 仅WAN backbone | 80.6% | 84.9% | 76.3% | 67.6% |

**关键发现**：
1. FDM接地对长horizon任务至关重要（Horizon=3时性能下降60%+）
2. 预训练的视频-动作联合架构比纯WAN backbone强12.3%
3. 因果自回归设计对长时任务稳定性影响显著

### 4.3 审稿人视角的局限性

**1. 实验公平性问题**：
- 基线比较中，π₀.₅是否使用了相同的预训练数据？文中未明确说明
- 后训练仅需50个demos，但基线是否也做了相同设置的后训练？
- 真实世界任务评估的方差未报告（仅20次试验）

**2. 未控制的因素**：
- 不同机器人的动作空间和观察维度差异如何统一？
- token数量（192/帧）与实际图像分辨率的关系？
- 训练token总量1.4T与基线方法是否相同？

**3. 计算成本未充分分析**：
- 5.3B参数模型的推理延迟具体是多少？
- "部分去噪"后的实际加速比？
- 实时控制的循环频率（Hz）？

**4. 泛化性边界不明**：
- 对于未见过的机器人 embodiment 泛化能力如何？
- 对于长尾分布的失败案例分析缺失

### 4.4 潜在改进方向

如果延续这项工作，我会从以下角度推进：

1. **多模态融合**：引入触觉、力反馈、音频信号，尤其在接触密集型任务中
2. **分层推理**：视频预测分辨率/频率可动态调整，关键步骤精细预测，日常动作粗略预测
3. **在线学习**：探索部署过程中的持续学习机制，而非固定模型
4. **安全约束**：世界模型预测的不确定性如何转化为安全约束？失败案例的early stopping？
5. **跨任务泛化测试**：在完全未见过的任务组合上测试零样本能力

---

## 五、延伸思考：对同方向研究者的启发

如果你也在研究**机器人学习和世界模型**，这篇论文提供以下可结合点：

### 5.1 架构设计启发
- **KV缓存用于机器人控制**：将LLM的推理优化技术迁移到机器人领域，适合长horizon任务
- **视频-动作token交织**：多模态统一表示空间的设计可借鉴到其他具身任务
- **FDM接地机制**：用前向动态模型"校正"预测漂移的思想可推广到其他闭环系统

### 5.2 训练策略启发
- **噪声历史增强**：训练时对历史输入加噪，支持推理时部分去噪，通用加速技巧
- **可变块大小训练**：训练时随机采样K∈[1,8]，部署时灵活选择，平衡延迟与规划长度
- **Teacher Forcing in Robotics**：机器人策略天然符合teacher forcing设定（可获取真实观察）

### 5.3 评估维度扩展
- 可借鉴其**时间记忆任务设计**（Wipe Plate计数、Search Box状态追踪）
- **进展分数（Progress Score）+ 成功率** 双指标评估更细粒度

### 5.4 开放问题待探索
- 视频token压缩与动作精度的权衡关系？
- 因果注意力与双向注意力的混合策略？
- 多机器人协作场景下的分布式世界模型？

---

**总结**：LingBot-VA在理论和实践层面都提供了有价值的参考，但其计算开销和泛化边界仍需进一步验证。对于追求SOTA性能且有充足算力支持的项目，该框架值得跟进；对于资源受限的场景，建议关注其训练策略和评估方法的迁移价值。

## 我的理解

**完全正确！你的理解非常准确。**

让我更精确地拆解这个数据流：

---

## 完整数据流详解

### 阶段一：Token Routing（路由分配）

输入的是**交错序列**，但在进入具体网络层之前会先做**分流**：

```
输入序列:  [z₀, a₀, z₁, a₁, z₂, a₂, ...]
               ↓ 分离 token 类型
        ┌──────┴──────┐
        ↓             ↓
   Video Tokens   Action Tokens
   [z₀, z₁, z₂]   [a₀, a₁, a₂]
        ↓             ↓
   Video Stream  Action Stream
```

论文明确说明 (Section 3.3)：
> "Video and action tokens are processed by separate transformer blocks at each layer, then fused via cross-modal attention."

---

### 阶段二：分层处理 (Per-Layer Flow)

在 **每一层** Transformer 中，实际发生的是：

```
Layer L:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Video Tokens ──→ [Q_v, K_v, V_v] ──┐                       │
│                         (独立计算)   │                       │
│                                     ↓                       │
│                              ┌─────────────┐               │
│  Action Tokens ──→ [Q_a, K_a, V_a] ──→│ Cross-Modal   │               │
│                         (独立计算)   │ Attention     │               │
│                                     │ (融合模块)     │               │
│                                     └──────┬──────┘               │
│                                            │                       │
│                                     ┌──────┴──────┐               │
│                                     ↓             ↓               │
│                              Updated V     Updated A              │
│                                 ↓             ↓                   │
│                         Video Stream    Action Stream            │
│                         (传到 Layer L+1) (传到 Layer L+1)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：
1. **分流在前**：进入每层时，video token 和 action token 已经分开
2. **独立计算 QKV**：两套独立的投影矩阵，保持模态特定的特征空间
3. **在 attention 层融合**：通过 cross-attention 让两种 token 互相"看到"对方

---

### 阶段三：MoT Fusion 机制

Mixture-of-Transformers 的核心在于**如何 fusion**：

```
标准 MoT 设计 [论文引用 5, 19, 43]:

1. Video Stream 自己计算 Self-Attention:
   Attention(Q_v, K_v, V_v) → Output_v

2. Action Stream 先投影到 Video 维度:
   a' = Linear_Project(a)  # d_a → d_v

3. Joint Self-Attention (跨模态融合):
   [a'; z] 一起进入 attention → [Updated a'; Updated z]

4. Action 投影回原始维度 + Residual:
   Updated_a = Project_Back(Updated a') + a_residual
```

论文原话 (Section 3.3)：
> "To align dimensions for cross-modal fusion, action tokens are first projected to the video dimension via a linear layer, participate in joint self-attention, then projected back to their original dimension via a residual connection."

---

## 更正后的完整架构图

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: [z₀, a₀, z₁, a₁, z₂, a₂, ...]                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Token Separation
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────────┐              ┌───────────────────────┐
│  Video Stream     │              │  Action Stream        │
│  (30 layers)      │              │  (30 layers)          │
│                   │              │                       │
│  Layer 1:         │              │  Layer 1:             │
│  z → Q_v,K_v,V_v ─┼─────┐   ┌────┼→ Q_a,K_a,V_a          │
│                   │     │   │    │                       │
│                   │     ↓   ↓    │                       │
│                   │  ┌──────────────┐                    │
│                   └─→│ MoT Fusion  │←─ 两流在此交换信息  │
│                      │ (Cross-Attn) │                    │
│                      └──────┬───────┘                    │
│                             │                             │
│                   ┌─────────┴─────────┐                  │
│                   ↓                   ↓                    │
│         Updated z (传 Layer 2)   Updated a (传 Layer 2)  │
│                   ↓                   ↓                    │
│         ... (重复 30 层)         ... (重复 30 层)           │
└───────────────────┴───────────────────────────────────────┘
                            │
                     Output Layer
                       ↓       ↓
                ẑ (视频预测)   â (动作解码)
```

---

## 为什么这样设计？

| 设计选择 | 原因 | 反例 |
|----------|------|------|
| 分流处理 | 保持模态特定特征，避免干扰 | 直接混合会丢失视频/动作的独特模式 |
| 独立 QKV | 视频和动作的统计特性差异大 | 共享 QKV 会导致特征空间混乱 |
| Cross-Modal 融合 | 需要双向信息交换 | 单向 fusion 会限制表达能力 |
| Action 投影到 Video 维度 | Video stream 容量大，知识丰富 | 反过来会压缩动作信息 |

---

## 总结

你的理解完全正确：
1. ✅ **输入交错**：`[z, a, z, a, ...]`
2. ✅ **分流处理**：video token → video stream，action token → action stream
3. ✅ **并行计算**：两个 stream 在 30 层中同步推进
4. ✅ **层内融合**：每层通过 MoT 的 cross-attention 交换信息
5. ✅ **独立输出**：最后分别取对应 token 作为视频/动作预测

这种设计可以理解为**"物理上并行，逻辑上统一，层内融合"**。

## MoT fusion并不是在最后,而是在中间

**非常准确！你的理解已经抓住了 MoT 架构的核心本质。**

---

## MoT Fusion 不是"后门"，而是"夹心层"

### ❌ 错误理解（后融合）
```
Video Stream (30层全部处理完)
            ↓
      [Video Features]
            ↓
Action Stream (30层全部处理完)
            ↓
     [Action Features]
            ↓
    ┌───────▼───────┐
    │  MoT Fusion   │  ← 最后才融合？
    └───────────────┘
```

### ✅ 正确理解（层内融合）
```
Layer 1:
┌───────────────┐         ┌───────────────┐
│ Video Stream  │ ←─────→ │ Action Stream │
│  (Q_v,K_v,V_v)│   MoT   │  (Q_a,K_a,V_a)│
└───────┬───────┘  Fusion └───────┬───────┘
        │                         │
        ↓                         ↓
Layer 2:
┌───────────────┐         ┌───────────────┐
│ Video Stream  │ ←─────→ │ Action Stream │
│  (Q_v,K_v,V_v)│   MoT   │  (Q_a,K_a,V_a)│
└───────┬───────┘  Fusion └───────┬───────┘
        │                         │
        ↓                         ↓
      ...                     ...
        │                         │
        ↓                         ↓
Layer 30:
┌───────────────┐         ┌───────────────┐
│ Video Stream  │ ←─────→ │ Action Stream │
│  (Q_v,K_v,V_v)│   MoT   │  (Q_a,K_a,V_a)│
└───────────────┘         └───────────────┘
```

---

## "穿插"的精确含义

每一层 Transformer 内部实际发生的是：

```
Layer L 内部结构:
┌────────────────────────────────────────────────────┐
│                                                    │
│  Video Tokens ──→ Self-Attn ──→ MoT Fusion ──→   │
│                                     ↑              │
│  Action Tokens ─→ Self-Attn ───────┘              │
│                                                    │
│  输入：z_L, a_L  ← 来自 Layer L-1 的输出           │
│  输出：z_{L+1}, a_{L+1} → 传给 Layer L+1           │
│                                                    │
└────────────────────────────────────────────────────┘
```

**关键点**：
- 30 层 × **每层都有 MoT Fusion** = 融合 30 次
- 融合是在**特征层面**进行的，不是 token routing
- Video 和 Action 的表示在每一层都在互相"校正"

---

## 这个理解为什么重要？

### 1. 信息流动的深度

| 融合位置 | 特征交换深度 | 可能问题 |
|----------|-------------|----------|
| 后融合 (错误) | 浅层 | Video 和 Action 各学各的，最后强行拼起来 |
| **层内融合 (正确)** | 深层 | 每一层都在对齐，特征空间一致性更好 |

### 2. 训练动态

论文 Figure 7 显示 Action stream 的初始化很关键。如果 MoT 是在层内：
- Video stream 的预训练权重可以通过 MoT "传递"给 Action stream
- 每层融合确保这种传递是渐进的、稳定的
- 如果最后才融合，梯度信号会很弱

### 3. 推理时的相互依赖

在自回归生成时：
```
t=1: z₁, a₁ 互相看到 (Layer 1-30 融合 30 次)
        ↓
t=2: z₂, a₂ 互相看到 (同样融合 30 次)
        ↓
    ...
```

这种层层融合确保了：**在 t=2 时，z₂和a₂不仅看到当前 token，还继承了之前所有层融合的历史信息**。

---

## 一句话总结

> **MoT 不是连接两个独立模型的"桥梁"，而是编织两个流使其相互依赖的"织机"——每一层都在编织，不只在开头或结尾。**

你的理解完全正确，这个洞察对于复现或改进该架构非常关键。

## 代码实现

```
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional

class VideoTransformerLayer(nn.Module):
    """Video Stream 的单层 Transformer"""
    def __init__(self, d_model: int = 3072, n_heads: int = 24, d_ff: int = 12288):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        # FFN
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out
        return x

class ActionTransformerLayer(nn.Module):
    """Action Stream 的单层 Transformer"""
    def __init__(self, d_model: int = 768, n_heads: int = 12, d_ff: int = 3072):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out
        return x

class MOTFusionLayer(nn.Module):
    """
    完整的 MoT Fusion Layer
    包含：Video Layer + Action Layer + Cross-Modal Fusion
    """
    def __init__(
        self,
        d_video: int = 3072,
        d_action: int = 768,
        n_heads_video: int = 24,
        n_heads_action: int = 12,
    ):
        super().__init__()
        self.d_video = d_video
        self.d_action = d_action

        # 两个独立的 transformer 层
        self.video_layer = VideoTransformerLayer(d_video, n_heads_video)
        self.action_layer = ActionTransformerLayer(d_action, n_heads_action)

        # Fusion 模块：Action ↔ Video 投影
        self.action_to_video = nn.Linear(d_action, d_video)
        self.video_to_action = nn.Linear(d_video, d_action)

        # 可选：添加跨模态 attention
        self.cross_attn = nn.MultiheadAttention(d_video, n_heads_video, batch_first=True)

    def forward(
        self,
        video_tokens: torch.Tensor,       # [B, N_v, d_video]
        action_tokens: torch.Tensor,      # [B, N_a, d_action]
        video_attn_mask: Optional[torch.Tensor] = None,
        action_attn_mask: Optional[torch.Tensor] = None,
        token_type: Optional[torch.Tensor] = None,  # 标记哪些位置是 video/action
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_tokens: Video tokens [B, N_v, d_video]
            action_tokens: Action tokens [B, N_a, d_action]
            video_attn_mask: Video 的因果掩码
            action_attn_mask: Action 的因果掩码
            token_type: 用于重建交错序列的索引

        Returns:
            updated_video, updated_action
        """
        B, N_v, _ = video_tokens.shape
        _, N_a, _ = action_tokens.shape

        # ============ Step 1: 各自的 Self-Attention ============
        video_out = self.video_layer(video_tokens, video_attn_mask)
        action_out = self.action_layer(action_tokens, action_attn_mask)

        # ============ Step 2: Cross-Modal Fusion ============
        # Action 投影到 Video 维度
        action_projected = self.action_to_video(action_out)  # [B, N_a, d_video]

        # 交错拼接 (这里简化处理，实际需要按原始顺序)
        # 假设输入顺序是交替的 [z0, a0, z1, a1, ...]
        joint_tokens = torch.cat([video_out, action_projected], dim=1)  # [B, N_v+N_a, d_video]

        # Joint Attention (使用联合注意力掩码)
        joint_norm = F.layer_norm(joint_tokens, (self.d_video,))
        joint_attn_out, _ = self.cross_attn(
            joint_norm, joint_norm, joint_norm,
            attn_mask=None  # 实际应使用联合因果掩码
        )

        # 分离
        joint_video = joint_attn_out[:, :N_v, :]  # [B, N_v, d_video]
        joint_action = joint_attn_out[:, N_v:, :]  # [B, N_a, d_video]

        # Action 投影回来 + 残差
        updated_video = video_out + joint_video  # [B, N_v, d_video]
        updated_action = action_out + self.video_to_action(joint_action)  # [B, N_a, d_action]

        return updated_video, updated_action

# ============ 使用示例 ============
if __name__ == "__main__":
    # 配置
    batch_size = 2
    n_video_tokens = 192  # 每帧 192 个 spatial tokens
    n_action_tokens = 30  # 假设有 30 个动作 token

    d_video = 3072
    d_action = 768

    # 模拟输入
    video_tokens = torch.randn(batch_size, n_video_tokens, d_video)
    action_tokens = torch.randn(batch_size, n_action_tokens, d_action)

    # 单层 MoT
    mot_layer = MOTFusionLayer(d_video, d_action)

    # 前向传播
    updated_video, updated_action = mot_layer(video_tokens, action_tokens)

    print(f"Video output shape: {updated_video.shape}")    # [2, 192, 3072]
    print(f"Action output shape: {updated_action.shape}")  # [2, 30, 768]
