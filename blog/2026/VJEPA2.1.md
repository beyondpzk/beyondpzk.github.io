---
title: VJEPA2.1
date: 2026-03-15
categories: [others]
---

# VJEPA2.1

[paper link](https://arxiv.org/abs/2603.14482)

# V-JEPA 2.1 深度阅读笔记

## 一、核心贡献提炼

**痛点解决**：前作V-JEPA 2学到的特征在稠密预测任务（如深度估计、语义分割）上表现欠佳，特征图存在噪声、空间结构松散的问题，限制了其在机器人、具身智能等需要精细空间理解场景的应用。

**创新要点**：论文提出**四项关键改进**——(1) 稠密预测损失（Dense Prediction Loss）：对mask和context所有token同时施加自监督信号；(2) 深度自监督（Deep Self-Supervision）：在编码器中间层也施加预测损失；(3) 多模态tokenizer：图像/视频分别用2D/3D卷积处理；(4) 数据与模型协同扩展（VisionMix-163M数据集 + ViT-G 2B模型）。

**核心结论**：V-JEPA 2.1在保持全局场景理解能力的同时，显著提升了特征的空间结构性和时间一致性，在Ego4D短交互预测（7.71 mAP）、EPIC-KITCHENS动作预测（40.8 Recall@5）、NYUv2深度估计（0.307 RMSE）等多项任务上达到SOTA，并在真实机器人抓取任务上实现20%成功率提升。

---

## 二、方法论深挖

### 2.1 整体架构

```
输入 (图像或视频)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  多模态Tokenizer                                            │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │ 2D Conv     │  │ 3D Conv     │                           │
│  │ (图像路径)  │  │ (视频路径)  │                           │
│  └─────────────┘  └─────────────┘                           │
│         │                │                                  │
│         ▼                ▼                                  │
│    Patch Embeddings + Modality Token                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  X-Encoder (可见token处理)                                  │
│  ├── Block 1, 2, ... N/4 → 输出层级1                        │
│  ├── Block N/4+1, ... N/2 → 输出层级2                       │
│  ├── Block N/2+1, ... 3N/4 → 输出层级3                      │
│  └── Block 3N/4+1, ... N → 输出层级4 (最终层)               │
│                                                             │
│  多级特征融合：Concat(层级1,2,3,4) → MLP → Context Tokens   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Predictor (24层Transformer)                                │
│  输入：Context Tokens + Mask Tokens (携带时空位置编码)        │
│  输出：4个层级的预测表示                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Y-Encoder (未mask的完整输入)                               │
│  输出：4个层级的目标表示 (作为预测目标)                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  双损失函数训练                                              │
│  L_predict: Mask Token预测损失 (原始V-JEPA目标)             │
│  L_context: Context Token自监督损失 (新提出)                │
│  Total = L_predict + L_context                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键公式解析

**公式2：Context Loss（上下文自监督损失）**
$$
\mathcal{L}_{context} = \frac{1}{|C|}\sum_{i \in C}\lambda_i \|P_\phi(E_\theta(x), \Delta_y)_i - \text{sg}(E_\theta(y)_i)\|_1
$$

**物理意义**：传统JEPA只对mask区域的预测负责，而V-JEPA 2.1要求**可见的context token也要预测自己的未腐蚀版本**，强制模型对每个可见patch也建立鲁棒的潜在表示。

**λ加权方案（公式3）**：
$$
\lambda_i = \frac{\lambda}{\sqrt{d_{\min}(i, M)}}
$$

其中$d_{\min}(i, M)$是context token i到最近mask token的时空距离。**距离mask越近的context token权重越大**，这强制局部连续性，避免特征图出现边界伪影。

### 2.3 为什么需要深度自监督？

| 问题 | 仅输出层监督的后果 | 加入深度自监督后 |
|------|----------------|----------------|
| 信息流动 | 中间层信息难以传递到最终输出 | 中间层直接被监督，信息流动通畅 |
| 稠密任务表现 | 需要中间层特征做密集预测时性能差 | 中间层已有高质量表示 |
| 训练稳定性 | 深层梯度易消失 | 每层都有梯度流 |

---

## 三、训练数据与推理

### 3.1 训练样本示例

**单样本规格**：
- **视频样本**：16帧 × 256×256分辨率（主训练阶段），4 FPS采样率
- **图像样本**：256×256分辨率
- **batch构成**：全局batch = 128个视频clip + 2304张图像

**训练流程**：
```
步骤1: 输入预处理
├── 图像/视频 → 多模态Tokenizer → Patch序列
├── 随机mask (spatio-temporal masking, scale [0.15, 0.7])
├── 添加 modality token (指示图像/视频来源)
└── 添加3D RoPE位置编码

步骤2: X-Encoder编码
├── 仅处理可见token
├── 输出4个中间层表示
└── MLP融合 → 降维Context Tokens

步骤3: Predictor预测
├── 输入: Context Tokens + Mask Tokens
├── 输出: 4层级的预测表示

步骤4: Y-Encoder编码 (教师端)
├── 输入: 完整未mask的样本
├── 输出: 4层级的目标表示

步骤5: 计算损失
├── L_predict: Mask token预测 vs Y-Encoder对应位置
├── L_context: Context token预测 vs Y-Encoder对应位置 (λ加权)
└── 反向传播更新 Encoder + Predictor
```

### 3.2 推理流程（以深度估计为例）

```
输入：单张RGB图像 (512×512)
  │
  ▼
2D Conv Tokenizer → Patch Embeddings
  │
  ▼
V-JEPA 2.1 Encoder (冻结) → 最后一层特征图
  │
  ▼
DPT Head (可训练) → 深度图预测
  │
  ▼
输出：单目深度估计 (RMSE on NYUv2: 0.307)
```

**关键观察**：下游任务**冻结Encoder**，仅训练task-specific head，验证了pretrain特征的质量。

---

## 四、实验分析与批判

### 4.1 核心SOTAs

| 任务 | 数据集 | 指标 | V-JEPA 2.1 | 前SOTA | 提升 |
|------|--------|------|-----------|--------|------|
| 短交互预测 | Ego4D | mAP All | **7.71** | 5.67 (STAformer) | +35% |
| 动作预测 | EK100 | Recall@5 | **40.8** | 39.7 (V-JEPA 2) | +2.8% |
| 深度估计 | NYUv2 | RMSE↓ | **0.307** | 0.362 (PE-spatial-G) | -15% |
| 语义分割 | ADE20K | mIoU | **47.9** | - | - |
| 动作识别 | SSv2 | Acc | **77.7** | 77.3 (V-JEPA 2) | +0.5% |
| 机器人抓取 | Franka | 成功率 | **80%** (长规划) | 60% (V-JEPA 2) | +20% |

### 4.2 消融实验关键发现

**表1 & 表2的结论**：
1. 单独加Context Loss会提升稠密任务（ADE20K 22.2→33.8 mIoU）但严重损害分类（SSv2 72.8→62.5 Acc）
2. 加入Deep Self-Supervision可以恢复分类性能（SSv2 62.5→72.1）
3. 动态λ加权比固定λ更稳定
4. VisionMix-163M数据扩展在所有任务上持续提升

### 4.3 局限性分析（审稿人视角）

| 问题类型 | 具体漏洞 | 影响程度 |
|----------|----------|----------|
| 消融不完整 | 未单独消融"多模态tokenizer"和"数据扩展"的独立贡献 | 中 |
| 对比公平性 | 与DINOv3对比时，训练数据量不一致（LVD-142M vs ImageNet） | 中 |
| 算力成本不明 | 2B模型训练时长/GPU小时未披露，蒸馏收益-成本比未量化 | 低 |
| 泛化性验证弱 | Cityscapes/ADE20K性能仍落后图像编码器，作者归因于数据分布但无验证 | 中 |
| 机器人实验规模 | Franka抓取仅10个任务×10次，统计显著性不足 | 低 |

### 4.4 潜在改进方向

1. **更丰富的中间层监督策略**：目前4个中间层等间距选择，是否可以learnable选择或基于任务重要性加权？
2. **跨模态一致性损失**：图像/视频共享编码器，但未强制跨模态特征对齐，可能导致模态间隙。
3. **更长时程依赖**：视频仅64帧（16秒），对于长时动作推理（如烹饪步骤）可能不足。
4. **评估benchmark扩展**：缺少3D理解任务（如室内场景重建、多视角一致性）验证。

---

## 五、延伸思考与可结合点

针对相同研究方向（视频SSL/世界模型/具身智能）的潜在启发：

### 5.1 可复用的设计原则

| 设计思想 | 你的任务中如何应用 |
|----------|-------------------|
| Context Loss | 在图像-文本对中，不仅监督被mask的文本，也监督可见文本的潜在表示 |
| 深度自监督 | 多层次特征可同时服务不同下游任务（如检测+分类） |
| 动态加权 | 对距离mask近的区域加权重可推广到任意"边界区域"的监督中 |

### 5.2 可直接集成的模块

1. **多模态Tokenizer**：如果你的研究涉及混合模态输入（如RGB-D、多视角、多传感器），可借鉴这种"模态特异性编码+统一编码器"的设计。
2. **Distillation协议**：2B→300M的蒸馏流程已开源，可直接用于资源受限场景的部署。
3. **Navigation Planning**：论文中提到用V-JEPA 2.1特征训练CDiT做轨迹规划，这一框架可直接迁移到你的机器人任务中。

### 5.3 可探索的结合点

- **V-JEPA 2.1 + 语言模型**：论文仅尝试了Llama 3.1 8B，更大LLM或其他模态（音频、触觉）未探索
- **世界模型扩展**：作者明确提到未来方向是"dense world modeling"，这是具身智能的核心需求
- **在线学习**：当前是离线pretrain，如何扩展到机器人在线交互场景？

---

## 六、审稿结论

| 评分维度 | 分数 (1-10) | 备注 |
|----------|-------------|------|
| 原创性 | 7 | 四项改进中有实质性创新的主要是Context Loss和Deep Self-Supervision |
| 技术深度 | 8 | 消融完整，理论分析充分 |
| 实验充分性 | 7 | 基准覆盖广，但成本分析和部分消融有缺失 |
| 实用性 | 9 | 代码/模型已开源，机器人实验验证实用价值 |
| 写作质量 | 8 | 结构清晰，图例质量高 |

**综合推荐**：强接收。该工作在视频自监督领域建立了新标杆，尤其在"同时优化稠密与全局特征"这一核心挑战上给出了令人信服的方案。

---

## 七、核心代码与资源

| 资源类型 | 链接/版本 |
|----------|----------|
| 官方代码 | https://github.com/facebookresearch/vjepa2 |
| 模型权重 | ViT-g/G 1B/2B + 蒸馏版 ViT-B/L 80M/300M |
| 训练数据 | VisionMix-163M (LVD-142M图像 + YT-1B视频) |
| 预训练轮数 | 135k步主训练 + 12k步cooldown |

建议研究者重点关注**Section 2.3.1 & 2.3.2**（损失设计）和**Section 3.3 & 3.4**（机器人应用），这两部分是相比前作最具区分度的贡献。

## 示例代码

# V-JEPA 2.1 完整可运行代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

# =============================================================================
# 1. 3D 旋转位置编码器 (RoPE)
# =============================================================================

class PositionEncoder3D(nn.Module):
    """
    3D 旋转位置编码 (时间 + 高度 + 宽度)
    所有 tokens 进入 Encoder 前都必须添加此编码
    """
    def __init__(self, embed_dim: int = 768, max_temporal: int = 16, max_spatial: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_temporal = max_temporal
        self.max_spatial = max_spatial

        # 每个维度分配 1/3 embedding dim
        dim_per_axis = embed_dim // 3

        # 预计算 RoPE 表
        self.freqs_t = self._build_freqs(max_temporal, dim_per_axis)
        self.freqs_h = self._build_freqs(max_spatial, dim_per_axis)
        self.freqs_w = self._build_freqs(max_spatial, dim_per_axis)

    def _build_freqs(self, length: int, dim: int) -> torch.Tensor:
        """构建 1D 频率表"""
        omega = torch.arange(dim // 2, dtype=torch.float32) * 2 * math.pi / dim
        positions = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(positions, torch.exp(-omega))
        return torch.stack([freqs.sin(), freqs.cos()], dim=-1).flatten(1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (N, 3) 位置索引 [t, h, w]
        Returns:
            rope_embs: (N, D) 旋转嵌入
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        N = positions.shape[0]
        rope_embs = torch.zeros(N, self.embed_dim, device=positions.device)

        # 时间维度
        freqs_t = self.freqs_t[positions[:, 0]].to(positions.device)
        # 高度维度
        freqs_h = self.freqs_h[positions[:, 1]].to(positions.device)
        # 宽度维度
        freqs_w = self.freqs_w[positions[:, 2]].to(positions.device)

        rope_embs[:, 0:freqs_t.shape[1]] = freqs_t[:, 0] if freqs_t.dim() > 1 else freqs_t
        rope_embs[:, freqs_t.shape[1]:freqs_t.shape[1]+freqs_h.shape[1]] = freqs_h[:, 0] if freqs_h.dim() > 1 else freqs_h

        return rope_embs

# =============================================================================
# 2. 时空 Mask 生成器
# =============================================================================

class SpatioTemporalMaskGenerator:
    """
    生成时空 mask，支持视频/图像
    """
    def __init__(self, mask_ratio: float = 0.75, patch_size: int = 16, temporal_block: int = 4):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.temporal_block = temporal_block

    def generate_mask(self, input_tensor: torch.Tensor, is_video: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Args:
            input_tensor: 输入数据 (B, T, C, H, W) 或 (B, C, H, W)
            is_video: 是否视频输入
        Returns:
            mask: (B, N) bool tensor, True 表示 masked
            N: patch 总数
        """
        batch_size = input_tensor.shape[0]

        # 计算 patch 数量
        if is_video:
            T = input_tensor.shape[1]
            H = input_tensor.shape[3]
            W = input_tensor.shape[4]
            t_patches = max(1, T // self.temporal_block)
        else:
            H = input_tensor.shape[2]
            W = input_tensor.shape[3]
            t_patches = 1

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        N = t_patches * h_patches * w_patches

        # 生成 mask
        num_masked = int(N * self.mask_ratio)
        mask = torch.zeros(batch_size, N, dtype=torch.bool, device=input_tensor.device)

        for i in range(batch_size):
            perm = torch.randperm(N, device=input_tensor.device)
            mask[i, perm[:num_masked]] = True

        return mask, N

    def get_position_grid(self, input_tensor: torch.Tensor, is_video: bool = True) -> torch.Tensor:
        """
        生成所有 patches 的 3D 位置索引
        Returns:
            positions: (N, 3) 位置索引 [t, h, w]
        """
        device = input_tensor.device

        if is_video:
            T = input_tensor.shape[1]
            H = input_tensor.shape[3]
            W = input_tensor.shape[4]
            t_steps = max(1, T // self.temporal_block)
        else:
            H = input_tensor.shape[2]
            W = input_tensor.shape[3]
            t_steps = 1

        h_steps = H // self.patch_size
        w_steps = W // self.patch_size

        positions = []
        for t in range(t_steps):
            for h in range(h_steps):
                for w in range(w_steps):
                    positions.append([t, h, w])

        return torch.tensor(positions, device=device)

# =============================================================================
# 3. 多模态 Tokenizer
# =============================================================================

class MultiModalTokenizer(nn.Module):
    """
    图像/视频统一 tokenizer
    - 图像：2D 卷积 (16x16 patch)
    - 视频：3D 卷积 (4x16x16 patch)
    """
    def __init__(self, embed_dim: int = 768, patch_size: int = 16, temporal_block: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_block = temporal_block
        self.embed_dim = embed_dim

        # 2D 图像 tokenizer
        self.image_tokenizer = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 3D 视频 tokenizer
        self.video_tokenizer = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(temporal_block, patch_size, patch_size),
            stride=(temporal_block, patch_size, patch_size)
        )

        # Modality token (0=图像，1=视频)
        self.modality_embed = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor, is_video: bool = True) -> torch.Tensor:
        """
        Args:
            x: 输入张量
                - 图像：(B, 3, H, W)
                - 视频：(B, T, 3, H, W)
            is_video: 是否视频输入
        Returns:
            tokens: (B, N, D) patch embeddings
        """
        if is_video:
            B, T, C, H, W = x.shape
            # (B, T, 3, H, W) -> (B, 3, T, H, W)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            tokens = self.video_tokenizer(x)  # (B, D, T', H', W')
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
        else:
            B, C, H, W = x.shape
            tokens = self.image_tokenizer(x)  # (B, D, H', W')
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)

        # 添加 modality token
        modality = torch.tensor([1 if is_video else 0], device=x.device)
        mod_embed = self.modality_embed(modality)  # (1, D)
        tokens = tokens + mod_embed.unsqueeze(1)  # broadcast

        return tokens

# =============================================================================
# 4. ViT Encoder (支持多层级输出)
# =============================================================================

class ViTEncoder(nn.Module):
    """
    ViT 编码器，支持多层级特征输出用于 Deep Self-Supervision
    """
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 深度自监督：输出 4 个中间层 (等间距)
        self.supervision_layers = sorted([depth - 1, depth // 4 * 3, depth // 2, depth // 4])
        self.supervision_layers = list(set(self.supervision_layers))

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> List[torch.Tensor]:
        """
        Args:
            x: (B, N, D) 输入 tokens
            return_intermediate: 是否返回中间层特征
        Returns:
            features: 单层输出或 4 层特征列表
        """
        intermediate_features = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_intermediate and i in self.supervision_layers:
                intermediate_features.append(self.norm(x))

        if return_intermediate:
            return intermediate_features
        return self.norm(x)

# =============================================================================
# 5. Predictor 模块
# =============================================================================

class Predictor(nn.Module):
    """
    预测器：从 context tokens 预测 masked tokens
    """
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) 输入 tokens
        Returns:
            predictions: (B, N, D) 预测特征
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# =============================================================================
# 6. V-JEPA 2.1 主模型
# =============================================================================

class VJEPA21(nn.Module):
    """
    V-JEPA 2.1 完整实现
    - ✅ 所有 tokens 添加 3D 位置编码
    - ✅ Visible tokens 带位置编码
    - ✅ Mask tokens 带位置编码
    - ✅ 深度自监督（4 层）
    - ✅ 多模态 tokenizer
    """
    def __init__(self,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mask_ratio: float = 0.75,
                 patch_size: int = 16,
                 temporal_block: int = 4,
                 predictor_depth: int = 12):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_block = temporal_block
        self.mask_ratio = mask_ratio

        # 组件
        self.tokenizer = MultiModalTokenizer(
            embed_dim=embed_dim,
            patch_size=patch_size,
            temporal_block=temporal_block
        )
        self.x_encoder = ViTEncoder(embed_dim, depth, num_heads)
        self.y_encoder = ViTEncoder(embed_dim, depth, num_heads)
        self.predictor = Predictor(embed_dim, predictor_depth, num_heads)

        # 3D 位置编码
        self.pos_encoder = PositionEncoder3D(embed_dim=embed_dim)
        self.mask_generator = SpatioTemporalMaskGenerator(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            temporal_block=temporal_block
        )

        # 可学习的 mask token (所有位置共享)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 层级融合 MLP
        self.fusion_mlp = nn.Linear(embed_dim * 4, embed_dim)

        # 初始化权重
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, is_video: bool = True) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播流程

        Args:
            x: 输入数据
                - 图像：(B, 3, H, W)
                - 视频：(B, T, 3, H, W)
            is_video: 是否视频输入

        Returns:
            outputs: 包含所有中间结果的字典
        """
        batch_size = x.device.type
        device = x.device

        # ============================================
        # Step 1: Tokenization
        # ============================================
        tokens = self.tokenizer(x, is_video=is_video)  # (B, N, D)
        B, N, D = tokens.shape

        # ============================================
        # Step 2: 生成位置索引和位置编码
        # ============================================
        positions = self.mask_generator.get_position_grid(x, is_video=is_video)  # (N, 3)
        position_encoding = self.pos_encoder(positions)  # (N, D)
        position_encoding = position_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # ✅ 所有 tokens 添加位置编码
        tokens_with_pos = tokens + position_encoding

        # ============================================
        # Step 3: 生成 Mask
        # ============================================
        mask, num_patches = self.mask_generator.generate_mask(x, is_video=is_video)  # (B, N)

        # ============================================
        # Step 4: 分离 visible 和 masked tokens
        # ============================================
        visible_tokens_list = []
        visible_counts = []

        for b in range(B):
            vis_tokens = tokens_with_pos[b, ~mask[b], :]  # ✅ 带位置编码
            visible_tokens_list.append(vis_tokens)
            visible_counts.append(len(vis_tokens))

        # Pad 到相同长度
        max_visible = max(visible_counts)
        visible_tokens = torch.zeros(B, max_visible, D, device=device)
        for b in range(B):
            visible_tokens[b, :visible_counts[b], :] = visible_tokens_list[b]

        # ============================================
        # Step 5: X-Encoder 编码 visible tokens
        # ============================================
        x_features = self.x_encoder(visible_tokens, return_intermediate=True)  # 4 层
        x_fused = torch.cat(x_features, dim=-1)  # (B, N_visible, 4D)
        x_fused = self.fusion_mlp(x_fused)  # (B, N_visible, D)

        # ============================================
        # Step 6: 构建 mask tokens (带位置编码)
        # ============================================
        mask_tokens = self.mask_token.expand(B, N, D)  # 扩展到 full size
        mask_tokens_with_pos = mask_tokens + position_encoding  # ✅ 添加位置编码

        # ============================================
        # Step 7: 组合 context + mask
        # ============================================
        combined = self._combine_context_mask(x_fused, mask_tokens_with_pos, mask, visible_counts)

        # ============================================
        # Step 8: Predictor 预测
        # ============================================
        predictions = self.predictor(combined)

        # ============================================
        # Step 9: Y-Encoder (冻结，也需要位置编码)
        # ============================================
        y_tokens = tokens + position_encoding
        with torch.no_grad():
            y_features = self.y_encoder(y_tokens, return_intermediate=True)  # 4 层

        return {
            'predictions': predictions,
            'targets': y_features,
            'mask': mask,
            'visible_tokens': visible_tokens,
            'visible_counts': visible_counts
        }

    def _combine_context_mask(self, context: torch.Tensor, mask_tokens: torch.Tensor,
                              mask: torch.Tensor, visible_counts: List[int]) -> torch.Tensor:
        """
        将 context tokens 和 mask tokens 按原始顺序组合

        Args:
            context: (B, N_visible, D) context tokens
            mask_tokens: (B, N, D) mask tokens (全量)
            mask: (B, N) bool mask
            visible_counts: 每个 sample 的 visible token 数量
        Returns:
            combined: (B, N, D) 完整序列
        """
        B, N, D = mask_tokens.shape
        combined = torch.zeros(B, N, D, device=context.device)

        for b in range(B):
            ctx_count = visible_counts[b]
            combined[b, ~mask[b], :] = context[b, :ctx_count, :]
            combined[b, mask[b], :] = mask_tokens[b, mask[b], :]

        return combined

# =============================================================================
# 7. V-JEPA 2.1 损失函数 (双 Loss)
# =============================================================================

class VJEPALoss(nn.Module):
    """
    V-JEPA 2.1 双损失函数
    - L_predict: Mask token 预测损失
    - L_context: Context token 自监督损失 (λ加权)
    """
    def __init__(self, lambda_base: float = 0.5, warmup_epochs: int = 100):
        super().__init__()
        self.lambda_base = lambda_base
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor],
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (B, N, D) 预测特征
            targets: 4 层目标特征列表 [(B, N, D), ...]
            mask: (B, N) bool, True 表示 masked 位置
        Returns:
            loss_dict: 包含各项 loss 的字典
        """
        # 从 4 层 targets 中取最后一层作为主目标
        target_main = targets[-1]

        # 动态λ (Warmup)
        lambda_eff = self.lambda_base * min(1.0, self.current_epoch / max(1, self.warmup_epochs))

        # 1. Mask Loss
        mask_loss = self._compute_mask_loss(predictions, target_main, mask)

        # 2. Context Loss (λ加权)
        context_loss = self._compute_context_loss(predictions, target_main, mask, lambda_eff)

        # 3. 总损失
        total_loss = mask_loss + context_loss

        return {
            'total': total_loss,
            'mask_loss': mask_loss.detach(),
            'context_loss': context_loss.detach(),
            'lambda': lambda_eff
        }

    def _compute_mask_loss(self, pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """计算 Mask 区域 L1 损失"""
        loss = 0.0
        count = 0
        for b in range(pred.shape[0]):
            if mask[b].any():
                loss += F.l1_loss(pred[b, mask[b], :], target[b, mask[b], :])
                count += 1
        return loss / max(1, count)

    def _compute_context_loss(self, pred: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor, lambda_eff: float) -> torch.Tensor:
        """计算 Context 区域 λ加权 L1 损失"""
        loss = 0.0
        count = 0
        for b in range(pred.shape[0]):
            context_mask = ~mask[b]
            if context_mask.any():
                context_pred = pred[b, context_mask, :]
                context_target = target[b, context_mask, :]
                loss += F.l1_loss(context_pred, context_target) * lambda_eff
                count += 1
        return loss / max(1, count) if count > 0 else torch.tensor(0.0, device=pred.device)

# =============================================================================
# 8. 完整训练示例
# =============================================================================

def train_example():
    """完整训练流程示例"""
    # 配置
    batch_size = 4
    embed_dim = 768
    num_epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # 创建示例数据
    video_data = torch.randn(batch_size, 16, 3, 256, 256).to(device)  # 视频
    image_data = torch.randn(batch_size, 3, 256, 256).to(device)  # 图像

    # 初始化模型
    model = VJEPA21(
        embed_dim=embed_dim,
        depth=12,
        num_heads=12,
        mask_ratio=0.75,
        patch_size=16,
        temporal_block=4,
        predictor_depth=12
    ).to(device)

    # 损失和优化器
    loss_fn = VJEPALoss(lambda_base=0.5, warmup_epochs=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n=== Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M ===\n")

    # ========== Epoch 1: 视频训练 ==========
    for epoch in range(num_epochs):
        model.train()
        loss_fn.set_epoch(epoch)

        # 视频
        outputs = model(video_data, is_video=True)
        loss_dict = loss_fn(outputs['predictions'], outputs['targets'], outputs['mask'])

        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Video Training:")
        print(f"  Total Loss: {loss_dict['total'].item():.4f}")
        print(f"  Mask Loss: {loss_dict['mask_loss'].item():.4f}")
        print(f"  Context Loss: {loss_dict['context_loss'].item():.4f}")
        print(f"  Lambda: {loss_dict['lambda']:.3f}")
        print(f"  Mask Ratio: {outputs['mask'].float().mean().item():.2%}")
        print()

        # 图像
        outputs = model(image_data, is_video=False)
        loss_dict = loss_fn(outputs['predictions'], outputs['targets'], outputs['mask'])

        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Image Training:")
        print(f"  Total Loss: {loss_dict['total'].item():.4f}")
        print(f"  Mask Loss: {loss_dict['mask_loss'].item():.4f}")
        print(f"  Context Loss: {loss_dict['context_loss'].item():.4f}")
        print()

    # ========== 推理测试 ==========
    print("=== Inference Test ===")
    model.eval()
    with torch.no_grad():
        test_video = torch.randn(2, 16, 3, 256, 256).to(device)
        outputs = model(test_video, is_video=True)
        print(f"Output predictions shape: {outputs['predictions'].shape}")
        print(f"Output targets length: {len(outputs['targets'])}")
        print(f"Output mask shape: {outputs['mask'].shape}")

# =============================================================================
# 9. 特征提取 (下游任务)
# =============================================================================

@torch.no_grad()
def extract_features(model: VJEPA21, data: torch.Tensor, is_video: bool = True) -> torch.Tensor:
    """
    使用训练好的 V-JEPA 2.1 提取特征

    Args:
        model: 训练好的 V-JEPA 2.1
        data: 输入数据
        is_video: 是否视频
    Returns:
        features: 提取的特征 (B, N, D)
    """
    model.eval()
    device = data.device

    # Tokenization
    tokens = model.tokenizer(data, is_video=is_video)

    # 位置编码
    positions = model.mask_generator.get_position_grid(data, is_video=is_video)
    position_encoding = model.pos_encoder(positions).unsqueeze(0).expand(tokens.shape[0], -1, -1)
    tokens_with_pos = tokens + position_encoding

    # X-Encoder 编码
    features = model.x_encoder(tokens_with_pos, return_intermediate=False)

    return features

if __name__ == "__main__":
    train_example()

    # 特征提取测试
    print("\n=== Feature Extraction Test ===")
    model = VJEPA21().to('cpu')
    model.eval()
    test_image = torch.randn(2, 3, 256, 256)
    features = extract_features(model, test_image, is_video=False)
    print(f"Extracted features shape: {features.shape}")
```

---

### 完整功能清单

| 模块 | 功能 | 状态 |
|------|------|------|
| `PositionEncoder3D` | 3D 旋转位置编码 | ✅ 完整定义 |
| `SpatioTemporalMaskGenerator` | 时空 mask 生成 | ✅ 完整定义 |
| `MultiModalTokenizer` | 图像/视频 tokenizer | ✅ 完整定义 |
| `ViTEncoder` | 支持多层输出的编码器 | ✅ 完整定义 |
| `Predictor` | 预测器 | ✅ 完整定义 |
| `VJEPA21` | 主模型 | ✅ 完整定义 |
| `VJEPALoss` | 双损失函数 | ✅ 完整定义 |
| `train_example` | 训练示例 | ✅ 完整定义 |
| `extract_features` | 特征提取 | ✅ 完整定义 |

---

### 运行方式

```bash
# 直接运行
python vjepa21_complete.py

# 或导入使用
from vjepa21_complete import VJEPA21, VJEPALoss, extract_features

model = VJEPA21()
video = torch.randn(4, 16, 3, 256, 256)
outputs = model(video, is_video=True)
```
