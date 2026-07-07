---
title: VLM 模型部署实战：推理流水线、预处理与 Token 拼接
date: 2025-05-20
categories: [Deploy]
---

# VLM 模型部署实战：推理流水线、预处理与 Token 拼接

VLM（Vision-Language Model，视觉语言模型）部署和纯文本 LLM 部署有一个本质区别：输入不仅有文本，还有图片。这就引出了三块核心工程代码——**推理流水线、预处理、Token 拼接**。

训练框架（如 Transformers、LLaVA 官方代码）通常已经把这些逻辑写好了，但上线部署时，你必须理解并自己封装它们，尤其是要做量化、服务化、动态 batch、Orin 边缘部署时。

这篇博客把 VLM 部署的核心流程和常见坑系统梳理一遍。

---

## 一、VLM 的典型架构

目前主流的 VLM 大多采用「视觉编码器 + 投影层 + 大语言模型」的三段式架构：

```text
图片
  ↓
Vision Encoder（ViT / CLIP / InternViT / SigLIP）
  ↓
Projector / Connector（MLP / Q-Former / Linear）
  ↓
Image Embeddings ──┐
                   ├──→ LLM Decoder → 文本输出
Text Embeddings ───┘
```

| 模块 | 代表模型 | 作用 |
|---|---|---|
| **Vision Encoder** | CLIP ViT（LLaVA）、InternViT（InternVL）、SigLIP（Qwen2-VL） | 把图片转成视觉 token |
| **Projector** | MLP、Q-Former、Perceiver Resampler | 对齐视觉空间和文本空间 |
| **LLM Decoder** | Vicuna、Llama、Qwen、Phi | 根据图文 token 生成回答 |

部署时，这三个部分通常要分开加载、分别优化，最后再拼起来跑推理。

---

## 二、推理流水线（Inference Pipeline）

一次 VLM 请求的完整流水线如下：

```text
用户请求（图片 + 问题）
         ↓
1. 图片预处理（resize、normalize、patch 切分）
         ↓
2. 文本 tokenization（问题 + 特殊 token）
         ↓
3. 视觉编码 → image tokens
         ↓
4. Projector → image embeddings（对齐到 LLM 空间）
         ↓
5. Token 拼接（image + text embeddings 合成一条序列）
         ↓
6. LLM 解码生成回答
         ↓
7. 后处理（decode、过滤特殊 token）
```

下面分步骤详细讲。

---

## 三、预处理：图片 + 文本

### 3.1 图片预处理

图片预处理的目标是把任意尺寸的图片转成模型训练时见过的固定格式。常见操作：

- **Resize**：缩放到模型要求的尺寸，如 224×224、336×336、448×448；
- **Pad**：保持长宽比，短边填充；
- **Normalize**：按训练时的 mean/std 归一化；
- **ToTensor**：HWC → CHW。

```python
from PIL import Image
import torchvision.transforms as T

image = Image.open("cat.jpg").convert("RGB")

transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

pixel_values = transform(image).unsqueeze(0)  # (1, 3, 448, 448)
```

#### 动态分辨率模型

InternVL2.5、MiniCPM-V 等模型支持动态分辨率：

```text
原图 896x448
   ↓
切成 2 个 448x448 的 tile
   ↓
每个 tile 过 ViT，得到 256 个 image tokens
   ↓
按空间位置拼接，共 512 个 image tokens
```

这类模型的预处理要自己实现 tile 切分和排序逻辑，不能简单用一个 `Resize`。

### 3.2 文本预处理

文本预处理就是 tokenization，但 VLM 比纯文本 LLM 多了**特殊 token**。

以 LLaVA 为例：

```text
USER: <image>
What is in this image?
ASSISTANT:
```

Tokenizer 后变成 token id 序列：

```python
input_ids = [
    user_token_id,
    image_token_id,          # <image>
    what_token_id,
    is_token_id,
    in_token_id,
    this_token_id,
    image_token_id,
    question_mark_token_id,
    assistant_token_id
]
```

不同模型的特殊 token 不同：

| 模型 | 图片占位 token |
|---|---|
| LLaVA | `<image>` |
| InternVL | `<image>` / `<IMG_CONTEXT>` |
| Qwen-VL | `<|image_start|>` `<|image_end|>` |
| MiniCPM-V | `<image>` / `<unk>` |

这些特殊 token 的位置必须和训练时一致，否则模型会「看不到」图片。

---

## 四、Token 拼接（Token Concatenation）

Token 拼接是 VLM 部署最核心的步骤之一。它的本质是：**把图片的 embedding 插入到文本序列中对应的位置**。

### 4.1 为什么需要拼接？

LLM 的输入是一个连续的 embedding 序列：

```text
[e1, e2, e3, e4, ..., eN]
```

对于纯文本 LLM，每个 `e` 来自 embedding 层的查表；对于 VLM，某些 `e` 需要替换成图片经过 ViT + Projector 得到的 image embedding。

### 4.2 单图场景

以 LLaVA 类模型为例：

```python
import torch

# 1. 文本 token ids
prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 2. 找到 <image> token 的位置
image_token_id = tokenizer.convert_tokens_to_ids("<image>")
image_token_index = (input_ids == image_token_id).nonzero(as_tuple=True)[1].item()

# 3. 图片 → image embeddings
with torch.no_grad():
    image_features = model.vision_model(pixel_values)       # (1, N, vision_dim)
    image_features = model.projector(image_features)         # (1, N, llm_dim)

# 4. 文本 → text embeddings
text_embeds = model.get_input_embeddings()(input_ids)       # (1, L, llm_dim)

# 5. 拼接
final_embeds = torch.cat([
    text_embeds[:, :image_token_index, :],   # image 之前的文本
    image_features,                           # 图片 embedding
    text_embeds[:, image_token_index+1:, :]  # image 之后的文本
], dim=1)
```

最后 `final_embeds` 就是 LLM 的输入。

### 4.3 多图场景

如果有多个 `<image>` 占位符，就逐一代替：

```python
image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[1]
final_embeds = text_embeds

offset = 0
for idx, pos in enumerate(image_positions):
    pos = pos.item() + offset  # 前面插入过 image 后位置会偏移
    final_embeds = torch.cat([
        final_embeds[:, :pos, :],
        image_embeds_list[idx],
        final_embeds[:, pos+1:, :]
    ], dim=1)
    offset += image_embeds_list[idx].size(1) - 1
```

### 4.4 多轮对话场景

多轮对话要把历史对话一起拼进去：

```text
SYSTEM: You are a helpful assistant.
USER: <image>
What is in this image?
ASSISTANT: A cat.
USER: What color is it?
ASSISTANT:
```

处理时：

- 历史文本 tokenize 后拼成完整序列；
- 每个历史 `<image>` 位置插入对应图片的 embedding；
- 新生成的 token 通过 KV cache 复用历史计算。

### 4.5 Batch 推理

Batch 场景下，不同样本的 image tokens 数量可能不同（动态分辨率），需要：

- 对文本序列做 left/right padding；
- 生成 attention mask；
- 为每个样本生成正确的 position ids；
- 或者使用变长 batch（如 FlashAttention 的 varlen 接口）。

这是 VLM 服务化比纯文本 LLM 更复杂的地方。

---

## 五、服务化部署还需要什么？

除了推理流水线、预处理、token 拼接，生产环境还要写：

| 模块 | 说明 |
|---|---|
| **模型加载** | 分别加载 vision tower、projector、LLM；权重可能分多个文件 |
| **KV Cache 管理** | 长序列生成时缓存 key/value，减少重复计算 |
| **量化/压缩** | INT4/INT8/AWQ/GPTQ，降低显存占用 |
| **动态批处理** | 多用户同时请求时的 padding、调度、超时 |
| **API 封装** | HTTP/gRPC 接口、请求队列、流式返回 |
| **后处理** | decode token、过滤特殊 token、stop words、截断 |

---

## 六、从 PyTorch 到 ONNX/TensorRT：图会怎么变？

很多人第一次把 VLM 的 PyTorch 模型导出成 ONNX 或 TensorRT engine 时，会发现计算图「面目全非」。这是正常的，因为部署引擎会做大量**图优化（graph optimization）**。

### 6.1 典型例子：QKV 融合

在 PyTorch 中，Transformer 的 Self-Attention 通常这样写：

```python
self.q_proj = nn.Linear(hidden_dim, head_dim * num_heads)
self.k_proj = nn.Linear(hidden_dim, head_dim * num_heads)
self.v_proj = nn.Linear(hidden_dim, head_dim * num_heads)

q = self.q_proj(x)  # (B, L, H)
k = self.k_proj(x)  # (B, L, H)
v = self.v_proj(x)  # (B, L, H)
```

导出的 ONNX 里会有三个独立的 `Gemm` 节点：

```text
x ─→ Gemm(Q) ─→ q
x ─→ Gemm(K) ─→ k
x ─→ Gemm(V) ─→ v
```

三个独立的矩阵乘法，各自启动 kernel、各自读写内存。

### 6.2 TensorRT / ONNX Runtime 会怎么优化？

部署引擎会做一个叫 **QKV fusion** 的优化：

1. 把三个 Linear 的权重 `W_q`、`W_k`、`W_v` 沿输出维度拼接成一个大权重 `W_qkv`；
2. 把三个偏置 `b_q`、`b_k`、`b_v` 拼接成 `b_qkv`；
3. 只做一次矩阵乘法；
4. 把结果切成三段，得到 q、k、v。

```text
x ─→ Gemm(QKV) ─→ Split ─→ q, k, v
```

这样做的收益：

- **减少 kernel launch 次数**：3 次变 1 次；
- **减少内存访问**：x 只读一次，输出也只写一次；
- **更容易利用 Tensor Core**：大矩阵乘法更容易占满 GPU 算力；
- ** fuse 后续操作**：切出来的 q、k、v 可以直接进入 `Reshape + Transpose + Scale` 等，进一步合并。

### 6.3 其他常见图优化

| 优化 | PyTorch 中的样子 | 优化后的样子 |
|---|---|---|
| **LayerNorm 融合** | `Sub → Pow → Mean → Add → Sqrt → Div → Mul → Add` | 一个 `LayerNormalization` 节点 |
| **GELU 融合** | `Mul + Erf + Add + Mul` | 一个 `Gelu` 节点 |
| **Attention 融合** | 多个 Gemm + Reshape + Transpose + Softmax | 一个 `MultiHeadAttention` 或 `FlashAttention` 节点 |
| **Conv + BN + ReLU 融合** | 三个独立节点 | 一个 Conv 节点 |

### 6.4 对部署代码的影响

这些优化意味着：

- 你写的 PyTorch 代码和最终执行的图可能差别很大；
- **不要试图用 ONNX 图的结构去 debug 模型语义**，要看输入输出是否一致；
- 有些优化需要静态 shape 才能做，动态 shape 会限制融合；
- 量化（INT8/INT4）通常在图优化之后做，对图的形态有进一步影响。

### 6.5 自己也可以先做 QKV 融合

如果不想依赖部署引擎自动优化，可以在 PyTorch 模型里就写成 fused 形式：

```python
self.qkv_proj = nn.Linear(hidden_dim, 3 * head_dim * num_heads)

qkv = self.qkv_proj(x)                        # (B, L, 3H)
q, k, v = qkv.chunk(3, dim=-1)                # 每个 (B, L, H)
```

这样导出的 ONNX 天然就是一次 Gemm，部署引擎更容易优化。很多开源 LLM/VLM（如 Llama、Qwen）的源码里已经这么写了。

---

## 七、Orin 上部署 VLM 的特殊注意点

结合 Jetson Orin 的特性，VLM 在 Orin 上部署有一些额外注意：

### 1. 视觉编码器基本只能跑 GPU

ViT 是 Transformer 结构，Attention、LayerNorm、MLP 这些算子 DLA 基本不支持，所以 vision encoder 老老实实跑 GPU。

### 2. LLM 必须量化

一个 7B 级别的 VLM：

- FP16 权重 ≈ 14 GB；
- KV cache + 激活 ≈ 2–6 GB；
- Orin NX 16GB 会爆，AGX Orin 32GB 也紧张。

通常要量化为 INT8 或 INT4（AWQ/GPTQ），权重降到 3.5–7 GB。

### 3. 图片预处理尽量用 GPU

resize、normalize、pad 这些操作如果放在 CPU，会频繁触发 CPU↔GPU 数据拷贝。可以用 OpenCV CUDA、NVIDIA `nvjpeg`、或把预处理写成 CUDA kernel。

### 4. Token 拼接通常是 CPU 代码

构造 input_ids、找 `<image>` 位置、拼 embedding 这些都是轻量操作，放在 CPU 上没问题。但要注意不要成为瓶颈——embedding lookup 可以直接在 GPU 上做。

### 5. 控制上下文长度

VLM 序列通常比纯文本 LLM 长（图片占大量 token），KV cache 很容易爆显存。要根据硬件设定合理的 `max_context_length`。

---

## 八、一个最简的 VLM 推理函数

```python
import torch
from PIL import Image

def vlm_inference(image_path, question, model, tokenizer, image_processor, device="cuda"):
    # 1. 图片预处理
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image).unsqueeze(0).to(device)

    # 2. 构造 prompt
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 3. 视觉编码 + projector
    with torch.no_grad():
        image_features = model.vision_model(pixel_values)
        image_features = model.projector(image_features)

    # 4. 找到 image token 位置并拼接
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_token_index = (input_ids == image_token_id).nonzero(as_tuple=True)[1].item()

    text_embeds = model.get_input_embeddings()(input_ids)
    final_embeds = torch.cat([
        text_embeds[:, :image_token_index, :],
        image_features,
        text_embeds[:, image_token_index+1:, :]
    ], dim=1)

    # 5. LLM 生成
    with torch.no_grad():
        output_ids = model.language_model.generate(
            inputs_embeds=final_embeds,
            max_new_tokens=512
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

> 这只是一个示意图。真实模型（LLaVA、InternVL、Qwen-VL）的具体 API、特殊 token、动态分辨率处理会不同，必须参考官方代码或 `modeling_xxx.py`。

---

## 九、常见坑

### 1. 预处理不一致

训练时用了 `BICUBIC` 插值，部署时用了 `BILINEAR`，精度可能掉 5% 以上。

### 2. 特殊 token 位置错了

图片 embedding 插到了错误的位置，模型会当成纯文本处理，输出胡说八道。

### 3. 动态分辨率没处理 tile 顺序

InternVL 的多图 tile 有固定顺序，顺序错了模型会把图片位置理解错。

### 4. Batch 时 attention mask 没对齐

不同样本图片 token 数量不同，attention mask 和 position ids 必须一一对应。

### 5. KV cache 没复用

多轮对话时如果每次把历史重新 encode，延迟会线性增长。正确做法是复用 KV cache。

---

## 十、小结

VLM 部署的核心比纯文本 LLM 多了视觉侧的处理，关键就是三块：

1. **预处理**：把图片和文本转成模型能接受的 tensor；
2. **推理流水线**：把 ViT、Projector、LLM 串起来执行；
3. **Token 拼接**：把 image embedding 插入到文本序列的正确位置。

在这之上，还要做量化、KV cache 管理、动态 batch、服务化封装，才能真正上线。

---

> **一句话总结**：VLM 部署的难点不是模型本身，而是把「图片」和「文本」这两种完全不同的输入，统一成 LLM 能理解的连续 embedding 序列。
