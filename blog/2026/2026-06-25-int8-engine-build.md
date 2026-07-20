---
title: INT8 Engine 构建实战：从 PyTorch 到 TensorRT 的量化、校准与部署
date: 2026-06-25
categories: [Deploy]
---

# INT8 Engine 构建实战：从 PyTorch 到 TensorRT 的量化、校准与部署

在模型部署中，INT8 量化是把模型从「能跑」推向「跑得快、省显存」的关键一步。但很多人第一次接触 TensorRT INT8 时会被几个概念绕晕：

- 为什么需要 calibration？
- scale 和 zero-point 到底是什么？
- `IEntropyCalibrator2` 和 `IMinMaxCalibrator` 选哪个？
- 精度掉了怎么回退某层到 FP16？

这篇博客把 INT8 Engine 的完整构建过程、底层原理和工程实践系统梳理一遍，并附上一个可复现的 ResNet-50 示例。

---

## 一、为什么要做 INT8 量化？

深度学习模型训练时通常用 FP32，推理时如果直接切到 INT8，主要有三个收益：

| 收益 | 说明 | 典型数值 |
|---|---|---|
| **显存/内存占用下降** | 权重和激活从 4 byte 降到 1 byte | 约 4x |
| **内存带宽需求下降** | 同样数据量减少，瓶颈模型吞吐提升 | 2–4x |
| **计算吞吐提升** | NVIDIA Tensor Core 支持 INT8 指令，峰值算力高于 FP16 | 1.5–2x |

代价是**精度损失**。对 CNN、部分 Transformer 来说，INT8 量化后的精度下降通常可以控制在 1% 以内；但对某些对数值敏感的层（如 LayerNorm、Softmax、小通道卷积），掉点可能比较明显。

---

## 二、INT8 量化的核心原理

### 2.1 浮点数与整数的映射

量化本质是把浮点范围 `[r_min, r_max]` 映射到 8-bit 整数范围（通常是 `[-128, 127]`），并尽可能减少误差。

**非对称量化**（asymmetric）：

$$
r = s \cdot (q - z)
$$

- `r`：原始浮点值
- `q`：量化后的整数
- `s`：scale（浮点比例因子）
- `z`：zero-point（零点偏移，保证 `r=0` 时 `q` 为整数）

**对称量化**（symmetric）：

$$
r = s \cdot q
$$

即 `z=0`，通常用于权重，因为权重分布大多以 0 为中心。

### 2.2 scale 怎么算？

对于对称量化：

$$
s = \frac{\max(|r_{\min}|, |r_{\max}|)}{127}
$$

对于非对称量化：

$$
s = \frac{r_{\max} - r_{\min}}{255}, \quad z = \text{round}\left(127 - \frac{r_{\max}}{s}\right)
$$

scale 的精度直接决定量化误差。scale 太大，小数值被「压缩」成一团；scale 太小，大数值被截断（clip）。

### 2.3 为什么权重可以直接量化，激活却需要 calibration？

| 对象 | 是否固定 | 量化方式 |
|---|---|---|
| **权重（Weight）** | 训练完固定 | 直接统计 min/max，计算 scale |
| **激活（Activation）** | 随输入变化 | 必须用真实数据跑一遍，统计分布后计算 scale |

激活的难点在于：

- 不同输入下，某层激活的分布范围可能差几十倍；
- 某些层存在极端 outlier（异常大值），如果按 min-max 量化，会把绝大多数有效值挤到很小的 INT8 区间；
- 需要 trade-off：是保留 outlier（大 scale），还是丢弃极端值（用 percentile 截断）。

这就是 **calibration（校准）** 的核心工作。

### 2.4 Per-Tensor vs Per-Channel

| 粒度 | 含义 | 精度 | 硬件友好度 |
|---|---|---|---|
| **Per-tensor** | 整个 tensor 共享一个 scale | 一般 | 高，几乎所有硬件支持 |
| **Per-channel** | 每个输出通道一个 scale | 更好，尤其 CNN | 中等，需要硬件支持 |

TensorRT 对卷积权重默认可以做 per-channel，激活通常是 per-tensor。在 Orin 等边缘设备上，per-channel 能显著降低 CNN 的精度损失。

---

## 三、TensorRT INT8 Engine 的完整构建流程

```text
1. 准备 PyTorch 模型
        ↓
2. 导出 ONNX（注意 dynamic shape、opset）
        ↓
3. 准备校准数据集（真实业务数据，100~1000 条）
        ↓
4. 选择校准策略（Entropy / MinMax / Percentile）
        ↓
5. 执行 Calibration，生成 calibration cache
        ↓
6. TensorRT Builder 构建 INT8 Engine
        ↓
7. 精度验证：INT8 vs FP32
        ↓
8. 敏感层回退到 FP16/FP32
        ↓
9. 序列化 .plan，部署加载
```

下面逐步展开。

---

## 四、代码实战：ResNet-50 转 INT8 Engine

这个例子覆盖：ONNX 导出 → calibration → INT8 build → 推理验证。

### 4.1 环境要求

```bash
pip install torch torchvision onnx tensorrt pycuda
```

> 注：TensorRT 版本建议 8.6 或 10.x。不同版本 API 略有差异，下面代码以 TensorRT 8.x 风格为主，与你其他博客保持一致。

### 4.2 导出 ONNX

```python
import torch
import torchvision

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
```

### 4.3 准备校准数据集

校准数据必须来自真实业务分布。这里用 ImageNet validation 的一个子集演示。

```python
import os
import cv2
import numpy as np
from glob import glob

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img[16:240, 16:240]  # center crop 224x224
    img = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    img = img / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(img)

def load_calib_data(image_dir, max_samples=500):
    paths = sorted(glob(os.path.join(image_dir, "*.JPEG")))[:max_samples]
    return [preprocess(p) for p in paths]

calib_data = load_calib_data("./imagenet_val_subset", max_samples=256)
print(f"Loaded {len(calib_data)} calibration samples")
```

### 4.4 实现 TensorRT Calibrator

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data, batch_size=16, cache_file="resnet50_calibration.cache"):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0
        
        # 分配 GPU buffer
        self.buffer = cuda.mem_alloc(
            self.batch_size * self.data[0].nbytes
        )
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        
        batch = []
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx >= len(self.data):
                # 末尾不足 batch_size 时，用最后一张填充
                idx = len(self.data) - 1
            batch.append(self.data[idx])
        
        batch = np.stack(batch, axis=0)
        cuda.memcpy_htod(self.buffer, batch)
        self.current_index += self.batch_size
        return [int(self.buffer)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Calibration cache saved to {self.cache_file}")
```

> `IInt8EntropyCalibrator2` 是 TensorRT 推荐的通用校准器，对 outlier 更鲁棒。

### 4.5 构建 INT8 Engine

```python
def build_int8_engine(onnx_path, engine_path, calib_data, batch_size=16):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())
    
    if parser.num_errors > 0:
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")
    
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # 允许精度回退
    
    # 注册 calibrator
    calibrator = Int8EntropyCalibrator(calib_data, batch_size=batch_size)
    config.int8_calibrator = calibrator
    
    # dynamic shape profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, 224, 224),
        opt=(16, 3, 224, 224),
        max=(32, 3, 224, 224)
    )
    config.add_optimization_profile(profile)
    
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")
    
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    print(f"INT8 engine saved to {engine_path}")
    return engine

build_int8_engine(
    "resnet50.onnx",
    "resnet50_int8.plan",
    calib_data,
    batch_size=16
)
```

### 4.6 运行推理

```python
import pycuda.driver as cuda
import pycuda.autoinit

def infer(engine_path, image_np):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 设置当前输入 shape
    context.set_binding_shape(0, image_np.shape)
    
    # 分配输入输出 buffer
    input_size = trt.volume(context.get_binding_shape(0)) * trt.float32.itemsize
    output_size = trt.volume(context.get_binding_shape(1)) * trt.float32.itemsize
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, image_np.astype(np.float32).ravel(), stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    output = np.empty(context.get_binding_shape(1), dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    
    return output

image = preprocess("test.jpg")[np.newaxis, ...]
output = infer("resnet50_int8.plan", image)
print("Top-1 class:", output.argmax())
```

### 4.7 INT8 + FP16 fallback 是什么意思？

在 4.5 的代码里，我们同时开了两个 flag：

```python
config.set_flag(trt.BuilderFlag.INT8)   # 启用 INT8
config.set_flag(trt.BuilderFlag.FP16)   # 允许回退到 FP16
```

这就是 **INT8 + FP16 fallback（混合精度回退）**：TensorRT 优先尝试把每一层跑在 INT8 上，如果某层不适合 INT8，就自动回退到 FP16，而不是报错或强行用 INT8 导致精度崩掉。

#### 为什么需要 fallback？

真实网络里经常存在这些情况：

| 情况 | 结果 |
|---|---|
| 某个算子没有 INT8 kernel 实现 | 不能跑 INT8 |
| 某层激活分布极差，INT8 scale 算不准 | 强行 INT8 精度掉很多 |
| 某层 channel 数太少，INT8 表达力不够 | INT8 反而不如 FP16 |
| 算子支持 INT8，但输入输出数据类型要求 FP16 | 需要中间转换 |

如果不开 FP16 flag，上面任何一层都会让 build 失败，或者被迫走 FP32，速度更慢。打开 FP16 fallback 后，TensorRT 会自动把这些「搞不定」的层交给 FP16 执行。

#### TensorRT 怎么做决策？

简化逻辑如下：

```text
某层能不能跑 INT8？ 且 INT8 的精度/速度优于 FP16？
    ├── 是 → 用 INT8
    └── 否 → 用 FP16
```

大部分情况下这个决策是自动的，你不需要干预。但如果你发现某层 INT8 精度差，也可以手动强制它回退 FP16：

```python
for layer in network:
    if layer.name in ["softmax_layer", "layernorm_1"]:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)
```

#### 一个 YOLO 的直观例子

```text
Backbone (CNN 卷积层)        → 跑 INT8 ✅
Neck (FPN/PAN，也是卷积)      → 跑 INT8 ✅
Head 的 reshape + concat      → 回退 FP16 ⚠️
Head 的小卷积                 → 可能 FP16（channel 太少）
NMS（如果加在 engine 里）      → 通常 FP16 或 CPU
```

最终 engine 是「大部分 INT8 + 小部分 FP16」的混合体，而不是纯 INT8。

#### 利弊

| | 说明 |
|---|---|
| **优点** | 兼顾 INT8 的速度/省显存收益，和 FP16 的兼容性/精度稳定性 |
| **优点** | build 成功率高，不容易因为某层不支持 INT8 而失败 |
| **缺点** | fallback 层多了，整体加速效果会打折扣 |
| **缺点** | INT8 和 FP16 层之间可能有数据格式转换开销 |

一句话：**INT8 + FP16 fallback = 能 INT8 的层尽量 INT8，搞不定的层自动用 FP16 兜底**，是生产环境最常用的混合精度策略。

---

## 五、Calibration 策略详解

### 5.1 TensorRT 提供的三种 Calibrator

| Calibrator | 算法 | 适用场景 |
|---|---|---|
| `IInt8EntropyCalibrator2` | KL 散度最小化 | **通用首选**，对 outlier 鲁棒 |
| `IInt8MinMaxCalibrator` | 直接 min/max | 分布稳定、无极端 outlier |
| `IInt8LegacyCalibrator` | 旧版 calibrator | 仅兼容老项目 |

### 5.2 Entropy Calibrator 在做什么？

它不是简单地取 min/max，而是：

1. 收集每层激活的 FP32 直方图；
2. 尝试不同的 threshold（截断点）；
3. 计算每个 threshold 下，INT8 分布与 FP32 分布的 KL 散度；
4. 选择 KL 散度最小的 threshold，反推 scale。

简单说：**它允许丢弃少量极端大值，换取绝大多数值的更高精度**。

### 5.3 校准数据集要多大？

| 模型类型 | 建议数量 | 说明 |
|---|---|---|
| CNN / ResNet / YOLO | 100–500 张 | 覆盖主要类别和输入尺寸 |
| ViT / DeiT | 500–1000 张 | attention 激活分布更复杂 |
| LLM / VLM | 几千条 prompt | 需要覆盖常见 token 分布 |

关键不是数量，而是**分布代表性**。100 张真实数据通常比 10000 张随机噪声更有效。

### 5.4 缓存复用

`read_calibration_cache` / `write_calibration_cache` 可以把 scale 存下来。只要模型和输入预处理不变，后续 build engine 可以直接读 cache，不用重新跑 calibration，能省十几分钟到几小时。

---

## 六、精度验证与层回退

### 6.1 如何验证 INT8 精度

准备 100~1000 条验证数据，分别用 FP32/FP16 engine 和 INT8 engine 推理，对比：

```python
from scipy.spatial.distance import cosine

fp32_out = infer("resnet50_fp32.plan", image)
int8_out = infer("resnet50_int8.plan", image)

cos_sim = 1 - cosine(fp32_out.flatten(), int8_out.flatten())
rel_err = np.abs(fp32_out - int8_out).mean() / (np.abs(fp32_out).mean() + 1e-6)

print(f"Cosine similarity: {cos_sim:.6f}")
print(f"Relative error: {rel_err:.6f}")
```

经验阈值：

| 指标 | 可接受 | 需警惕 |
|---|---|---|
| 余弦相似度 | > 0.999 | < 0.995 |
| 相对误差 | < 1% | > 5% |
| 任务指标（如 Top-1）| 下降 < 0.5% | 下降 > 1% |

### 6.2 敏感层回退到 FP16

如果某层 INT8 精度损失大，可以强制它用 FP16：

```python
for layer in network:
    if layer.name in ["layer_name_1", "layer_name_2"]:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)
```

TensorRT 会自动处理 INT8 和 FP16 层之间的转换。

### 6.3 常见掉点原因

| 原因 | 表现 | 解决 |
|---|---|---|
| 校准数据分布不对 | 整体精度掉很多 | 换真实数据重跑 calibration |
| 某层 outlier 严重 | 特定层输出偏离大 | 该层回退 FP16，或用 percentile 截断 |
| 小通道卷积 | 小 channel 数 INT8 表达力不足 | per-channel 或 FP16 fallback |
| LayerNorm / Softmax | 对数值敏感 | 通常直接回退 FP16 |
| Batch size 变化 | dynamic shape 范围没设好 | 正确设置 min/opt/max |

---

## 七、YOLO / Transformer / LLM 的特殊注意

### 7.1 YOLO

YOLO 的 backbone 大多是 CNN，INT8 收益明显。但要注意：

- 检测头（head）中的 reshape、concat、gather 经常触发 GPU fallback；
- NMS 通常不在 engine 里做，需要单独实现；
- 小目标检测对量化误差敏感，建议 head 部分回退 FP16。

### 7.2 Vision Transformer

ViT 的 attention 层对数值敏感，INT8 掉点通常比 CNN 大。实践建议：

- Softmax、LayerNorm、GELU 回退 FP16；
- QKV projection、MLP 可以做 INT8；
- 如果精度不够，考虑 SmoothQuant 而不是普通 PTQ。

### 7.3 LLM / VLM

LLM 的 INT8 通常走 TensorRT-LLM、llama.cpp、vLLM 等框架，而不是原生 TensorRT API。常见路径：

| 方案 | 量化对象 | 是否需要 calibration | 精度 |
|---|---|---|---|
| **SmoothQuant** | 权重 + 激活 | 需要（用框架内置 calibration）| 高 |
| **AWQ / GPTQ** | 仅权重 | 不需要 activation calibration | 中高 |
| **FP8** | 权重 + 激活 | 需要 | 高，但需 Hopper/Ada |
| **KV Cache INT8** | KV cache | 不需要 | 长上下文收益大 |

这些框架已经把 calibration、量化、engine build 封装成了脚本，不需要手写 `IInt8EntropyCalibrator2`。

---

## 八、常见坑与最佳实践

### 1. 校准数据不能是随机噪声

随机输入的激活分布和真实数据完全不同，calibration 出来的 scale 会严重偏离。务必用真实业务数据。

### 2. 预处理和训练时保持一致

归一化 mean/std、resize 方式、颜色通道顺序必须和训练时一致。预处理不同，校准结果就不可靠。

### 3. Batch size 和 dynamic shape 要匹配实际场景

build engine 时指定的 `min/opt/max` shape 必须覆盖运行时所有可能的 shape。如果运行时 shape 超出范围，会触发重新构建或报错。

### 4. 不要迷信 INT8 一定更快

INT8 更快的前提是：

- 模型计算密度高（CNN、大矩阵乘）；
- 硬件支持 INT8 Tensor Core；
- 量化后没有大量层回退到 FP16/FP32。

如果模型很小、算子很多、回退严重，INT8 可能不如 FP16。

### 5. 保存 calibration cache

Calibration 可能很慢，保存 cache 后下次 build 可以直接复用：

```python
def read_calibration_cache(self):
    if os.path.exists(self.cache_file):
        with open(self.cache_file, "rb") as f:
            return f.read()
    return None
```

### 6. 区分 Weight-only 和 Full INT8

- **Weight-only**（AWQ/GPTQ）：只省显存，计算时反量化回 FP16，速度提升有限；
- **Full INT8**（TensorRT native INT8 / SmoothQuant）：权重和激活都走 INT8，速度提升更明显。

### 7. 在目标硬件上 build engine

TensorRT engine 对 GPU 架构和驱动版本敏感。在 x86 工作站上 build 的 engine 不一定能在 Orin 上直接用，建议在目标设备上重新 build。

### 8. 量化后一定要做端到端验证

不要只看输出张量的余弦相似度，要在真实任务上测指标：

- 分类：Top-1 / Top-5 accuracy；
- 检测：mAP；
- 分割：mIoU；
- LLM：perplexity、下游任务 score。

---

## 九、完整流程的 Bash 脚本化

把上面的步骤打包成一个可复现脚本：

```bash
#!/bin/bash
set -e

MODEL=resnet50
ONNX_PATH=${MODEL}.onnx
ENGINE_PATH=${MODEL}_int8.plan
CALIB_DIR=./imagenet_val_subset

echo "Step 1: Export ONNX"
python export_onnx.py --model ${MODEL} --output ${ONNX_PATH}

echo "Step 2: Build INT8 engine with calibration"
python build_int8_engine.py \
    --onnx ${ONNX_PATH} \
    --output ${ENGINE_PATH} \
    --calib_dir ${CALIB_DIR} \
    --calib_samples 256 \
    --batch_size 16

echo "Step 3: Validate accuracy"
python validate.py --engine ${ENGINE_PATH} --test_data ./imagenet_test

echo "Done: ${ENGINE_PATH}"
```

---

## 十、小结

INT8 Engine 构建不是简单的「把 FP32 改成 INT8」，而是一个完整的工程流程：

1. **导出 ONNX**：注意 dynamic shape 和 opset；
2. **准备校准数据**：真实业务分布，100~1000 条；
3. **选择 calibrator**：一般用 `IInt8EntropyCalibrator2`；
4. **构建 engine**：开 INT8 flag，注册 calibrator，允许 FP16 fallback；
5. **精度验证**：对比 FP32 输出，必要时层回退；
6. **部署运行**：在目标硬件上加载 .plan 文件。

核心原理记住两点：

- **量化公式**：`r = s · (q - z)`，scale 决定精度；
- **calibration 目的**：用真实数据找最优 scale，平衡 outlier 和大多数值的表达精度。

对于 LLM/VLM，建议直接用 TensorRT-LLM、llama.cpp、vLLM 等框架的量化脚本，它们已经把 calibration 和 engine build 封装好了。

---

> **一句话总结**：INT8 量化的本质是「用真实数据校准 scale，把 FP32 映射到 INT8」；Engine 构建的本质是「让 TensorRT 在精度和速度之间自动做最优选择」。
