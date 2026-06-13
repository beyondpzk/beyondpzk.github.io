---
title: 从 PyTorch InternVL2.5-1B 到 Jetson Orin NX 16GB 的 VLA/VLM 嵌入式部署实战
date: 2025-10-04
categories: [Deploy]
---

# 从 PyTorch InternVL2.5-1B 到 Jetson Orin NX 16GB 的 VLA/VLM 嵌入式部署实战

> 这篇文章记录了我们把 **InternVL2.5-1B** 这一多模态视觉-语言模型（VLM），从 PyTorch 环境完整部署到 **NVIDIA Jetson Orin NX 16GB** 的全过程。最终系统能够实时从摄像头采集画面，对当前场景进行理解（例如生成场景描述、回答“前方有什么？”“是否有人？”等问题），并在本地显示结果，开机后自动启动。
>
> 与 YOLO 这类纯检测模型不同，VLM 的部署链路更长：它同时包含 **Vision Transformer（ViT）视觉编码器** 和 **LLM 语言模型**，模型体积更大、推理延迟更高、内存占用更敏感。因此这篇文章会重点讲解如何在 16GB 边缘设备上做 **量化（Quantization）、KV Cache 管理、TensorRT-LLM / llama.cpp 加速**，以及如何把摄像头、前端界面、语音/文本输出整合成一套实时系统。
>
> 同样，我会在最后专门总结：**什么时候需要写 C++，什么时候需要写嵌入式相关代码**。

---

## 0. 项目背景与最终效果

我们做的是一个**边缘端实时场景理解系统**，部署在机器人/无人车上：

- 输入：1 路 CSI/USB 摄像头，分辨率 1280×720，30 FPS。
- 模型：**OpenGVLab/InternVL2.5-1B**，基于 InternViT + Qwen2.5 的多模态模型。
- 平台：**NVIDIA Jetson Orin NX 16GB**（共享内存，实际可用 GPU 显存约 12~14 GB）。
- 任务：
  - 场景描述（Image Captioning）
  - 视觉问答（Visual Question Answering, VQA）
  - 关键目标存在性判断（“画面中是否有行人/车辆/红绿灯？”）
- 输出：本地 HDMI 显示 + 文字转语音播报 + MQTT/HTTP 上报给上位机。
- 可靠性：上电自动启动，程序崩溃后自动恢复。

最终性能：在 **INT4 量化 + 1024 token KV Cache** 配置下，端到端延迟（摄像头取图 → 视觉编码 → LLM 生成 50 token 回答）稳定在 **1.2~1.8 秒**，基本满足“近实时场景理解”的需求。如果只做场景描述（约 20 token），延迟可降到 **0.8~1.0 秒**。

---

## 1. 整体技术链路概览

VLM 的部署链路比 CNN/YOLO 复杂很多：

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. 模型准备     │ --> │  2. 量化/优化    │ --> │  3. 板端引擎    │
│  PyTorch +      │     │  AWQ/GPTQ/      │     │ TensorRT-LLM /  │
│  Transformers   │     │  TensorRT-LLM   │     │ llama.cpp       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  6. 开机自启     │ <-- │  5. 实时系统     │ <-- │  4. 板端环境     │
│  systemd service│     │  capture+encode │     │ JetPack/CUDA/   │
│                 │     │  +VLM+display   │     │ TensorRT/Camera │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

每个阶段需要的技能栈：

| 阶段 | 主要技术 | 是否需要 C++ | 是否需要嵌入式 |
|------|---------|------------|--------------|
| 1. 模型准备 | PyTorch、Transformers、InternVL | 否 | 否 |
| 2. 量化/优化 | AutoAWQ、AutoGPTQ、TensorRT-LLM、ONNX | 可选 | 否 |
| 3. 板端引擎 | TensorRT-LLM engine、GGUF、KV Cache | **是（生产）** | 否 |
| 4. 板端环境 | JetPack、L4T、CUDA、cuDNN、TensorRT | 否 | 是 |
| 5. 摄像头调试 | GStreamer、`nvarguscamerasrc`、V4L2 | 否 | 是 |
| 6. 实时系统 | C++ 推理服务、多线程、ZeroMQ/MQTT | **是** | 是 |
| 7. 开机自启 | systemd、shell 脚本、权限配置 | 否 | 是 |

---

## 2. 阶段一：模型准备与 PyTorch 验证

### 2.1 InternVL2.5-1B 模型结构

InternVL2.5-1B 是一个 **Multimodal Large Language Model（MLLM/VLM）**，结构上分两块：

1. **Vision Encoder**：`InternViT-300M-448px` 或类似规模，把输入图像编码成视觉 token（例如 256 个 image tokens）。
2. **Language Model**：基于 `Qwen2.5-0.5B` 或 `Qwen2.5-1.5B` 的小尺寸 LLM，接收视觉 token + 文本 token，自回归生成回答。

总参数量约 1B，其中 LLM 占大头。

### 2.2 工作站环境准备

```bash
conda create -n internvl python=3.10 -y
conda activate internvl

pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0 accelerate sentencepiece pillow
pip install timm einops
```

### 2.3 下载模型

Hugging Face 下载：

```bash
pip install huggingface-hub
huggingface-cli download OpenGVLab/InternVL2_5-1B --local-dir ./InternVL2_5-1B
```

如果网络受限，可以用 modelscope：

```bash
pip install modelscope
modelscope download --model OpenGVLab/InternVL2_5-1B --local_dir ./InternVL2_5-1B
```

下载后目录结构：

```text
InternVL2_5-1B/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── preprocessor_config.json
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### 2.4 PyTorch 推理验证

先在工作站上跑通原生 PyTorch 推理，确认模型能正常生成文本。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

# 加载模型
model_path = "./InternVL2_5-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 准备图片和提示
image = Image.open("scene.jpg").convert("RGB")
question = "<image>\n请描述这张图片中的场景。"

# InternVL 的特殊 chat 模板
conversation = [
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

# 预处理
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
pixel_values = model.process_images([image], model.config).to(torch.float16).to("cuda")

# 生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        pixel_values=pixel_values,
        max_new_tokens=100,
        do_sample=False,
        num_beams=1
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

> ⚠️ **坑点 1**：`trust_remote_code=True` 必须加，因为 InternVL 的模型实现不是 transformers 内置架构，而是自定义的 `modeling_internvl_chat.py`。
>
> ⚠️ **坑点 2**：`model.process_images` 是 InternVL 特有的方法，会把图片 resize/pad 到模型要求的尺寸（例如 448×448），并归一化成 ViT 输入。

### 2.5 评估内存占用

在 PyTorch FP16 下，用 `nvidia-smi` 观察：

```bash
nvidia-smi
```

对于 1B 模型，FP16 权重大约占 **2~2.5 GB**，加上 KV Cache 和激活，峰值可能在 **4~6 GB**。这在 16GB Orin NX 上理论上能跑，但留给系统和其他进程的空间很小。因此**量化是必须的**。

---

## 3. 阶段二：模型量化与引擎生成

这是 VLM 部署中最关键、也最容易踩坑的一步。Orin NX 16GB 上必须做量化才能跑得稳。

### 3.1 量化方案选择

| 方案 | 精度 | 速度 | 复杂度 | 推荐度 |
|------|------|------|--------|--------|
| FP16 原生 | 最高 | 慢 | 低 | ⭐⭐ |
| INT8 SmoothQuant | 高 | 中等 | 中 | ⭐⭐⭐ |
| AWQ/GPTQ INT4 | 中高 | 快 | 中 | ⭐⭐⭐⭐ |
| TensorRT-LLM INT4/FP8 | 高 | 很快 | 高 | ⭐⭐⭐⭐⭐ |
| llama.cpp Q4_K_M GGUF | 中高 | 快 | 低 | ⭐⭐⭐⭐ |

对于 Orin NX 16GB，推荐两条路：

1. **TensorRT-LLM（生产首选）**：NVIDIA 官方优化最好，能充分利用 Orin 的 Tensor Core 和 FP16/INT8/INT4 能力。但需要自己处理 InternVL 的多模态结构。
2. **llama.cpp + CLIP（快速落地）**：把 LLM 转成 GGUF，把 ViT 用 llama.cpp 的 `clip` 项目或 ONNX 跑。社区支持成熟，适合快速验证和小批量部署。

下面分别讲解。

### 3.2 方案 A：TensorRT-LLM 部署

TensorRT-LLM 是 NVIDIA 针对 LLM 的高性能推理框架，Orin 上也能用（JetPack 6.x 开始支持更好）。

#### 3.2.1 安装 TensorRT-LLM

在 x86 工作站上（用于准备 checkpoint）：

```bash
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com
```

在 Orin NX 上：

```bash
# JetPack 6.x 之后可以用 pip 安装对应 aarch64 版本
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com
```

> ⚠️ **坑点 3**：TensorRT-LLM 的版本必须与 JetPack / CUDA / TensorRT 版本严格匹配。 mismatch 会导致 engine build 失败。

#### 3.2.2 导出 InternVL 到 TensorRT-LLM

TensorRT-LLM 官方支持 LLaVA、VILA 等模型，对 InternVL 没有开箱即用的 example。需要手动改造：

1. **把 LLM backbone 从 safetensors 转成 TensorRT-LLM checkpoint**：
   - 用 `examples/llama/convert_checkpoint.py` 类似的脚本，加载 Qwen2.5-1B 权重。
   - InternVL2.5-1B 的 LLM 部分通常是 Qwen2.5，可以复用 Qwen 的转换脚本。

2. **把 Vision Encoder（InternViT）转成 TensorRT engine**：
   - 用 `torch.onnx.export` 导出 ViT 为 ONNX。
   - 用 `trtexec` 转成 engine。

3. **写多模态推理代码**：
   - 图片 → ViT TensorRT engine → 视觉 token。
   - 视觉 token + 文本 token → TensorRT-LLM engine → 自回归生成。

下面给出 ViT 导出 ONNX 的代码：

```python
import torch
from transformers import AutoModelForCausalLM

model_path = "./InternVL2_5-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu"
)

# 取出 vision encoder
vision_model = model.vision_model
vision_model.eval()

# dummy input: 3x448x448
dummy_input = torch.randn(1, 3, 448, 448).half()

# 导出 ONNX
torch.onnx.export(
    vision_model,
    dummy_input,
    "internvit.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "image_embeds": {0: "batch"}
    },
    opset_version=14
)
```

然后转 TensorRT engine：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=internvit.onnx \
  --saveEngine=internvit_orin_fp16.engine \
  --fp16 \
  --minShapes=pixel_values:1x3x448x448 \
  --optShapes=pixel_values:1x3x448x448 \
  --maxShapes=pixel_values:4x3x448x448 \
  --workspace=4096
```

#### 3.2.3 LLM 部分转 TensorRT-LLM engine

假设已经把 LLM 权重转成 TensorRT-LLM checkpoint：

```bash
python TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir ./InternVL2_5-1B/llm \
  --output_dir ./tllm_checkpoint_1b_int4_awq \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4 \
  --quant_ckpt_path ./internvl_1b_awq.pth

# Build engine
python TensorRT-LLM/examples/qwen/build.py \
  --checkpoint_dir ./tllm_checkpoint_1b_int4_awq \
  --output_dir ./internvl_1b_orin_engine \
  --dtype float16 \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_output_len 256 \
  --use_gpt_attention_plugin float16 \
  --use_gemm_plugin float16 \
  --use_weight_only \
  --weight_only_precision int4 \
  --max_beam_width 1
```

> ⚠️ **坑点 4**：`max_input_len` 和 `max_output_len` 在 build engine 时就固定了，后续运行时不能超过。如果业务需要更长回答，必须重新 build。
>
> ⚠️ **坑点 5**：Orin NX 上 INT4/AWQ 需要 TensorRT-LLM 版本支持。如果报错 `unsupported quantization`，可能需要回退到 INT8 SmoothQuant。

### 3.3 方案 B：llama.cpp + CLIP（推荐快速落地）

llama.cpp 对边缘设备支持极好，Orin NX 上编译运行都很成熟。

#### 3.3.1 转换 LLM 为 GGUF

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 安装依赖
pip install -r requirements.txt

# 把 InternVL 的 LLM 部分导出为 GGUF
python convert_hf_to_gguf.py \
  ../InternVL2_5-1B \
  --outfile ../internvl2_5_1b_q4_k_m.gguf \
  --outtype q4_k_m
```

`q4_k_m` 是推荐量化类型：速度、精度、体积平衡。

#### 3.3.2 处理 Vision Encoder

llama.cpp 的 `examples/llava` 目录下有对 LLaVA 系列多模态模型的支持。InternVL 的 ViT 和投影层需要单独导出：

```bash
# 导出 vision tower 和 mm_projector 为 ggml 格式
python examples/llava/llava_surgery.py \
  -m ../InternVL2_5-1B \
  --skip-llm \
  --output ../internvl_vision_tower.gguf
```

> InternVL 的具体转换脚本可能需要根据模型结构微调，因为 llama.cpp 原生主要支持 LLaVA 结构。实际操作中可能需要修改 `modeling_internvl_chat.py` 中 vision tower 的导出逻辑。

#### 3.3.3 在 Orin 上编译 llama.cpp

```bash
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

关键参数 `-DGGML_CUDA=ON` 启用 CUDA 加速。

#### 3.3.4 运行多模态推理

```bash
./build/bin/llava \
  -m ../internvl2_5_1b_q4_k_m.gguf \
  --mmproj ../internvl_vision_tower.gguf \
  --image ../scene.jpg \
  -p "请描述这张图片中的场景。" \
  --temp 0.2 \
  -n 100
```

### 3.4 方案 C：Transformers + bitsandbytes（仅限验证）

如果只是临时验证，不想转 engine，可以在 Orin 上用 `bitsandbytes` 做 4-bit/8-bit 量化：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "./InternVL2_5-1B",
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True
)
```

> ⚠️ **坑点 6**：bitsandbytes 在 Jetson/ARM64 上支持不稳定，而且速度比 TensorRT-LLM 和 llama.cpp 慢很多。**不建议用于生产**。

---

## 4. 阶段三：Jetson Orin NX 16GB 环境搭建

### 4.1 硬件清单

- NVIDIA Jetson Orin NX 16GB 核心模块 + 载板
- 256GB NVMe SSD（系统盘，16GB 内存很容易被占满，SSD .swap 必须够大）
- CSI 摄像头（IMX219/IMX477）或 USB 摄像头
- 显示器 + 键鼠（调试用）
- 网线/WiFi
- DC 12V~19V 电源

### 4.2 烧录 JetPack

Orin NX 推荐 **JetPack 6.x**（对应 L4T 36.x），因为：

- TensorRT-LLM 对 JetPack 6 支持更好。
- CUDA 12.x，cuDNN 9.x，TensorRT 10.x。
- 对 Transformer Engine/FP8 支持更好。

烧录方式与 AGX Orin 类似，用 SDK Manager 或 SDK Manager CLI：

```bash
sudo apt install ./sdkmanager_2.x.x-xxxx_amd64.deb
sdkmanager --cli install --logintype devzone --product Jetson --version 6.0 --targetos Linux --host --target P3767-0000 --flash all
```

> P3767-0000 是 Orin NX 16GB 的模块型号。

### 4.3 系统基础配置

#### 4.3.1 换源与更新

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo nano /etc/apt/sources.list
```

替换为国内 ARM64 源：

```text
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports jammy-security main restricted universe multiverse
```

```bash
sudo apt update && sudo apt upgrade -y
```

#### 4.3.2 增大 Swap

VLM 推理时内存很容易爆，必须增大 swap：

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

> ⚠️ **坑点 7**：Orin NX 16GB 内存紧张，build TensorRT engine 时如果内存不够会 segfault。建议 build 时关闭其他程序，必要时加大 swap 到 32GB。

#### 4.3.3 安装必要工具

```bash
sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  vim \
  htop \
  tmux \
  v4l-utils \
  python3-pip \
  python3-venv \
  python3-numpy \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools \
  libopencv-dev \
  python3-opencv
```

### 4.4 验证 CUDA / TensorRT / PyTorch

```bash
nvcc -V
dpkg -l | grep -E "cuda|cudnn|tensorrt"

python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

JetPack 6.x 通常对应：

- CUDA 12.2
- cuDNN 8.9
- TensorRT 8.6 或 10.x

### 4.5 安装 Jetson Stats

```bash
sudo -H pip3 install -U jetson-stats
jtop
```

在 `jtop` 里把功耗模式设为 **MAXN**：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

> `jetson_clocks` 会锁定 CPU/GPU 频率到最高，对 LLM 推理有明显提升。

---

## 5. 阶段四：摄像头安装与调试

### 5.1 摄像头选型

| 类型 | 推荐度 | 原因 |
|------|--------|------|
| CSI (IMX219/IMX477) | ⭐⭐⭐⭐⭐ | 低延迟、Jetson 原生支持、ISP 自动处理 |
| USB 3.0 | ⭐⭐⭐ | 即插即用，但延迟和 CPU 占用稍高 |
| GMSL | ⭐⭐⭐⭐ | 车载场景，但需要专用解串板 |

我们使用 **IMX477 CSI 摄像头**，因为它支持 1200 万像素，ISP 输出的画质更好，有利于 VLM 理解细节。

### 5.2 连接与测试

#### 5.2.1 查看设备

```bash
v4l2-ctl --list-devices
```

#### 5.2.2 GStreamer 预览

```bash
gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! \
  nvvidconv flip-method=0 ! \
  'video/x-raw,width=1280,height=720' ! \
  xvimagesink -e
```

### 5.3 在 Python/OpenCV 中捕获

```python
import cv2

GST_STR = (
    "nvarguscamerasrc sensor_id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
)

cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
```

> ⚠️ **坑点 8**：VLM 对图像质量敏感，摄像头曝光、白平衡、对焦不良会直接影响场景理解效果。调试时建议先用 `nvarguscamerasrc` 的 ISP 自动模式，必要时通过 `argus_camera` 工具手动调参。

### 5.4 图像预处理给 VLM

InternVL 通常要求输入 448×448。需要从 1280×720 resize：

```python
import cv2
from PIL import Image

def capture_and_preprocess(cap, input_size=448):
    ret, frame = cap.read()
    if not ret:
        return None

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # 后续由 model.process_images 做 resize/pad/normalize
    return pil_image, frame
```

---

## 6. 阶段五：在 Orin NX 上部署 VLM 推理

### 6.1 Python 快速验证版（llama-cpp-python）

最快验证端到端的方式是用 `llama-cpp-python`。

#### 6.1.1 安装

```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir
```

#### 6.1.2 多模态推理代码

```python
from llama_cpp import Llama
from llama_cpp.llava_chat_handler import Llava15ChatHandler
import cv2
from PIL import Image

# 加载 vision tower 和 LLM
chat_handler = Llava15ChatHandler(
    clip_model_path="./internvl_vision_tower.gguf",
    verbose=False
)

llm = Llama(
    model_path="./internvl2_5_1b_q4_k_m.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    n_gpu_layers=-1,  # 全部 offload 到 GPU
    verbose=False
)

# 打开摄像头
GST_STR = (
    "nvarguscamerasrc sensor_id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
)
cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 保存当前帧
    cv2.imwrite("/tmp/current_frame.jpg", frame)

    # 调用 VLM
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file:///tmp/current_frame.jpg"}},
                    {"type": "text", "text": "请简要描述画面内容。"}
                ]
            }
        ],
        max_tokens=50,
        temperature=0.2
    )

    answer = response["choices"][0]["message"]["content"]
    print(answer)

    # 显示
    cv2.putText(frame, answer[:80], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("VLM Scene Understanding", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

> ⚠️ **坑点 9**：`llama-cpp-python` 的 vision 支持目前主要针对 LLaVA。InternVL 需要确认 chat handler 和 projector 是否兼容。如果不兼容，需要回到 C++ llama.cpp 社区分支或自己修改 projector 加载逻辑。

### 6.2 C++ 生产部署版

生产环境建议用 C++ 重写，原因和 YOLO 部署一样：

- 避免 Python GIL 限制。
- 减少内存拷贝。
- 长期运行更稳定。
- 方便和摄像头、语音合成、MQTT 等其他 C/C++ 组件集成。

#### 6.2.1 C++ 工程结构

```text
cpp_vlm_orin/
├── CMakeLists.txt
├── include/
│   ├── llm_engine.h          # llama.cpp / TensorRT-LLM 封装
│   ├── vision_encoder.h      # ViT engine 或 clip 封装
│   ├── image_processor.h     # resize/pad/normalize
│   ├── camera_capture.h      # GStreamer 捕获
│   └── scene_understander.h  # 业务逻辑封装
├── src/
│   ├── main.cpp
│   ├── llm_engine.cpp
│   ├── vision_encoder.cpp
│   ├── image_processor.cpp
│   ├── camera_capture.cpp
│   └── scene_understander.cpp
└── third_party/
    ├── llama.cpp/            # git submodule
    └── tensorrt_llm/         # 可选
```

#### 6.2.2 核心流程

```cpp
// main.cpp 伪代码
#include "camera_capture.h"
#include "vision_encoder.h"
#include "llm_engine.h"
#include "scene_understander.h"

int main() {
    CameraCapture cam("nvarguscamerasrc ...");
    VisionEncoder vit("internvit_orin_fp16.engine");
    LLMEngine llm("internvl_1b_orin_engine");
    SceneUnderstander app(&vit, &llm);

    while (true) {
        cv::Mat frame = cam.capture();
        if (frame.empty()) continue;

        // 1. 预处理图像
        ImageTensor img_tensor = preprocess_for_vit(frame, 448);

        // 2. ViT 编码
        std::vector<float> image_embeds = vit.encode(img_tensor);

        // 3. 构造 prompt
        std::string prompt = build_prompt("请描述当前场景。");
        std::vector<int> input_ids = tokenizer.encode(prompt);

        // 4. LLM 生成
        std::string answer = llm.generate(input_ids, image_embeds, max_tokens=50);

        // 5. 显示与播报
        render(frame, answer);
        tts_speak(answer);
    }

    return 0;
}
```

#### 6.2.3 关键优化点

1. **零拷贝图像预处理**：用 CUDA kernel 把 `NvBufSurface` 上的 NV12/BGRA 直接转成 ViT 需要的 `NCHW float16`。
2. **KV Cache 复用**：连续帧如果 prompt 类似，可以复用 prefix 的 KV Cache，减少重复计算。
3. **Batch Prefill**：把视觉 token 和文本 token 一起做 prefill，用 TensorRT-LLM 的 `gather_generation` 接口。
4. **异步流水线**：
   - 线程 A：持续捕获最新帧。
   - 线程 B：对上一帧做 ViT 编码。
   - 线程 C：对当前 visual tokens 做 LLM 生成。
   - 线程 D：显示结果、TTS、MQTT 上报。

### 6.3 TensorRT-LLM C++ 推理示例

如果使用 TensorRT-LLM，C++ 侧主要调用 `GptSession` 或 `Executor` API：

```cpp
#include "tensorrt_llm/runtime/gptSession.h"

// 加载 engine
auto runtime = tensorrt_llm::runtime::GptSession::create(...);

// 准备输入
tensorrt_llm::runtime::GenerationInput input{
    endId, padId,
    input_ids_tensor,
    input_lengths_tensor,
    max_new_tokens
};

// 把 image_embeds 作为 prompt embedding 传入
input.promptEmbedding = image_embeds_tensor;

// 生成
tensorrt_llm::runtime::GenerationOutput output;
session->generate(output, input, generationConfig);
```

> TensorRT-LLM 的具体 C++ API 版本差异较大，这里只给出概念性代码。实际使用时需要对照对应版本的 `cpp/include` 头文件。

---

## 7. 阶段六：实时场景理解系统设计

### 7.1 系统架构

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  摄像头采集  │ -> │  图像预处理  │ -> │  ViT 编码    │ -> │  LLM 生成    │
│  GStreamer  │    │ CUDA Kernel │    │ TensorRT/   │    │ TRT-LLM/    │
│  30 FPS     │    │ 448x448     │    │ ONNX        │    │ llama.cpp   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                              ┌-------------------------------┼---------------┐
                              ▼                               ▼               ▼
                        ┌─────────┐                    ┌──────────┐     ┌──────────┐
                        │ 本地显示 │                    │ 语音播报  │     │ MQTT上报 │
                        │ OpenCV  │                    │  TTS     │     │ HTTP     │
                        └─────────┘                    └──────────┘     └──────────┘
```

### 7.2 关键设计取舍

#### 7.2.1 帧率 vs 延迟

VLM 生成一次回答需要 1~2 秒，不可能每帧都跑。因此采用 **“滑动窗口触发”** 策略：

- 每隔 N 秒（例如 2 秒）取一帧最新画面送入 VLM。
- 如果上一帧还在生成中，丢弃中间帧，保证系统始终处理最新画面。
- 对关键问题（“是否有人？”）可以走轻量模型（YOLO）做快速过滤，再决定是否调用 VLM。

#### 7.2.2 提示词工程（Prompt Engineering）

不同任务用不同 prompt：

```python
PROMPTS = {
    "caption": "请用一句话描述这张图片中的场景。",
    "vqa": "<question>",
    "existence": "图片中是否有<target>？请只回答“是”或“否”。",
    "hazard": "图片中是否存在危险情况？请简要说明。"
}
```

#### 7.2.3 并发与队列

```python
import threading
import queue
import time

frame_queue = queue.Queue(maxsize=1)   # 只保留最新帧
result_queue = queue.Queue(maxsize=1)  # 只保留最新结果


def capture_loop(cap):
    while True:
        ret, frame = cap.read()
        if ret:
            # 丢弃旧帧
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)


def inference_loop(vlm):
    while True:
        frame = frame_queue.get()
        pil_img = cv2_to_pil(frame)
        answer = vlm.predict(pil_img, PROMPTS["caption"])
        if not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put((frame, answer))


def render_loop():
    while True:
        frame, answer = result_queue.get()
        cv2.putText(frame, answer[:100], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("VLM", frame)
        cv2.waitKey(1)
```

### 7.3 TTS 语音播报

可以用 `pyttsx3`（离线）或 `edge-tts`（在线）：

```python
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()
```

### 7.4 MQTT/HTTP 上报

```python
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("broker_ip", 1883, 60)

def publish_result(timestamp, image_path, answer):
    payload = {
        "timestamp": timestamp,
        "image": image_path,
        "scene_understanding": answer
    }
    client.publish("robot/vlm/result", json.dumps(payload))
```

---

## 8. 阶段七：开机自启与守护

与 YOLO 部署相同，用 systemd 服务实现开机自启。

### 8.1 启动脚本 `start_vlm.sh`

```bash
#!/bin/bash
set -e

export HOME=/home/nvidia
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 等摄像头和驱动就绪
sleep 15

cd /home/nvidia/vlm_deploy
exec ./cpp_vlm_orin/build/vlm_orin \
  --llm ./models/internvl_1b_orin_engine \
  --vit ./models/internvit_orin_fp16.engine \
  --camera csi
```

### 8.2 systemd 服务

```ini
[Unit]
Description=InternVL2.5 Scene Understanding on Orin NX
After=network.target multi-user.target graphical.target
Wants=network.target

[Service]
Type=simple
User=nvidia
Group=nvidia
WorkingDirectory=/home/nvidia/vlm_deploy
ExecStart=/home/nvidia/vlm_deploy/start_vlm.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vlm-orin

[Install]
WantedBy=multi-user.target
```

### 8.3 启用服务

```bash
sudo cp vlm-orin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vlm-orin.service
sudo systemctl start vlm-orin.service

# 查看日志
sudo journalctl -u vlm-orin.service -f
```

---

## 9. 什么时候写 C++？什么时候写嵌入式代码？

### 9.1 什么时候 Python 就够了？

- **模型验证与 prompt 调试**：快速验证 InternVL 在业务数据上的效果。
- **单帧/离线推理**：处理图片文件、批量生成场景描述。
- **低帧率原型**：对延迟不敏感的 demo，比如 5 秒处理一帧。
- **团队没有 C++ 能力时**：用 `llama-cpp-python` 也能做出可用版本。

### 9.2 什么时候必须写 C++？

- **生产部署**：长期运行、低延迟、内存可控。
- **多路输入/多任务并发**：Python GIL 会严重限制并发。
- **NVMM 零拷贝**：摄像头 buffer 直接在 GPU 上，避免 CPU-GPU 来回拷贝。
- **自定义 CUDA 预处理**：把 NV12/BGRA 直接转成 ViT 输入。
- **TensorRT-LLM C++ Runtime**：生产级 LLM 推理一般都用 C++ API。
- **和车载/机器人中间件对接**：ROS2、DDS、CAN、MQTT 等通常用 C++ 生态更好。
- **TTS/语音合成实时性要求高**：需要低延迟音频播放。

### 9.3 什么时候需要写嵌入式相关代码？

- **摄像头驱动与 ISP 调试**：CSI 排线、device tree、`nvarguscamerasrc` pipeline。
- **系统启动优化**：uboot、kernel、systemd 服务、自动登录。
- **功耗管理**：`nvpmodel`、`jetson_clocks`、风扇调速、散热设计。
- **存储与内存管理**：Orin NX 16GB 内存紧张，swap、zram、模型分片加载。
- **硬件接口**：GPIO、UART、I2C、CAN、PWM（控制云台、补光灯、报警器）。
- **看门狗与恢复机制**：硬件 watchdog、systemd restart、日志监控。

### 9.4 决策树

```text
是否只需要离线验证或低帧率 demo？
  ├─ 是 -> Python + transformers / llama-cpp-python
  └─ 否
       是否对延迟/稳定性要求很高？
         ├─ 否 -> Python 多线程 + llama-cpp-python
         └─ 是
              是否需要 TensorRT-LLM 极致优化？
                ├─ 是 -> C++ + TensorRT-LLM + CUDA 预处理
                └─ 否 -> C++ + llama.cpp + CUDA
```

---

## 10. 完整可复现的代码结构

最终 Orin NX 上的工程目录：

```text
/home/nvidia/vlm_deploy/
├── models/
│   ├── InternVL2_5-1B/              # 原始 HuggingFace 模型
│   ├── internvl2_5_1b_q4_k_m.gguf   # llama.cpp 量化模型
│   ├── internvl_vision_tower.gguf   # vision tower
│   ├── internvit.onnx               # ViT ONNX
│   ├── internvit_orin_fp16.engine   # ViT TensorRT engine
│   └── internvl_1b_orin_engine/     # TensorRT-LLM engine
├── python_demo/
│   ├── verify_transformers.py       # PyTorch 验证
│   ├── verify_llama_cpp.py          # llama.cpp Python 验证
│   └── real_time_vlm.py             # 实时 Python demo
├── cpp_vlm_orin/
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── scripts/
│   ├── export_vit_onnx.py
│   ├── build_vit_engine.sh
│   ├── convert_to_gguf.sh
│   └── test_camera.sh
├── start_vlm.sh
└── vlm-orin.service
```

---

## 11. 踩坑记录与性能数据

### 11.1 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `trtexec` build engine 时 killed | 16GB 内存 + swap 不够 | 加大 swap 到 16~32GB，关闭其他程序 |
| LLM 生成结果乱码 | tokenizer/chat template 不对 | 严格复用 HuggingFace 的 chat template |
| ViT 输出维度不匹配 | image resize/pad 方式与训练时不同 | 使用 `model.process_images` 或严格复现预处理 |
| llama.cpp 加载失败 | GGUF 版本不兼容 | 更新 llama.cpp 到最新版 |
| 摄像头画面延迟大 | OpenCV 默认 backend 不是 GStreamer | 用 `CAP_GSTREAMER` + `drop=true max-buffers=1` |
| 开机后服务启动但无画面 | DISPLAY/XAUTHORITY 未设置 | 在 start.sh 里 export，或改用 headless 模式 |
| TTS 播报卡顿 | 和 VLM 推理抢 CPU/GPU | TTS 单独线程，VLM 生成期间只缓存文本 |

### 11.2 性能数据

在 Jetson Orin NX 16GB，MAXN 模式：

| 方案 | 量化 | 输入尺寸 | 生成 50 token 延迟 | 峰值显存 |
|------|------|---------|-------------------|---------|
| Transformers FP16 | FP16 | 448×448 | ~5.0 s | ~10 GB |
| llama.cpp Q4_K_M | INT4 | 448×448 | ~1.5 s | ~3.5 GB |
| TensorRT-LLM INT4 | INT4 | 448×448 | ~1.2 s | ~3.0 GB |
| TensorRT-LLM INT8 | INT8 | 448×448 | ~1.6 s | ~3.8 GB |
| TensorRT-LLM FP16 | FP16 | 448×448 | ~2.8 s | ~6.5 GB |

> 以上数据为近似值，实际受 prompt 长度、KV Cache、batch、token 生成策略影响较大。

---

## 12. 总结

把 InternVL2.5-1B 这类 VLM 部署到 Jetson Orin NX 16GB，核心挑战不是“能不能跑”，而是**如何在有限内存和算力下跑得又快又稳**。完整链路总结：

1. **模型准备**：用 PyTorch + Transformers 验证模型能力。
2. **量化优化**：INT4/AWQ/TensorRT-LLM 是必须的，否则 16GB 内存不够。
3. **引擎生成**：ViT 走 TensorRT，LLM 走 TensorRT-LLM 或 llama.cpp GGUF。
4. **板端环境**：JetPack 6.x、大 swap、MAXN 功耗模式、`jetson_clocks`。
5. **摄像头调试**：GStreamer + `nvarguscamerasrc` + OpenCV `CAP_GSTREAMER`。
6. **实时系统**：多线程 + 队列 + 滑动窗口触发，避免每帧都跑 VLM。
7. **外围输出**：本地显示、TTS、MQTT/HTTP 上报。
8. **开机自启**：systemd service + 自动登录 + 看门狗。

与 YOLO 部署最大的不同是：VLM 同时涉及 **CV 模型（ViT）**、**NLP 模型（LLM）**、**多模态对齐（projector）** 三个部分，需要分别优化再组合。C++ 和 CUDA 的介入程度也比 YOLO 更深，尤其是在 KV Cache 管理、零拷贝预处理和 TensorRT-LLM 运行时方面。

如果你刚开始做 VLM 边缘部署，建议先用 `llama.cpp` 的 Python 接口把整条链路跑通，再逐步把瓶颈迁移到 C++/CUDA。不要一开始就碰 TensorRT-LLM 的多模态改造，除非你对 TRT-LLM 已经比较熟悉。

希望这篇文章能帮到你。如果有具体的量化失败或摄像头问题，欢迎在评论区留言。
