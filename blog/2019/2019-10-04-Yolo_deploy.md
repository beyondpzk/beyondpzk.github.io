---
title: 从 PyTorch YOLO 到 Jetson Orin 的完整嵌入式部署实战
date: 2019-10-04
categories: [Deploy]
---

# 从 PyTorch YOLO 到 Jetson Orin 的完整嵌入式部署实战

> 这篇文章记录了我们把一套基于 PyTorch 的 YOLO 行人与车辆检测模型，完整部署到 **NVIDIA Jetson AGX Orin** 嵌入式平台的全过程。最终系统能够实时从摄像头采集画面、执行目标检测、在本地显示并推流，开机后无需人工干预自动启动。
>
> 整篇文章会按照“模型准备 → 转换 → 嵌入式环境 → 摄像头 → 推理 → 实时系统 → 开机自启”的完整链路展开，尽可能详细地写出每一步的命令、代码、踩坑点和思考。特别是对“什么时候需要写 C++”“什么时候需要写嵌入式相关代码”这两个问题，我会在第 9 节专门总结。

---

## 0. 项目背景与最终效果

我们做的是一个路口/园区监控场景下的**行人与车辆实时检测系统**。需求很明确：

- 输入：1 路摄像头（最开始用 CSI，后来也支持了 USB 和 GMSL）。
- 模型：YOLOv5 / YOLOv7 家族的检测模型，类别只保留 `person`、`car`、`bus`、`truck`、`bicycle`、`motorcycle`。
- 平台：NVIDIA Jetson AGX Orin（32GB 显存版本）。
- 输出：本地 HDMI 显示 + RTSP 推流 + 本地保存告警截图。
- 可靠性：上电后自动启动，掉电/崩溃后能够自动恢复。

最终性能：在 1280×720 输入下，YOLOv5s 模型通过 TensorRT FP16 加速后，单路检测可以达到 **~45 FPS**，端到端（含采集、预处理、推理、后处理、显示）稳定运行在 **30 FPS** 以上。

---

## 1. 整体技术链路概览

一条完整的嵌入式 AI 部署链路，通常可以拆成下面几个阶段：

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. 模型训练     │ --> │  2. 模型导出     │ --> │  3. 转换加速    │
│  PyTorch YOLO   │     │    ONNX         │     │ TensorRT Engine │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  6. 开机自启     │ <-- │  5. 实时系统     │ <-- │  4. 板端环境     │
│  systemd service│     │  capture+infer+ │     │ JetPack/CUDA/   │
│                 │     │  display+stream │     │ TensorRT/Camera │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

每个阶段需要的技能栈不同：

| 阶段 | 主要技术 | 是否需要 C++ | 是否需要嵌入式 |
|------|---------|------------|--------------|
| 1. 模型训练 | PyTorch、YOLO、数据集标注 | 否 | 否 |
| 2. 模型导出 | `torch.onnx.export`、ONNX 简化 | 否 | 否 |
| 3. 转换加速 | ONNX → TensorRT、`trtexec` | 可选（可用 Python） | 否 |
| 4. 板端环境 | JetPack、Linux for Tegra (L4T) | 否 | 是 |
| 5. 摄像头调试 | GStreamer、`nvarguscamerasrc`、V4L2 | 否 | 是 |
| 6. 实时推理 | TensorRT、CUDA、OpenCV | **是（生产环境）** | 是 |
| 7. 开机自启 | systemd、shell 脚本、权限配置 | 否 | 是 |

接下来按顺序展开。

---

## 2. 阶段一：训练/准备 PyTorch YOLO 模型

### 2.1 训练环境

我们在一台 x86 工作站（Ubuntu 20.04）上训练：

```bash
# 创建环境
conda create -n yolo python=3.9 -y
conda activate yolo

# 克隆 ultralytics yolov5（当时用的是 v6.2 版本）
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.2
pip install -r requirements.txt
```

### 2.2 准备自己的数据集

我们的数据集格式是 YOLO 格式：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

每张图片对应一个同名的 `.txt` 标签文件，每行一个目标：

```text
<class_id> <x_center> <y_center> <width> <height>
```

坐标都是相对于图像宽高的 0~1 归一化值。类别映射 `data.yaml`：

```yaml
path: /home/user/dataset
train: images/train
val: images/val

names:
  0: person
  1: car
  2: bus
  3: truck
  4: bicycle
  5: motorcycle
```

### 2.3 训练命令

```bash
python train.py \
  --data data.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --device 0 \
  --project runs/train \
  --name traffic_det
```

这里选了 `yolov5s` 作为 baseline，因为它在速度和精度之间最平衡。如果后续在 Orin 上精度不够，可以换 `yolov5m` 或做知识蒸馏。

训练完成后，权重在：

```text
runs/train/traffic_det/weights/best.pt
```

### 2.4 验证模型

```bash
python val.py \
  --weights runs/train/traffic_det/weights/best.pt \
  --data data.yaml \
  --img 640 \
  --conf-thres 0.001 \
  --iou-thres 0.6
```

这一步必须在导出前做。因为有些操作在导出时会变化（比如 `Detect` 层的 `export` 模式），如果验证时 mAP 就不对，说明训练本身有问题。

---

## 3. 阶段二：模型转换 —— ONNX 与 TensorRT Engine

### 3.1 为什么需要先转 ONNX，再转 TensorRT？

PyTorch 的 `.pt` 文件不能直接在 Jetson 上高效运行，原因：

1. **PyTorch 运行时太重**：需要在板端安装 PyTorch，占用空间大，推理速度也不如 TensorRT。
2. **TensorRT 能充分利用 Orin 的 GPU**：Orin 的 Ampere/Ada 架构 GPU 对 FP16/INT8 支持很好，TensorRT 会做算子融合、内存优化、Kernel Auto-Tuning。
3. **ONNX 是中间桥梁**：PyTorch → ONNX → TensorRT 是最成熟的链路。ONNX 让我们可以在 x86 工作站上先做转换验证，再把 engine 文件拷贝到 Jetson。

### 3.2 PyTorch → ONNX

YOLOv5 官方已经提供了导出脚本，但我们要理解里面的细节。

```bash
python export.py \
  --weights runs/train/traffic_det/weights/best.pt \
  --img 640 \
  --batch 1 \
  --include onnx \
  --simplify \
  --opset 12
```

参数含义：

- `--img 640`：导出时固定输入尺寸为 `1×3×640×640`。TensorRT 需要静态 shape 才能做最优优化（动态 batch 也可以，但初学者建议先固定）。
- `--batch 1`：单 batch，适合单路摄像头。
- `--include onnx`：只导出 ONNX。
- `--simplify`：用 `onnxsim` 简化计算图，去掉无用节点。
- `--opset 12`：ONNX opset 版本。JetPack 自带的 ONNX/TensortRT 对 opset 12/13 支持最好。

导出后会得到：

```text
runs/train/traffic_det/weights/best.onnx
```

**关键注意点**：YOLOv5 的 `Detect` 层在导出时会变成 `Concat` + `Reshape`，输出 shape 是 `1×25200×85`（COCO 80 类时）。对我们 6 类任务，输出是 `1×25200×11`，其中 11 = 5（xywh + obj）+ 6（class scores）。

如果你想在 ONNX 输出中**不包含 NMS**，就用默认导出；如果希望把 NMS 也封装进 engine，需要额外处理（例如用 `EfficientNMS_TRT` 插件），这里先按不包含 NMS 的方案讲解，后处理自己写。

### 3.3 ONNX → TensorRT Engine

#### 方案 A：在工作站用 `trtexec` 转换（推荐）

如果你的工作站也装了 TensorRT，可以先把 engine 转好，再拷贝到 Orin。

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best_fp16.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640 \
  --workspace=4096
```

参数说明：

- `--fp16`：开启 FP16 精度。Orin 对 FP16 支持极好，速度和精度损失通常都可接受。
- `--minShapes/--optShapes/--maxShapes`：指定输入 shape 范围。固定 batch 时三者相同。
- `--workspace`：builder 可用的最大工作内存（MB）。

> ⚠️ **坑点 1**：TensorRT engine 是**平台相关的**！在工作站（x86 + RTX 显卡）上生成的 engine **不能**直接在 Orin 上用。最终必须在 Orin 上重新生成 engine。
>
> 所以工作站上的转换只是**验证** ONNX 是否正确，真正部署用的 engine 必须在 Orin 上重新 build。

#### 方案 B：在 Orin 上转换

把 `best.onnx` 拷贝到 Orin：

```bash
scp best.onnx nvidia@orin-ip:/home/nvidia/yolo_deploy/
```

在 Orin 上执行：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best_orin_fp16.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640 \
  --workspace=4096
```

转换成功会看到类似输出：

```text
[MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +0, now: CPU 1234, GPU 4567 (MiB)
Engine built in 45.23 seconds.
Serialized Engine Size: 25.6 MiB
```

### 3.4 验证 engine 是否正常

`trtexec` 也可以直接跑 inference 测试：

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=best_orin_fp16.engine \
  --fp16 \
  --batch=1
```

如果输出末尾有 `mean: X ms`，说明 engine 可以正常推理。

---

## 4. 阶段三：Jetson Orin 嵌入式环境搭建

### 4.1 硬件清单

- NVIDIA Jetson AGX Orin 32GB 开发套件
- 128GB NVMe SSD（系统盘）
- CSI 摄像头：Raspberry Pi Camera V2（IMX219）
- 显示器 + 键鼠（调试阶段需要）
- 网线或 WiFi（下载 JetPack 和包）
- DC 19V 电源

### 4.2 烧录 JetPack

Jetson 的系统是 NVIDIA 定制的 **L4T（Linux for Tegra）**，上面再跑 Ubuntu。我们用 **NVIDIA SDK Manager** 或 **balenaEtcher + SD 卡镜像** 烧录。

#### 使用 SDK Manager（推荐）

1. 在一台 Ubuntu x86 主机上安装 SDK Manager：

```bash
sudo dpkg -i sdkmanager_2.x.x-xxxx_amd64.deb
sudo apt-get install -f
```

2. 用 USB-C 线把 Orin 连到主机，Orin 进入 Recovery 模式（按住 RECOVERY 键，再按 RESET）。
3. 打开 SDK Manager，登录 NVIDIA 账号，选择目标设备 **Jetson AGX Orin**。
4. 选择 JetPack 版本（例如 JetPack 5.1.2，对应 L4T 35.4.1）。
5. 勾选：
   - Jetson OS（系统镜像）
   - Jetson SDK Components（CUDA、cuDNN、TensorRT、OpenCV、VPI 等）
6. 先刷 OS，再刷 SDK Components。整个过程大约 40~60 分钟。

> ⚠️ **坑点 2**：如果 SDK Manager 在刷 SDK Components 时失败，经常是网络问题。可以只刷 OS，进系统后手动安装 CUDA/cuDNN/TensorRT：
>
> ```bash
> sudo apt update
> sudo apt install nvidia-jetpack -y
> ```

### 4.3 进系统后的基础配置

首次开机，完成 Ubuntu 初始设置（用户名、密码、时区等）。建议创建一个非 root 用户，比如 `nvidia`。

#### 4.3.1 换源

JetPack 默认源在国外，下载很慢。换成国内源：

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo nano /etc/apt/sources.list
```

替换为（以 Ubuntu 20.04 + ARM64 为例）：

```text
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports focal-security main restricted universe multiverse
```

然后：

```bash
sudo apt update
sudo apt upgrade -y
```

#### 4.3.2 安装必要工具

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
  libcanberra-gtk-module \
  libcanberra-gtk3-module \
  python3-pip \
  python3-venv \
  python3-numpy
```

#### 4.3.3 安装 Jetson Stats（强烈推荐）

```bash
sudo -H pip3 install -U jetson-stats
```

运行 `jtop` 可以实时看 GPU、CPU、内存、温度、风扇、功耗等状态，是调试 Jetson 的神器。

### 4.4 验证 CUDA / cuDNN / TensorRT

```bash
# CUDA 版本
nvcc -V

# cuDNN 版本
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# TensorRT 版本
dpkg -l | grep tensorrt
```

正常情况下应该看到：

- CUDA 11.4（JetPack 5.x）
- cuDNN 8.6
- TensorRT 8.5

Python 环境里验证：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())

import tensorrt as trt
print(trt.__version__)
```

> ⚠️ **坑点 3**：Jetson 上的 PyTorch 不是通过 `pip install torch` 直接装的官方包，因为官方包是 x86 的。必须从 NVIDIA 论坛下载对应 JetPack 版本的 wheel 包。例如：
>
> ```bash
> wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
> pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
> ```

---

## 5. 阶段四：摄像头选型、安装与调试

摄像头是嵌入式部署中**最容易踩坑**的环节之一。不同接口的摄像头对应完全不同的调试方法。

### 5.1 CSI / USB / GMSL 摄像头对比

| 类型 | 接口 | 延迟 | 带宽 | 调试难度 | 适用场景 |
|------|------|------|------|----------|----------|
| CSI | MIPI CSI-2 | 低 | 高 | 中 | 固定场景首选 |
| USB | USB 3.0 | 中 | 中 | 低 | 快速验证、临时方案 |
| GMSL | 同轴/FAKRA | 极低 | 极高 | 高 | 车载、长距离、多路 |

我们最终用了 **IMX219 CSI 摄像头**，因为成本低、延迟低、Jetson 原生支持好。

### 5.2 连接 CSI 摄像头

Jetson AGX Orin 的摄像头接口是 **CSI 排线接口**。连接时注意：

1. 先断电！
2. 打开摄像头接口卡扣。
3. 摄像头排线**蓝色面朝上**（或根据丝印方向）插入接口。
4. 扣紧卡扣。
5. 上电。

### 5.3 用 GStreamer 测试摄像头

Jetson 上最可靠的摄像头测试方式是 GStreamer，而不是直接用 OpenCV 的 `cv2.VideoCapture(0)`。

#### 5.3.1 列出可用视频设备

```bash
v4l2-ctl --list-devices
```

输出示例：

```text
vi-output, imx219 9-0010 (platform:tegra-capture-vi:0):
	/dev/video0

NVIDIA Tegra Video Input Device (platform:tegra-capture-vi):
	/dev/media0
```

#### 5.3.2 用 GStreamer 预览

对于 IMX219（CSI），标准 pipeline 是：

```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! \
  nvvidconv flip-method=0 ! \
  'video/x-raw,width=1280,height=720' ! \
  xvimagesink -e
```

解释：

- `nvarguscamerasrc`：Jetson 专用的 CSI 摄像头源，通过 ARGUS 驱动。
- `memory:NVMM`：使用 NVIDIA 的硬件内存（零拷贝）。
- `format=NV12`：YUV 420 semi-planar，ISP 输出格式。
- `nvvidconv`：硬件格式转换/缩放。
- `xvimagesink`：显示到屏幕。

如果看到画面，说明摄像头硬件和驱动都正常。

#### 5.3.3 在 OpenCV 中使用 GStreamer 作为后端

```python
import cv2

# 注意：字符串里的单引号必须保留
 gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("CSI Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

> ⚠️ **坑点 4**：`cv2.VideoCapture(0)` 在 Jetson 上经常黑屏或报错，因为默认后端不是 GStreamer。一定要用 `cv2.CAP_GSTREAMER`。
>
> ⚠️ **坑点 5**：如果画面是上下颠倒的，调整 `flip-method`：
> - `0`：不翻转
> - `2`：180 度旋转
> - 其他值对应不同翻转方向

### 5.4 USB 摄像头调试

如果是 USB 摄像头，pipeline 更简单：

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
  'video/x-raw,width=1280,height=720,framerate=30/1' ! \
  xvimagesink
```

在 OpenCV 中：

```python
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
```

### 5.5 常见问题与调试

#### 问题 1：摄像头打不开，报错 `Failed to connect to camera` 或 `No cameras available`

可能原因：

1. 排线没插好或方向反了。
2. 摄像头型号不被当前 JetPack 的 device tree 支持。
3. 设备树（device tree）需要重新配置。

排查：

```bash
# 查看内核日志
dmesg | grep imx219
dmesg | grep tegra-camrtc
```

如果看不到 `imx219` 相关日志，说明硬件或 device tree 有问题。IMX219 是官方支持的，通常不需要改 device tree；如果是其他摄像头，可能需要重新编译 kernel 或 dtb。

#### 问题 2：画面撕裂/帧率低

1. 检查 GStreamer pipeline 是否走了硬件加速（有没有 `nvvidconv`、`nvarguscamerasrc`）。
2. 用 `jtop` 看 CPU/GPU 占用，确认不是 CPU 在软解。
3. 降低分辨率或帧率测试。

#### 问题 3：颜色异常

IMX219 输出的是 NV12，需要经过 `nvvidconv` 转成 BGR。如果直接拿 NV12 当 BGR 用，画面会发绿。

---

## 6. 阶段五：在 Orin 上运行 TensorRT 推理

### 6.1 Python 方案（快速验证）

先写一版 Python 的推理脚本，验证端到端流程是否跑通。

#### 6.1.1 安装 Python 依赖

```bash
pip3 install opencv-python numpy
```

> Jetson 上的 OpenCV 通常是 NVIDIA 预编译好的，带 CUDA 支持。如果 `cv2.cuda.getCudaEnabledDeviceCount()` 返回 1，说明 OpenCV 启用了 CUDA。

#### 6.1.2 Python TensorRT 推理类

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


class TRTInference:
    def __init__(self, engine_path, input_shape=(1, 3, 640, 640)):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 反序列化 engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.input_size = trt.volume(input_shape)

        # 绑定输入输出
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # 分配 GPU 内存
        self.d_input = cuda.mem_alloc(self.input_size * np.dtype(np.float32).itemsize)
        self.output_shape = self.context.get_tensor_shape(self.output_name)
        self.output_size = trt.volume(self.output_shape)
        self.d_output = cuda.mem_alloc(self.output_size * np.dtype(np.float32).itemsize)

        # 创建 stream
        self.stream = cuda.Stream()

    def infer(self, image):
        """
        image: np.ndarray, shape (1, 3, 640, 640), float32, normalized
        """
        cuda.memcpy_htod_async(self.d_input, image.ravel(), self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return output
```

> 注意：TensorRT 8.6 之后推荐用 `execute_async_v3` + `set_tensor_address`。如果是 TensorRT 7.x，接口会不同，需要用 `execute_async_v2` + binding。

#### 6.1.3 预处理与后处理

```python
import cv2

CLASS_NAMES = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]


def preprocess(img, input_size=640):
    """letterbox + normalize + HWC->CHW"""
    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)
    pad_h = (input_size - nh) // 2
    pad_w = (input_size - nw) // 2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized

    # BGR -> RGB, normalize to 0~1
    padded = padded[:, :, ::-1].astype(np.float32) / 255.0
    # HWC -> CHW, add batch
    padded = np.transpose(padded, (2, 0, 1))[np.newaxis, ...]
    return padded, scale, pad_w, pad_h


def postprocess(output, conf_thres=0.25, iou_thres=0.45,
                scale=1.0, pad_w=0, pad_h=0, orig_w=1280, orig_h=720):
    """
    output: (1, 25200, 11) for 6 classes
    """
    preds = output[0]  # (25200, 11)

    # 过滤低置信度
    conf = preds[:, 4]
    mask = conf > conf_thres
    preds = preds[mask]

    boxes = []
    scores = []
    class_ids = []

    for det in preds:
        x, y, w, h, obj_conf = det[:5]
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        final_conf = obj_conf * class_conf

        if final_conf < conf_thres:
            continue

        # xywh -> xyxy
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # 去掉 letterbox 的 padding，缩放到原图
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # 裁剪到图像边界
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(class_id)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    indices = indices.flatten() if len(indices) > 0 else []

    results = []
    for i in indices:
        results.append({
            "box": boxes[i].astype(int).tolist(),
            "score": float(scores[i]),
            "class": CLASS_NAMES[int(class_ids[i])]
        })
    return results
```

#### 6.1.4 完整推理脚本

```python
import cv2
from trt_infer import TRTInference
from utils import preprocess, postprocess

GST_STR = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true"
)


def main():
    model = TRTInference("best_orin_fp16.engine")
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_blob, scale, pad_w, pad_h = preprocess(frame)
        output = model.infer(input_blob)
        dets = postprocess(
            output, scale=scale, pad_w=pad_w, pad_h=pad_h,
            orig_w=frame.shape[1], orig_h=frame.shape[0]
        )

        for det in dets:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class']} {det['score']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv5 TRT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### 6.2 C++ 方案（生产部署）

Python 方案验证通过后，如果要做**生产部署**，强烈建议用 C++ 重写。原因：

1. **Python GIL**：多线程时 Python 的全局解释器锁会限制并行度。
2. **内存拷贝开销**：Python 的 NumPy/OpenCV 与 CUDA 之间来回拷贝数据，会有额外开销。
3. **启动速度和稳定性**：C++ 程序启动更快，长期运行更稳定，适合开机自启。
4. **硬件内存零拷贝**：C++ 可以更方便地使用 `cudaMalloc`、`cudaMemcpyAsync`、`NvBufSurface` 等，配合 GStreamer 的 NVMM 内存做零拷贝。

#### 6.2.1 C++ 工程结构

```text
cpp_yolo/
├── CMakeLists.txt
├── include/
│   ├── trt_engine.h
│   ├── yolo_detector.h
│   └── utils.h
└── src/
    ├── main.cpp
    ├── trt_engine.cpp
    ├── yolo_detector.cpp
    └── utils.cpp
```

#### 6.2.2 `trt_engine.h`

```cpp
#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

class TrtEngine {
public:
    TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    // 输入：CPU 上的 float32 CHW 数据
    // 输出：CPU 上的检测结果
    std::vector<float> infer(const std::vector<float>& input);

    int input_h() const { return input_h_; }
    int input_w() const { return input_w_; }

private:
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    cudaStream_t stream_;

    int input_h_ = 640;
    int input_w_ = 640;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    std::vector<int> output_shape_;
};

#endif
```

#### 6.2.3 `trt_engine.cpp`

```cpp
#include "trt_engine.h"
#include <fstream>
#include <iostream>
#include <numeric>

TrtEngine::TrtEngine(const std::string& engine_path) {
    // 读取 engine 文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    context_.reset(engine_->createExecutionContext());

    // 获取输入输出 shape
    auto in_name = engine_->getIOTensorName(0);
    auto out_name = engine_->getIOTensorName(1);
    in_name_ = in_name;
    out_name_ = out_name;

    auto in_dims = engine_->getTensorShape(in_name);
    auto out_dims = engine_->getTensorShape(out_name);

    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];
    input_size_ = std::accumulate(in_dims.d, in_dims.d + in_dims.nbDims, 1, std::multiplies<int>());
    output_size_ = std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims, 1, std::multiplies<int>());

    for (int i = 0; i < out_dims.nbDims; ++i) {
        output_shape_.push_back(out_dims.d[i]);
    }

    cudaMalloc(&d_input_, input_size_ * sizeof(float));
    cudaMalloc(&d_output_, output_size_ * sizeof(float));
    cudaStreamCreate(&stream_);
}

TrtEngine::~TrtEngine() {
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaStreamDestroy(stream_);
}

std::vector<float> TrtEngine::infer(const std::vector<float>& input) {
    cudaMemcpyAsync(d_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);

    context_->setTensorAddress(in_name_.c_str(), d_input_);
    context_->setTensorAddress(out_name_.c_str(), d_output_);
    context_->enqueueV3(stream_);

    std::vector<float> output(output_size_);
    cudaMemcpyAsync(output.data(), d_output_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return output;
}
```

> 这里为了简洁，省略了部分成员变量声明，完整代码见文末仓库。

#### 6.2.4 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(yolo_orin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDA REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu)

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(yolo_orin
    src/main.cpp
    src/trt_engine.cpp
    src/yolo_detector.cpp
    src/utils.cpp
)

target_link_libraries(yolo_orin
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${OpenCV_LIBS}
    cuda
)
```

#### 6.2.5 编译与运行

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行
./yolo_orin ../best_orin_fp16.engine
```

---

## 7. 阶段六：多线程实时检测系统

单线程版本（采集 → 预处理 → 推理 → 后处理 → 显示串行执行）帧率往往上不去。实际生产系统中，我们会把整个 pipeline 拆成多个阶段，用**多线程/多进程 + 队列**并行。

### 7.1 系统架构

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  采集线程    │ -> │  预处理线程  │ -> │  推理线程    │ -> │  后处理线程  │
│  GStreamer  │    │ resize/norm │    │ TensorRT    │    │ NMS/draw    │
│   producer  │    │  consumer/   │    │  consumer/   │    │  consumer/   │
│             │    │  producer   │    │  producer   │    │  producer   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────┐
                                                    │  显示/推流   │
                                                    │ OpenGL/RTSP │
                                                    └─────────────┘
```

每个阶段之间用线程安全的队列（Python 用 `queue.Queue`，C++ 用 `std::queue` + `std::mutex` + `std::condition_variable`）连接。

### 7.2 Python 多线程版本

```python
import threading
import queue
import cv2
import numpy as np
import time
from trt_infer import TRTInference
from utils import preprocess, postprocess

GST_STR = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
)


def capture_thread(cap, q_raw):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 如果队列满，丢弃旧帧，保证实时性
        if q_raw.full():
            try:
                q_raw.get_nowait()
            except queue.Empty:
                pass
        q_raw.put(frame)


def preprocess_thread(q_raw, q_input):
    while True:
        frame = q_raw.get()
        blob, scale, pad_w, pad_h = preprocess(frame)
        q_input.put((frame, blob, scale, pad_w, pad_h))


def inference_thread(model, q_input, q_output):
    while True:
        frame, blob, scale, pad_w, pad_h = q_input.get()
        output = model.infer(blob)
        q_output.put((frame, output, scale, pad_w, pad_h))


def postprocess_thread(q_output):
    while True:
        frame, output, scale, pad_w, pad_h = q_output.get()
        dets = postprocess(
            output, scale=scale, pad_w=pad_w, pad_h=pad_h,
            orig_w=frame.shape[1], orig_h=frame.shape[0]
        )
        for det in dets:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class']} {det['score']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("YOLO Multi-Thread", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    model = TRTInference("best_orin_fp16.engine")

    q_raw = queue.Queue(maxsize=2)
    q_input = queue.Queue(maxsize=2)
    q_output = queue.Queue(maxsize=2)

    threads = [
        threading.Thread(target=capture_thread, args=(cap, q_raw), daemon=True),
        threading.Thread(target=preprocess_thread, args=(q_raw, q_input), daemon=True),
        threading.Thread(target=inference_thread, args=(model, q_input, q_output), daemon=True),
        threading.Thread(target=postprocess_thread, args=(q_output,), daemon=True),
    ]

    for t in threads:
        t.start()

    # 主线程保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

关键点：

- `max-buffers=1` 和队列满时丢弃旧帧：保证我们看到的是最新画面，而不是积压的历史帧。
- `daemon=True`：主线程退出时子线程自动退出。
- 预处理、推理、后处理分离：当推理耗时 20ms 时，采集线程仍在继续取下一帧。

### 7.3 C++ 高性能版本

C++ 的多线程版本思路相同，但可以用 `std::thread`、`std::mutex`、`std::condition_variable`。更重要的是，C++ 可以直接用 **GStreamer 的 NVMM buffer + CUDA 零拷贝** 来进一步减少 CPU-GPU 之间的 memcpy。

这里不贴完整代码，只给出核心优化点：

1. **用 `nvvidconv` 把 CSI 输出直接转成 GPU 上的 `RGBA` 或 `BGRA`**，通过 `appsink` 拿到 `NvBufSurface` 指针。
2. **用 CUDA Kernel 做预处理**：把 `NvBufSurface` 上的 BGRA 数据直接通过 CUDA 做 resize + normalize + HWC→CHW，结果直接写到 TensorRT 的 `d_input` 里面。
3. **推理输出直接送给另一个 CUDA Kernel 做后处理**：可以在 GPU 上做部分 NMS 前置过滤。

这一步是真正需要写 **CUDA C++** 的地方。如果你不会做 CUDA kernel，至少也要用 C++ 把流程串起来，避免 Python 的 GIL 和内存拷贝。

### 7.4 RTSP 推流

要把检测结果推流到监控中心，可以用 `gst-rtsp-server` 或者 `ffmpeg`。

用 GStreamer 的 `nvvidconv` + `nvv4l2h264enc` 硬件编码：

```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! \
  nvvidconv ! \
  'video/x-raw,width=1280,height=720,format=NV12' ! \
  nvv4l2h264enc bitrate=4000000 ! \
  'video/x-h264,stream-format=byte-stream' ! \
  h264parse ! \
  rtph264pay mtu=1400 ! \
  udpsink host=192.168.1.100 port=5000
```

在 C++ 中，可以把 OpenCV 画好框的 `cv::Mat` 转成 GStreamer pipeline 推出去。如果追求性能，直接在 GPU 上完成编码。

---

## 8. 阶段七：开机自启与守护

最后一步：让系统上电后自动启动检测程序，崩溃后自动重启。

### 8.1 创建 systemd service

假设我们用 C++ 可执行文件 `/home/nvidia/yolo_deploy/cpp_yolo/build/yolo_orin`，启动脚本 `/home/nvidia/yolo_deploy/start.sh`。

#### 8.1.1 启动脚本 `start.sh`

```bash
#!/bin/bash
set -e

export HOME=/home/nvidia
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority

# 等待系统完全启动（网络、摄像头驱动等）
sleep 10

# 切换到工作目录
cd /home/nvidia/yolo_deploy

# 启动检测程序
exec ./cpp_yolo/build/yolo_orin ./best_orin_fp16.engine
```

```bash
chmod +x /home/nvidia/yolo_deploy/start.sh
```

> 如果程序需要显示画面，`DISPLAY` 和 `XAUTHORITY` 必须设置正确。否则可以用无头（headless）模式，只推流不显示。

#### 8.1.2 创建 service 文件

```bash
sudo nano /etc/systemd/system/yolo-orin.service
```

内容：

```ini
[Unit]
Description=YOLO Real-time Detection on Jetson Orin
After=network.target multi-user.target graphical.target
Wants=network.target

[Service]
Type=simple
User=nvidia
Group=nvidia
WorkingDirectory=/home/nvidia/yolo_deploy
ExecStart=/home/nvidia/yolo_deploy/start.sh
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=yolo-orin

[Install]
WantedBy=multi-user.target
```

参数说明：

- `After=network.target graphical.target`：等网络和图形界面起来后再启动。
- `User=nvidia`：用普通用户运行，避免 root 权限带来的问题。
- `Restart=always`：程序退出后自动重启。
- `RestartSec=5`：5 秒后重启。

#### 8.1.3 启用并启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable yolo-orin.service
sudo systemctl start yolo-orin.service
```

查看状态：

```bash
sudo systemctl status yolo-orin.service
```

查看日志：

```bash
sudo journalctl -u yolo-orin.service -f
```

### 8.2 自动登录（如果需要显示 GUI）

如果程序依赖 X11 显示，需要设置开机自动登录：

```bash
sudo nano /etc/gdm3/custom.conf
```

找到并修改：

```ini
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=nvidia
```

保存后重启。

### 8.3 硬件看门狗（可选）

如果系统可能完全卡死，可以启用 Jetson 的硬件 watchdog：

```bash
sudo systemctl enable nvwatchdog.service
sudo systemctl start nvwatchdog.service
```

或者在程序里定期喂狗，一旦程序卡住就触发系统复位。

---

## 9. 什么时候写 C++？什么时候写嵌入式代码？

这是大家最关心的问题。根据我们的经验，可以按下面的原则判断：

### 9.1 什么时候用 Python 就够了？

- **模型验证阶段**：快速验证 ONNX/TensorRT engine 是否能跑通。
- **原型开发**：只做单路、低帧率、不要求极致性能的 demo。
- **非实时任务**：比如定时截图上传、后台统计、日志分析。
- **团队没有 C++ 能力**：如果团队只有算法工程师，Python 方案能更快出结果。

### 9.2 什么时候必须写 C++？

- **生产部署**：需要长期稳定运行、低延迟、高帧率。
- **多路摄像头**：Python GIL 会限制多路并行。
- **NVMM 零拷贝**：需要直接操作 GStreamer 的硬件 buffer，必须用 C/C++。
- **自定义 CUDA 预处理/后处理**：例如把 YUV/NV12 直接通过 CUDA kernel 转成模型输入。
- **多线程架构**：需要精确控制线程、锁、队列、内存池。
- **与其他 C++ 系统对接**：比如 ROS2、自动驾驶中间件、车载以太网等。

### 9.3 什么时候需要写嵌入式相关代码？

- **摄像头驱动调试**：device tree、V4L2、GStreamer pipeline、ISP 参数。
- **系统启动优化**：uboot、kernel、systemd、自动登录、看门狗。
- **硬件接口**：GPIO、UART、CAN、I2C、PWM（比如控制补光灯、报警器、云台）。
- **功耗/散热管理**：设置 Jetson 的功耗模式（MAXN、15W、30W 等）。
- **固件烧录与恢复**：JetPack、SDK Manager、备份系统镜像。
- **实时性要求高的场景**：用 `jetson_clocks` 锁定频率，关闭 CPU 动态调频。

### 9.4 一个简洁的决策树

```text
是否只需要跑通模型验证？
  ├─ 是 -> Python + TensorRT Python API
  └─ 否
       是否对帧率/延迟要求很高？
         ├─ 否 -> Python 多线程 + 队列
         └─ 是
              是否需要多路摄像头/NVMM零拷贝/CUDA预处理？
                ├─ 否 -> C++ 标准 TensorRT + OpenCV
                └─ 是 -> C++ + CUDA + GStreamer NVMM
```

---

## 10. 完整可复现的代码结构

最终我们在 Orin 上的工程目录结构如下：

```text
/home/nvidia/yolo_deploy/
├── models/
│   ├── best.onnx              # 从工作站拷贝来的 ONNX
│   └── best_orin_fp16.engine  # 在 Orin 上生成的 engine
├── python_demo/
│   ├── trt_infer.py           # TensorRT Python 推理类
│   ├── utils.py               # 预处理/后处理
│   ├── single_thread.py       # 单线程 demo
│   └── multi_thread.py        # 多线程 demo
├── cpp_yolo/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── trt_engine.h
│   │   ├── yolo_detector.h
│   │   └── utils.h
│   ├── src/
│   │   ├── main.cpp
│   │   ├── trt_engine.cpp
│   │   ├── yolo_detector.cpp
│   │   └── utils.cpp
│   └── build/
│       └── yolo_orin          # 编译产物
├── scripts/
│   ├── build_engine.sh        # 在 Orin 上转 engine
│   ├── export_onnx.sh         # 在工作站导出 ONNX
│   └── test_camera.sh         # 摄像头测试脚本
├── start.sh                   # 自启脚本
└── yolo-orin.service          # systemd 服务文件（已复制到 /etc/systemd/system）
```

---

## 11. 踩坑记录与性能数据

### 11.1 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `trtexec` 转换失败，提示 shape 不匹配 | ONNX 的 dynamic axis 没固定 | 导出时指定 `--batch 1` 并设置 `--minShapes/--optShapes/--maxShapes` |
| Engine 在 x86 上转好但 Orin 无法加载 | TensorRT engine 平台相关 | 必须在目标设备上重新 build engine |
| OpenCV 打不开摄像头 | 默认后端不是 GStreamer | 用 `cv2.CAP_GSTREAMER` 并写完整 pipeline |
| 画面发绿 | 把 NV12 当成 BGR 用了 | 加 `nvvidconv` 转成 BGR/BGRx |
| 帧率只有 10 FPS | 预处理在 CPU 做，且没用队列 | 用多线程，C++ 版本用 CUDA 预处理 |
| 开机后服务启动但黑屏 | 没设置 `DISPLAY` 和 `XAUTHORITY` | 在 start.sh 里 export 这两个变量，或用 headless 模式 |
| 服务崩溃后没有自动重启 | systemd 配置缺少 `Restart=always` | 加上并 `systemctl daemon-reload` |

### 11.2 性能数据

在 Jetson AGX Orin 32GB，MAXN 功耗模式下：

| 方案 | 输入分辨率 | 模型 | 端到端 FPS | GPU 占用 |
|------|-----------|------|-----------|---------|
| Python 单线程 | 640×640 | YOLOv5s | ~18 | 45% |
| Python 多线程 | 640×640 | YOLOv5s | ~28 | 55% |
| C++ 标准版 | 640×640 | YOLOv5s | ~35 | 50% |
| C++ + CUDA 预处理 | 640×640 | YOLOv5s | ~45 | 60% |
| C++ + CUDA 预处理 | 1280×1280 | YOLOv5m | ~22 | 75% |

---

## 12. 总结

从 PyTorch YOLO 到 Jetson Orin 的嵌入式部署，完整链路可以概括为：

1. **训练**：用 PyTorch + YOLO 训练检测模型，得到 `best.pt`。
2. **导出**：用 `export.py` 转成 ONNX，注意 shape 和 opset。
3. **转换**：在 Orin 上用 `trtexec` 把 ONNX 转成 TensorRT engine。
4. **环境**：烧录 JetPack，安装 CUDA/cuDNN/TensorRT，配置摄像头驱动。
5. **摄像头**：用 GStreamer pipeline 调试，确认 ISP 和驱动正常。
6. **推理**：先写 Python 验证，再写 C++ 做生产版本。
7. **实时系统**：多线程 + 队列，必要时用 CUDA 做零拷贝预处理。
8. **自启**：systemd service + 自动登录 + 看门狗。

最关键的认知是：**模型训练和模型部署是完全不同的两件事**。训练时我们只关心 mAP，部署时我们要关心延迟、内存、功耗、稳定性、启动时间、摄像头驱动、系统服务。只有把整条链路都打通，才是真正的“模型落地”。

如果你刚开始做嵌入式部署，建议先用 Python 把整个流程跑通，再逐步把瓶颈环节（预处理、推理、编码）迁移到 C++/CUDA。不要一开始就追求纯 C++，也不要一直停留在 Python 原型。

希望这篇文章能帮到你。有问题欢迎在评论区讨论。
