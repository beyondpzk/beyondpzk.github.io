---
title: PrepareData
date: 2023-05-02
categories: [自动驾驶]
---

<!--more-->

# 自动驾驶数据准备：从采集车 Rosbag 到算法训练燃料

在自动驾驶的研发流程中，数据是算法迭代的根基。然而，采集车录制的原始 Rosbag 并不能直接被算法工程师拿来训练模型，中间需要经过一套完整的数据准备（Data Preparation）流程。本文将详细拆解这套流程：原始 Rosbag 里有什么、如何读取和解压、各种 Topic 怎么处理、数据如何上云（OSS），以及最终如何变成算法同学手中的训练数据。

---

## 一、原始数据：Rosbag 里到底装了什么？

采集车在日常道路或封闭场地中行驶时，车上的传感器会持续产生数据。这些数据通常以 **ROS Bag** 的形式录制下来。一个 Rosbag 本质上是一个时序数据库，按时间戳存储了多个 Topic 的消息流。

常见的自动驾驶 Rosbag Topic 包括：

| Topic 类型 | 典型 Topic 名称 | 说明 |
|------------|----------------|------|
| 相机图像 | `/camera/front/image_raw` | 前视摄像头原始图像，通常是压缩或 RAW 格式 |
| 激光雷达 | `/lidar/top/pointcloud` | 顶部激光雷达点云 |
| 毫米波雷达 | `/radar/front/targets` | 前向毫米波雷达目标 |
| GNSS/IMU | `/gnss/ins` | 定位、姿态、速度信息 |
| CAN 总线 | `/can/vehicle_speed` | 车辆速度、方向盘转角、油门刹车 |
| 控制信号 | `/vehicle/ctrl_cmd` | 规划控制输出 |
| 标注触发 | `/record/trigger` | 事件触发标记 |

### 1.1 视频 Topic 的组成

以相机为例，一个典型的视频 Topic 消息结构如下：

```
sensor_msgs/Image
├── header          # 时间戳、坐标系
├── height          # 图像高度
├── width           # 图像宽度
├── encoding        # 编码格式，如 bgr8、rgb8、mono8
├── is_bigendian    # 字节序
├── step            # 每行字节数
└── data            # 原始图像字节流
```

实际生产中为了节省空间，更多使用 `sensor_msgs/CompressedImage`，其 `format` 字段可能是 `jpeg` 或 `png`，`data` 字段直接是压缩后的图像字节。

### 1.2 点云 Topic 的组成

激光雷达点云通常以 `sensor_msgs/PointCloud2` 发布：

```
sensor_msgs/PointCloud2
├── header
├── height          # 点云高度（1 表示无序点云）
├── width           # 点的数量
├── fields          # 每个点的字段定义（x, y, z, intensity, ring, timestamp 等）
├── is_bigendian
├── point_step      # 每个点占用的字节数
├── row_step
├── data            # 原始点云二进制数据
└── is_dense
```

理解 `fields` 和 `point_step` 非常重要，因为不同雷达型号（如 Velodyne、Hesai、Robosense）的点云字段排列可能不同。

---

## 二、Rosbag 读取与解压

### 2.1 使用 rosbag 命令行查看

在本地有 ROS 环境的情况下，可以先对 Rosbag 做快速探查：

```bash
# 查看 bag 信息：包含哪些 topic、消息数量、时间跨度
rosbag info xxx.bag

# 播放 bag
rosbag play xxx.bag

# 提取某个 topic 到新的 bag
rosbag filter xxx.bag output.bag "topic == '/camera/front/image_raw'"
```

### 2.2 使用 Python 读取 Rosbag

在数据准备流水线中，更常见的是用 Python 批量处理。`rosbag` 库可以直接读取 `.bag` 文件：

```python
import rosbag
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2

bag_path = "xxx.bag"
bag = rosbag.Bag(bag_path)

bridge = CvBridge()

# 遍历指定 topic
for topic, msg, t in bag.read_messages(topics=[
    '/camera/front/image_raw',
    '/lidar/top/pointcloud',
    '/gnss/ins'
]):
    timestamp = msg.header.stamp.to_sec()
    
    if topic == '/camera/front/image_raw':
        # 解码图像
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(f"images/{timestamp:.6f}.jpg", cv_img)
    
    elif topic == '/lidar/top/pointcloud':
        # 读取点云
        points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        pts = np.array(list(points))
        np.save(f"points/{timestamp:.6f}.npy", pts)
    
    elif topic == '/gnss/ins':
        # 解析位姿
        pose = parse_ins_msg(msg)
        save_pose(timestamp, pose)

bag.close()
```

### 2.3 处理 CompressedImage

如果录制的是压缩图像，需要额外解压缩：

```python
if topic == '/camera/front/image_raw/compressed':
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(f"images/{timestamp:.6f}.jpg", cv_img)
```

### 2.4 时间同步

自动驾驶是多传感器系统，不同传感器的数据频率和触发方式不同。例如：

- 相机：10 Hz 或 30 Hz
- 激光雷达：10 Hz 或 20 Hz
- GNSS/IMU：100 Hz 或 200 Hz
- CAN 信号：100 Hz

为了让算法能同时使用多个传感器的数据，必须进行 **时间同步**。常见策略有：

1. **最近邻匹配**：以激光雷达时间戳为基准，找相机、GNSS、CAN 中最接近的帧。
2. **插值**：对高频信号（如 GNSS/IMU、CAN）在目标时间戳处做线性插值。
3. **硬件同步**：通过 PTP/gPTP 或外部触发信号，让多个传感器共享同一时间源。

一个简化的时间同步示例：

```python
from bisect import bisect_left

def find_nearest(timestamps, target):
    idx = bisect_left(timestamps, target)
    if idx == 0:
        return timestamps[0]
    if idx == len(timestamps):
        return timestamps[-1]
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return after if after - target < target - before else before

# 以 lidar 为主键
for lidar_ts in lidar_timestamps:
    cam_ts = find_nearest(cam_timestamps, lidar_ts)
    pose_ts = find_nearest(pose_timestamps, lidar_ts)
    can_ts = find_nearest(can_timestamps, lidar_ts)
    
    sample = {
        'lidar': load_lidar(lidar_ts),
        'camera': load_camera(cam_ts),
        'pose': load_pose(pose_ts),
        'vehicle_state': load_can(can_ts),
        'timestamp': lidar_ts
    }
    samples.append(sample)
```

---

## 三、数据处理：从原始消息到算法可用格式

### 3.1 图像处理

算法通常不需要原始 Bayer RAW 图像，而是需要：

- 去畸变（Undistort）：根据相机内参和畸变系数消除镜头畸变。
- 缩放/裁剪：统一到模型输入尺寸，如 1920×1080 → 1600×900。
- 颜色空间转换：BGR → RGB。
- 归一化：将像素值除以 255 或减去均值。

```python
# 去畸变示例
K = camera_intrinsics       # 3x3 内参矩阵
D = distortion_coeffs       # 畸变系数 [k1, k2, p1, p2, k3]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
undistorted = cv2.remap(raw_img, map1, map2, cv2.INTER_LINEAR)
```

### 3.2 点云处理

点云从 Rosbag 解出来后，通常还要做：

- **坐标系转换**：将点云从雷达坐标系转换到车体坐标系或世界坐标系。
- **运动补偿**：由于车辆在高速行驶，一帧点云内不同点采集时刻不同，需要根据 IMU/轮速做去畸变。
- **ROI 过滤**：去掉地面以上太远或太近的点。
- **强度归一化**：不同雷达的 intensity 范围不同，需要统一。

```python
def compensate_motion(points, poses, timestamps):
    """
    根据每个点的 ring/timestamp 和车辆位姿做运动补偿。
    """
    compensated = []
    for pt in points:
        # 假设点云中包含每个点的时间偏移
        dt = pt['timestamp'] - frame_timestamp
        T = interpolate_pose(poses, frame_timestamp + dt)
        pt_world = T @ np.array([pt['x'], pt['y'], pt['z'], 1.0])
        compensated.append(pt_world[:3])
    return np.array(compensated)
```

### 3.3 位姿与地图处理

GNSS/IMU 提供的是 WGS84 坐标系下的经纬度、高度和姿态。为了和点云、图像对齐，通常需要：

- **坐标转换**：经纬度 → 局部 ENU（East-North-Up）坐标系。
- **SLAM 优化**：使用 LIO/SLAM 对原始位姿做平滑和优化，消除 GNSS 跳变。
- **地图对齐**：将多趟采集的数据对齐到同一地图坐标系。

### 3.4 数据组织格式

处理完成后，数据通常按如下结构组织：

```
data/
├── scenes/
│   ├── scene_001/
│   │   ├── images/
│   │   │   ├── 1682995200.000000.jpg
│   │   │   └── ...
│   │   ├── lidar/
│   │   │   ├── 1682995200.000000.npy
│   │   │   └── ...
│   │   ├── poses/
│   │   │   ├── 1682995200.000000.txt
│   │   │   └── ...
│   │   └── vehicle_state/
│   │       ├── 1682995200.000000.json
│   │       └── ...
│   └── scene_002/
│       └── ...
└── metadata.json
```

其中 `metadata.json` 记录每个场景的基本信息：

```json
{
  "scenes": [
    {
      "scene_id": "scene_001",
      "start_time": 1682995200.0,
      "end_time": 1682995300.0,
      "num_frames": 100,
      "topics": [...]
    }
  ]
}
```

---

## 四、上传阿里云 OSS

处理好的数据量通常非常大。一小时的采集数据可能包含：

- 6~8 路相机 × 10 Hz × 1 小时 ≈ 20~30 万张图
- 1~3 个激光雷达 × 10 Hz × 1 小时 ≈ 3~10 亿个点

这些原始图像和点云动辄几百 GB 甚至上 TB，因此必须上传到对象存储。阿里云 OSS 是常用的选择。

### 4.1 使用 ossutil 命令行上传

```bash
# 配置 ossutil
ossutil config -e oss-cn-beijing.aliyuncs.com -i <AccessKeyId> -k <AccessKeySecret>

# 上传整个目录
ossutil cp -r ./data/scenes/scene_001 oss://my-autodrive-bucket/scenes/scene_001
```

### 4.2 使用 Python SDK 上传

```python
import oss2
import os

auth = oss2.Auth('<AccessKeyId>', '<AccessKeySecret>')
bucket = oss2.Bucket(auth, 'oss-cn-beijing.aliyuncs.com', 'my-autodrive-bucket')

def upload_dir(local_dir, oss_prefix):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            oss_path = os.path.join(oss_prefix, relative_path)
            bucket.put_object_from_file(oss_path, local_path)
            print(f"Uploaded: {oss_path}")

upload_dir('./data/scenes/scene_001', 'scenes/scene_001')
```

### 4.3 断点续传与分片上传

对于大文件（如未切分的原始 Rosbag），建议使用分片上传：

```python
oss2.resumable_upload(
    bucket,
    'raw_bags/xxx.bag',
    './xxx.bag',
    store=oss2.ResumableStore(root='/tmp/.ossutil_checkpoint'),
    multipart_threshold=100 * 1024 * 1024,  # 100MB
    part_size=100 * 1024 * 1024,
    num_threads=4
)
```

### 4.4 数据版本管理

在 OSS 上，通常会按日期、场景、版本号组织路径：

```
oss://my-autodrive-bucket/
├── raw_bags/
│   └── 2023-05-02/
│       └── xxx.bag
├── processed/
│   └── v1.0/
│       └── 2023-05-02/
│           └── scene_001/
└── annotations/
    └── v1.0/
        └── 2023-05-02/
            └── scene_001/
```

---

## 五、从 OSS 到算法训练：数据如何变成燃料？

数据上传到 OSS 后，算法工程师并不会直接下载所有数据到本地，而是通过各种方式按需读取。

### 5.1 制作 Dataset 索引

训练框架（如 PyTorch）需要一个 Dataset 类来告诉它样本在哪里、怎么读。通常会先生成一个索引文件：

```python
import json

def build_index(scenes_dir):
    index = []
    for scene in os.listdir(scenes_dir):
        scene_dir = os.path.join(scenes_dir, scene)
        lidar_dir = os.path.join(scene_dir, 'lidar')
        for fname in sorted(os.listdir(lidar_dir)):
            timestamp = fname.replace('.npy', '')
            index.append({
                'scene': scene,
                'timestamp': timestamp,
                'lidar_path': f'{scene_dir}/lidar/{fname}',
                'image_path': f'{scene_dir}/images/{timestamp}.jpg',
                'pose_path': f'{scene_dir}/poses/{timestamp}.txt',
            })
    with open('dataset_index.json', 'w') as f:
        json.dump(index, f)
```

### 5.2 Dataset 类读取 OSS 数据

```python
import torch
from torch.utils.data import Dataset
import oss2
import numpy as np
import cv2

class AutonomousDrivingDataset(Dataset):
    def __init__(self, index_file, bucket):
        with open(index_file) as f:
            self.samples = json.load(f)
        self.bucket = bucket

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 从 OSS 读取点云
        lidar_obj = self.bucket.get_object(s['lidar_path'])
        points = np.load(lidar_obj)
        
        # 从 OSS 读取图像
        img_obj = self.bucket.get_object(s['image_path'])
        img = cv2.imdecode(np.frombuffer(img_obj.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取位姿
        pose_obj = self.bucket.get_object(s['pose_path'])
        pose = np.loadtxt(pose_obj)
        
        return {
            'points': torch.from_numpy(points),
            'image': torch.from_numpy(img).permute(2, 0, 1),
            'pose': torch.from_numpy(pose),
        }
```

### 5.3 数据加载优化

直接每次从 OSS 读取样本会面临延迟问题，常见优化手段：

1. **本地缓存**：第一次从 OSS 下载后缓存到本地 NVMe SSD。
2. **预取（Prefetch）**：使用 `DataLoader` 的多进程和 `prefetch_factor` 提前加载。
3. **TFRecord / LMDB 格式**：将多个小文件打包成少量大文件，减少随机 IO。
4. **对象存储 + CDN / 内部高速通道**：在有内网专线的场景下，OSS 可以直接挂载为存储卷。

```python
from torch.utils.data import DataLoader

dataset = AutonomousDrivingDataset('dataset_index.json', bucket)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True
)
```

### 5.4 训练流程

最终，数据会进入模型训练循环：

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['image'].cuda()
        points = batch['points'].cuda()
        poses = batch['pose'].cuda()
        
        # 模型前向、计算损失、反向传播
        pred = model(images, points, poses)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

至此，采集车里的原始 Rosbag 才真正变成了驱动算法迭代的「数据燃料」。

---

## 六、总结

一条完整的自动驾驶数据准备链路可以概括为：

```
采集车 Rosbag
    │
    ▼
Rosbag 信息查看 / Topic 梳理
    │
    ▼
Python 读取 Topic（图像、点云、GNSS、CAN 等）
    │
    ▼
时间同步 + 多传感器对齐
    │
    ▼
图像去畸变 / 点云运动补偿 / 位姿转换
    │
    ▼
按场景组织数据，生成 metadata + index
    │
    ▼
上传到阿里云 OSS
    │
    ▼
Dataset + DataLoader 按需读取
    │
    ▼
模型训练
```

每一个环节都影响着最终数据的质量。图像没有正确去畸变，多相机融合就会错位；点云没有运动补偿，远处的目标就会变形；时间同步没做好，前视相机和激光雷达看到的就是两个世界。

因此，数据准备团队的工作虽然不像模型训练那么「光鲜」，却是整个自动驾驶系统中最基础、也最重要的一环。

---

## 参考

- [ROS Wiki - rosbag](http://wiki.ros.org/rosbag)
- [sensor_msgs/Image](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html)
- [sensor_msgs/PointCloud2](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html)
- [阿里云 OSS Python SDK 文档](https://help.aliyun.com/document_detail/32026.html)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
