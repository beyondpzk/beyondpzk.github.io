---
title: 高德地图语音导航逻辑与纯 Python Demo
date: 2026-07-01
categories: [others]
---

# 高德地图语音导航逻辑与纯 Python Demo

最近在看一个机器狗户外步行导航项目（`/Users/jingyu/robot_dog_nav`），它基于 ROS2 Humble 调用高德地图步行路径规划 API，把路线转成 `nav_msgs/Path`，再根据实时位姿做语音播报。为了让核心逻辑不依赖 ROS2 也能快速验证，项目里还写了一个**纯 Python Demo**。这篇博客把高德 App 的导航逻辑和这个 Demo 的实现整理出来，方便后续复用和扩展。

---

## 一、高德 App 的语音导航核心逻辑

高德、百度、腾讯等导航 App 的语音播报，本质上是一个循环：

```text
路径规划 → 实时定位 → 地图匹配 → 触发判断 → TTS 播报
```

### 1. 路径规划

用户输入起点 A 和终点 B 后，后台依次完成：

| 步骤 | 做什么 | 输出 |
|---|---|---|
| 地址解析 | 把「天安门」等地址转成经纬度 | 起点/终点坐标 |
| 路径规划 | 调用步行/驾车路径规划 API，结合路况、道路等级等 | 路径点序列 + 路段元数据 |
| 路口拆解 | 把路线拆成多个 step | 每个 step 含动作类型、距离、路口坐标、道路名称 |

高德步行路径规划 API 返回的 step 示例：

```json
{
  "instruction": "沿长安街向东步行500米",
  "orientation": "东",
  "road_name": "长安街",
  "distance": 500,
  "polyline": "116.397,39.909;116.398,39.909;...",
  "action": "直行"
}
```

每个 step 的**终点坐标**被当作**转向点**，后续播报就是围绕这些转向点展开的。

### 2. 实时定位与地图匹配

导航开始后，App 持续循环：

```text
GPS/基站/WiFi/惯性导航 → 原始定位 → 地图匹配 → 当前在路径上的精确位置
```

| 技术 | 作用 |
|---|---|
| GPS/北斗 | 绝对定位，主要用于户外 |
| 基站/WiFi | 辅助定位，GPS 信号弱时补充 |
| IMU/计步 | 惯性导航，隧道/高楼间短时补盲 |
| 地图匹配 | 把 GPS 点吸附到最近的道路上，避免漂到楼里 |

地图匹配后得到两个关键信息：

- **当前位置**（已吸附到路径）
- **到下一个转向点的剩余距离**

### 3. 播报触发

播报不是定时念，而是按**事件和距离阈值**触发。

#### 转向播报（核心）

| 触发距离 | 典型播报内容 |
|---|---|
| 约 300~500 米 | 「前方 500 米左转」 |
| 约 50 米 | 「前方 50 米左转」 |
| 约 20 米 | 「即将左转」 |
| 到达路口 | 「左转」 |

不同模式阈值不同：

- **驾车导航**：500m、200m、100m、50m，车速快；
- **步行导航**：50m、20m、10m，步行速度慢；
- **AR 步行导航**：结合摄像头视觉识别路口，提示「在这里左转」。

机器狗户外步行的速度更接近步行导航，所以项目里采用了 **50/20/10 米** 的阈值。

### 4. 语音合成 TTS

播报文本生成后，调用 TTS 引擎：

```text
播报文本 → TTS 引擎 → 音频 → 扬声器/耳机
```

| TTS 方式 | 说明 |
|---|---|
| 云端 TTS | 音质好、音色自然，需联网 |
| 本地 TTS | 离线可用，音质较机械 |
| 预录音频 | 把常见指令提前录好，直接播放 |

高德 App 通常优先使用云端 TTS，网络不好时降级到本地合成。

---

## 二、机器狗项目与高德 App 的对应关系

这个机器狗导航项目抽象出了三个 ROS2 节点，正好对应高德 App 的核心逻辑：

```text
gaode_route_node   →  调用高德 API，得到路径 step（对应 App 路径规划）
       ↓
map_matcher_node   →  把 /current_pose 吸附到路径上（对应 App 地图匹配）
       ↓
nav_announcer_node →  距离 50/20/10 米触发播报（对应 App 语音播报）
```

| 高德 App | 本项目 |
|---|---|
| 手机 GPS + 多源融合定位 | 机器狗 GNSS/RTK + IMU + 轮速 → EKF |
| 手机扬声器/耳机 | 机器狗扬声器 |
| 高德内置 TTS | pyttsx3 / edge-tts / 讯飞等 |
| 云端完整地图和实时路况 | 只调用路径规划 API，无实时路况 |

---

## 三、纯 Python Demo：不依赖 ROS2 也能跑通

为了快速验证核心逻辑，项目在 `demo/` 目录下写了一个纯 Python 脚本 `gaode_nav_demo.py`，功能包括：

1. 调用高德步行路径规划 API，获取 A 到 B 的路线；
2. 解析路径 step，得到每个转向点；
3. 支持两种定位模式：
   - **模拟模式**：沿规划路径以固定速度自动移动；
   - **手动模式**：用户手动输入当前 WGS-84 经纬度；
4. 根据到下一个转向点的距离，在 **50/20/10 米** 时触发语音或文字播报。

### 依赖安装

```bash
cd ~/robot_dog_nav/demo
pip install requests

# 如需语音播报
pip install pyttsx3
```

### 快速运行

用地址作为起终点：

```bash
python gaode_nav_demo.py \
  --key 你的高德Key \
  --start "北京市天安门" \
  --end "北京市西单" \
  --mode simulate \
  --speed 2.0 \
  --voice
```

用经纬度作为起终点（WGS-84）：

```bash
python gaode_nav_demo.py \
  --key 你的高德Key \
  --start "116.397428,39.90923" \
  --end "116.372,39.914" \
  --mode simulate \
  --speed 2.0
```

手动输入实时位置：

```bash
python gaode_nav_demo.py \
  --key 你的高德Key \
  --start "北京市天安门" \
  --end "北京市西单" \
  --mode manual
```

---

## 四、关键代码解析

### 1. 地理编码与路径规划

`geocode()` 把地址转成经纬度，`plan_walking_route()` 调高德步行路径规划 API。注意高德 API 使用 **GCJ-02（火星坐标系）**，而机器狗 GPS/RTK 通常输出 **WGS-84**，所以入参出参都要做坐标转换。

```python
def geocode(amap_key: str, address: str):
    """调用高德地理编码 API，返回 WGS-84 经纬度。"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"key": amap_key, "address": address}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    location = data["geocodes"][0]["location"]
    lon, lat = [float(x) for x in location.split(",")]
    # 高德返回 GCJ-02，转成 WGS-84
    return gcj02_to_wgs84(lon, lat)


def plan_walking_route(amap_key, start_lonlat, end_lonlat):
    """调用高德步行路径规划 API。"""
    # 高德 API 需要 GCJ-02 坐标
    start_gcj = wgs84_to_gcj02(start_lonlat[0], start_lonlat[1])
    end_gcj = wgs84_to_gcj02(end_lonlat[0], end_lonlat[1])

    url = "https://restapi.amap.com/v3/direction/walking"
    params = {
        "key": amap_key,
        "origin": f"{start_gcj[0]},{start_gcj[1]}",
        "destination": f"{end_gcj[0]},{end_gcj[1]}",
    }
    resp = requests.get(url, params=params, timeout=10)
    return resp.json()
```

### 2. 解析路径 step

高德返回的每个 step 包含 `polyline`、`instruction`、`action`、`distance` 等字段。Demo 把 `polyline` 中所有点转成 WGS-84，并把最后一个点作为**转向点**。

```python
def parse_route_steps(route_data: dict):
    steps = []
    for step in route_data["route"]["paths"][0]["steps"]:
        polyline = step.get("polyline", "")  # "lon1,lat1;lon2,lat2;..."
        points = []
        for pt in polyline.split(";"):
            lon, lat = [float(x) for x in pt.split(",")]
            wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)
            points.append((wgs_lon, wgs_lat))

        turn_point = points[-1]
        steps.append({
            "instruction": step.get("instruction", ""),
            "action": step.get("action", ""),
            "road_name": step.get("road_name", ""),
            "distance": float(step.get("distance", 0)),
            "turn_point": turn_point,
            "polyline": points,
            "distance_along_path": cumulative_distance,
        })
    return steps
```

### 3. 距离计算

地球表面两点距离用 Haversine 公式，输入为 WGS-84 经纬度。

```python
EARTH_RADIUS = 6371000.0

def haversine(lon1, lat1, lon2, lat2):
    """计算两点间大地线距离（米）。"""
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c
```

### 4. 模拟定位器

模拟模式下，`Simulator` 沿路径 polyline 以固定速度移动，按时间步长推进，输出当前 WGS-84 位置。

```python
class Simulator:
    def __init__(self, steps, speed_mps=1.0):
        self.steps = steps
        self.speed_mps = speed_mps
        self.path_points = self._build_path()
        self.total_distance = self._compute_path_length()
        self.current_distance = 0.0

    def advance(self, dt):
        """按时间步长推进模拟位置。"""
        self.current_distance += self.speed_mps * dt
        return self.current_distance >= self.total_distance

    def get_position(self):
        """返回当前模拟位置（在路径上做线性插值）。"""
        # ... 根据 current_distance 在 path_points 中插值
```

### 5. 播报器

`Announcer` 同时支持文字输出和语音输出。语音基于 `pyttsx3`，初始化失败时自动降级为文字。

```python
class Announcer:
    def __init__(self, use_voice=False):
        self.use_voice = use_voice
        self.engine = None
        if use_voice:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 180)

    def announce(self, text: str):
        print(f"\n🗣️  {text}\n")
        if self.use_voice and self.engine:
            self.engine.say(text)
            self.engine.runAndWait()
```

### 6. 播报文本生成

根据动作类型和距离生成自然语言。

```python
def generate_announcement(action: str, distance: float) -> str:
    action_text = action or "转弯"
    if distance >= 45:
        return f"前方约{int(distance)}米，{action_text}"
    elif distance >= 15:
        return f"前方{int(distance)}米，{action_text}"
    else:
        return f"即将{action_text}"
```

### 7. 核心循环

无论模拟模式还是手动模式，核心逻辑都一样：

```text
loop:
  ├── 获取当前定位（模拟/手动/GPS）
  ├── 找到下一个转向点
  ├── 计算到转向点的距离
  ├── 距离 <= 50/20/10 米且未播报过？生成文本并播报
  └── 到达终点？播报「目的地已到达，导航结束」
```

模拟模式下的实现：

```python
announced = {i: set() for i in range(len(steps))}

while not finished:
    pos = simulator.get_position()
    next_idx = 0
    for i, step in enumerate(steps):
        if simulator.current_distance < step["distance_along_path"] - 2.0:
            next_idx = i
            break
        next_idx = i

    next_step = steps[next_idx]
    distance = haversine(pos[0], pos[1], next_step["turn_point"][0], next_step["turn_point"][1])

    for threshold in [50.0, 20.0, 10.0]:
        if distance <= threshold and threshold not in announced[next_idx]:
            text = generate_announcement(next_step["action"], distance)
            announcer.announce(text)
            announced[next_idx].add(threshold)
            break

    if next_idx == len(steps) - 1 and distance < 5.0:
        announcer.announce("目的地已到达，导航结束")
        finished = True

    finished = simulator.advance(dt)
    time.sleep(dt)
```

---

## 五、坐标系：GCJ-02 与 WGS-84

这是国内地图项目绕不开的问题：

| 来源 | 坐标系 | 说明 |
|---|---|---|
| 高德地图 API | GCJ-02（火星坐标系） | 中国对 WGS-84 的加密偏移 |
| GPS/RTK | WGS-84 | 国际标准坐标系 |
| 百度地图 API | BD-09 | 在 GCJ-02 基础上再次加密 |

如果直接把高德的 GCJ-02 和 GPS 的 WGS-84 混用，会有几十到几百米的偏差。Demo 内部统一转成 WGS-84 计算距离和匹配，所以手动输入也请用 WGS-84。

坐标转换代码在项目 `coord_utils.py` 中实现，基于公开的中国坐标系偏移公式，反算精度约 1~2 米，对步行导航通常足够。

---

## 六、Mock 测试

`demo/test_mock_nav.py` 构造了一条假路径，不调用任何 API，只验证模拟移动、距离计算、阈值播报逻辑是否正确。这对于 CI 或离线调试非常有用。

```bash
python test_mock_nav.py
```

输出示例：

```text
📍 当前位置: 116.397428, 39.909230 | 已走: 0.0 米 | 下一动作: 直行 | 距离转向点: 100.0 米
...
🗣️  前方约50米，直行
🗣️  前方20米，直行
🗣️  即将直行
🗣️  目的地已到达，导航结束
```

---

## 七、已知局限与可扩展方向

1. **地图匹配较简单**：只是找最近的转向点，没有道路拓扑约束，复杂路口可能误判；
2. **不会自动重规划**：偏航后不会重新调用高德 API；
3. **没有实时路况**：调用的是步行路径规划 API，不带实时路况；
4. **播报阈值固定**：50/20/10 米，没有根据速度动态调整；
5. **TTS 音质一般**：默认 pyttsx3 较机械，可替换为 edge-tts、Piper 或云厂商 TTS。

后续可以做的改进：

- 引入更鲁棒的地图匹配算法（如隐马尔可夫模型 HMM）；
- 加入偏航检测与自动重规划；
- 根据机器狗速度动态调整播报阈值；
- 把 Demo 中的逻辑完整迁移到 ROS2 节点。

---

## 八、小结

高德 App 的语音导航并不神秘，核心就是 **「路径规划 → 实时定位 → 地图匹配 → 距离阈值触发 → TTS 播报」** 的闭环。机器狗项目把这个闭环搬到了 ROS2 上，同时用一个纯 Python Demo 把关键逻辑抽离出来，方便快速验证。

这个 Demo 的意义在于：

- **脱离 ROS2 也能跑**，降低调试门槛；
- **清晰展示坐标转换**，解决国内地图 API 与 GPS 的坐标系差异；
- **事件驱动播报**，而非定时播报，更符合真实导航体验；
- **易于扩展**，可以把 `pyttsx3` 替换成更好的 TTS，也可以接入真实 GPS。

如果你也在做户外机器人导航或地图 API 接入，这个思路应该可以直接借鉴。

---

> **参考项目**：`/Users/jingyu/robot_dog_nav`
>
> **核心文件**：
> - `demo/gaode_nav_demo.py`：纯 Python 导航 Demo
> - `demo/test_mock_nav.py`：本地 Mock 测试
> - `demo/coord_utils.py`：GCJ-02 / WGS-84 坐标转换
> - `src/dog_nav/dog_nav/gaode_route_node.py`：ROS2 路径规划节点
> - `src/dog_nav/dog_nav/map_matcher_node.py`：ROS2 地图匹配节点
> - `src/dog_nav/dog_nav/nav_announcer_node.py`：ROS2 语音播报节点
