---
title: CityWalker
date: 2024-11-26
categories: [VLN]
---

# CityWalker：从网络规模视频中学习具身城市导航

> **论文**：*CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos*  
> **作者**：Xinhao Liu*, Jintong Li*, Yicheng Jiang, Niranjan Sujay, Zhicheng Yang, Juexiao Zhang, John Abanes, Jing Zhang, Chen Feng  
> **单位**：New York University  
> **发布时间**：2024-11-26（arXiv），CVPR 2025  
> **arXiv**：[https://arxiv.org/abs/2411.17820](https://arxiv.org/abs/2411.17820)  
> **项目主页**：[https://ai4ce.github.io/CityWalker/](https://ai4ce.github.io/CityWalker/)

---

## 摘要

CityWalker 提出了一种**从网络规模城市步行与驾驶视频中学习具身城市导航策略**的可扩展方案。它把导航任务建模为**基于导航工具给出的 waypoint（GPS 坐标）进行点到点导航**，利用 DPVO 等现成的视觉里程计从 YouTube 等平台上的 in-the-wild 视频中提取伪动作标签，从而无需昂贵的遥操作或人工标注即可构造 2000+ 小时的训练数据。模型以冻结的 DINOv2 作为视觉编码器，用 Transformer 融合历史图像、历史轨迹与目标坐标，预测未来路点动作并判断是否到达子目标。在真实 Unitree Go1 四足机器人上的实验表明，CityWalker 在复杂城市环境（十字路口、转弯、人群、绕行等）中的导航成功率显著超过 GNM、ViNT、NoMaD 等基线，并且数据量越大，零样本性能持续提升。

---

## 一、研究背景与动机

### 1.1 城市导航的特殊挑战

视觉导航在静态室内环境中已被认为接近“解决”，但在真实城市公共空间中仍是难题：

- **动态性强**：行人、车辆、红绿灯、施工障碍随时变化；
- **遵循常识规范**：走人行道、等红绿灯、保持社交距离；
- **场景复杂多样**：路口转弯、绕行、拥挤人群、proximity 等关键场景；
- **无 HD Map**：最后一公里配送机器人无法依赖高精度地图。

### 1.2 数据瓶颈与规模化思路

传统方法靠遥操作收集专家数据，规模小、多样性差、成本高。CityWalker 的核心洞察是：

> 互联网上存在大量第一人称城市步行/驾驶视频，它们天然包含人类如何在城市中移动的示范。只要用廉价的视觉里程计提取出相对运动，就可以把这些视频转化为大规模模仿学习数据。

这与近期依赖 VLM prompt 生成动作标签的工作不同：CityWalker 完全使用 **off-the-shelf VO（DPVO）** 提取伪标签，可并行扩展，成本极低。

---

## 二、核心贡献

| 贡献 | 内容 |
|------|------|
| **问题定义** | 明确提出 embodied urban navigation 任务：在无 HD Map 的动态城市环境中，跟随导航工具给出的 waypoint 序列进行移动 |
| **可扩展数据管道** | 用 DPVO 从 2000+ 小时 in-the-wild 城市步行/驾驶视频中自动提取动作监督，无需人工标注或昂贵 VLM prompt |
| **跨域、跨本体训练** | 证明步行视频预训练 + 少量四足机器人专家数据微调，即可迁移到真实机器人 |
| **新评测指标** | 提出 Average Orientation Error（AOE）和 Max AOE（MAOE），比纯 L2 距离更能反映导航方向质量 |
| **关键场景分析** | 针对 Turn、Crossing、Detour、Proximity、Crowd 等城市关键场景进行细粒度评测 |
| **SOTA 性能** | 在离线测试与真实 Unitree Go1 部署中均显著优于 GNM、ViNT、NoMaD |

---

## 三、问题定义与评测指标

### 3.1 任务形式化：Waypoint-Goal Navigation

CityWalker 把任务定义为**连续的点到点导航**：

- 当前 RGB 观测 $o_t$；
- 当前 GPS 位置 $p_t$；
- 导航工具给出的下一个子目标 waypoint $w_t$（GPS 坐标）。

策略表示为：

$$
\pi(a_t \mid o_{(t-k):t},\; p_{(t-k):t},\; w_t)
$$

其中 $k = 5$ 为历史帧数。动作空间 $\mathcal{A}$ 是欧氏空间中的未来路点序列，通常预测未来 5 步动作。当模型判断已到达当前子目标后，自动切换到下一个 waypoint。

### 3.2 评测指标

#### 平均方向误差 AOE

L2 距离不能很好反映方向是否正确。CityWalker 提出 AOE：预测动作与真值动作之间的夹角：

$$
\text{AOE}(k) = \frac{1}{n} \sum_{i=1}^{n} \arccos \frac{\langle \hat{a}_{i_k}, a_{i_k} \rangle}{\|\hat{a}_{i_k}\| \|a_{i_k}\|}
$$

#### 最大平均方向误差 MAOE

为进一步捕捉最坏情况，定义：

$$
\text{MAOE} = \frac{1}{n} \sum_{i=1}^{n} \max_k \theta_{i_k}
$$

#### 关键场景

| 场景 | 定义 |
|------|------|
| **Turn** | 真值动作方向角 $\varphi_{\text{action}} > 20°$ |
| **Crossing** | 检测到红绿灯（置信度 > 0.5） |
| **Detour** | 动作方向偏离目标方向，$|\varphi_{\text{action}} - \varphi_{\text{target}}| > 45°$ |
| **Proximity** | 画面中最大行人框面积 > 25% |
| **Crowd** | 同时检测到 ≥ 5 个行人 |
| **Other** | 其余普通前进行驶 |

这些场景在数据中占比不到一半，但决定了城市导航能否成功。

---

## 四、数据管道：从野外视频到动作监督

### 4.1 数据来源

- **城市步行视频**：2000+ 小时，来自公开网络视频，覆盖不同地理位置、天气、时段；
- **驾驶视频**：同样用 VO 提取，证明管道不限于步行数据。

### 4.2 用视觉里程计生成伪标签

CityWalker 使用 **DPVO（Deep Patch Visual Odometry）** 从视频中估计帧间相对位姿。面临两个问题：

1. **全局漂移**：VO 长轨迹会累积误差。但 CityWalker 只预测短窗口（5 步）内的相对动作，因此局部相对位姿足够可靠；
2. **尺度歧义**：不同视频、不同本体（人 vs 车）的步长/速度不同。CityWalker 对每条轨迹内的动作按**平均步长归一化**：

$$
\tilde{a}_t = \frac{a_t}{\bar{s}_{\text{trajectory}}}
$$

部署时再根据目标机器人的实际步长反归一化。

### 4.3 可扩展性

相比用 VLM 逐帧生成动作标签，VO 管道可以高度并行。论文指出处理 2000 小时视频所需的 wall-clock 时间“可忽略”，成本远低于 VLM prompt。

---

## 五、模型与训练

### 5.1 模型架构

CityWalker 的整体结构如图 2 所示：

| 组件 | 说明 |
|------|------|
| **视觉编码器** | 冻结的 DINOv2，提取当前及历史 RGB 图像特征 |
| **坐标编码器** | 可训练，编码历史位置 $p_{(t-k):t}$ 与目标位置 $w_t$ |
| **Transformer** | 融合图像 token 与坐标 token，输出同长度序列 |
| **Action Head** | 解码为未来 $k = 5$ 个归一化路点动作 |
| **Arrival Head** | 二分类：当前子目标是否已到达 |

### 5.2 输入与输出

**输入**：
- 过去 $k = 5$ 帧 RGB 图像 $o_{(t-k):t}$；
- 过去 $k = 5$ 个 GPS/里程计位置 $p_{(t-k):t}$；
- 当前子目标 waypoint $w_t$。

**输出**：
- 未来 5 个动作路点 $\hat{a}_{t+1}, \dots, \hat{a}_{t+5}$；
- 到达标志 $\hat{c}_t \in \{0, 1\}$。

### 5.3 Feature Hallucination

CityWalker 在 Transformer 输出上增加了一个辅助任务：让输出 token 尽可能逼近未来帧直接用 DINOv2 提取的特征。

$$
\mathcal{L}_{\text{feat}} = \text{MSE}(\text{TransformerOutput},\; \text{DINOv2}(o_{t+1:t+k}))
$$

这个辅助损失帮助模型“想象”未来状态，使动作头和到达头获得更有预测性的表示。论文发现，在**人类步行视频上预训练时**，feature hallucination 在零样本迁移到四足机器人上反而略有负面影响，推测是因为它迫使模型预测人类视角的未来，而与机器人视角存在域差异；但在**用机器人专家数据微调后**，该损失带来稳定收益。

### 5.4 损失函数

方向损失直接监督动作方向：

$$
\mathcal{L}_{\text{ori}} = -\frac{1}{k} \sum_{i=1}^{k} \frac{\langle \hat{a}_i, a_i \rangle}{\|\hat{a}_i\| \|a_i\|}
$$

总损失为四项加权：

$$
\mathcal{L} = \omega_{\text{l1}} \mathcal{L}_{\text{l1}} + \omega_{\text{ori}} \mathcal{L}_{\text{ori}} + \omega_{\text{arr}} \mathcal{L}_{\text{arr}} + \omega_{\text{feat}} \mathcal{L}_{\text{feat}}
$$

其中：
- $\mathcal{L}_{\text{l1}}$：预测路点的 L1 回归损失；
- $\mathcal{L}_{\text{ori}}$：方向损失；
- $\mathcal{L}_{\text{arr}}$：到达状态二分类交叉熵损失；
- $\mathcal{L}_{\text{feat}}$：特征 hallucination 损失。

权重按“各项量级相近”原则选取，论文中 $\omega_{\text{l1}} = 1.0, \omega_{\text{arr}} = 1.0, \omega_{\text{ori}} = 5.0$。

---

## 六、实验

### 6.1 实验设置

- **机器人平台**：Unitree Go1 四足机器人；
- **传感器**：Livox Mid-360 LiDAR（仅用于 SLAM 真值）、普通 webcam（RGB 观测）、智能手机 GPS；
- **遥操作数据**：共 15 小时，纽约市多区域采集；6 小时用于微调，9 小时用于离线测试；
- **基线**：GNM、ViNT、NoMaD（均用 CityWalker 收集的目标图像适配为 image-goal 形式，ViNT 额外做微调）。

### 6.2 离线基准测试

下表汇总了各方法在不同关键场景下的 **Mean（场景平均）** 指标：

| 方法 | L2 ↓ (m) | MAOE ↓ (°) | Arrival ↑ (%) |
|------|----------|------------|---------------|
| GNM [14] (fine-tuned) | 0.74 | 12.1 | 70.0 |
| ViNT [44] (fine-tuned) | 0.70 | 12.6 | 70.7 |
| NoMaD [45] (fine-tuned) | 0.74 | 12.1 | 70.0 |
| **Ours (zero-shot)** | 1.38 | 12.7 | 84.1 |
| **Ours (fine-tuned)** | **1.07** | **11.5** | **87.8** |

关键发现：
- CityWalker **零样本模型** 的到达率（84.1%）已超过所有在机器人数据上微调的基线；
- 微调后进一步把 MAOE 降到 11.5°，到达率提升到 87.8%；
- 在 **Crowd、Proximity、Other** 等场景中优势尤为明显。

### 6.3 真实世界部署

在未见过的城市环境中，目标距离 50–100 m，成功标准为预测到达位置距目标 5 m 以内。

| 方法 | All | Forward | Left Turn | Right Turn |
|------|-----|---------|-----------|------------|
| ViNT [44] (zero-shot) | 37.7 | 62.5 | 0.0 | 50.0 |
| ViNT [44] (fine-tuned) | 57.1 | 100.0 | 25.0 | 25.0 |
| NoMaD [45] (zero-shot) | 42.9 | 75.0 | 16.7 | 28.6 |
| **Ours (fine-tuned)** | **77.3** | **100.0** | **62.5** | **66.7** |

CityWalker 在总体成功率上大幅领先，尤其在**左转**和**右转**等需要精细方向控制的场景中优势显著。

### 6.4 数据规模的影响

论文通过改变训练视频时长验证 scaling law：

- 当训练数据超过 1000 小时时，CityWalker **零样本** 模型的表现超过用少量专家数据微调的 ViNT；
- 仅使用**驾驶视频**训练时，零样本性能与基线相近，说明跨本体数据单独使用效果有限；
- 但 **250 小时步行 + 驾驶混合数据** 即可接近 1000 小时单独步行数据的性能，证明**跨域、跨本体数据混合能显著提升样本效率**。

### 6.5 消融实验

在 1000 小时步行数据上，论文消融了方向损失、feature hallucination 和机器人专家微调的作用：

| 方向损失 | Feature Hall. | 机器人微调 | MAOE ↓ (°) |
|----------|---------------|-----------|------------|
| | | | 17.03 |
| ✓ | | | 17.00 |
| | ✓ | | 17.02 |
| | | ✓ | 15.23 |
| ✓ | | ✓ | 15.21 |
| ✓ | ✓ | ✓ | **15.16** |

结论：
- 在纯步行视频预训练阶段，方向损失和 feature hallucination 的单独收益较小；
- 一旦加入少量机器人专家数据微调，所有组件共同作用，MAOE 从 17.03° 降到 15.16°。

---

## 七、优势与局限

### 7.1 优势

1. **可扩展性极强**：VO 标签生成廉价、可并行，能轻松扩展到数万小时视频；
2. **零样本迁移能力**：仅用人类步行视频预训练即可在四足机器人上取得有竞争力的性能；
3. **跨域数据互补**：步行 + 驾驶混合数据比单一数据源更高效；
4. **面向真实城市场景**：专门定义并评测了 Turn、Crossing、Crowd 等关键场景；
5. **方向损失更有效**：AOE/MAOE 比 L2 距离更能反映导航动作质量。

### 7.2 局限

1. **依赖 GPS/waypoint 输入**：需要导航工具提供 waypoint，未处理纯语言或图像目标；
2. **GPS 噪声敏感**：当前系统使用 iPhone GPS，位置噪声较大时可能影响性能；
3. **VO 尺度与漂移**：虽然通过短窗口归一化缓解，但 VO 本身在遮挡、快速运动中仍可能失效；
4. **Detour 场景较弱**：绕行数据在步行视频中占比较少，微调后才有明显改善；
5. **无显式社交规则建模**：模型从数据中隐式学习规范，但未显式编码红绿灯状态、人行道边界等。

---

## 八、历史意义与后续影响

CityWalker 代表了视觉导航从“机器人自己收集数据”向“利用互联网开放视频”扩展的重要一步。它与相关工作的关系：

| 相关工作 | 关系 |
|----------|------|
| **GNM / ViNT / NoMaD** | CityWalker 把它们作为基线，证明城市环境下需要更大规模、更贴近人类行为的数据 |
| **LeLaN（同期）** | 同样从野外视频学习导航，但依赖 VLM prompt 生成标签，CityWalker 使用更廉价的 VO |
| **CoNVOI** | 类似城市导航设定，但依赖闭源 VLM 和特定 prompt |
| **DA-Nav / ABot-N0 (2026)** | 延续城市级导航愿景；CityWalker 展示了“用网络视频规模化数据”是可行路径之一 |

---

## 九、总结

CityWalker 提出了一条**低成本、可扩展**的城市导航学习路径：利用网络上海量的第一人称步行/驾驶视频，通过视觉里程计提取伪动作标签，在共享的归一化动作空间中预训练 Transformer 策略，再用少量机器人专家数据微调。它在离线评测和真实 Unitree Go1 部署中均显著优于 GNM、ViNT、NoMaD 等现有方法，并展示了数据规模越大、零样本性能越强的 scaling 趋势。尽管仍依赖 waypoint 输入和 GPS，且对 VO 质量敏感，CityWalker 为后续城市级 VLN/VLA 系统如何利用互联网视频数据提供了重要范式。
