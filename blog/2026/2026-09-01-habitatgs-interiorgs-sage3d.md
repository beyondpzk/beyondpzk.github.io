---
title: "Habitat-GS 与 InteriorGS / SAGE-3D：3DGS 如何变成可执行的导航环境"
date: 2026-09-01
categories: [具身智能]
description: "围绕Habitat-GS 与 InteriorGS / SAGE-3D：3DGS 如何变成可执行的导航环境整理研究背景、核心方法、实验结果与应用边界。"
topic: navigation
type: 论文精读
summary: "围绕Habitat-GS 与 InteriorGS / SAGE-3D：3DGS 如何变成可执行的导航环境整理研究背景、核心方法、实验结果与应用边界。"
---

# Habitat-GS 与 InteriorGS / SAGE-3D：3DGS 如何变成可执行的导航环境

> **Habitat-GS**：*A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting*  
> 作者：Ziyuan Xia, Jingyi Xu, Chong Cui 等  
> arXiv：https://arxiv.org/abs/2604.12626  
> 项目：https://zju3dv.github.io/habitat-gs/

> **InteriorGS / SAGE-3D**：*Towards Physically Executable 3D Gaussian for Embodied Navigation*  
> 作者：Bingchen Miao, Rong Wei, Zhiqi Ge 等  
> arXiv：https://arxiv.org/abs/2510.21307  
> 项目：https://sage-3d.github.io

---

## TL;DR

这两篇论文都在做同一件事：把 **3D Gaussian Splatting（3DGS）从“好看的三维表示”升级成“能训练和评测导航模型的环境基础”**。

- **Habitat-GS** 侧重 **simulator**：在 Habitat-Sim 上集成 3DGS 场景渲染和可驱动的 Gaussian avatar，解决“更像真实世界、有动态行人”的导航仿真问题。
- **InteriorGS / SAGE-3D** 侧重 **dataset + benchmark + executable environment**：给 3DGS 加物体级语义和物理碰撞，发布 1K 场景的 InteriorGS 数据集，以及 2M 轨迹-指令对的 SAGE-Bench。

两者不是竞争关系，反而互补：Habitat-GS 需要高保真 3DGS 场景资产，而 InteriorGS 正好提供带语义和物理接口的 3DGS 场景；SAGE-3D 强调“语义 + 物理可执行”，Habitat-GS 强调“渲染 + 动态人物 + 生态兼容”。

在 LightNav-0 的 INSIGHT-Bench 里，Habitat-GS 和 InteriorGS 也都是场景来源之一。

---

## 一、背景：3DGS 为什么适合具身导航

传统具身智能仿真器，例如 Habitat、iGibson、AI2-THOR，主要使用 **mesh + 纹理栅格化**。它们的问题是：

- 高频细节、镜面反射、复杂光照很难真实还原；
- 真实场景的扫描 mesh 容易出现纹理拉伸、接缝和模糊；
- 动态人物大多用低质量 mesh avatar，视觉真实感不足；
- 高质量 mesh 资产制作成本高，扩展场景数量困难。

3DGS 的优势正好补上这些：

- 用大量各向异性高斯显式表示场景；
- tile-based rasterization 能实时渲染；
- 新视角合成更自然，外观细节更丰富；
- 显式表示更适合与 CUDA / OpenGL 图形管线结合。

但 3DGS 本身也有明显短板：

- 没有物体级语义，只是颜色/密度信息；
- 没有明确几何表面，难以直接做碰撞检测；
- 原生 3DGS 通常是“可看不可走”，无法直接作为导航环境。

这正是 Habitat-GS 和 SAGE-3D 各自要解决的问题。

---

## 二、Habitat-GS：高保真、有动态行人的导航仿真器

### 2.1 核心定位

Habitat-GS 是在 **Habitat-Sim** 上做的升级，目标不是发明新的 3DGS 表示，而是把它接入成熟的 Habitat 生态：

- 保留 Habitat-Lab 的任务、训练和评测 API；
- 把渲染后端从 mesh 升级为 3DGS；
- 加入可驱动 Gaussian avatar；
- 保持 open source，不依赖 RTX 专用硬件。

它的一个核心设计原则是 **visual–navigation decoupling**：

```text
视觉渲染：由 3DGS 负责
导航逻辑：由传统 NavMesh 负责
```

这样避免了 3DGS 没有明确几何表面带来的碰撞难题。

### 2.2 3DGS 场景渲染如何接入 Habitat

技术难点是：Habitat 的传感器管线基于 OpenGL，而高性能 3DGS 栅格化基于 CUDA。

Habitat-GS 的做法是 **zero-copy CUDA–OpenGL interop**：

1. CUDA rasterizer 完成 3DGS forward splatting；
2. 把 color / depth 写到 GPU buffer；
3. 通过 intra-GPU copy 直接传到预注册的 OpenGL texture；
4. 全流程不经过 CPU，避免数据来回搬运。

一个场景里可能同时有：

- 3DGS scene；
- 传统 mesh 物体；
- 多个 Gaussian avatar。

Habitat-GS 通过 **depth compositing** 把 3DGS depth 和 OpenGL depth 合成，保证不同表示之间遮挡关系正确。所有 active avatars 的变形高斯会被拼到一个 GPU buffer，一次 CUDA pass 栅格化，再和主 framebuffer 合成。

### 2.3 动态 Gaussian Avatar

这是 Habitat-GS 与普通 3DGS 场景仿真器最大的不同。

每个 avatar 身份准备三样资产：

1. **Canonical Gaussians**：位置、SH 系数、透明度、尺度、旋转、LBS 权重；
2. **GAMMA 生成的运动轨迹**：转换成逐帧 SMPL-X joint 变换矩阵；
3. **Proxy capsules**：用来近似 avatar 每帧碰撞体积。

运行时：

- 用 CUDA Linear Blend Skinning 把 canonical gaussians 变形到当前 SMPL-X pose；
- 不需要运行时神经网络推理；
- GAMMA 轨迹从场景 NavMesh 上的路径生成，使人物沿着场景合理路线行走；
- proxy capsules 被注入 NavMesh，作为动态障碍物。

这样每个 avatar 同时是：

```text
照片级视觉实体 + 有效的动态导航障碍
```

### 2.4 实验结果

#### 视觉质量

作者用 Gemini 3.0 Pro 对 240 张渲染图打分，维度是 rendering quality、realism、scene diversity。结论是 **3DGS 场景在三个维度都明显优于 mesh 场景**。

#### PointNav：混合域训练最划算

固定 5×10⁷ 训练步，五种训练配置如下：

| 训练配置 | Mesh Test SR | GS Test SR | 关键结论 |
|---|---|---|---|
| A：100 Mesh | 59.00 | 61.30 | 收敛快，但 GS 泛化弱 |
| B：100 GS | 53.00 | 70.70 | GS 能力更强，但收敛慢 |
| D：50 Mesh + 50 GS | 61.80 | 78.10 | 混合训练显著提升 |
| E：20 Mesh + 80 GS | 59.60 | 79.60 | GS 泛化最好，Mesh 能力不塌 |

最有价值的发现是：

> mesh 负责快速建立基础导航能力，GS 负责提升视觉鲁棒性；二者混合训练是最优策略。

#### Avatar-aware PointNav：Gaussian avatar 能教碰撞避免

在 GS 场景中加入三个动态 Gaussian avatar 后，模型要边导航边避免撞人。

结果：

- 相比没有 avatar 的 baseline，用 GS avatar 训练的模型在 in-domain GS test 上 **CR 和 PSI 都下降**；
- 更关键的是，迁移到低质量 mesh avatar 场景时，碰撞率也从 2.521% 降到 2.342%，个人空间入侵从 0.075 降到 0.068。

这说明从照片级 Gaussian avatar 学到的“识别人体、预判走向、保持安全距离”，可以迁移到视觉质量较差的 mesh avatar 环境。

#### 性能

在 RTX 4090、256×256 分辨率下：

- 典型 RL 负载下，1–2 个 avatar 的中等规模场景仍能 **>50 FPS**；
- 5M Gaussians 无 avatar 时约 51.46 FPS，4.091 GB；
- GPU 内存随场景和 avatar 数量近似线性增长。

相对 mesh 渲染的吞吐下降，被视觉质量和泛化收益所抵消。

### 2.5 局限

Habitat-GS 明确说明，因为采用 visual–navigation decoupling：

- 只能做 **navigation-level obstacle avoidance**；
- 不支持 force / impulse 级接触；
- 抓取、推物体等 manipulation 任务仍不在范围内。

所以它是导航仿真器，不是完整物理仿真器。

---

## 三、InteriorGS / SAGE-3D：给 3DGS 加语义和物理

### 3.1 论文要解决的问题

3DGS 有两个关键缺陷，导致它不能直接用于 VLN：

1. **没有细粒度物体语义**  
   只有颜色和密度，没有 instance ID、类别、属性。类似 “go to the red chair next to the white bookshelf” 的指令无法可靠 grounding。

2. **没有物理可执行结构**  
   高斯是体积渲染原语，没有明确表面；碰撞几何难提取，容易穿透物体。

SAGE-3D 的形式化定义是：

```text
G + M + Φ → E_exec
```

- `G`：3DGS 高斯原语集合；
- `M`：语义层，如 instance / category / 属性；
- `Φ`：物理层，如 collision body / dynamics；
- `E_exec`：可执行环境。

最终得到一个 semantics- and physics-augmented POMDP。

### 3.2 Object-Level Semantic Grounding

#### InteriorGS 数据集

InteriorGS 包含：

- **1,000 个高保真室内 3DGS 场景**；
- 其中 752 个住宅场景 + 248 个公共空间，如音乐厅、游乐园、健身房；
- **超过 554k 个物体实例**；
- **755 个物体类别**；
- 人工标注 + double verification，包含类别、instance ID、bounding box。

3DGS 数据是从 artist-created mesh 场景采样生成：

- 每个场景平均渲染约 3,000 个 camera view；
- 使用开源 gsplat 估计 3DGS 参数。

#### 2D 语义俯视图

有了物体级标注后，作者把 3D 物体投影到地面，得到 2D semantic top-down map：

- 物体 footprint 通过表面点采样、俯视投影、2D convex hull 得到；
- 门被标注为 open / closed / half-open；
- 墙标记为不可通行。

这个语义地图随后用于路径规划和指令生成。

### 3.3 Physics-Aware Execution Jointing

SAGE-3D 采用 **3DGS–Mesh Hybrid Representation**：

- 3DGS 负责外观；
- mesh 负责物理碰撞。

具体做法：

1. 对 artist-created triangle mesh 使用 **CoACD** 做 convex decomposition；
2. 得到每个物体的 collision body；
3. 在 USDA 场景中，collision body 是 invisible rigid shape，负责接触和动力学；
4. 3DGS 保持可见，负责照片级渲染；
5. 静态物体默认 static body，一部分物体配置为 movable / articulated，支持扩展交互。

这种设计不需要在运行时 ray-trace 高精度 artist mesh，兼顾渲染质量和碰撞精度。

#### 机器人与控制

SAGE-3D 暴露多种机器人 API：

- 腿式/轮式机器人，如 Unitree G1 / Go2 / H1；
- 无人机，如四旋翼；
- 支持 discrete actions 和 continuous velocity commands；
- 连续环境，不是全景图节点跳转；
- 提供 RGB、depth、semantic segmentation、pose、contact events；
- 内置碰撞检测、卡住/穿透监测和恢复。

### 3.4 SAGE-Bench：首个 3DGS 基 VLN benchmark

SAGE-Bench 包含：

- **2M 条新的 trajectory–instruction pairs**；
- 554k 个详细 collision bodies；
- 1,148 个 test samples；
- 944 个 high-level + 204 个 low-level；
- 35 个 distinct scenes。

#### 层级指令生成

High-level instructions 有 5 类：

1. **Add Object**：加入因果对象，使轨迹上下文更有意义；
2. **Scenario Driven**：带场景动机，例如 “I’m thirsty, please bring me a drink from the fridge.”；
3. **Relative Relationship**：用空间关系区分相似目标，例如 “Move to the chair next to that table.”；
4. **Attribute-based**：用颜色、状态、内容等属性定位目标，例如 “Find an empty table in the dining hall.”；
5. **Area-based**：导航到某个功能区，例如 “Walk from here to the kitchen area.”。

Low-level instructions 则由 start/end waypoint 模板生成，偏动作级控制。

#### 三轴评测框架

- Task types：VLN 和 Nogoal-Nav；
- Instruction level：high-level / low-level；
- Episode complexity：scene complexity 和 path complexity。

#### 三个自然连续性指标

传统指标只关心终点，SAGE-Bench 增加了过程质量：

| 指标 | 含义 |
|---|---|
| CSR | Continuous Success Ratio，轨迹中有多少比例保持在参考路径容许走廊内 |
| ICP | Integrated Collision Penalty，把碰撞频率和持续时间都积起来 |
| PS | Path Smoothness，路径转角平滑度 |

例如传统 CR 可能很低，但模型可能长时间贴墙刮擦。ICP 能把这种“持续碰撞”暴露出来。论文中有一个案例：CR 只有 1，但 ICP 达到 0.87。

### 3.5 实验结果

#### 3DGS 渲染更快，但更难收敛

| 环境 | 单帧渲染 | 内存 | 达到 40% SR 所需迭代 | 达到 40% SR 时间 |
|---|---|---|---|---|
| Scanned Mesh | 16.7 ms | 850 MB | 120k | 4.8 h |
| 3DGS–Mesh Hybrid | 6.2 ms | 220 MB | 160k | 6.2 h |

这说明：

- 3DGS 渲染效率更好；
- 但训练更难收敛，因为数据更丰富、更接近真实分布。

#### 3DGS 数据泛化性更强

只用 SAGE-Bench 数据训练，不在 VLN-CE 上训练，然后去 VLN-CE R2R Val-Unseen 评测：

| 模型 | SR ↑ | OSR ↑ | SPL ↑ |
|---|---|---|---|
| NaVILA-base | 0.29 | 0.38 | 0.27 |
| **NaVILA-SAGE** | **0.38** | **0.51** | **0.36** |
| NaVid-base | 0.22 | 0.32 | 0.17 |
| **NaVid-SAGE** | **0.31** | **0.42** | **0.29** |

其中 NaVILA-SAGE 的 SR 相对提升约 **31%**。这支持了一个核心观点：

> 照片级、语义丰富的 3DGS 数据，能带来比传统扫描 mesh 数据更强的跨域泛化。

#### 场景多样性比样本密度更重要

论文比较了训练场景数量和样本数量：

- 相同场景数下，增加样本密度收益有限；
- 相同样本量下，增加场景数收益更大；
- 结论：**scene diversity > sample density**。

这跟 LightNav-0 的 scaling analysis 是同一类结论。

#### 高层指令比低层指令更难

所有模型在 high-level instruction 上的表现都明显低于 low-level。即便 SOTA 的 NaVILA，high-level SR 也只有 0.39，而 low-level 是 0.56。

在不同切片中，**Relative Relationship 和 Attribute-based** 这两类指令更困难。

---

## 四、两篇论文怎么比较

| 维度 | Habitat-GS | InteriorGS / SAGE-3D |
|---|---|---|
| 核心产出 | 仿真器 | 数据集 + benchmark + executable environment |
| 基座 | Habitat-Sim | Isaac Sim + gsplat / CoACD |
| 场景表示 | 3DGS + mesh 混合 | 3DGS + mesh hybrid |
| 语义 | 主要依赖场景本身 | 显式 object-level annotation |
| 物理 | NavMesh + proxy capsule | CoACD collision body + USD physics |
| 动态人 | Gaussian avatar + GAMMA | 未作为核心贡献 |
| 导航任务 | PointNav、avatar-aware PointNav | VLN、Nogoal-Nav |
| 关键发现 | 混合域训练最有效；avatar 训练可迁移 | 3DGS 更难收敛但泛化强；场景多样性 > 样本密度 |
| 与 Habitat 生态兼容 | 强 | 不强调 Habitat，偏向 Isaac Sim / robot API |

简单说：

- **Habitat-GS 回答的是“如何把 3DGS 和动态人高效接入成熟导航训练生态”。**
- **SAGE-3D 回答的是“如何让 3DGS 拥有语义和物理，从而真正可执行、可评测”。**

两者合在一起，正好构成 3DGS 具身导航环境的两个关键侧面：**视觉真实性** 和 **语义/物理可执行性**。

---

## 五、和 LightNav-0 的联系

这两篇工作和 LightNav-0 有直接联系：

- LightNav-0 的 INSIGHT-Bench 场景源里包含 **Habitat-GS** 和 **InteriorGS**；
- Habitat-GS 提供户外/室内高保真 3DGS 场景和动态人物能力；
- InteriorGS 提供 1K 带物体级语义的室内 3DGS 场景；
- 二者共同支撑了 LightNav-0 所说的 Real2Sim2Real：用真实场景来源的仿真资产，规模化生成轨迹和指令。

所以如果只看 LightNav-0 的 4K+ 小时数据，背后很大一部分并不是普通 mesh 场景，而是这些 3DGS / semantic / physics-aware 场景。

---

## 六、我看到的共同趋势

1. **3DGS 正在从渲染工具变成环境基础**  
   不只是“好看”，而是要求能 navigation、能 grounding、能 collision。

2. **外观与物理/语义解耦是务实路线**  
   Habitat-GS 是 visual–navigation decoupling；SAGE-3D 是 3DGS appearance + mesh collision。这说明大家暂时不会试图从 Gaussian 里精确恢复完整物理表面。

3. **环境多样性比数据小时数更关键**  
   Habitat-GS 的混合域实验、InteriorGS 的 scenes-vs-samples 实验，都指向场景多样性比简单堆样本更有效。

4. **动态人物正在成为导航仿真的标配**  
   Habitat-GS 的 Gaussian avatar 不是装饰，而是能让模型学会碰撞避免和个人空间意识。

5. **评测从“终点成功”走向“过程质量”**  
   SAGE-Bench 的 CSR、ICP、PS 说明，只看 SR/CR 会漏掉持续碰撞、不自然轨迹等问题。

---

## 七、仍然值得追问的地方

- **Habitat-GS 的 avatar 数量/行为多样性仍有限**：主要是 GAMMA 轨迹，离真实人群交互还有距离。
- **SAGE-3D 的语义标注依赖 artist mesh**：其 3DGS 数据不是从原始真实扫描直接重建，而是从 artist-created mesh 采样，因此和“真实世界 3DGS 重建”仍有差异。
- **物理能力仍偏导航级**：两篇工作都明确不是 manipulation 级物理仿真。
- **动态对象支持仍不完整**：SAGE-3D 中 movable/articulated 对象是 curated subset。
- **跨平台迁移仍需更多实验**：Habitat 生态和 Isaac Sim / robot API 生态之间的模型迁移还不完全统一。

---

## 总结

Habitat-GS 和 InteriorGS / SAGE-3D 共同说明了一个方向：

> 下一代具身导航仿真，不只是提高渲染分辨率，而是要让 3DGS 同时具备照片级外观、物体级语义、可执行物理，以及动态人物。

如果要做 Real2Sim2Real 或大规模 VLN 数据生成，这两类基础设施缺一不可。
