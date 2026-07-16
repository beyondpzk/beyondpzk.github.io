---
title: 具身智能机器狗 Agent 架构设计
date: 2026-01-22
categories: [Agents]
---

# 具身智能机器狗 Agent 架构设计

这个项目的本质是一个**"具身多模态多Agent系统"**（Embodied Multimodal Multi-Agent System）。它不是云端大模型，而是**长在机器狗硬件上的"数字生命"**。

## 一、整体架构：三层 Agent 协作模型

```
┌─────────────────────────────────────────────────────────────┐
│                    🧠 认知层（大脑 Agent）                      │
│         语言理解 │ 任务规划 │ 世界模型 │ 情感记忆              │
└──────────────────────┬────────────────────────────────────┘
                       │ 指令/目标
┌──────────────────────▼────────────────────────────────────┐
│                   🎯 协调层（中枢 Agent）                      │
│      任务分解 │ 资源调度 │ 异常处理 │ 多Agent 编排              │
└──────┬────────┬────────┬────────┬────────┬─────────────────┘
       │        │        │        │        │
┌──────▼──┐ ┌───▼───┐ ┌──▼──┐ ┌──▼──┐ ┌──▼────┐
│ 🗣️ 交互  │ │ 👁️ 感知 │ │ 🧭  │ │ 🦴  │ │ ⚠️ 安全 │
│  Agent  │ │ Agent │ │导航 │ │运动 │ │ Agent │
│(语音/情感)│ │(视觉/听觉)│ │Agent│ │Agent│ │(紧急/边界)│
└─────────┘ └───────┘ └─────┘ └─────┘ └───────┘
```

## 二、核心 Agent 角色详解

### 🧠 大脑 Agent（认知中枢）

这是机器狗的"灵魂"，通常由一个强大大模型（如 GPT-4o、Claude 或本地大模型）驱动。

| 模块 | 功能 |
|------|------|
| **自然语言理解** | 解析用户意图、提取实体和约束 |
| **任务规划器** | 将高层指令分解为可执行子任务 |
| **世界模型** | 维护对环境的认知（地图、物体位置、状态） |
| **情感与陪伴引擎** | 生成性格、情绪状态、主动互动策略 |

**指令分类路由**：

```python
def classify_intent(command):
    if "背诗/讲笑话/聊天" in command:
        return "PURE_INTERACTION"
    elif "去XX看看/检查一下" in command:
        return "NAVIGATE + INSPECT"
    elif "带我去/指引我" in command:
        return "GUIDED_NAVIGATION"
    elif "去XX转一圈/巡检" in command:
        return "PATROL + EXPLORE"
```

### 🎯 中枢 Agent（协调调度器）

整个系统的"项目经理"，负责把大脑的计划变成各执行 Agent 的协作流。

以 **"去厨房看看关火了没有"** 为例：

```
用户指令
    ↓
大脑Agent：解析为 [导航到厨房] + [视觉检查灶台] + [语音汇报]
    ↓
中枢Agent创建任务流：
    Step 1: 导航Agent → 路径规划到厨房
    Step 2: 感知Agent → 视觉识别"灶台/火焰/开关状态"
    Step 3: 交互Agent → 语音合成："主人，火已经关了/还在烧"
    Step 4: 安全Agent → 全程监控（台阶、障碍物、电量）
```

**关键机制**：
- **优先级抢占**：如果走到一半听到"停下"，安全Agent可立即中断导航Agent
- **状态同步**：中枢维护一个**实时状态板**（黑板架构），各Agent上报状态

### 👁️ 感知 Agent（感官系统）

| 传感器 | Agent 功能 | 技术栈 |
|--------|-----------|--------|
| **RGB-D 相机** | 物体识别、场景理解、火焰检测 | YOLO、CLIP、Segment Anything |
| **激光雷达/LiDAR** | 建图、定位、障碍物检测 | SLAM (Cartographer, LIO-SAM) |
| **麦克风阵列** | 声源定位、语音唤醒、情绪识别 | 波束成形、ASR (Whisper) |
| **IMU/触觉传感器** | 姿态感知、地面材质识别 | 卡尔曼滤波、步态自适应 |

**主动感知（Active Perception）**：不是被动接收，而是为了完成目标主动调整感知策略——检查关火时主动抬头对准灶台，夜间巡检时主动开启补光灯。

### 🧭 导航 Agent（空间智能）

```
┌─────────────────────────────────────┐
│           导航 Agent 架构            │
├─────────────────────────────────────┤
│  全局规划 │ 基于先验地图的路径规划     │
│  局部规划 │ 实时避障、动态路径调整     │
│  定位     │ SLAM + 视觉重定位         │
│  语义导航 │ "去厨房"→地图中的语义标签 │
└─────────────────────────────────────┘
```

### 🦴 运动控制 Agent（小脑）

最底层的物理执行者。分层控制架构：

```
大脑："以0.5m/s速度向正前方移动3米"
    ↓
运动Agent → 步态规划（trot/walk/pace）
    ↓
关节控制 → 12个伺服电机协调（前腿、后腿、躯干）
    ↓
硬件执行 → 电机驱动 + 力控反馈
```

- **地形自适应**：感知到草地/瓷砖/楼梯，自动调整步态和刚度
- **平衡恢复**：被踢一脚或踩到球，快速调整姿态不摔倒
- **能耗管理**：低电量时自动切换节能步态

### 🗣️ 交互 Agent（情感与表达）

让机器狗不只是工具，而是有"个性"的伙伴。

| 功能 | 实现 |
|------|------|
| **语音合成** | 根据情绪状态调整语调（开心时轻快、抱歉时低沉） |
| **肢体语言** | 摇尾巴、歪头、趴下、转圈——配合语音表达情绪 |
| **主动互动** | 长时间无交互时，主动发起："主人，要出去走走吗？" |
| **用户识别** | 人脸识别区分家庭成员，对不同人不同态度 |

### ⚠️ 安全 Agent（底线守护）

最高优先级的 Agent，拥有"一票否决权"：

```
安全Agent持续监控：
├── 物理安全：前方有悬崖/楼梯？→ 立即停步
├── 电量安全：电量<<20%？→ 拒绝远途任务，提示充电
├── 边界安全：超出设定活动范围？→ 强制返回
├── 交互安全：检测到儿童/老人摔倒？→ 报警+陪伴
└── 网络安全：指令来源验证，防止被恶意控制
```

## 三、记忆系统：机器狗的"经验"

```
┌─────────────────────────────────────────┐
│              记忆系统架构                │
├─────────────────────────────────────────┤
│  短期记忆 │ 当前对话上下文、正在执行的任务 │
├─────────────────────────────────────────┤
│  工作记忆 │ 当前环境地图、障碍物位置       │
├─────────────────────────────────────────┤
│  长期记忆 │                              │
│  • 环境地图 │ 家的布局、公园的路径、常去地点 │
│  • 用户画像 │ 主人喜欢李白、怕热、晚上散步 │
│  • 经验库  │ "上次去公园走了A路线，有狗"   │
│  • 技能库  │ 学会了"握手"、"转圈"等指令    │
└─────────────────────────────────────────┘
```

## 四、四个场景完整流程解析

### 场景1："去厨房看看关火了没有"

```
交互Agent → 语音识别："去厨房看看关火了没有"
    ↓
大脑Agent → 意图解析：目的地=厨房，任务=检查火焰状态
    ↓
中枢Agent → 创建任务链：
    1. 导航Agent：当前位置→厨房（全局路径规划）
    2. 感知Agent：到达后，视觉检测"灶台区域"
    3. 交互Agent：语音汇报结果
    4. 安全Agent：全程监控
    ↓
交互Agent："主人，灶台上的火已经关了，但锅里还有水"
```

### 场景2："给我背一首李白的诗"

```
交互Agent → 语音识别
    ↓
大脑Agent → 分类：PURE_INTERACTION（纯交互，无需移动）
    ↓
中枢Agent → 直接调用交互Agent + 知识库
    ↓
交互Agent → 语音合成："好的，我来背一首《静夜思》..."
    ↓
（运动Agent可配合：趴下、摇尾巴，进入"陪伴模式"姿态）
```

### 场景3："请下楼去公园里面转一圈再回来"

```
大脑Agent → 解析：长距离导航 + 探索 + 返回
    ↓
中枢Agent → 任务分解：
    Phase 1: 导航Agent → 出门→等电梯→下楼→出单元门
    Phase 2: 导航Agent → 导航到公园（语义地图匹配）
    Phase 3: 导航Agent → 进入"探索模式"（绕圈巡逻）
             感知Agent → 记录感兴趣物体
    Phase 4: 导航Agent → 原路返回（或根据记忆选最优路径）
    Phase 5: 交互Agent → 汇报见闻："公园里樱花开了，还有一只橘猫"
    ↓
安全Agent → 全程：电子围栏、电量监控、定时报平安
```

### 场景4："去某个地方巡检一圈，并指引我到某个地方"

**双轨并行**——中枢同时调度两条独立任务线：

```
中枢Agent → 分解为两个任务：
    
    任务A：巡检（机器狗独自完成）
    ├── 导航Agent → 按预设路线巡逻
    ├── 感知Agent → 拍照/录像/检测异常
    └── 安全Agent → 独立运行
    
    任务B：指引主人（人机协作）
    ├── 大脑Agent → 理解目的地
    ├── 导航Agent → 规划从主人当前位置到目的地的路径
    ├── 运动Agent → 以"领航者"步态前进（慢速、频繁回头确认）
    ├── 交互Agent → 语音引导："跟我来，前面左转，注意台阶"
    └── 感知Agent → 检测主人是否跟上（视觉跟踪）
```

## 五、技术栈建议

| 层级 | 推荐技术 | 说明 |
|------|---------|------|
| **大脑** | 本地大模型（Llama 3、Qwen 2.5）或云端API | 需要低延迟可本地部署，复杂推理可上云 |
| **感知** | ROS2 + OpenCV + YOLO + Whisper | ROS2是机器人中间件标准 |
| **导航** | ROS2 Navigation2 + SLAM Toolbox | 成熟开源框架 |
| **运动** | ROS2 Control + 四足步态库（如 MIT Cheetah） | 或购买成熟硬件（宇树Go2、Unitree） |
| **记忆** | ChromaDB / Milvus（向量库）+ SQLite | 本地轻量存储 |
| **多Agent** | AutoGen / 自研状态机 | 中枢调度可自研，更可控 |
| **硬件** | Unitree Go2 / 自研四足平台 | Go2有成熟SDK和ROS2支持 |

## 六、关键设计原则

### 1. 分层解耦
大脑负责"做什么"，小脑负责"怎么做"，运动层负责"动起来"。不要用一个模型管所有事。

### 2. 安全优先
安全Agent的权限高于一切，可以直接切断其他Agent的执行。

### 3. 渐进智能
- V1：先实现"听懂指令 + 基础导航"
- V2：加入"视觉检查 + 主动汇报"
- V3：加入"长期记忆 + 情感陪伴"
- V4：加入"自主决策 + 主动发起互动"

### 4. 人机回环（Human-in-the-loop）
复杂任务不要自己硬决策，主动汇报："主人，我发现厨房地上有水，要清理吗？"

## 七、总结：机器狗 Agent 本质

| 维度 | 特征 |
|------|------|
| **具身性** | 不是聊天机器人，是物理世界的"数字生命" |
| **多模态** | 眼（视觉）+ 耳（听觉）+ 身（运动）+ 嘴（语音）协同 |
| **多Agent** | 一个"大脑"指挥多个"专家"，像一支球队 |
| **持续性** | 有记忆、有经验、越用越懂你 |
| **社会性** | 能陪伴、能互动、有情感表达 |

## 附：中枢 Agent 核心调度代码框架

```python
# 中枢 Agent (Orchestrator) - 核心调度器

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional
import asyncio

class TaskType(Enum):
    INSPECT = auto()
    INTERACT = auto()
    NAVIGATE_EXPLORE = auto()
    GUIDE_HUMAN = auto()
    COMPOUND = auto()

@dataclass
class Task:
    task_id: str
    task_type: TaskType
    intent: str
    target_location: Optional[str] = None
    status: str = "PENDING"

class OrchestratorAgent:
    def __init__(self):
        self.agents = {}  # 各Agent实例
        self.active_tasks: Dict[str, Task] = {}
        self.blackboard = {}  # 共享状态板
        
    async def handle_command(self, voice_text: str):
        # Step 1: 大脑解析
        intent = await self.agents["cortex"].parse_intent(voice_text)
        
        # Step 2: 任务分类
        task = self._create_task(intent)
        self.active_tasks[task.task_id] = task
        
        # Step 3: 启动安全Agent全程监控
        asyncio.create_task(self.agents["safety"].monitor(task.task_id))
        
        # Step 4: 按类型分发
        if task.task_type == TaskType.INTERACT:
            await self._execute_pure_interaction(task)
        elif task.task_type == TaskType.INSPECT:
            await self._execute_inspect(task)
        elif task.task_type == TaskType.NAVIGATE_EXPLORE:
            await self._execute_navigate_explore(task)
        elif task.task_type == TaskType.GUIDE_HUMAN:
            await self._execute_guide_human(task)
    
    async def _execute_inspect(self, task: Task):
        """任务一：检查类"""
        await self._call_agent("navigation", "navigate_to", target=task.target_location)
        result = await self._call_agent(
            "perception", "visual_inspect",
            target=task.check_object,
            criteria=["fire", "switch_status"]
        )
        report = self._generate_report(result)
        await self._call_agent("interaction", "speak", content=report)
        task.status = "DONE"
    
    async def _execute_pure_interaction(self, task: Task):
        """任务二：纯交互"""
        content = await self.agents["cortex"].generate_response(task.intent)
        await self._call_agent("locomotion", "set_posture", posture="companionship_mode")
        await self._call_agent("interaction", "speak", content=content, emotion="warm")
        task.status = "DONE"
    
    async def _execute_navigate_explore(self, task: Task):
        """任务三：导航+探索+返回"""
        stages = [
            {"name": "indoor_nav", "agent": "navigation", "action": "navigate_to_door"},
            {"name": "elevator", "agent": "navigation", "action": "use_elevator"},
            {"name": "outdoor_nav", "agent": "navigation", "action": "navigate_to", "target": "park"},
            {"name": "patrol", "agent": "navigation", "action": "patrol", "pattern": "circle"},
            {"name": "return_home", "agent": "navigation", "action": "return"},
            {"name": "report", "agent": "interaction", "action": "summarize_trip"}
        ]
        for stage in stages:
            if await self._safety_check() == "ABORT":
                await self._emergency_return(task)
                return
            await self._call_agent(stage["agent"], stage["action"])
        task.status = "DONE"
    
    async def _execute_guide_human(self, task: Task):
        """任务四：带路 (领航者模式)"""
        path = await self._call_agent("navigation", "plan_path", start="current", end=task.target_location)
        await self._call_agent("locomotion", "set_mode", mode="guided_leader")
        for waypoint in path.waypoints:
            move_task = asyncio.create_task(self._call_agent("navigation", "move_to", waypoint=waypoint))
            track_task = asyncio.create_task(self._track_human_loop())
            await move_task
            await self._call_agent("locomotion", "pause_and_look_back")
        await self._call_agent("interaction", "speak", content=f"主人，{task.target_location}到了")
        task.status = "DONE"
    
    async def _track_human_loop(self):
        """持续跟踪主人位置"""
        while True:
            human_pos = await self._call_agent("perception", "track_human")
            if human_pos.distance > 3.0:
                await self._call_agent("interaction", "speak", content="主人，我在这里等您")
                await self._call_agent("locomotion", "stop")
            await asyncio.sleep(0.5)
    
    async def _safety_check(self) -> str:
        return (await self._call_agent("safety", "check")).result
    
    async def _emergency_return(self, task: Task):
        await self._call_agent("interaction", "speak", content="遇到情况，我先回去了")
        await self._call_agent("navigation", "return_home")
        task.status = "FAILED"
    
    async def _call_agent(self, role: str, action: str, **kwargs):
        return await self.agents[role].execute(action, **kwargs)
```

## 附：Agent 通信协议（标准化消息格式）

```json
{
  "message_id": "msg_001",
  "timestamp": "2026-06-10T21:24:00Z",
  "from": "中枢Agent",
  "to": "导航Agent",
  "task_id": "task_001",
  "message_type": "COMMAND",
  "action": "navigate_to",
  "payload": {
    "target": "厨房",
    "speed": 0.5,
    "avoid_obstacles": true
  },
  "priority": 5,
  "timeout_ms": 30000
}
```

| 消息类型 | 说明 |
|---------|------|
| `COMMAND` | 中枢下发指令 |
| `STATUS` | Agent上报状态（进行中/完成/失败） |
| `EVENT` | 异步事件（如安全Agent发现障碍物） |
| `QUERY` | 查询请求（如中枢问感知Agent"看到什么"） |
| `RESPONSE` | 查询响应 |
