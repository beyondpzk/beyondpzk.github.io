---
title: Agent、Skill 与 Tool：三者的关系与协作边界
date: 2026-07-01
categories: [Agent]
---

# Agent、Skill 与 Tool：三者的关系与协作边界

随着 LLM 应用的深入，「Agent」「Skill」「Tool」这三个词出现的频率越来越高，但它们的边界却常常让人混淆。有人说 Agent 就是会调 Tool 的大模型，有人说 Skill 是封装好的 Tool，还有人说多 Agent 就是开多个对话窗口。这些说法都不完全准确。

这篇博客尝试用一个清晰的层次模型，把 Agent、Skill、Tool 的关系讲清楚，并通过具体例子说明：什么时候只需要一个 Agent 调几个 Tool，什么时候需要多个 Skill，什么时候又该把任务拆给多个 Agent。

---

## 一、先定义三个核心概念

### 1. Tool：最底层的「能力原子」

Tool 是环境暴露给 Agent 的**单个能力单元**。它通常对应一个函数调用，有明确的输入、输出和副作用。

在 Kimi Code CLI 里，Tool 就是 `Read`、`Write`、`Edit`、`Bash`、`Grep`、`Agent`、`Skill` 这些。每个 Tool 只做一件事：

- `Read`：读文件；
- `Bash`：执行 shell 命令；
- `Grep`：搜索文件内容；
- `Agent`：启动一个子代理；
- `Skill`：调用一个已注册的技能。

Tool 本身不蕴含复杂策略。它回答不了「我该什么时候用你」，只会响应「用户调用我，我执行」。

### 2. Skill：可复用的「能力模块」

Skill 是**围绕一类任务封装的可复用能力**。它通常由一段说明、若干示例、推荐工具组合构成，目的是让 Agent 不必每次都从零思考「这件事该怎么做」。

比如 Kimi Code CLI 里的 `kimi-webbridge` skill：

- 它内部可能会调用浏览器控制相关的工具；
- 对外则暴露成「帮我操作网页」这个高层接口；
- Agent 只需要说「调用 kimi-webbridge 去登录并截图」，而不需要了解 click、type、screenshot 的调用顺序。

Skill 可以：

- 封装领域知识（如 PDF 处理、Git 提交、Web 自动化）；
- 提供推荐的工作流；
- 被多个 Agent 共享。

### 3. Agent：拥有目标并自主决策的「执行者」

Agent 是**能够理解目标、制定计划、选择工具/技能并持续执行直到任务完成的实体**。它是三者中最高层、最主动的存在。

一个 Agent 的典型行为循环：

```text
1. 理解用户目标
2. 拆解任务、制定计划
3. 选择合适的 Tool 或 Skill
4. 执行并观察结果
5. 根据反馈调整计划
6. 重复 3~5，直到完成或需要用户确认
```

Agent 可以是一个主助手，也可以是某个子任务里的专用代理（subagent）。

---

## 二、三者之间的关系

用一个简单的层次图来概括：

```text
┌─────────────────────────────────────┐
│            Agent（决策者）            │
│   理解目标 · 规划 · 调度 · 反思       │
└──────────────┬──────────────────────┘
               │ 调用
┌──────────────▼──────────────────────┐
│           Skill（能力模块）           │
│   封装领域知识 · 提供标准工作流        │
└──────────────┬──────────────────────┘
               │ 组合/调用
┌──────────────▼──────────────────────┐
│            Tool（能力原子）           │
│   Read / Write / Bash / Grep / ...  │
└─────────────────────────────────────┘
```

### 关系说明

- **Agent 指挥 Skill 和 Tool**：Agent 根据任务判断，是直接调 Tool，还是先调 Skill。
- **Skill 是对 Tool 的封装和编排**：一个 Skill 可能内部用到多个 Tool，也可能用到其他 Skill。
- **Tool 是最终落地的执行器**：所有 Skill 和 Agent 的意图，最终都要靠 Tool 来执行。
- **Skill 不是 Agent**：Skill 不自己做决策，它只是给 Agent 提供了一条「已知的高效路径」。
- **Agent 可以调用 Agent**：当一个任务太大或需要不同角色时，主 Agent 可以启动子 Agent（subagent），让子 Agent 独立完成子任务后返回结果。

---

## 三、案例说明

### 案例 1：只用一个 Agent + 几个 Tool

**任务**：把项目里的 `methodName` 改成 snake_case。

这个任务很直接：

1. Agent 用 `Grep` 找到所有 `methodName` 出现的位置；
2. 用 `Read` 确认上下文；
3. 用 `Edit` 逐个替换；
4. 用 `Bash` 跑一下测试验证。

这里不需要什么领域封装，也不需要多个角色。一个 Agent 直接调 Tool 就足够了。

```text
User: 把 methodName 改成 snake_case
Agent:
  → Grep("methodName")
  → Read(file)
  → Edit(file, old="methodName", new="method_name")
  → Bash("pytest")
```

**结论**：任务边界清晰、步骤可控、不需要领域知识封装时，Agent 直接调 Tool。

---

### 案例 2：一个 Agent + 一个 Skill

**任务**：帮我在某个网站上登录账号并截取登录后的页面。

这个任务涉及浏览器操作：导航、输入、点击、等待、截图。如果让 Agent 自己用底层工具一步步做，也能完成，但很容易出错。

更好的方式是调用 `kimi-webbridge` skill：

```text
User: 登录 example.com 并截图
Agent:
  → Skill("kimi-webbridge", "navigate to example.com, login with user/pwd, screenshot dashboard")
```

`kimi-webbridge` skill 内部知道如何控制浏览器、处理弹窗、等待元素加载。Agent 不需要关心这些细节。

**结论**：当任务属于某个常见领域，且有现成 Skill 时，优先用 Skill。Skill 让 Agent 更聚焦在「目标」上，而不是「步骤」上。

---

### 案例 3：一个 Agent + 多个 Skill

**任务**：帮我分析一份 PDF 财报，提取关键指标，生成一份 PPT，并自动把结果文件提交到 Git。

这个任务横跨多个领域：

- 读 PDF → 需要 `pdf` skill；
- 生成 PPT → 需要 `slides` skill（如果有）；
- Git 提交 → 需要 `commit` skill。

Agent 的规划可能是：

```text
User: 分析财报 PDF 并生成 PPT
Agent:
  → Skill("pdf", "extract tables from report.pdf")
  → （自己分析提取出的数据）
  → Skill("slides", "create slides with key metrics")
  → Skill("commit", "-m 'add report analysis slides'")
```

Agent 仍然只有一个，但它通过组合多个 Skill 来完成跨领域任务。

**结论**：当任务涉及多个不同领域，而每个领域都有现成 Skill 时，一个 Agent + 多个 Skill 是最自然的组合。

---

### 案例 4：多个 Agent 协作

**任务**：重构一个复杂模块，要求不影响现有功能。

这个任务可以拆成两个相对独立的子任务：

1. **探索阶段**：先搞清楚当前模块的结构、调用关系、测试覆盖；
2. **实现阶段**：基于探索结果进行重构，并更新相关调用方。

可以让两个 Agent 分别负责：

```text
User: 重构 auth 模块
Main Agent:
  → Agent("explore", "read auth module, find all callers, summarize refactor risks")
       ↓ 返回分析报告
  → Agent("coder", "refactor auth module based on the report, update callers, run tests")
       ↓ 返回修改结果
  → 审核结果，必要时要求补充测试
```

`explore` Agent 只负责读和理解，不动代码；`coder` Agent 只负责改代码和跑测试，不被海量无关信息干扰。两者上下文隔离，避免互相污染。

**结论**：当任务复杂、可以拆分、且不同阶段需要不同能力或不同上下文时，使用多个 Agent。

---

### 案例 5：多 Agent + 多 Skill

**任务**：调研一个新技术方向，写一份调研报告，并把报告内容自动发布到博客。

这可以拆成：

- **研究 Agent**：用 `WebSearch`、浏览器 Skill 收集资料；
- **写作 Agent**：基于资料写 Markdown 报告；
- **发布 Agent**：用文件操作 Tool 和 `commit` skill 把报告提交到博客仓库。

```text
Main Agent:
  → Agent("research", "调研 RLHF 最新进展", skills=["websearch", "kimi-webbridge"])
  → Agent("writer", "基于调研结果写报告", tools=["Read", "Write"])
  → Agent("publisher", "发布到博客", skills=["commit"])
```

**结论**：大型任务通常需要「多 Agent 分工 + 每个 Agent 按需调用 Skill/Tool」的组合。

---

## 四、什么时候需要一个 Agent？

不是每个交互都需要 Agent。如果用户只是问「Python 里 list 和 tuple 的区别是什么」，直接回答即可，不需要调用 Tool，也不需要 Agent。

需要 Agent 的典型信号：

| 信号 | 说明 |
|---|---|
| 目标需要多步骤完成 | 不是一句话能回答的，需要计划、执行、验证 |
| 执行过程中需要决策 | 要根据中间结果动态调整下一步 |
| 需要与环境交互 | 必须读写文件、运行命令、查询外部系统等 |
| 结果不确定 | 可能失败、需要重试、需要用户确认 |
| 需要持续追踪状态 | 有 TODO 列表、长期目标、多轮反馈 |

简单说：**当任务需要「自主推进」而不是「单次回答」时，就需要 Agent**。

---

## 五、什么时候需要多个 Skill？

一个 Agent 的能力边界取决于它能调用的 Skill 和 Tool。当任务横跨多个领域，而每个领域都有成熟封装时，就该引入多个 Skill。

| 场景 | 需要的 Skill |
|---|---|
| 处理 PDF、图片、音视频 | `pdf`、`image`、`media` skill |
| 网页自动化 | `kimi-webbridge` skill |
| Git 操作 | `commit` skill |
| 生成文档/PPT | `slides`、`docx` skill |
| 调用内部 API | 自定义的 API skill |

多个 Skill 的意义不是「让 Agent 变复杂」，而是**把复杂留到 Skill 内部，让 Agent 的决策层保持简洁**。

---

## 六、什么时候需要多个 Agents？

多 Agent 是有成本的：上下文切换、结果合并、协调开销。所以不要为了用而用。

适合多 Agent 的场景：

| 场景 | 原因 |
|---|---|
| 任务可拆成独立子任务 | 每个子任务交给一个 Agent，并行或串行执行 |
| 需要不同角色 | 探索者、实现者、审查者、测试者 |
| 上下文需要隔离 | 避免一个 Agent 的临时文件/中间结果污染主上下文 |
| 需要并行验证 | 多个 Agent 同时尝试不同方案，最后比较 |
| 任务太大，单 Agent 上下文装不下 | 把子任务分出去，只把结论收回来 |

不适合多 Agent 的场景：

- 任务本身只有一两步；
- 子任务之间高度耦合，必须频繁同步；
- 引入多 Agent 后的协调成本大于收益。

---

## 七、一个实用的判断框架

遇到任务时，可以按下面这个顺序思考：

```text
1. 这个任务需要 Agent 吗？
   ├── 不需要 → 直接回答/给出建议
   └── 需要 → 进入下一步

2. 这个 Agent 能直接调 Tool 完成吗？
   ├── 能 → 直接调 Tool
   └── 不能/重复劳动 → 进入下一步

3. 有没有合适的 Skill？
   ├── 有 → 调用 Skill
   └── 没有 → 进入下一步

4. 任务能否拆成独立的子任务？
   ├── 能 → 启动多个 Sub-Agent
   └── 不能 → Agent 自己一步步做，必要时中途再拆分
```

这个框架的核心思想是：**能用简单方案就不用复杂方案，能复用 Skill 就不重写流程，能拆分出去就不要把所有上下文塞到一个 Agent 里。**

---

## 八、小结

| 概念 | 层级 | 职责 | 比喻 |
|---|---|---|---|
| **Tool** | 原子层 | 执行单个操作 | 手脚、感官 |
| **Skill** | 模块层 | 封装领域能力，提供标准工作流 | 专业技能（开车、做饭） |
| **Agent** | 决策层 | 理解目标、规划、调度、反思 | 大脑 + 项目经理 |

三者不是竞争关系，而是协作关系：

- **Tool 被 Agent 和 Skill 调用**；
- **Skill 帮 Agent 更高效地调用 Tool**；
- **多个 Agent 可以分工协作，各自调用合适的 Skill 和 Tool**。

理解这三者的边界，是设计清晰、可维护、可扩展的 LLM 应用系统的第一步。

---

> **一句话总结**：Tool 是做一件事，Skill 是会做一类事，Agent 是决定做什么事、怎么组织别人一起做。
