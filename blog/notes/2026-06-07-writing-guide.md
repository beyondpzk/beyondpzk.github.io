---
title: 写作约定
date: 2026-06-07
categories: [工具与约定]
topic: foundations
type: 工程实践
tags: []
summary: BEYOND 的文章组织、元数据、图片处理和发布约定，让笔记能持续积累，也能被可靠地检索与阅读。
---

# 写作约定

BEYOND 用研究主题组织文章，用专题路线表达阅读顺序。写作时先明确文章要回答的问题，再补充方法、证据和自己的判断。

## 文件与内容组织

文章保存在 `blog/<年份>/<日期>-<slug>.md`，图片建议保存在 `public/images/<slug>/`。网址由文件路径决定，文章发布后尽量保持路径稳定。

主题、文章类型和技术标签分别回答不同的问题：

| 字段 | 回答的问题 | 示例 |
| --- | --- | --- |
| topic | 属于哪个主要研究方向？ | world-models、navigation、vla、driving、deployment |
| type | 这篇文章适合怎样阅读？ | 论文精读、技术分析、工程实践 |
| tags | 涉及哪些具体技术？ | PPO、LoRA、INT8、TensorRT |

完整主题见[研究主题目录](/topics/)。首页、文章目录、专题和相关文章由统一索引生成，不需要手动修改侧边栏。

## 文章元数据

新文章可以从仓库中的 `templates/article.md` 开始：

```yaml
---
title: 文章标题
date: 2026-09-19
topic: world-models
type: 论文精读
tags: [Transformer]
summary: 用一到两句话说明问题、关键发现和阅读价值。
draft: true
---
```

`date` 表示笔记日期。历史笔记中该字段含义不完全一致，所以不自动将它解释为论文日期。

确认论文发表日期后，可以另外填写 `paperDate`。`publishedAt` 表示首次收录日期，未填写时取 Git 首次加入日期；`updatedAt` 表示实际修订日期，未填写时取最新 Git 修改日期。日期使用 `YYYY-MM-DD`。

## 正文结构

每篇文章只保留一个一级标题，章节使用二级及以下标题。代码块中的 `#` 不属于文章标题。

建议按下面的顺序组织长文：

1. 先给核心结论，说明它解决什么问题。
2. 展开关键机制，必要时用公式、图和代码解释。
3. 用实验或可核实的事实支撑判断。
4. 说明局限，区分作者报告和自己的推断。
5. 链接原论文、项目页和阅读过的代码版本。

摘要应能独立帮助读者决定是否继续阅读。优先手写 `summary`，避免用作者列表、论文链接或 Markdown 表格充当摘要。

## 图片与引用

```markdown
![清楚描述图中信息](/images/my-article/method.png)
```

为图像写有意义的 alt；引用论文图时注明出处及许可，自行重绘则说明重绘依据。文章页支持点击或用 Enter 放大图片，Esc 关闭。

图片也可以与文章放在同一目录并用相对路径引用。私人附件不要放进 `public/`，该目录会直接复制到网站。

## 草稿与本地笔记

`draft: true` 或 `visibility: private` 会将文章排除出网站页面、首页摘要、搜索和站点地图。准备发布时移除草稿标记。

这两个字段控制的是网站可见性，不能阻止 Git 提交源文件。真正只留本地的笔记，需要同时加入 `.gitignore` 和 `scripts/local-only-posts.js`；已跟踪的文件还需要先取消 Git 跟踪。

## 预览与发布

使用 Node.js 22 安装依赖后，运行 `npm run docs:dev`，打开终端显示的本地地址。

发布前运行：

```bash
npm run check
npm run docs:build
npm run check:output
```

这些检查覆盖元数据、专题引用、主标题、站内链接、章节锚点和未发布文章的排除。检查差异后明确选择要提交的文件，再推送到 GitHub；CI 使用相同检查，成功后部署。
