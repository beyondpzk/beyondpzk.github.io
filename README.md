# BEYOND

使用 VitePress 的技术博客：研究主题、专题路线、全文搜索与可筛选文章目录。线上地址：https://beyondpzk.github.io/

## 本地运行

使用 Node.js 22（见 `.nvmrc`）：

```bash
npm ci
npm run docs:dev
```

打开终端显示的地址，默认是 http://localhost:5173/ 。

## 写一篇文章

复制 `templates/article.md` 到 `blog/<年份>/<日期>-<slug>.md`，填写标题、日期、主题、类型和摘要。模板默认 `draft: true`；准备发布时改成 `false` 或移除该字段。不要在复制模板后立即提交尚未检查的私人内容。

- `date`：笔记日期；老文章保留原有含义，不能自动当作论文发表日期。
- `paperDate`：可选的论文发表日期，只有核实后才填写。
- `publishedAt`：可选的首次收录日期；不填写则使用 Git 首次加入的日期。
- `updatedAt`：可选的实际修订日期；不填写则使用最新 Git 修改日期。
- `topic`：在 `lib/taxonomy.js` 的主题中选择一个 ID。
- 个人思考使用 `topic: thinking`，在「Thinking · 思考」中单独展示；历史分类 `Thinking` 和 `Thinkings` 统一兼容。
- `type`：`论文精读`、`技术分析` 或 `工程实践`。
- `tags`：具体技术名称数组，如 `[PPO, LoRA]`。
- `summary`：推荐手写一到两句摘要。不填写时从正文提取普通文本。
- `featured: true`：进入首页精选候选，首页按笔记日期展示前三篇。

正文只保留一个一级标题；其他章节使用二级及以下标题。代码块中的 `#` 不受影响。图片放在 `public/images/<slug>/`，用 `/images/<slug>/...` 引用，并写明 alt 与来源。

## 公开与本地内容

`draft: true` 或 `visibility: private` 会同时从页面、首页、文章索引、搜索、专题和站点地图排除；这是网站发布开关，**不能阻止 Git 提交源文件**。

只保留在本地的文件，应同时加入 `.gitignore` 和 `scripts/local-only-posts.js`。已经跟踪的文件还需先取消 Git 跟踪。不要把私人附件放进 `public/`，该目录会直接复制到发布产物中。

## 内容结构

`lib/content.js` 是文章索引的唯一入口：解析 Markdown 元数据、生成摘要和真实日期，并应用公开范围规则。

- 首页、文章目录和专题使用 `posts.data.js` 调用它。
- VitePress 配置从同一索引生成文章元数据与相关文章。
- `lib/taxonomy.js` 定义主题；历史分类保留兼容映射，新文章显式填写 topic/type/tags。
- `content/collections.js` 定义专题阅读顺序和每一步的说明。
- `.vitepress/book-sidebar.js` 单独维护书籍章节顺序。

不再手工编辑文章侧边栏。`npm run sync` 作为兼容命令只检查索引，不修改文件。

## 检查与发布

```bash
npm run check
npm run docs:build
npm run check:output
```

构建会检查站内死链；内容检查会验证主题、日期、专题引用和主标题；产物检查会确认排除的文章没有进入发布页面或数据。

明确选择需要提交的文件，检查差异后提交并推送到 `main`。GitHub Actions 使用相同检查，成功后部署 GitHub Pages。`push.sh` 只推送已有提交，不会自动 `git add .`。

首页和专题为精选入口；`/blog/` 提供主题、类型、年份、标签和关键词筛选，分页与筛选保存在 URL 中。文章图片可点击或用 Enter 放大，Esc 关闭。
