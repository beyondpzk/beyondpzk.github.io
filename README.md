# 我的博客

用 [VitePress](https://vitepress.dev/) 搭建的个人博客，部署在 GitHub Pages 上。

## 特点

- 📖 **像书一样阅读**：左侧侧边栏按主题分卷，方便系统浏览
- ⚡ **极速体验**：基于 Vite，开发时热更新，生产环境纯静态
- ✍️ **Markdown 写作**：专注内容，支持代码高亮、公式、表格
- 🖼️ **图片友好**：支持 `public` 目录和相对路径两种图片引用方式
- 🌓 **明暗主题**：自动适配系统主题
- 🔍 **本地搜索**：内置搜索，快速定位文章

## 快速开始

### 1. 安装依赖

```bash
cd my-blog
npm install
```

### 2. 本地预览

```bash
npm run docs:dev
```

然后打开 <http://localhost:5173/my-blog/>（如果 base 配置为 `/my-blog/`）

### 3. 构建

```bash
npm run docs:build
```

构建结果在 `.vitepress/dist` 目录。

## 如何写作

### 新建文章

在 `blog/` 下的对应主题目录创建 Markdown 文件：

```bash
blog/tech/my-new-post.md
```

文件开头加上 frontmatter：

```markdown
---
title: 文章标题
date: 2026-06-12
---

# 文章标题

正文内容...
```

### 添加图片

#### 方法 1：放入 public 目录（适合全局复用图片）

```bash
public/my-image.jpg
```

```markdown
![说明](/my-image.jpg)
```

#### 方法 2：与文章同目录（适合文章专属图片）

```bash
blog/tech/my-new-post.md
blog/tech/my-image.jpg
```

```markdown
![说明](./my-image.jpg)
```

### 复制粘贴图片

推荐使用以下编辑器实现"复制粘贴图片"：

- **[Obsidian](https://obsidian.md/)**：设置附件文件夹后，粘贴图片自动保存
- **[Typora](https://typora.io/)**：偏好设置中指定图片保存路径
- **VS Code + 插件**：安装 `Markdown Paste` 等插件

### 更新侧边栏

新增文章后，在 `.vitepress/config.mjs` 的 `sidebar` 中添加链接，读者才能在侧边栏看到。

## 部署到 GitHub Pages

### 1. 创建 GitHub 仓库

在 GitHub 上创建一个名为 `my-blog` 的仓库（或 `username.github.io`）。

### 2. 修改 base 配置

打开 `.vitepress/config.mjs`，根据实际情况修改 `base`：

- 如果仓库名为 `my-blog`：
  ```js
  base: '/my-blog/',
  ```
- 如果仓库名为 `username.github.io`：
  ```js
  base: '/',
  ```

### 3. 创建 GitHub Actions 工作流

创建文件 `.github/workflows/deploy.yml`：

```yaml
name: Deploy VitePress to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run docs:build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: .vitepress/dist

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 4. 启用 GitHub Pages

1. 进入仓库 **Settings → Pages**
2. **Source** 选择 **GitHub Actions**
3. 推送代码到 `main` 分支

### 5. 访问博客

部署完成后，访问：

- `https://yourusername.github.io/my-blog/`
- 或 `https://yourusername.github.io/`

## 自定义

### 修改头像

替换 `public/avatar.svg` 为你自己的头像图片，并更新：

- `index.md` 中的 `hero.image.src`
- `about.md` 中的头像引用

### 修改站点信息

编辑 `.vitepress/config.mjs`：

- `title`：站点标题
- `description`：站点描述
- `nav`：顶部导航
- `sidebar`：侧边栏书式目录
- `socialLinks`：社交媒体链接

### 修改个人信息

编辑 `about.md` 和 `resume.md`，替换为你的真实信息。

## 目录结构

```
my-blog/
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions 部署配置
├── .vitepress/
│   ├── config.mjs           # 站点配置
│   └── theme/
│       ├── index.js         # 主题入口
│       └── style.css        # 自定义样式
├── blog/
│   ├── index.md             # 博客目录页
│   ├── tech/                # 技术思考
│   ├── life/                # 生活随笔
│   ├── reading/             # 读书笔记
│   └── notes/               # 写作约定等
├── public/
│   └── avatar.svg           # 头像
├── about.md                 # 关于我
├── resume.md                # 简历
├── index.md                 # 首页
├── package.json
└── README.md
```

## 学习资源

- [VitePress 官方文档](https://vitepress.dev/)
- [Markdown 语法](https://vitepress.dev/guide/markdown)
- [GitHub Pages 文档](https://docs.github.com/en/pages)

---

祝你写作愉快！📝
