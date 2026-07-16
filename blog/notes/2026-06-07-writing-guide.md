---
title: 写作约定
date: 2026-06-07
categories: [工具与约定]
---

# 写作约定

为了让博客长期保持一致的阅读和写作体验，我制定了以下约定。

## 文件组织

```
my-blog/
├── blog/
│   ├── tech/          # 技术思考
│   ├── life/          # 生活随笔
│   ├── reading/       # 读书笔记
│   └── notes/         # 写作约定、工具说明等
├── public/            # 公共资源：头像、图片等
│   ├── avatar.png
│   └── *.jpg
└── .vitepress/
    └── config.mjs     # 站点配置
```

## 图片处理

### 方法 1：放入 public 目录（推荐用于全局图片）

把图片放到 `public/` 目录下，然后在 Markdown 中这样引用：

```markdown
![说明文字](/image-name.jpg)
```

### 方法 2：与文章放在同一目录（推荐用于文章专属图片）

把图片和 Markdown 文件放在同一文件夹：

```
blog/tech/
  ├── why-vitepress.md
  └── screenshot.png
```

引用方式：

```markdown
![说明文字](./screenshot.png)
```

### 推荐的编辑器工作流

想要"复制粘贴图片"，建议使用以下编辑器之一：

- **Obsidian**：设置附件文件夹为 `public` 或当前文件夹，粘贴图片会自动保存并生成相对路径。
- **Typora**：在偏好设置中指定图片保存路径，复制粘贴图片时会自动处理。
- **VS Code + Markdown Paste 插件**：支持直接粘贴图片到指定目录。

## Frontmatter 格式

每篇文章开头建议包含：

```yaml
---
title: 文章标题
date: 2026-06-12
---
```

## 侧边栏更新

新增文章后，记得在 `.vitepress/config.mjs` 的 `sidebar` 中添加对应链接，这样读者才能在侧边栏看到它。

## 发布流程

1. 在本地用 `npm run docs:dev` 预览
2. 确认无误后，提交到 GitHub
3. GitHub Actions 自动构建并部署到 GitHub Pages
