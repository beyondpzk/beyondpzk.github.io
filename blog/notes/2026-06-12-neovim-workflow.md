---
title: Neovim + iTerm2 工作流
date: 2026-06-12
categories: [工具与约定]
---

# Neovim + iTerm2 工作流

如果你平时主要用 **Neovim + iTerm2**，这个工作流会让你在终端里高效地写作、管理图片、预览和发布博客。

---

## 1. 推荐插件

### 1.1 图片粘贴：`img-clip.nvim`

在 macOS 下，`img-clip.nvim` 可以直接从剪贴板粘贴图片到指定路径，是 Neovim 写博客的必备插件。

**安装（lazy.nvim）**：

```lua
{
  "HakonHarnes/img-clip.nvim",
  event = "VeryLazy",
  opts = {
    default = {
      embed_image_as_base64 = false,
      prompt_for_file_name = false,
      drag_and_drop = {
        insert_mode = true,
      },
      -- 关键配置：根据文件路径自动决定图片保存位置
      file_name = "%Y-%m-%d-%H-%M-%S",
      -- 使用相对路径引用
      relative_to_current_file = true,
      -- 图片保存在当前文件同级目录的 .assets 子目录
      dir_path = function()
        return vim.fn.expand("%:t:r") .. ".assets"
      end,
    },
  },
  keys = {
    { "<leader>p", "<cmd>PasteImage<cr>", desc = "Paste image from clipboard" },
  },
}
```

**使用**：

1. 截图（`Cmd + Shift + 4`）或复制图片到剪贴板
2. 在 Neovim 中按 `<leader>p`
3. 图片自动保存为 `文章名.assets/2026-06-12-xxx.png`
4. Markdown 中自动插入相对路径

### 1.2 Markdown 预览：`markdown-preview.nvim`

```lua
{
  "iamcco/markdown-preview.nvim",
  cmd = { "MarkdownPreviewToggle", "MarkdownPreview", "MarkdownPreviewStop" },
  ft = { "markdown" },
  build = function() vim.fn["mkdp#util#install"]() end,
  keys = {
    { "<leader>mp", "<cmd>MarkdownPreviewToggle<cr>", desc = "Toggle Markdown Preview" },
  },
}
```

> 注意：这个预览是 Markdown 的通用预览，不是 VitePress 主题预览。要看 VitePress 实际效果，还是需要 `npm run docs:dev`。

### 1.3 文件浏览：`oil.nvim`

推荐用 `oil.nvim`，它把文件管理器变成了可编辑的缓冲区，非常适合批量重命名、移动图片等操作。

```lua
{
  "stevearc/oil.nvim",
  opts = {},
  dependencies = { "nvim-tree/nvim-web-devicons" },
  keys = {
    { "-", "<cmd>Oil<cr>", desc = "Open parent directory" },
  },
}
```

### 1.4 模糊查找：`telescope.nvim`

快速在博客文章中搜索内容：

```lua
{
  "nvim-telescope/telescope.nvim",
  dependencies = { "nvim-lua/plenary.nvim" },
  keys = {
    { "<leader>ff", "<cmd>Telescope find_files<cr>", desc = "Find files" },
    { "<leader>fg", "<cmd>Telescope live_grep<cr>", desc = "Live grep" },
  },
}
```

### 1.5 Markdown 增强：`render-markdown.nvim`

让 Markdown 在 Neovim 里渲染得更美观：

```lua
{
  "MeanderingProgrammer/render-markdown.nvim",
  opts = {},
  dependencies = { "nvim-treesitter/nvim-treesitter", "nvim-tree/nvim-web-devicons" },
}
```

---

## 2. 项目内的便捷脚本

我已经在 `scripts/` 目录下准备了两个脚本：

### 2.1 `scripts/generate-sidebar.js` —— 自动生成侧边栏

扫描 `blog/` 目录，自动更新 `.vitepress/config.mjs` 中的侧边栏。

```bash
npm run sync
```

### 2.2 `scripts/preview.sh` —— 一键预览

自动同步侧边栏并启动 VitePress 开发服务器。

```bash
chmod +x scripts/preview.sh
./scripts/preview.sh
```

---

## 3. 推荐的目录组织

```
my-blog/
├── blog/
│   ├── 2026/
│   │   ├── RLNeeds.md
│   │   ├── RLNeeds.assets/
│   │   │   └── image-20260612.png
│   │   └── pi0.6.md
│   ├── 2025/
│   └── notes/
├── scripts/
│   ├── generate-sidebar.js
│   └── preview.sh
└── .vitepress/
    └── config.mjs
```

**为什么把图片放在 `文章名.assets/`？**

- 和文章一一对应，找图方便
- 移动/删除文章时不会遗留无用图片
- Typora 默认也使用这种命名方式，方便混用

---

## 4. 完整的写作流程

### 4.1 快速打开博客项目

在 iTerm2 中：

```bash
alias blog='cd ~/Desktop/my-blog && nvim .'
```

把上面这行加入 `~/.zshrc` 或 `~/.bashrc`，以后输入 `blog` 就能一键进入。

### 4.2 新建文章

```bash
# 进入博客目录
cd ~/Desktop/my-blog

# 新建文章
nvim blog/2026/my-new-article.md
```

文件开头写上 frontmatter：

```markdown
---
title: 文章标题
date: 2026-06-12
---

# 文章标题

正文...
```

### 4.3 粘贴图片

在 Neovim 中：

1. 截图或复制图片
2. 按 `<leader>p`（如果你按上面的配置）
3. 自动生成：`![image](./my-new-article.assets/2026-06-12-xxx.png)`

### 4.4 同步侧边栏并预览

```bash
# 方式一：分别执行
npm run sync
npm run docs:dev

# 方式二：一键执行
./scripts/preview.sh
```

然后浏览器访问：`http://localhost:5173/`

### 4.5 发布

```bash
git add .
git commit -m "add: 新文章"
git push origin main
```

GitHub Actions 会自动部署到 GitHub Pages。

---

## 5. iTerm2 推荐配置

### 5.1 设置博客项目的 Profile

在 iTerm2 中新建一个 Profile：

- **Name**: Blog
- **Command**: `cd ~/Desktop/my-blog && nvim .`
- **Working Directory**: `~/Desktop/my-blog`

这样每次点一下就能进入博客写作环境。

### 5.2 分屏工作流

一个屏幕分两个 pane：

- 左边：Neovim 写文章
- 右边：`npm run docs:dev` 实时预览

### 5.3 快捷键建议

在 iTerm2 里给常用命令设置快捷键，或者直接在 Neovim 里用 `:term`：

```vim
:term
npm run docs:dev
```

按 `<C-\><C-n>` 回到普通模式。

---

## 6. 可选：Tmux 会话管理

如果你用 Tmux，可以创建一个固定的博客会话：

```bash
# ~/.config/tmux/blog-session.sh
#!/bin/bash
SESSION="blog"

tmux has-session -t $SESSION 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION -c ~/Desktop/my-blog
    tmux split-window -h -t $SESSION -c ~/Desktop/my-blog
    tmux send-keys -t $SESSION:0.0 'nvim .' C-m
    tmux send-keys -t $SESSION:0.1 'npm run docs:dev' C-m
fi
tmux attach -t $SESSION
```

运行这个脚本，就会同时打开 Neovim 和预览服务器。

---

## 7. 推荐命令行工具

| 工具 | 用途 | 安装 |
|------|------|------|
| `fd` | 快速查找文件 | `brew install fd` |
| `ripgrep` | 快速搜索内容 | `brew install ripgrep` |
| `fzf` | 模糊查找 | `brew install fzf` |
| `glow` | 终端 Markdown 渲染 | `brew install glow` |

示例：

```bash
# 查找所有 markdown 文件
fd .md blog/

# 搜索包含某个关键词的文章
rg "VLA" blog/

# 终端渲染 markdown
glow blog/2026/RLNeeds.md
```

---

## 8. 最小化插件配置示例

如果你只想快速开始，下面是一份最小化的 Neovim 配置：

```lua
-- init.lua 或 ~/.config/nvim/lua/plugins/blog.lua
return {
  -- 图片粘贴
  {
    "HakonHarnes/img-clip.nvim",
    event = "VeryLazy",
    opts = {
      default = {
        file_name = "%Y-%m-%d-%H-%M-%S",
        relative_to_current_file = true,
        dir_path = function()
          return vim.fn.expand("%:t:r") .. ".assets"
        end,
      },
    },
    keys = {
      { "<leader>p", "<cmd>PasteImage<cr>", desc = "Paste image" },
    },
  },

  -- Markdown 预览
  {
    "iamcco/markdown-preview.nvim",
    cmd = { "MarkdownPreviewToggle" },
    ft = { "markdown" },
    build = function() vim.fn["mkdp#util#install"]() end,
    keys = {
      { "<leader>mp", "<cmd>MarkdownPreviewToggle<cr>", desc = "Markdown preview" },
    },
  },
}
```

---

## 9. 总结

Neovim + iTerm2 写博客的核心是：

1. **`img-clip.nvim`** 解决图片粘贴
2. **`npm run sync`** 自动同步侧边栏
3. **`npm run docs:dev`** 实时预览
4. **Tmux / iTerm2 Profile** 管理写作环境

这套工作流完全在终端里完成，不需要离开 Neovim 就能写文章、插图片、预览和发布。
