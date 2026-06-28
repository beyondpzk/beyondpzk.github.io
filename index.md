---
layout: home

hero:
  name: WordLapse(文字时光)
  text: Tenacious life, proud journey
  tagline: AI 研究笔记 · 论文精读 · 技术思考
  image:
    src: /assets/img/prof_pic.jpg
    alt: 头像
  actions:
    - theme: brand
      text: 开始阅读
      link: /blog/
    - theme: alt
      text: 关于我
      link: /about

features:
  - icon: 📖
    title: 像书一样组织
    details: 侧边栏按主题分卷，论文笔记、技术思考一目了然，方便系统阅读和回顾。
  - icon: 🤖
    title: AI 论文精读
    details: VLA、世界模型、自动驾驶、机器人学习等领域的论文笔记与思考。
  - icon: ✍️
    title: Markdown 写作
    details: 用纯 Markdown 写作，支持代码高亮、数学公式、表格，专注于内容本身。
  - icon: ⚡
    title: 极速访问
    details: 基于 VitePress 静态生成，页面加载飞快，完美部署在 GitHub Pages 上。
---

## 最新文章

<PostList :posts="data" :per-page="20" />

<script setup>
import { data } from './posts.data.js'
</script>
