import { defineConfig } from 'vitepress'
import { getPosts, excludedPages, relatedPosts } from '../lib/content.js'
import { topics } from '../lib/taxonomy.js'
import { collections } from '../content/collections.js'
import { bookSidebar } from './book-sidebar.js'

const posts = getPosts()
const postByFile = new Map(posts.map((post) => [post.file, post]))
const siteUrl = 'https://beyondpzk.github.io'
const browseSidebar = [
  {
    text: '浏览笔记',
    items: [
      { text: '全部文章', link: '/blog/' },
      { text: '研究主题', link: '/topics/' },
      { text: '专题阅读', link: '/series/' },
    ],
  },
  {
    text: '研究主题',
    items: topics.map((topic) => ({
      text: topic.name,
      link: `/topics/${topic.id}`,
    })),
  },
  {
    text: '专题路线',
    collapsed: true,
    items: collections.map((item) => ({
      text: item.title,
      link: `/series/${item.id}`,
    })),
  },
]
export default defineConfig({
  title: 'BEYOND',
  description: '世界模型、机器人导航与具身智能的论文精读、技术判断和工程实践。',
  lang: 'zh-CN',
  base: '/',
  srcExclude: [
    ...excludedPages(),
    'README.md',
    'templates/**',
    'tests/**',
    'books/my-first-book/**',
  ],
  ignoreDeadLinks: false,
  lastUpdated: true,
  sitemap: { hostname: siteUrl },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
  ],
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '文章', link: '/blog/' },
      { text: '主题', link: '/topics/' },
      { text: '专题', link: '/series/' },
      { text: '书籍', link: '/books/' },
      { text: '关于我', link: '/about' },
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/beyondpzk' }],
    sidebar: {
      '/blog/': browseSidebar,
      '/topics/': browseSidebar,
      '/series/': browseSidebar,
      ...bookSidebar,
    },
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: { buttonText: '搜索', buttonAriaLabel: '搜索文章' },
              modal: {
                noResultsText: '没有找到相关内容',
                resetButtonTitle: '清除搜索',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                },
              },
            },
          },
        },
      },
    },
    outline: { label: '本页目录', level: [2, 3] },
    sidebarMenuLabel: '浏览目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '切换明暗主题',
    docFooter: { prev: '上一篇', next: '下一篇' },
    editLink: {
      pattern:
        'https://github.com/beyondpzk/beyondpzk.github.io/edit/main/:path',
      text: '编辑此页',
    },
    footer: {
      message: 'BEYOND · 理解模型，走向真实世界。',
      copyright: 'Copyright © 2026',
    },
    lastUpdated: {
      text: '实际更新于',
      formatOptions: {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        timeZone: 'Asia/Shanghai',
      },
    },
  },
  transformPageData(pageData) {
    const post = postByFile.get(pageData.filePath)
    if (!post) return
    pageData.description = post.summary
    pageData.frontmatter.article = post
    pageData.frontmatter.relatedPosts = relatedPosts(post, posts)
    pageData.frontmatter.prev = false
    pageData.frontmatter.next = false
    if (post.updatedAt)
      pageData.lastUpdated = Date.parse(post.updatedAt + 'T00:00:00+08:00')
  },
  transformHead({ pageData }) {
    const pathname = pageData.relativePath
      .replace(/index\.md$/, '')
      .replace(/\.md$/, '.html')
    const url = siteUrl + '/' + pathname
    return [
      ['link', { rel: 'canonical', href: url }],
      [
        'meta',
        {
          property: 'og:title',
          content: `${pageData.title || 'BEYOND'} | BEYOND`,
        },
      ],
      [
        'meta',
        {
          property: 'og:description',
          content:
            pageData.description ||
            '世界模型、机器人导航与具身智能的研究笔记。',
        },
      ],
      [
        'meta',
        {
          property: 'og:type',
          content: pageData.frontmatter.article ? 'article' : 'website',
        },
      ],
      ['meta', { property: 'og:url', content: url }],
      ['meta', { property: 'og:locale', content: 'zh_CN' }],
    ]
  },
  markdown: {
    lineNumbers: true,
    math: true,
    image: { lazyLoading: true },
    config(md) {
      // Legacy notes occasionally skip heading levels. Keep a readable outline
      // without touching code blocks, equations, or the text of headings.
      md.core.ruler.push('continuous-heading-levels', (state) => {
        let previous = 0
        for (let i = 0; i < state.tokens.length; i++) {
          const token = state.tokens[i]
          if (token.type !== 'heading_open') continue
          const requested = Number(token.tag.slice(1))
          const level = previous ? Math.min(requested, previous + 1) : 1
          token.tag = `h${level}`
          state.tokens[i + 2].tag = `h${level}`
          previous = level
        }
      })
    },
  },
})
