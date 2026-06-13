#!/usr/bin/env node
// 自动扫描 blog/ 目录生成 VitePress 侧边栏配置
// 支持按年份和按分类两种视图
import fs from 'fs'
import path from 'path'

const BLOG_DIR = path.resolve('blog')
const CONFIG_PATH = path.resolve('.vitepress/config.mjs')

function getFrontmatter(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8')
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) return {}
  const meta = {}
  match[1].split('\n').forEach(line => {
    const sepIndex = line.indexOf(':')
    if (sepIndex > 0) {
      const key = line.slice(0, sepIndex).trim()
      let value = line.slice(sepIndex + 1).trim()
      value = value.replace(/^["']|["']$/g, '')
      meta[key] = value
    }
  })
  return meta
}

function parseCategories(meta) {
  const raw = meta.categories
  if (!raw) return ['others']

  // 解析数组格式: [A, B] 或 [A]
  if (typeof raw === 'string') {
    const inlineMatch = raw.match(/^\[(.*)\]$/)
    if (inlineMatch) {
      return inlineMatch[1]
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)
    }
    // 单个字符串
    return [raw.trim()]
  }

  if (Array.isArray(raw)) {
    return raw.map(c => String(c).trim()).filter(Boolean)
  }

  return ['others']
}

function scanPosts(dir, urlPrefix = '') {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  const files = entries.filter(e => e.isFile() && e.name.endsWith('.md') && e.name !== 'index.md')
  const dirs = entries.filter(e => e.isDirectory())

  const posts = []

  files.forEach(file => {
    const filePath = path.join(dir, file.name)
    const meta = getFrontmatter(filePath)
    const name = file.name.replace('.md', '')
    const link = `${urlPrefix}/${name}`
    const year = path.basename(dir)
    const displayName = meta.title || name
    const displayText = meta.date ? `${displayName} (${meta.date})` : displayName

    posts.push({
      text: displayText,
      link,
      date: meta.date || '',
      year,
      categories: parseCategories(meta),
    })
  })

  // 递归扫描子目录
  dirs.forEach(subdir => {
    const subPath = path.join(dir, subdir.name)
    const children = scanPosts(subPath, `${urlPrefix}/${subdir.name}`)
    posts.push(...children)
  })

  return posts
}

function sortByDate(items) {
  return [...items].sort((a, b) => {
    if (a.date && b.date) return new Date(b.date) - new Date(a.date)
    if (a.date) return -1
    if (b.date) return 1
    return a.text.localeCompare(b.text)
  })
}

function buildYearGroups(posts) {
  const yearMap = {}

  posts.forEach(post => {
    // 只把数字年份（2018-2026）归入按年份查看
    if (!/^\d{4}$/.test(post.year)) return
    if (!yearMap[post.year]) yearMap[post.year] = []
    yearMap[post.year].push(post)
  })

  return Object.keys(yearMap)
    .sort((a, b) => parseInt(b) - parseInt(a))
    .map(year => ({
      text: `📅 ${year} 年`,
      collapsed: true,
      items: sortByDate(yearMap[year]).map(({ text, link }) => ({ text, link })),
    }))
}

function buildCategoryGroups(posts) {
  const categoryMap = {}

  posts.forEach(post => {
    post.categories.forEach(cat => {
      if (!categoryMap[cat]) categoryMap[cat] = []
      categoryMap[cat].push(post)
    })
  })

  // 按文章数量降序，others 放最后
  const categories = Object.keys(categoryMap).sort((a, b) => {
    if (a === 'others') return 1
    if (b === 'others') return -1
    return categoryMap[b].length - categoryMap[a].length
  })

  return categories.map(cat => ({
    text: `📂 ${cat}`,
    collapsed: true,
    items: sortByDate(categoryMap[cat]).map(({ text, link }) => ({ text, link })),
  }))
}

function formatSidebar(items, indent = 6) {
  const prefix = ' '.repeat(indent)
  return items.map(item => {
    const lines = []
    lines.push(`${prefix}{`)
    lines.push(`${prefix}  text: '${item.text}',`)
    if (item.link) {
      lines.push(`${prefix}  link: '${item.link}',`)
    }
    if (item.collapsed !== undefined) {
      lines.push(`${prefix}  collapsed: ${item.collapsed},`)
    }
    if (item.items) {
      lines.push(`${prefix}  items: [`)
      lines.push(formatSidebar(item.items, indent + 4))
      lines.push(`${prefix}  ],`)
    }
    lines.push(`${prefix}},`)
    return lines.join('\n')
  }).join('\n')
}

function generateSidebar() {
  if (!fs.existsSync(BLOG_DIR)) {
    console.error(`❌ 目录不存在: ${BLOG_DIR}`)
    process.exit(1)
  }

  const posts = scanPosts(BLOG_DIR, '/blog')
  const yearGroups = buildYearGroups(posts)
  const categoryGroups = buildCategoryGroups(posts)

  const fullSidebar = [
    {
      text: '📚 博客目录',
      items: [{ text: '全部文章', link: '/blog/' }],
    },
    {
      text: '📅 按年份查看',
      collapsed: false,
      items: yearGroups,
    },
    {
      text: '📂 按分类查看',
      collapsed: false,
      items: categoryGroups,
    },
  ]

  let config = fs.readFileSync(CONFIG_PATH, 'utf-8')

  // 精确定位 sidebar: { '/blog/': [ ... ] } 块
  const sidebarStart = config.indexOf("sidebar: {")
  if (sidebarStart === -1) {
    console.error('⚠️ 无法定位 sidebar 配置，请手动检查 .vitepress/config.mjs')
    process.exit(1)
  }

  // 从 sidebar: { 开始找到匹配的闭合 }
  let braceCount = 0
  let inString = false
  let stringChar = ''
  let i = sidebarStart + "sidebar: {".length - 1
  let found = false

  for (; i < config.length; i++) {
    const char = config[i]

    // 处理字符串
    if (char === '"' || char === "'") {
      if (!inString) {
        inString = true
        stringChar = char
      } else if (stringChar === char) {
        inString = false
      }
      continue
    }

    if (inString) continue

    if (char === '{') braceCount++
    else if (char === '}') {
      braceCount--
      if (braceCount === 0) {
        found = true
        break
      }
    }
  }

  if (!found) {
    console.error('⚠️ 无法解析 sidebar 配置的闭合括号')
    process.exit(1)
  }

  const newSidebarBlock = `sidebar: {
      '/blog/': [
${formatSidebar(fullSidebar, 8)}
      ],
    }`

  config = config.slice(0, sidebarStart) + newSidebarBlock + config.slice(i + 1)
  fs.writeFileSync(CONFIG_PATH, config)
  console.log('✅ 侧边栏已自动更新')
  console.log(`📝 共 ${posts.length} 篇文章`)
  console.log(`📅 ${yearGroups.length} 个年份`)
  console.log(`📂 ${categoryGroups.length} 个分类`)
}

generateSidebar()
