#!/usr/bin/env node
// 自动扫描 blog/ 目录生成 VitePress 侧边栏配置
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

function scanDir(dir, urlPrefix = '') {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  const files = entries.filter(e => e.isFile() && e.name.endsWith('.md') && e.name !== 'index.md')
  const dirs = entries.filter(e => e.isDirectory())

  const items = []

  files.forEach(file => {
    const filePath = path.join(dir, file.name)
    const meta = getFrontmatter(filePath)
    const name = file.name.replace('.md', '')
    const link = `${urlPrefix}/${name}`
    items.push({
      text: meta.title || name,
      link,
      date: meta.date || '',
    })
  })

  // 同一文件夹内按 date 从新到旧排序
  items.sort((a, b) => {
    if (a.date && b.date) return new Date(b.date) - new Date(a.date)
    if (a.date) return -1
    if (b.date) return 1
    return a.text.localeCompare(b.text)
  })

  // 移除临时排序字段，避免写入配置
  items.forEach(item => delete item.date)

  dirs.forEach(subdir => {
    const subPath = path.join(dir, subdir.name)
    const children = scanDir(subPath, `${urlPrefix}/${subdir.name}`)
    if (children.length > 0) {
      items.push({
        text: subdir.name,
        collapsed: false,
        items: children,
      })
    }
  })

  return items
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

  const scanned = scanDir(BLOG_DIR, '/blog')

  // 按年份排序：从新到旧（2018-2026）
  const sortedGroups = [...scanned].sort((a, b) => {
    const yearA = parseInt(a.text, 10)
    const yearB = parseInt(b.text, 10)
    if (!isNaN(yearA) && !isNaN(yearB)) return yearB - yearA
    return a.text.localeCompare(b.text)
  })

  const sidebarItems = sortedGroups.map(group => {
    const year = parseInt(group.text, 10)
    const isYear = !isNaN(year)
    const textMap = {
      notes: '📝 工具与约定',
    }
    return {
      text: isYear ? `📅 ${year} 年` : (textMap[group.text] || group.text),
      collapsed: isYear ? true : false,
      items: group.items,
    }
  })

  const fullSidebar = [
    {
      text: '📚 博客目录',
      items: [{ text: '全部文章', link: '/blog/' }],
    },
    ...sidebarItems,
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
  console.log(`📝 共扫描到 ${scanned.length} 个主题`)
}

generateSidebar()
