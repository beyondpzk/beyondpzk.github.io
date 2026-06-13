#!/usr/bin/env node
// 从旧 Jekyll 文章读取 categories，写入到已迁移的 VitePress 文章中
import fs from 'fs'
import path from 'path'

const NOTES_DIR = path.resolve('../../notes')
const POSTS_DIR = path.join(NOTES_DIR, '_posts')
const BLOG_DIR = path.resolve('blog')

// 简单的分类规范化：修正明显的拼写错误
const CATEGORY_NORMALIZATION = {
  'Peception': 'Perception',
  'WorldModel': 'WorldModels',
}

function normalizeCategory(cat) {
  const trimmed = cat.trim()
  return CATEGORY_NORMALIZATION[trimmed] || trimmed
}

function parseFrontmatter(content) {
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

function parseCategories(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) return []

  const fmLines = match[1].split('\n')
  let inCategories = false
  const categories = []

  for (const line of fmLines) {
    // categories: [A, B]
    const inlineMatch = line.match(/^categories:\s*\[(.*)\]\s*$/)
    if (inlineMatch) {
      return inlineMatch[1]
        .split(',')
        .map(s => normalizeCategory(s.trim()))
        .filter(Boolean)
    }

    // categories:
    //   - A
    //   - B
    if (line.match(/^categories:\s*$/)) {
      inCategories = true
      continue
    }

    if (inCategories) {
      const listMatch = line.match(/^\s*-\s*(.+)$/)
      if (listMatch) {
        categories.push(normalizeCategory(listMatch[1].trim()))
      } else if (line.trim() === '' || line.match(/^[a-zA-Z]/)) {
        inCategories = false
      }
    }
  }

  return categories
}

function slugify(filename) {
  return filename.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '')
}

function extractYear(filename) {
  const match = filename.match(/^(\d{4})-/)
  return match ? match[1] : null
}

function updatePostCategories(slug, year, categories) {
  const postPath = path.join(BLOG_DIR, year, `${slug}.md`)
  if (!fs.existsSync(postPath)) {
    console.warn(`  ⚠️  找不到已迁移文章: ${postPath}`)
    return false
  }

  let content = fs.readFileSync(postPath, 'utf-8')
  const meta = parseFrontmatter(content)

  // 构建新的 frontmatter
  const title = meta.title || slug
  const date = meta.date || `${year}-01-01`
  const cats = categories.length > 0 ? categories : ['others']

  const newFrontmatter = `---\ntitle: ${title}\ndate: ${date}\ncategories: [${cats.join(', ')}]\n---`

  // 替换原来的 frontmatter
  content = content.replace(/^---\n[\s\S]*?\n---/, newFrontmatter)
  fs.writeFileSync(postPath, content)
  return true
}

function main() {
  if (!fs.existsSync(POSTS_DIR)) {
    console.error(`❌ 找不到旧博客目录: ${POSTS_DIR}`)
    process.exit(1)
  }

  const files = fs.readdirSync(POSTS_DIR).filter(f => f.endsWith('.md'))
  let updatedCount = 0
  let noCategoryCount = 0
  const allCategories = {}

  files.forEach(file => {
    const filePath = path.join(POSTS_DIR, file)
    const content = fs.readFileSync(filePath, 'utf-8')
    const categories = parseCategories(content)

    if (categories.length === 0) {
      noCategoryCount++
    }

    categories.forEach(cat => {
      allCategories[cat] = (allCategories[cat] || 0) + 1
    })

    const slug = slugify(file)
    const year = extractYear(file)
    if (!year) {
      console.warn(`  ⚠️  无法提取年份: ${file}`)
      return
    }

    if (updatePostCategories(slug, year, categories)) {
      updatedCount++
      console.log(`  ✓ ${year}/${slug}.md → ${categories.length > 0 ? categories.join(', ') : 'others'}`)
    }
  })

  console.log(`\n✅ 已更新 ${updatedCount} 篇文章的分类`)
  console.log(`   无分类（归入 others）: ${noCategoryCount} 篇`)
  console.log(`\n分类统计：`)
  Object.entries(allCategories)
    .sort((a, b) => b[1] - a[1])
    .forEach(([cat, count]) => {
      console.log(`   ${cat}: ${count}`)
    })
  console.log(`\n💡 提示：你可以直接修改每篇文章 frontmatter 中的 categories 字段来调整分类`)
}

main()
