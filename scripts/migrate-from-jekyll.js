#!/usr/bin/env node
// 将 Jekyll /_posts 目录下的文章迁移到 VitePress 结构
import fs from 'fs'
import path from 'path'

const NOTES_DIR = path.resolve('../../notes')  // 旧博客路径
const POSTS_DIR = path.join(NOTES_DIR, '_posts')
const ASSETS_DIR = path.join(NOTES_DIR, 'assets/img')
const TARGET_DIR = path.resolve('blog')
const PUBLIC_ASSETS_DIR = path.resolve('public/assets/img')

function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) return { meta: {}, body: content }
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
  return { meta, body: content.slice(match[0].length).trim() }
}

function slugify(filename) {
  // 2026-06-02-RLNeeds.md -> RLNeeds
  return filename.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '')
}

function extractYear(filename) {
  const match = filename.match(/^(\d{4})-/)
  return match ? match[1] : 'other'
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true })
}

function copyImage(src, dest) {
  if (!fs.existsSync(src)) {
    console.warn(`  ⚠️  图片不存在: ${src}`)
    return false
  }
  ensureDir(path.dirname(dest))
  fs.copyFileSync(src, dest)
  return true
}

function migrate() {
  if (!fs.existsSync(POSTS_DIR)) {
    console.error(`❌ 找不到旧博客目录: ${POSTS_DIR}`)
    console.error('请确认 ~/notes 仓库已克隆')
    process.exit(1)
  }

  ensureDir(TARGET_DIR)
  ensureDir(PUBLIC_ASSETS_DIR)

  const files = fs.readdirSync(POSTS_DIR).filter(f => f.endsWith('.md'))
  console.log(`📚 发现 ${files.length} 篇文章，开始迁移...\n`)

  let migratedCount = 0
  let imageCount = 0

  files.forEach(file => {
    const filePath = path.join(POSTS_DIR, file)
    const content = fs.readFileSync(filePath, 'utf-8')
    const { meta, body } = parseFrontmatter(content)

    const slug = slugify(file)
    const year = extractYear(file)
    const yearDir = path.join(TARGET_DIR, year)
    ensureDir(yearDir)

    const targetPath = path.join(yearDir, `${slug}.md`)

    // 处理内容
    let newBody = body
      .replace(/^\[TOC\]\n?/m, '')  // 移除 Jekyll 的 TOC 标记
      .replace(/\n{3,}/g, '\n\n')   // 清理多余空行

    // 提取并复制图片
    const imgRegex = /!\[([^\]]*)\]\((\/assets\/img\/[^)]+)\)/g
    const imgRefs = [...newBody.matchAll(imgRegex)]

    imgRefs.forEach(match => {
      const altText = match[1]
      const oldPath = match[2]
      const imgName = path.basename(oldPath)
      const srcImgPath = path.join(NOTES_DIR, oldPath)
      const destImgPath = path.join(PUBLIC_ASSETS_DIR, imgName)

      if (copyImage(srcImgPath, destImgPath)) {
        // VitePress 中 public 目录下的资源保持相同路径
        const newRef = `![${altText}](${oldPath})`
        newBody = newBody.replace(match[0], newRef)
        imageCount++
      }
    })

    // 构建新的 frontmatter
    const title = meta.title || slug
    const date = meta.date || `${year}-01-01`

    const newContent = `---\ntitle: ${title}\ndate: ${date}\n---\n\n${newBody}\n`

    fs.writeFileSync(targetPath, newContent)
    migratedCount++
    console.log(`  ✓ ${year}/${slug}.md`)
  })

  console.log(`\n✅ 迁移完成`)
  console.log(`   文章: ${migratedCount} 篇`)
  console.log(`   图片: ${imageCount} 张`)
  console.log(`\n下一步运行: npm run sync`)
}

migrate()
