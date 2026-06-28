import { readFileSync } from 'node:fs'
import { createContentLoader } from 'vitepress'

export default createContentLoader('blog/**/*.md', {
  transform(raw) {
    return raw
      .filter(({ url }) => !url.endsWith('/blog/') && !url.endsWith('/blog.html'))
      .map(({ url, frontmatter }) => {
        // 从 URL 反推文件路径，例如 /blog/2026/RLNeeds.html -> blog/2026/RLNeeds.md
        const filePath = url.replace(/^\//, '').replace(/\.html$/, '') + '.md'
        const content = readFileSync(filePath, 'utf-8')
        const body = content.replace(/^---\n[\s\S]*?\n---/, '').trim()

        // 取正文中第一段非空、非标题、非图片、非分隔线的内容作为摘要
        const excerpt = body
          .split('\n')
          .map(line => line.trim())
          .filter(line => line && !line.startsWith('#') && !line.startsWith('![') && !line.startsWith('---'))
          .slice(0, 2)
          .join(' ')
          .replace(/\*\*/g, '')
          .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
          .slice(0, 160) + '…'

        return {
          title: frontmatter.title || url.split('/').pop(),
          date: frontmatter.date,
          url,
          excerpt,
          categories: parseCategories(frontmatter.categories)
        }
      })
      .filter(post => post.date)
      .sort((a, b) => new Date(b.date) - new Date(a.date))
  }
})

function parseCategories(categories) {
  if (!categories) return ['others']
  if (typeof categories === 'string') {
    const m = categories.match(/^\[(.*)\]$/)
    if (m) return m[1].split(',').map(s => s.trim()).filter(Boolean)
    return [categories.trim()]
  }
  if (Array.isArray(categories)) return categories.map(String).filter(Boolean)
  return ['others']
}
