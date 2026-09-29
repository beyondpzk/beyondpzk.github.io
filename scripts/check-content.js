import fs from 'node:fs'
import MarkdownIt from 'markdown-it'
import matter from 'gray-matter'
import { getPosts } from '../lib/content.js'
import { resolveCollections } from '../content/collections.js'
import { topics } from '../lib/taxonomy.js'
const posts = getPosts(),
  md = new MarkdownIt({ html: true })
const errors = []
for (const post of posts) {
  const { data, content } = matter(fs.readFileSync(post.file, 'utf8'))
  if (!data.topic || !data.type)
    errors.push(`${post.file}: specify topic and type`)
  const h1 = md
    .parse(content, {})
    .filter((t) => t.type === 'heading_open' && t.tag === 'h1')
  if (h1.length !== 1)
    errors.push(`${post.file}: expected one H1, found ${h1.length}`)
  if (!post.summary) errors.push(`${post.file}: missing summary`)
}
for (const topic of topics)
  if (!fs.existsSync(`topics/${topic.id}.md`))
    errors.push(`Missing topic page: ${topic.id}`)
resolveCollections(posts)
if (errors.length) {
  console.error(errors.join('\n'))
  process.exit(1)
}
console.log(
  `内容检查通过：${posts.length} 篇文章，单一主标题、主题、类型和专题链接均有效。`,
)
