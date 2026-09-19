import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import matter from 'gray-matter'
import MarkdownIt from 'markdown-it'
import {
  topics,
  articleTypes,
  inferTopic,
  inferType,
  inferTags,
} from './taxonomy.js'
import { localOnlyPosts } from '../scripts/local-only-posts.js'

const markdown = new MarkdownIt({ html: true })
export function markdownFiles(dir) {
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .flatMap((entry) => {
      const file = path.join(dir, entry.name)
      return entry.isDirectory()
        ? markdownFiles(file)
        : entry.name.endsWith('.md')
          ? [file]
          : []
    })
    .sort()
}
export function isPublicPost(file, data) {
  if (data.draft !== undefined && typeof data.draft !== 'boolean')
    throw new Error(`${file}: draft must be a boolean`)
  if (
    data.visibility !== undefined &&
    !['public', 'private'].includes(data.visibility)
  )
    throw new Error(`${file}: visibility must be public or private`)
  return (
    !localOnlyPosts.includes(file) &&
    data.draft !== true &&
    data.visibility !== 'private'
  )
}
export function excludedPages(root = process.cwd()) {
  return [
    ...new Set([
      ...localOnlyPosts,
      ...markdownFiles(path.join(root, 'blog')).flatMap((file) => {
        const relative = path.relative(root, file).split(path.sep).join('/')
        // Do not read notes already declared local-only.
        if (localOnlyPosts.includes(relative)) return []
        return isPublicPost(
          relative,
          matter(fs.readFileSync(file, 'utf8')).data,
        )
          ? []
          : [relative]
      }),
    ]),
  ]
}
export function dateOnly(value) {
  if (value instanceof Date && Number.isFinite(value.getTime()))
    return value.toISOString().slice(0, 10)
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}/.test(value)) return ''
  const date = value.slice(0, 10)
  return Number.isFinite(Date.parse(date)) &&
    new Date(date).toISOString().slice(0, 10) === date
    ? date
    : ''
}
export function plainText(source) {
  const text =
    markdown
      .parseInline(source, {})[0]
      ?.children?.map((token) => {
        if (['text', 'code_inline'].includes(token.type)) return token.content
        if (['softbreak', 'hardbreak'].includes(token.type)) return ' '
        return ''
      })
      .join('') || ''
  return text
    .replace(/\$\$?[\s\S]*?\$\$?/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}
export function summarize(body, explicit) {
  if (explicit) return plainText(String(explicit)).slice(0, 200)
  const tokens = markdown.parse(body, {})
  let candidate = ''
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i - 1].type !== 'paragraph_open' || tokens[i].type !== 'inline')
      continue
    const text = plainText(tokens[i].content)
    if (
      text.length < 35 ||
      /^(论文|作者|机构|链接|项目|paper|http|好的[，,])/i.test(text)
    )
      continue
    candidate = text
    break
  }
  if (!candidate) candidate = '阅读这篇研究笔记，了解方法、关键设计与技术分析。'
  return candidate.length > 150
    ? candidate.slice(0, 150).replace(/[，、；：\s]+$/, '') + '…'
    : candidate
}
function gitDates(root, addedOnly = false) {
  const dates = new Map()
  try {
    const args = [
      '-c',
      'core.quotepath=false',
      'log',
      '--format=%x1e%cI',
      '--name-only',
      ...(addedOnly ? ['--diff-filter=A'] : []),
      '--',
      'blog',
    ]
    const log = execFileSync('git', args, {
      cwd: root,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      maxBuffer: 16 * 1024 * 1024,
    })
    for (const record of log.split('\x1e')) {
      const [stamp, ...files] = record.trim().split('\n')
      if (!stamp) continue
      for (const file of files.filter(Boolean))
        if (!dates.has(file) || addedOnly) dates.set(file, dateOnly(stamp))
    }
  } catch {
    /* Source archives may omit .git; explicit metadata still works. */
  }
  return dates
}
export function getPosts(root = process.cwd()) {
  const updated = gitDates(root),
    created = gitDates(root, true)
  return markdownFiles(path.join(root, 'blog'))
    .flatMap((absolute) => {
      const file = path.relative(root, absolute).split(path.sep).join('/')
      if (path.basename(file) === 'index.md' || localOnlyPosts.includes(file))
        return []
      const source = fs.readFileSync(absolute, 'utf8')
      const parsed = matter(source)
      const header =
        source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)?.[1] || ''
      const { data, content } = parsed
      // YAML's timestamp resolver normalizes impossible dates; validate the source
      // before relying on parsed Date objects.
      for (const match of header.matchAll(
        /^(date|paperDate|publishedAt|updatedAt):\s*['"]?(\d{4}-\d{2}-\d{2})/gm,
      )) {
        if (!dateOnly(match[2])) throw new Error(`${file}: invalid ${match[1]}`)
      }
      if (!isPublicPost(file, data)) return []
      const topic = inferTopic(data, file),
        type = inferType(data, content, file)
      if (!topics.some((t) => t.id === topic))
        throw new Error(`${file}: unknown topic ${topic}`)
      if (!articleTypes.includes(type))
        throw new Error(`${file}: unknown article type ${type}`)
      for (const field of ['paperDate', 'publishedAt', 'updatedAt']) {
        if (data[field] !== undefined && !dateOnly(data[field]))
          throw new Error(`${file}: invalid ${field}`)
      }
      const date = dateOnly(data.date || data.publishedAt)
      if (!data.title || !date)
        throw new Error(
          `${file}: title and a valid date (YYYY-MM-DD) are required`,
        )
      return [
        {
          file,
          url: '/' + file.replace(/\.md$/, '.html'),
          title: String(data.title),
          date,
          year: date.slice(0, 4),
          publishedAt: dateOnly(data.publishedAt) || created.get(file) || '',
          updatedAt: dateOnly(data.updatedAt) || updated.get(file) || '',
          paperDate: dateOnly(data.paperDate),
          topic,
          topicName: topics.find((t) => t.id === topic).name,
          type,
          tags: inferTags(data, content),
          summary: summarize(content, data.summary),
          readingMinutes: Math.max(1, Math.ceil(content.length / 900)),
          featured: data.featured === true,
        },
      ]
    })
    .sort(
      (a, b) => b.date.localeCompare(a.date) || a.file.localeCompare(b.file),
    )
}
export function relatedPosts(post, posts, limit = 3) {
  return posts
    .filter((p) => p.file !== post.file && p.topic === post.topic)
    .map((p) => ({
      post: p,
      score: p.tags.filter((t) => post.tags.includes(t)).length,
    }))
    .sort((a, b) => b.score - a.score || b.post.date.localeCompare(a.post.date))
    .slice(0, limit)
    .map(({ post: p }) => ({ title: p.title, url: p.url, summary: p.summary }))
}
