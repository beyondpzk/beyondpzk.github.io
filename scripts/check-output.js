import fs from 'node:fs'
import path from 'node:path'
import { excludedPages } from '../lib/content.js'
const root = '.vitepress/dist'
const forbidden = excludedPages().map((file) => file.replace(/\.md$/, ''))
const errors = [],
  pages = new Map(),
  files = new Set()
const origin = 'https://build.invalid'
const decode = (text) =>
  text
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#(?:x([\da-f]+)|(\d+));/gi, (_, hex, decimal) =>
      String.fromCodePoint(parseInt(hex || decimal, hex ? 16 : 10)),
    )
function walk(dir) {
  for (const item of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, item.name)
    if (item.isDirectory()) {
      walk(file)
      continue
    }
    const relative = '/' + path.relative(root, file).split(path.sep).join('/')
    files.add(relative)
    if (!/\.(html|js|json|xml)$/.test(item.name)) continue
    const content = fs.readFileSync(file, 'utf8')
    for (const slug of forbidden)
      if (content.includes(slug))
        errors.push(`${relative}: references an excluded article`)
    if (!file.endsWith('.html')) continue
    if (
      content.includes('<mjx-merror') ||
      content.includes('data-mml-node="merror"')
    )
      errors.push(`${relative}: invalid rendered formula`)
    // VitePress emits quoted HTML attributes. Check generated Vue links too,
    // which the Markdown-only dead-link check cannot inspect.
    const ids = new Set(
      [...content.matchAll(/\sid="([^"]*)"/g)].map((m) => decode(m[1])),
    )
    const links = [...content.matchAll(/\shref="([^"]*)"/g)].map((m) =>
      decode(m[1]),
    )
    pages.set(relative, { ids, links })
    if (
      relative.startsWith('/blog/') &&
      relative !== '/blog/index.html' &&
      [...content.matchAll(/<h1(?:\s|>)/g)].length !== 1
    )
      errors.push(`${relative}: expected one rendered H1`)
  }
}
if (!fs.existsSync(root))
  throw new Error('Build the site before checking its output')
walk(root)
for (const [source, page] of pages) {
  for (const link of page.links) {
    const url = new URL(link, origin + source)
    if (url.origin !== origin) continue
    const pathname = decodeURIComponent(url.pathname)
    const candidates = [
      pathname,
      pathname + '.html',
      pathname.replace(/\/$/, '') + '/index.html',
    ]
    const target = candidates.find((file) => files.has(file))
    if (!target) {
      errors.push(`${source}: missing link ${pathname}`)
      continue
    }
    if (
      url.hash &&
      pages.has(target) &&
      !pages.get(target).ids.has(decodeURIComponent(url.hash.slice(1)))
    )
      errors.push(`${source}: missing anchor ${target}${url.hash}`)
  }
}
for (const file of forbidden)
  if (files.has('/' + file + '.html'))
    errors.push(`Excluded page emitted: ${file}`)
if (errors.length) {
  console.error([...new Set(errors)].join('\n'))
  process.exit(1)
}
console.log(
  `构建产物检查通过：${pages.size} 个页面，站内链接、章节锚点与主标题有效；未发布文章未进入页面、索引、脚本或站点地图。`,
)
