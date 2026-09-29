import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  getPosts,
  excludedPages,
  dateOnly,
  summarize,
  plainText,
  relatedPosts,
} from '../lib/content.js'

function fixture(t, files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'beyond-content-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  for (const [name, source] of Object.entries(files)) {
    const file = path.join(root, name)
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fs.writeFileSync(file, source)
  }
  return root
}
const post = (title, extra = '') =>
  `---\ntitle: '${title}'\ndate: 2026-09-19\ntopic: world-models\ntype: 论文精读\n${extra}\n---\n\n# ${title}\n\n这是一段足够完整的研究说明，用来验证摘要保留正文语义，并且不会泄露未发布文章中的内容。\n`

test('public index and page exclusions agree, even when private files exist locally', (t) => {
  const root = fixture(t, {
    'blog/2026/public.md': post('Public'),
    'blog/2026/draft.md': post('SECRET-DRAFT', 'draft: true'),
    'blog/2026/private.md': post('SECRET-PRIVATE', 'visibility: private'),
    'blog/2026/2026-07-23-outdoor-long-range-navigation-solution.md':
      'invalid frontmatter SECRET-LOCAL',
  })
  const posts = getPosts(root)
  assert.deepEqual(
    posts.map((p) => p.title),
    ['Public'],
  )
  assert.deepEqual(
    getPosts(root),
    posts,
    'repeated consumers receive the same index',
  )
  assert.equal(JSON.stringify(posts).includes('SECRET'), false)
  assert.ok(excludedPages(root).includes('blog/2026/draft.md'))
  assert.ok(excludedPages(root).includes('blog/2026/private.md'))
  assert.ok(
    excludedPages(root).includes(
      'blog/2026/2026-07-23-outdoor-long-range-navigation-solution.md',
    ),
  )
})
test('publication metadata rejects invalid dates, topics and ambiguous draft flags', (t) => {
  const root = fixture(t, { 'blog/invalid.md': post('Bad', "draft: 'true'") })
  assert.throws(() => getPosts(root), /draft must be a boolean/)
  fs.writeFileSync(
    path.join(root, 'blog/invalid.md'),
    post('Bad').replace('world-models', 'typo'),
  )
  assert.throws(() => getPosts(root), /unknown topic/)
  fs.writeFileSync(
    path.join(root, 'blog/invalid.md'),
    post('Bad', 'paperDate: 2026-02-30'),
  )
  assert.throws(() => getPosts(root), /invalid paperDate/)
  assert.equal(dateOnly('2026-02-30'), '')
  assert.equal(dateOnly('2024-02-29'), '2024-02-29')
})
test('summaries contain prose rather than markdown, images, source metadata or fenced code', () => {
  const text =
    '这段**技术分析**通过[原论文](https://example.com)解释模型的学习过程，并区分实验结论和推断，帮助读者理解方法的适用条件。'
  const summary = summarize(
    '# 标题\n\n![图片](x.png)\n\n```python\nSECRET_CODE\n```\n\n论文：https://example.com\n\n' +
      text,
  )
  assert.match(summary, /技术分析通过原论文解释/)
  assert.doesNotMatch(summary, /SECRET_CODE|\*|https:|!\[/)
  assert.equal(plainText('*强调* 与 `代码`'), '强调 与 代码')
  assert.equal(summarize('ignored', '**简明摘要**'), '简明摘要')
})
test('related reading stays in the same topic, excludes itself and prioritizes shared tags', () => {
  const a = {
    file: 'a',
    topic: 'world-models',
    tags: ['PPO'],
    date: '2026-01-01',
  }
  const b = { ...a, file: 'b', title: 'Shared tag', url: '/b', summary: 'b' }
  const c = { ...a, file: 'c', tags: [], date: '2026-09-01' }
  const d = { ...b, file: 'd', topic: 'driving' }
  assert.equal(relatedPosts(a, [a, c, b, d])[0].title, 'Shared tag')
  assert.equal(relatedPosts(a, [a, c, b, d]).length, 2)
})

test('index preserves quoted YAML metadata, deterministic ordering and separate date meanings', (t) => {
  const root = fixture(t, {
    'blog/2026/older.md': post(
      'Older: a title',
      'publishedAt: 2026-09-20\nupdatedAt: 2026-09-21\npaperDate: 2025-01-16\ntags:\n  - PPO\n  - Flow Matching',
    ).replace('date: 2026-09-19', 'date: 2026-09-18'),
    'blog/2026/newer.md': post('Newer'),
  })
  const result = getPosts(root)
  assert.deepEqual(
    result.map((p) => p.title),
    ['Newer', 'Older: a title'],
  )
  assert.equal(result[1].date, '2026-09-18')
  assert.equal(result[1].paperDate, '2025-01-16')
  assert.equal(result[1].publishedAt, '2026-09-20')
  assert.equal(result[1].updatedAt, '2026-09-21')
  assert.deepEqual(result[1].tags, ['PPO', 'Flow Matching'])
})
