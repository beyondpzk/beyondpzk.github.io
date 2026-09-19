#!/usr/bin/env node
// Compatibility command: navigation now reads the same article index at build time.
import { getPosts } from '../lib/content.js'
import { topics } from '../lib/taxonomy.js'
import { resolveCollections } from '../content/collections.js'
const posts = getPosts()
resolveCollections(posts)
console.log(
  `文章索引有效：${posts.length} 篇公开文章，${topics.length} 个主题。导航无需手动同步。`,
)
