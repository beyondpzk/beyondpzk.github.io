import { getPosts } from './lib/content.js'
import { resolveCollections } from './content/collections.js'

export default {
  watch: [
    'blog/**/*.md',
    'lib/*.js',
    'content/*.js',
    'scripts/local-only-posts.js',
  ],
  load() {
    const posts = getPosts()
    return { posts, collections: resolveCollections(posts) }
  },
}
