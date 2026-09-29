<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'
const { frontmatter } = useData()
const related = computed(() => frontmatter.value.relatedPosts || [])
</script>
<template>
  <aside
    v-if="related.length"
    class="related-posts"
    aria-labelledby="related-title"
  >
    <h2 id="related-title">继续阅读</h2>
    <div class="related-grid">
      <article v-for="post in related" :key="post.url">
        <h3>
          <a :href="post.url">{{ post.title }}</a>
        </h3>
        <p>{{ post.summary }}</p>
      </article>
    </div>
  </aside>
</template>
