<template>
  <div class="post-list">
    <div v-for="post in paginatedPosts" :key="post.url" class="post-card">
      <h3><a :href="post.url">{{ post.title }}</a></h3>
      <div class="meta">{{ formatDate(post.date) }} · {{ post.categories.join(' / ') }}</div>
      <div class="excerpt">{{ post.excerpt }}</div>
    </div>

    <div class="pagination" v-if="totalPages > 1">
      <button
        class="page-btn"
        :disabled="currentPage === 1"
        @click="currentPage--"
      >
        上一页
      </button>
      <span class="page-info">{{ currentPage }} / {{ totalPages }}</span>
      <button
        class="page-btn"
        :disabled="currentPage === totalPages"
        @click="currentPage++"
      >
        下一页
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  posts: {
    type: Array,
    required: true
  },
  perPage: {
    type: Number,
    default: 20
  }
})

const currentPage = ref(1)

const totalPages = computed(() => Math.ceil(props.posts.length / props.perPage))

const paginatedPosts = computed(() => {
  const start = (currentPage.value - 1) * props.perPage
  return props.posts.slice(start, start + props.perPage)
})

function formatDate(date) {
  if (!date) return ''
  const d = new Date(date)
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}
</script>

<style scoped>
.post-card {
  margin-bottom: 24px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.post-card:last-child {
  border-bottom: none;
}

.post-card h3 {
  margin: 0 0 8px;
  font-size: 1.2rem;
  line-height: 1.4;
}

.post-card h3 a {
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.post-card h3 a:hover {
  text-decoration: underline;
}

.meta {
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
  margin-bottom: 8px;
}

.excerpt {
  color: var(--vp-c-text-1);
  font-size: 0.95rem;
  line-height: 1.6;
}

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 32px;
}

.page-btn {
  padding: 8px 16px;
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 6px;
  background: transparent;
  color: var(--vp-c-brand-1);
  cursor: pointer;
  font-size: 0.9rem;
}

.page-btn:hover:not(:disabled) {
  background: var(--vp-c-brand-soft);
}

.page-btn:disabled {
  border-color: var(--vp-c-gray-2);
  color: var(--vp-c-text-3);
  cursor: not-allowed;
}

.page-info {
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}
</style>
