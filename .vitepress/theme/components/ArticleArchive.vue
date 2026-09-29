<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { topics, articleTypes } from '../../../lib/taxonomy.js'
import PostCard from './PostCard.vue'
const props = defineProps({
  posts: { type: Array, required: true },
  topic: { type: String, default: '' },
})
const query = ref(''),
  selectedTopic = ref(props.topic),
  type = ref(''),
  year = ref(''),
  tag = ref(''),
  page = ref(1)
const resultsEl = ref(null),
  perPage = 12
const years = computed(() =>
  [...new Set(props.posts.map((p) => p.year))].sort().reverse(),
)
const tags = computed(() =>
  [
    ...new Set(
      props.posts
        .filter((p) => !selectedTopic.value || p.topic === selectedTopic.value)
        .flatMap((p) => p.tags),
    ),
  ].sort(),
)
const filtered = computed(() => {
  const terms = query.value
    .trim()
    .toLocaleLowerCase()
    .split(/\s+/)
    .filter(Boolean)
  return props.posts.filter(
    (p) =>
      (!selectedTopic.value || p.topic === selectedTopic.value) &&
      (!type.value || p.type === type.value) &&
      (!year.value || p.year === year.value) &&
      (!tag.value || p.tags.includes(tag.value)) &&
      terms.every((term) =>
        `${p.title} ${p.summary} ${p.tags.join(' ')}`
          .toLocaleLowerCase()
          .includes(term),
      ),
  )
})
const pages = computed(() =>
  Math.max(1, Math.ceil(filtered.value.length / perPage)),
)
const current = computed(() =>
  filtered.value.slice((page.value - 1) * perPage, page.value * perPage),
)
function readUrl() {
  const p = new URLSearchParams(window.location.search)
  query.value = p.get('q') || ''
  selectedTopic.value =
    props.topic ||
    (topics.some((t) => t.id === p.get('topic')) ? p.get('topic') : '')
  type.value = articleTypes.includes(p.get('type')) ? p.get('type') : ''
  year.value = years.value.includes(p.get('year')) ? p.get('year') : ''
  tag.value = tags.value.includes(p.get('tag')) ? p.get('tag') : ''
  page.value = Math.min(
    pages.value,
    Math.max(1, Number.parseInt(p.get('page'), 10) || 1),
  )
}
function urlFor(targetPage = page.value) {
  const p = new URLSearchParams()
  if (query.value.trim()) p.set('q', query.value.trim())
  if (!props.topic && selectedTopic.value) p.set('topic', selectedTopic.value)
  if (type.value) p.set('type', type.value)
  if (year.value) p.set('year', year.value)
  if (tag.value) p.set('tag', tag.value)
  if (targetPage > 1) p.set('page', String(targetPage))
  return (p.size ? '?' + p.toString() : '?') + '#articles'
}
function updateFilters() {
  if (!tags.value.includes(tag.value)) tag.value = ''
  page.value = 1
  window.history.replaceState({}, '', urlFor())
}
async function goToPage(event, value) {
  if (
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  )
    return
  event.preventDefault()
  page.value = Math.min(pages.value, Math.max(1, value))
  window.history.pushState({}, '', urlFor())
  await nextTick()
  resultsEl.value?.focus({ preventScroll: true })
  resultsEl.value?.scrollIntoView({ block: 'start', behavior: 'auto' })
}
function reset() {
  query.value = ''
  selectedTopic.value = props.topic
  type.value = ''
  year.value = ''
  tag.value = ''
  updateFilters()
}
watch(() => props.topic, readUrl)
onMounted(() => {
  readUrl()
  window.addEventListener('popstate', readUrl)
})
onUnmounted(() => window.removeEventListener('popstate', readUrl))
</script>

<template>
  <section class="article-archive" aria-label="文章筛选与列表">
    <form class="archive-filters" role="search" @submit.prevent="updateFilters">
      <label class="search-field"
        >搜索文章<input
          v-model="query"
          type="search"
          placeholder="标题、摘要或技术关键词"
          @input="updateFilters"
      /></label>
      <div class="filter-row">
        <label v-if="!topic"
          >研究主题<select
            aria-label="研究主题"
            v-model="selectedTopic"
            @change="updateFilters"
          >
            <option value="">全部主题</option>
            <option v-for="item in topics" :key="item.id" :value="item.id">
              {{ item.name }}
            </option>
          </select></label
        >
        <label
          >文章类型<select
            aria-label="文章类型"
            v-model="type"
            @change="updateFilters"
          >
            <option value="">全部类型</option>
            <option v-for="item in articleTypes" :key="item">{{ item }}</option>
          </select></label
        >
        <label
          >笔记年份<select
            aria-label="笔记年份"
            v-model="year"
            @change="updateFilters"
          >
            <option value="">全部年份</option>
            <option v-for="item in years" :key="item">{{ item }}</option>
          </select></label
        >
        <label
          >技术标签<select
            aria-label="技术标签"
            v-model="tag"
            @change="updateFilters"
          >
            <option value="">全部标签</option>
            <option v-for="item in tags" :key="item">{{ item }}</option>
          </select></label
        >
      </div>
    </form>
    <div id="articles" ref="resultsEl" class="results-heading" tabindex="-1">
      <p role="status" aria-live="polite">
        找到 {{ filtered.length }} 篇文章<span v-if="filtered.length">
          · 第 {{ page }} / {{ pages }} 页</span
        >
      </p>
      <button type="button" class="text-button" @click="reset">重置筛选</button>
    </div>
    <div v-if="current.length" class="archive-grid">
      <PostCard v-for="post in current" :key="post.url" :post="post" />
    </div>
    <div v-else class="empty-state">
      <h2>没有找到匹配的文章</h2>
      <p>试试更短的关键词，或减少筛选条件。</p>
      <button class="action-button" @click="reset">查看全部文章</button>
    </div>
    <nav v-if="pages > 1" class="pagination" aria-label="文章分页">
      <a
        v-if="page > 1"
        :href="urlFor(page - 1)"
        target="_self"
        @click="goToPage($event, page - 1)"
        >上一页</a
      ><span v-else aria-disabled="true">上一页</span>
      <span class="page-label">{{ page }} / {{ pages }}</span>
      <a
        v-if="page < pages"
        :href="urlFor(page + 1)"
        target="_self"
        @click="goToPage($event, page + 1)"
        >下一页</a
      ><span v-else aria-disabled="true">下一页</span>
    </nav>
  </section>
</template>
