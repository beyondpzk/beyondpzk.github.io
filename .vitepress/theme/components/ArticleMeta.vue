<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'
const { frontmatter } = useData()
const post = computed(() => frontmatter.value.article)
</script>
<template>
  <div v-if="post" class="article-meta">
    <div class="article-breadcrumb">
      <a href="/blog/">全部文章</a><span aria-hidden="true">/</span
      ><a :href="`/topics/${post.topic}.html`">{{ post.topicName }}</a
      ><span class="type-badge">{{ post.type }}</span>
    </div>
    <div class="article-dates">
      <span
        >笔记日期 <time :datetime="post.date">{{ post.date }}</time></span
      ><span v-if="post.paperDate"
        >论文日期
        <time :datetime="post.paperDate">{{ post.paperDate }}</time></span
      ><span v-if="post.publishedAt"
        >首次收录
        <time :datetime="post.publishedAt">{{ post.publishedAt }}</time></span
      ><span>约 {{ post.readingMinutes }} 分钟</span>
    </div>
    <div v-if="post.tags.length" class="article-tags">
      <a
        v-for="tag in post.tags"
        :key="tag"
        :href="`/blog/?tag=${encodeURIComponent(tag)}#articles`"
        >{{ tag }}</a
      >
    </div>
  </div>
</template>
