<script setup>
import { computed } from 'vue'
import PostCard from './PostCard.vue'
import TopicGrid from './TopicGrid.vue'
import CollectionGrid from './CollectionGrid.vue'
const props = defineProps({ data: { type: Object, required: true } })
const featured = computed(() =>
  props.data.posts.filter((p) => p.featured).slice(0, 3),
)
</script>
<template>
  <main class="reading-home">
    <header class="home-intro">
      <p class="eyebrow">BEYOND · AI RESEARCH NOTES</p>
      <h1>理解模型，走向真实世界。</h1>
      <p class="intro-copy">
        关于世界模型、机器人导航与具身智能的论文精读、技术判断和工程实践。
      </p>
      <div class="home-actions">
        <a class="action-button" href="/blog/"
          >浏览 {{ data.posts.length }} 篇笔记
          <span aria-hidden="true">→</span></a
        ><a class="quiet-link" href="/about.html">关于我</a>
      </div>
    </header>
    <section class="home-section" aria-labelledby="featured-title">
      <div class="section-heading">
        <div>
          <p class="eyebrow">START HERE</p>
          <h2 id="featured-title">精选阅读</h2>
        </div>
        <a href="/blog/">全部文章 →</a>
      </div>
      <div class="featured-grid">
        <PostCard v-for="post in featured" :key="post.url" :post="post" />
      </div>
    </section>
    <section class="home-section topic-section" aria-labelledby="topics-title">
      <div class="section-heading">
        <h2 id="topics-title">按研究主题探索</h2>
        <a href="/topics/">主题目录 →</a>
      </div>
      <TopicGrid :posts="data.posts" compact />
    </section>
    <section class="home-section" aria-labelledby="series-title">
      <div class="section-heading">
        <div>
          <p class="eyebrow">READ IN CONTEXT</p>
          <h2 id="series-title">沿着问题读下去</h2>
        </div>
        <a href="/series/">所有专题 →</a>
      </div>
      <CollectionGrid :collections="data.collections" />
    </section>
    <section class="home-section" aria-labelledby="latest-title">
      <div class="section-heading">
        <h2 id="latest-title">最新笔记</h2>
        <a href="/blog/">更多笔记 →</a>
      </div>
      <div class="latest-grid">
        <PostCard
          v-for="post in data.posts.slice(0, 6)"
          :key="post.url"
          :post="post"
          compact
        />
      </div>
    </section>
    <p class="home-signoff">Tenacious life, proud journey.</p>
  </main>
</template>
