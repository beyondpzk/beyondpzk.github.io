<script setup>
defineProps({ post: { type: Object, required: true }, compact: Boolean })
</script>

<template>
  <article class="post-card" :class="{ compact }">
    <div class="post-kicker">
      <span>{{ post.topicName }}</span
      ><span>{{ post.type }}</span>
    </div>
    <h3>
      <a :href="post.url">{{ post.title }}</a>
    </h3>
    <p v-if="!compact" class="post-summary">{{ post.summary }}</p>
    <div class="post-meta">
      <time :datetime="post.date">{{ post.date }}</time
      ><span>约 {{ post.readingMinutes }} 分钟</span>
    </div>
  </article>
</template>
