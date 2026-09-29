<script setup>
import { topics } from '../../../lib/taxonomy.js'
defineProps({ posts: { type: Array, required: true }, compact: Boolean })
</script>
<template>
  <div :class="compact ? 'topic-chips' : 'topic-grid'">
    <a
      v-for="topic in topics"
      :key="topic.id"
      :href="`/topics/${topic.id}.html`"
      class="topic-card"
    >
      <span class="topic-title"
        >{{ topic.name
        }}<span class="topic-count">{{
          posts.filter((p) => p.topic === topic.id).length
        }}</span></span
      >
      <p v-if="!compact">{{ topic.description }}</p>
    </a>
  </div>
</template>
