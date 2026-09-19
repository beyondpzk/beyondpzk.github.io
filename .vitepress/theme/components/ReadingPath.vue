<script setup>
defineProps({ collection: { type: Object, required: true } })
</script>
<template>
  <ol class="reading-path">
    <li v-for="(step, i) in collection.steps" :key="step.url">
      <span class="step-number">{{ i + 1 }}</span>
      <div>
        <h2>
          <a :href="step.url">{{ step.title }}</a>
        </h2>
        <p>{{ step.note }}</p>
        <span class="post-meta">约 {{ step.readingMinutes }} 分钟</span>
      </div>
    </li>
  </ol>
  <p class="reading-note">
    这些路线按概念依赖组织。笔记保留各自的写作背景，实验结论请结合原论文和版本阅读。
  </p>
</template>
