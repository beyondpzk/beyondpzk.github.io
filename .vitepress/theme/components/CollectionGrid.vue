<script setup>
defineProps({ collections: { type: Array, required: true } })
</script>
<template>
  <div class="collection-grid">
    <a
      v-for="(collection, i) in collections"
      :key="collection.id"
      :href="`/series/${collection.id}.html`"
      class="collection-card"
      ><span class="collection-number"
        >0{{ i + 1 }} / {{ collection.steps.length }} 篇</span
      >
      <h3>{{ collection.title }}</h3>
      <p>{{ collection.description }}</p>
      <span class="collection-link">开始这条阅读路线 →</span></a
    >
  </div>
</template>
