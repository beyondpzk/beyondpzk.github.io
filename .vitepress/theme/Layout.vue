<script setup>
import DefaultTheme from 'vitepress/theme'
import ArticleMeta from './components/ArticleMeta.vue'
import RelatedPosts from './components/RelatedPosts.vue'
import ImageViewer from './components/ImageViewer.vue'
</script>
<template>
  <DefaultTheme.Layout
    ><template #doc-before><ArticleMeta /></template
    ><template #doc-footer-before><RelatedPosts /></template
    ><template #layout-bottom><ImageViewer /></template
  ></DefaultTheme.Layout>
</template>
