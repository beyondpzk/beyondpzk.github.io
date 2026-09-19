<script setup>
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vitepress'
const route = useRoute(),
  dialog = ref(null),
  source = ref(''),
  caption = ref(''),
  expanded = ref(false)
let origin = null,
  observer
function prepareImages() {
  document.querySelectorAll('.vp-doc img:not(a img)').forEach((img) => {
    img.tabIndex = 0
    img.setAttribute('role', 'button')
    img.setAttribute('aria-label', `放大图片：${img.alt || '文章插图'}`)
    img.classList.add('zoomable-image')
  })
}
async function open(event) {
  const image = event.target
  if (
    !(image instanceof HTMLImageElement) ||
    !image.matches('.vp-doc .zoomable-image')
  )
    return
  if (event.type === 'keydown' && !['Enter', ' '].includes(event.key)) return
  event.preventDefault()
  origin = image
  source.value = image.currentSrc || image.src
  caption.value = image.alt
  expanded.value = false
  await nextTick()
  dialog.value.showModal()
}
function close() {
  dialog.value?.close()
  origin?.focus({ preventScroll: true })
}
watch(
  () => route.path,
  async () => {
    close()
    await nextTick()
    prepareImages()
  },
)
onMounted(() => {
  prepareImages()
  document.addEventListener('click', open)
  document.addEventListener('keydown', open)
  observer = new MutationObserver(prepareImages)
  observer.observe(document.querySelector('#app'), {
    childList: true,
    subtree: true,
  })
})
onUnmounted(() => {
  observer?.disconnect()
  document.removeEventListener('click', open)
  document.removeEventListener('keydown', open)
})
</script>
<template>
  <dialog
    ref="dialog"
    class="image-dialog"
    aria-label="文章图片预览"
    @cancel.prevent="close"
    @click="
      (event) => {
        if (event.target === dialog) close()
      }
    "
  >
    <div class="image-toolbar">
      <span>{{ caption || '文章插图' }}</span
      ><button type="button" @click="expanded = !expanded">
        {{ expanded ? '适合窗口' : '原始尺寸' }}</button
      ><button type="button" aria-label="关闭图片预览" autofocus @click="close">
        关闭 ×
      </button>
    </div>
    <div class="image-canvas" :class="{ expanded }">
      <img v-if="source" :src="source" :alt="caption" />
    </div>
  </dialog>
</template>
