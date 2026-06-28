import DefaultTheme from 'vitepress/theme'
import './style.css'
import PostList from './components/PostList.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router, siteData }) {
    app.component('PostList', PostList)
  },
}
