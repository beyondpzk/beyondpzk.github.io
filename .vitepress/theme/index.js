import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import HomePage from './components/HomePage.vue'
import ArticleArchive from './components/ArticleArchive.vue'
import TopicGrid from './components/TopicGrid.vue'
import CollectionGrid from './components/CollectionGrid.vue'
import ReadingPath from './components/ReadingPath.vue'
import './style.css'
export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    for (const [name, component] of Object.entries({
      HomePage,
      ArticleArchive,
      TopicGrid,
      CollectionGrid,
      ReadingPath,
    }))
      app.component(name, component)
  },
}
