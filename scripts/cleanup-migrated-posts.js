#!/usr/bin/env node
// 清理迁移后的文章，修复会导致 VitePress/Vue 解析错误的自定义标签
import fs from 'fs'
import path from 'path'

// 只处理从 Jekyll 迁移过来的文章（按年份分类的目录）
const MIGRATED_DIRS = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']
  .map(year => path.resolve('blog', year))

function cleanupFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf-8')

  // 1. 移除 alphaxiv-paper-citation 标签
  content = content.replace(/<alphaxiv-paper-citation[^>]*\/>/g, '')

  // 2. 移除 alphaxiv-chart 标签
  content = content.replace(/<alphaxiv-chart[^>]*>/g, '')
  content = content.replace(/<\/alphaxiv-chart>/g, '')

  // 3. 把 AI 模型输出的自定义标签转成文本标记，保留内容
  const tagReplacements = {
    think: { open: '**[思考]** ', close: ' **[思考结束]**' },
    answer: { open: '**[回答]** ', close: ' **[回答结束]**' },
    box: { open: '**[框]** ', close: ' **[框结束]**' },
    s: { open: '**[系统]** ', close: ' **[系统结束]**' },
    coor: { open: '**[坐标]** ', close: ' **[坐标结束]**' },
  }

  Object.entries(tagReplacements).forEach(([tag, repl]) => {
    const escaped = tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    content = content.replace(new RegExp(`<${escaped}>`, 'g'), repl.open)
    content = content.replace(new RegExp(`</${escaped}>`, 'g'), repl.close)
  })

  // 4. 把变量/按键占位符转成代码格式
  content = content.replace(/<v_([A-Za-z0-9]+)>/g, '`v_$1`')
  content = content.replace(/<v_N>/g, '`v_N`')
  content = content.replace(/<leader>/g, '`<leader>`')
  content = content.replace(/<cr>/g, '`<cr>`')
  content = content.replace(/<cmd>/g, '`<cmd>`')
  content = content.replace(/<action_i>/g, '`action_i`')
  content = content.replace(/<System_Prompt>/g, '`System_Prompt`')
  content = content.replace(/<latex_guidelines>/g, '`latex_guidelines`')
  content = content.replace(/<Video, Action>/g, '`Video, Action`')

  // 5. 清理因处理产生的多余空格和空行
  content = content.replace(/ +\n/g, '\n')
  content = content.replace(/\n{3,}/g, '\n\n')

  fs.writeFileSync(filePath, content)
}

function walkDir(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  entries.forEach(entry => {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      walkDir(fullPath)
    } else if (entry.name.endsWith('.md')) {
      cleanupFile(fullPath)
    }
  })
}

console.log('🧹 开始清理迁移后的文章...')
MIGRATED_DIRS.forEach(dir => {
  if (fs.existsSync(dir)) {
    walkDir(dir)
  }
})
console.log('✅ 清理完成')
