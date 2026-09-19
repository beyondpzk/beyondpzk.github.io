export const collections = [
  {
    id: 'world-models',
    title: '从 World Models 到 Dreamer',
    subtitle: '理解潜在空间中的预测与决策',
    description: '沿着表示、动力学和策略学习这条线，建立世界模型的知识骨架。',
    steps: [
      [
        '2018-03-27-worldmodels',
        '先建立整体视角：表示、记忆和控制分别负责什么。',
      ],
      ['2018-11-12-planet', '理解如何在学习到的潜在动力学中规划。'],
      ['2019-12-03-dreamerv1', '观察策略学习如何进入潜在空间。'],
      ['2020-10-05-dreamerv2', '继续比较表示设计与训练方法的变化。'],
      [
        '2025-10-19-surveyonworldmodelsforembodiedai',
        '回到综述，整理不同路线的适用问题。',
      ],
    ],
  },
  {
    id: 'visual-navigation',
    title: '视觉导航的表示与泛化',
    subtitle: '从通用导航策略到跨本体执行',
    description:
      '围绕观测、目标、动作接口和数据覆盖，阅读机器人导航的几条代表路线。',
    steps: [
      ['2022-10-07-GNM', '从通用导航模型入手，明确观测和目标接口。'],
      ['2023-06-26-ViNT', '理解导航预训练与任务适配的关系。'],
      ['2023-10-11-NoMaD', '关注多模态动作与探索行为如何表达。'],
      ['2026-07-30-ea-nav', '加入本体条件，思考不同机器人如何共享策略。'],
    ],
  },
  {
    id: 'vla-actions',
    title: 'VLA 如何表示与学习动作',
    subtitle: '沿着 π 系列理解动作建模',
    description:
      '将架构、动作表示与训练目标放在一起读，区分训练效率和执行能力。',
    steps: [
      ['2024-10-31-pi0', '先了解视觉语言模型与动作生成的连接方式。'],
      ['2026-09-18-pi0-fast', '理解动作压缩如何影响自回归训练。'],
      ['2025-04-22-pi05', '继续阅读泛化能力与训练数据的扩展。'],
      ['2025-11-18-pi06', '关注经验反馈如何进入策略改进过程。'],
    ],
  },
  {
    id: 'quantization',
    title: '从量化原理到部署验证',
    subtitle: '精度、吞吐和硬件约束一起看',
    description:
      '按误差来源、校准方法和落地约束组织阅读，避免只比较位宽与榜单数字。',
    steps: [
      ['2022-10-31-gptq', '从权重量化的误差补偿方法开始。'],
      ['2022-11-18-smoothquant', '进一步理解激活分布对量化的影响。'],
      ['2023-06-01-awq', '比较不同方法如何利用激活信息。'],
      [
        '2024-07-20-vit-nvidia-quantization-guide',
        '将方法带到视觉模型与具体部署环境中。',
      ],
    ],
  },
]
export function resolveCollections(posts) {
  return collections.map((collection) => ({
    ...collection,
    steps: collection.steps.map(([slug, note]) => {
      const post = posts.find((p) => p.file.endsWith('/' + slug + '.md'))
      if (!post)
        throw new Error(
          `Reading collection ${collection.id}: missing article ${slug}`,
        )
      return {
        title: post.title,
        url: post.url,
        note,
        readingMinutes: post.readingMinutes,
      }
    }),
  }))
}
