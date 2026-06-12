import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "P's Notes",
  description: 'Tenacious life, proud journey. AI research notes and beyond.',
  lang: 'zh-CN',

  // GitHub Pages 部署配置
  // 因为是 username.github.io 仓库，base 用 '/'
  base: '/',

  // 忽略 README 中的本地开发链接检查
  ignoreDeadLinks: [
    /^http:\/\/localhost/,
  ],

  // 简洁美观的默认主题
  themeConfig: {
    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      { text: '博客', link: '/blog/' },
      { text: '关于我', link: '/about' },
      { text: '简历', link: '/resume' },
    ],

    // 头像（放在右上角）
    socialLinks: [
      { icon: 'github', link: 'https://github.com/beyondpzk' },
    ],

    // 书式侧边栏：按主题组织
    sidebar: {
      '/blog/': [
        {
          text: '📚 博客目录',
          items: [
            {
              text: '全部文章',
              link: '/blog/',
            },
          ],
        },
        {
          text: '📅 2026 年',
          collapsed: true,
          items: [
            {
              text: 'C_RADIOv4',
              link: '/blog/2026/C_RADIOv4',
            },
            {
              text: 'DreamZero',
              link: '/blog/2026/DreamZero',
            },
            {
              text: 'DriveJEPA',
              link: '/blog/2026/DriveJEPA',
            },
            {
              text: 'DriveWorldVLA',
              link: '/blog/2026/DriveWorldVLA',
            },
            {
              text: 'FastWAM',
              link: '/blog/2026/FastWAM',
            },
            {
              text: 'FineGrainedAlignmentedVLN',
              link: '/blog/2026/FineGrainedAlignmentedVLN',
            },
            {
              text: 'GeRo',
              link: '/blog/2026/GeRo',
            },
            {
              text: 'GigaWorldPolicy',
              link: '/blog/2026/GigaWorldPolicy',
            },
            {
              text: 'LAPose',
              link: '/blog/2026/LAPose',
            },
            {
              text: 'LearningLatentActionWM',
              link: '/blog/2026/LearningLatentActionWM',
            },
            {
              text: 'LingBotVA',
              link: '/blog/2026/LingBotVA',
            },
            {
              text: 'LowerMemory',
              link: '/blog/2026/LowerMemory',
            },
            {
              text: 'ManyForcing',
              link: '/blog/2026/ManyForcing',
            },
            {
              text: 'ModelSpeed',
              link: '/blog/2026/ModelSpeed',
            },
            {
              text: 'Muon',
              link: '/blog/2026/Muon',
            },
            {
              text: 'RAENWM',
              link: '/blog/2026/RAENWM',
            },
            {
              text: 'RISE',
              link: '/blog/2026/RISE',
            },
            {
              text: 'RLNeeds',
              link: '/blog/2026/RLNeeds',
            },
            {
              text: 'SimVLA',
              link: '/blog/2026/SimVLA',
            },
            {
              text: 'TrainingSpeed',
              link: '/blog/2026/TrainingSpeed',
            },
            {
              text: 'VJEPA2.1',
              link: '/blog/2026/VJEPA2.1',
            },
            {
              text: 'VLM4VLA',
              link: '/blog/2026/VLM4VLA',
            },
            {
              text: 'pi0.7',
              link: '/blog/2026/pi0.7',
            },
          ],
        },
        {
          text: '📅 2025 年',
          collapsed: true,
          items: [
            {
              text: 'AutoVLA',
              link: '/blog/2025/AutoVLA',
            },
            {
              text: 'BAGEL',
              link: '/blog/2025/BAGEL',
            },
            {
              text: 'ChatVLA',
              link: '/blog/2025/ChatVLA',
            },
            {
              text: 'DeepSpeed',
              link: '/blog/2025/DeepSpeed',
            },
            {
              text: 'DinoWorld',
              link: '/blog/2025/DinoWorld',
            },
            {
              text: 'FSDP',
              link: '/blog/2025/FSDP',
            },
            {
              text: 'GoalVLA',
              link: '/blog/2025/GoalVLA',
            },
            {
              text: 'ImprovedMeanFlows',
              link: '/blog/2025/ImprovedMeanFlows',
            },
            {
              text: 'JEPA_WM',
              link: '/blog/2025/JEPA_WM',
            },
            {
              text: 'KVCACHE',
              link: '/blog/2025/KVCACHE',
            },
            {
              text: 'LongScape',
              link: '/blog/2025/LongScape',
            },
            {
              text: 'MatrixGame2.0',
              link: '/blog/2025/MatrixGame2.0',
            },
            {
              text: 'MeanFlows',
              link: '/blog/2025/MeanFlows',
            },
            {
              text: 'Motus',
              link: '/blog/2025/Motus',
            },
            {
              text: 'Pi05',
              link: '/blog/2025/Pi05',
            },
            {
              text: 'RAD',
              link: '/blog/2025/RAD',
            },
            {
              text: 'ROPE',
              link: '/blog/2025/ROPE',
            },
            {
              text: 'RoboScape',
              link: '/blog/2025/RoboScape',
            },
            {
              text: 'RoboScapeR',
              link: '/blog/2025/RoboScapeR',
            },
            {
              text: 'SurveyOnWorldModelsForEmbodiedAI',
              link: '/blog/2025/SurveyOnWorldModelsForEmbodiedAI',
            },
            {
              text: 'TrackVLA',
              link: '/blog/2025/TrackVLA',
            },
            {
              text: 'VJEPA2',
              link: '/blog/2025/VJEPA2',
            },
            {
              text: 'VLA0S',
              link: '/blog/2025/VLA0S',
            },
            {
              text: 'VLAAN',
              link: '/blog/2025/VLAAN',
            },
            {
              text: 'VLACompare',
              link: '/blog/2025/VLACompare',
            },
            {
              text: 'VLA_embodedAI',
              link: '/blog/2025/VLA_embodedAI',
            },
            {
              text: 'VLJEPA',
              link: '/blog/2025/VLJEPA',
            },
            {
              text: 'WMcompare',
              link: '/blog/2025/WMcompare',
            },
            {
              text: 'pi0.6',
              link: '/blog/2025/pi0.6',
            },
          ],
        },
        {
          text: '📅 2024 年',
          collapsed: true,
          items: [
            {
              text: 'DROID',
              link: '/blog/2024/DROID',
            },
            {
              text: 'DinoWM',
              link: '/blog/2024/DinoWM',
            },
            {
              text: 'E2E',
              link: '/blog/2024/E2E',
            },
            {
              text: 'FLIP',
              link: '/blog/2024/FLIP',
            },
            {
              text: 'MAR',
              link: '/blog/2024/MAR',
            },
            {
              text: 'NWM',
              link: '/blog/2024/NWM',
            },
            {
              text: 'OpenVLA',
              link: '/blog/2024/OpenVLA',
            },
            {
              text: 'Pi0',
              link: '/blog/2024/Pi0',
            },
            {
              text: 'PredictiveWM_VS_GenerativeWM',
              link: '/blog/2024/PredictiveWM_VS_GenerativeWM',
            },
            {
              text: 'RDT1B',
              link: '/blog/2024/RDT1B',
            },
            {
              text: 'SelfForcing',
              link: '/blog/2024/SelfForcing',
            },
            {
              text: 'SparseAD',
              link: '/blog/2024/SparseAD',
            },
            {
              text: 'SparseDrive',
              link: '/blog/2024/SparseDrive',
            },
            {
              text: 'Survey_Occupancy',
              link: '/blog/2024/Survey_Occupancy',
            },
            {
              text: 'TinyVLM',
              link: '/blog/2024/TinyVLM',
            },
            {
              text: 'VAR',
              link: '/blog/2024/VAR',
            },
            {
              text: 'VJEPA',
              link: '/blog/2024/VJEPA',
            },
            {
              text: 'pytorch_bug',
              link: '/blog/2024/pytorch_bug',
            },
            {
              text: 'pytorch_weights_datasets',
              link: '/blog/2024/pytorch_weights_datasets',
            },
          ],
        },
        {
          text: '📅 2023 年',
          collapsed: true,
          items: [
            {
              text: 'Sparse4DV2',
              link: '/blog/2023/2023-5-23-Sparse4DV2',
            },
            {
              text: 'SparseBEV',
              link: '/blog/2023/2023-8-18-SparseBEV',
            },
            {
              text: '3DpointCloudGenerative',
              link: '/blog/2023/3DpointCloudGenerative',
            },
            {
              text: 'ACT',
              link: '/blog/2023/ACT',
            },
            {
              text: 'DE_VS_SDE',
              link: '/blog/2023/DE_VS_SDE',
            },
            {
              text: 'DeramerV3',
              link: '/blog/2023/DeramerV3',
            },
            {
              text: 'DiffusionPolicy',
              link: '/blog/2023/DiffusionPolicy',
            },
            {
              text: 'Emu1',
              link: '/blog/2023/Emu1',
            },
            {
              text: 'Emu2',
              link: '/blog/2023/Emu2',
            },
            {
              text: 'IJEPA',
              link: '/blog/2023/Ijepa',
            },
            {
              text: 'MAGVITV2',
              link: '/blog/2023/MAGVITV2',
            },
            {
              text: 'RT2',
              link: '/blog/2023/RT2',
            },
            {
              text: 'RenderOcc',
              link: '/blog/2023/RenderOcc',
            },
            {
              text: 'Sparse4DV3',
              link: '/blog/2023/Sparse4DV3',
            },
            {
              text: 'SparseOcc',
              link: '/blog/2023/SparseOcc',
            },
            {
              text: 'StreamPETR',
              link: '/blog/2023/StreamPETR',
            },
            {
              text: 'Vidar',
              link: '/blog/2023/Vidar',
            },
            {
              text: 'WM',
              link: '/blog/2023/WM',
            },
            {
              text: 'offlineonlineworldmodel',
              link: '/blog/2023/offlineonlineworldmodel',
            },
            {
              text: 'verl',
              link: '/blog/2023/verl',
            },
          ],
        },
        {
          text: '📅 2022 年',
          collapsed: true,
          items: [
            {
              text: 'DayDreamer',
              link: '/blog/2022/DayDreamer',
            },
            {
              text: 'FlowAndDataGeneration',
              link: '/blog/2022/FlowAndDataGeneration',
            },
            {
              text: 'FlowMatching',
              link: '/blog/2022/FlowMatching',
            },
            {
              text: 'Gato',
              link: '/blog/2022/Gato',
            },
            {
              text: 'IRIS',
              link: '/blog/2022/IRIS',
            },
            {
              text: 'MILE',
              link: '/blog/2022/MILE',
            },
            {
              text: 'PETR',
              link: '/blog/2022/PETR',
            },
            {
              text: 'PETRV2',
              link: '/blog/2022/PETRV2',
            },
            {
              text: 'RT1',
              link: '/blog/2022/RT1',
            },
            {
              text: 'RectifiedFlow',
              link: '/blog/2022/RectifiedFlow',
            },
            {
              text: 'Sparse4D',
              link: '/blog/2022/Sparse4D',
            },
            {
              text: 'TransDreamer',
              link: '/blog/2022/TransDreamer',
            },
            {
              text: 'VPT',
              link: '/blog/2022/VPT',
            },
            {
              text: 'Wayformer',
              link: '/blog/2022/Wayformer',
            },
            {
              text: 'mu_law',
              link: '/blog/2022/mu_law',
            },
          ],
        },
        {
          text: '📅 2021 年',
          collapsed: true,
          items: [
            {
              text: 'CaDDN',
              link: '/blog/2021/CaDDN',
            },
            {
              text: 'DETR3D',
              link: '/blog/2021/DETR3D',
            },
            {
              text: 'IntentNet',
              link: '/blog/2021/IntentNet',
            },
            {
              text: 'Mask2Former',
              link: '/blog/2021/Mask2Former',
            },
            {
              text: 'MaskFormer',
              link: '/blog/2021/MaskFormer',
            },
          ],
        },
        {
          text: '📅 2020 年',
          collapsed: true,
          items: [
            {
              text: 'CVAE_VS_VAE',
              link: '/blog/2020/CVAE_VS_VAE',
            },
            {
              text: 'DETR',
              link: '/blog/2020/DETR',
            },
            {
              text: 'DreamerV2',
              link: '/blog/2020/DreamerV2',
            },
            {
              text: 'SparseRCNN',
              link: '/blog/2020/SparseRCNN',
            },
            {
              text: 'TNT',
              link: '/blog/2020/TNT',
            },
            {
              text: 'VectorNet',
              link: '/blog/2020/VectorNet',
            },
          ],
        },
        {
          text: '📅 2019 年',
          collapsed: true,
          items: [
            {
              text: 'DreamerV1',
              link: '/blog/2019/DreamerV1',
            },
            {
              text: 'MultiPath',
              link: '/blog/2019/MultiPath',
            },
            {
              text: 'l1l2',
              link: '/blog/2019/l1l2',
            },
          ],
        },
        {
          text: '📅 2018 年',
          collapsed: true,
          items: [
            {
              text: 'PlaNet',
              link: '/blog/2018/PlaNet',
            },
            {
              text: 'WorldModels',
              link: '/blog/2018/WorldModels',
            },
          ],
        },
        {
          text: '📝 工具与约定',
          collapsed: false,
          items: [
            {
              text: 'Neovim + iTerm2 工作流',
              link: '/blog/notes/neovim-workflow',
            },
            {
              text: '写作约定',
              link: '/blog/notes/writing-guide',
            },
          ],
        },
      ],
    },

    // 搜索
    search: {
      provider: 'local',
    },

    // 大纲
    outline: {
      label: '本页目录',
      level: [2, 3],
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/beyondpzk/beyondpzk.github.io/edit/main/:path',
      text: '在 GitHub 上编辑此页',
    },

    // 页脚
    footer: {
      message: '用 ❤️ 和 VitePress 构建',
      copyright: 'Copyright © 2026',
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short',
      },
    },
  },

  // Markdown 配置
  markdown: {
    lineNumbers: true,
    math: true, // 启用数学公式支持，保护 $$...$$ 中的内容不被 Vue 解析
    image: {
      // 懒加载图片
      lazyLoading: true,
    },
  },
})
