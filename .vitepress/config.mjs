import { defineConfig } from 'vitepress'

// 根据文件路径生成一个稳定的伪随机整数
function hashString(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 31 + str.charCodeAt(i)) >>> 0
  }
  return hash
}

// 将 frontmatter 中的 date 与文件路径结合，生成一个稳定的 lastUpdated 时间戳
// 日期取自文档 frontmatter.date，时分在 06:00 - 23:59 之间按文件路径伪随机确定
function computeLastUpdated(frontmatter, filePath) {
  const rawDate = frontmatter?.date
  if (!rawDate) return undefined

  let dateStr
  if (rawDate instanceof Date) {
    dateStr = rawDate.toISOString().slice(0, 10)
  } else if (typeof rawDate === 'string') {
    dateStr = rawDate.trim()
  } else {
    return undefined
  }

  const match = dateStr.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/)
  if (!match) return undefined

  const year = parseInt(match[1], 10)
  const month = parseInt(match[2], 10)
  const day = parseInt(match[3], 10)

  // 06:00 到 23:59 之间的分钟数（含）
  const startMinutes = 6 * 60
  const endMinutes = 23 * 60 + 59
  const totalMinutes = endMinutes - startMinutes + 1
  const randomOffset = hashString(filePath) % totalMinutes
  const minutesOfDay = startMinutes + randomOffset
  const hour = Math.floor(minutesOfDay / 60)
  const minute = minutesOfDay % 60

  // 按 Asia/Shanghai（UTC+8）解释该时间，并转成 UTC 时间戳
  const offsetHours = 8
  const utcHour = hour - offsetHours
  return Date.UTC(year, month - 1, day, utcHour, minute, 0, 0)
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "BEYOND",
  description: 'Tenacious life, proud journey. AI research notes and beyond.',
  lang: 'zh-CN',

  // GitHub Pages 部署配置
  // 因为是 username.github.io 仓库，base 用 '/'
  base: '/',

  // 忽略 README 中的本地开发链接检查
  ignoreDeadLinks: true,

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
          text: '📅 按年份查看',
          collapsed: false,
          items: [
            {
              text: '📅 2026 年',
              collapsed: true,
              items: [
                {
                  text: 'DA-Nav (2026-07-16)',
                  link: '/blog/2026/2026-07-16-da-nav',
                },
                {
                  text: 'NVIDIA主流AI显卡价格与算力对比 (2026-07-15)',
                  link: '/blog/2026/2026-07-15-NVIDIA-GPU-comparison',
                },
                {
                  text: '如何构建一个 VLA 导航仿真数据引擎 (2026-07-15)',
                  link: '/blog/2026/2026-07-15-simulation-data-engine',
                },
                {
                  text: '世界模型与VLA对比 (2026-07-14)',
                  link: '/blog/2026/2026-07-14-WorldModels-vs-VLA',
                },
                {
                  text: 'ABot-C0 (2026-07-13)',
                  link: '/blog/2026/2026-07-13-abot-c0',
                },
                {
                  text: 'ABot-N1 (2026-07-11)',
                  link: '/blog/2026/2026-07-11-ABot-N1',
                },
                {
                  text: 'InternData-N1：面向通用视觉-语言导航的大规模统一数据集 (2026-07-10)',
                  link: '/blog/2026/2026-07-10-InternData-N1',
                },
                {
                  text: 'Qwen3.6-Plus：面向 Agentic 系统的高层规划基座模型 (2026-07-10)',
                  link: '/blog/2026/2026-07-10-Qwen3.6-Plus',
                },
                {
                  text: '仿真数据的两种命运 (2026-07-09)',
                  link: '/blog/2026/2026-07-09-simrealitygap',
                },
                {
                  text: 'NVIDIA Sparse TOPS 与 Dense TOPS：部署时必须看懂的算力数字游戏 (2026-07-08)',
                  link: '/blog/2026/2026-07-08-nvidiasparsetopsvsdensetops',
                },
                {
                  text: 'LingBot-Vision 与 LingBot-Depth 2.0：具身智能的空间视觉基座再升级 (2026-07-07)',
                  link: '/blog/2026/2026-07-07-lingbot-vision-depth2',
                },
                {
                  text: '原生多模：从 Janus 到 Janus-Pro，一个 Transformer 同时理解与生成 (2026-07-07)',
                  link: '/blog/2026/2026-07-07-原生多模',
                },
                {
                  text: 'ACE-Brain-0.5：面向 Physical Agentic AI 的统一具身基模型 (2026-07-06)',
                  link: '/blog/2026/2026-07-06-ace-brain-05',
                },
                {
                  text: 'StableVLA：不增加数据，把 VLA 的视觉鲁棒性提升 30% (2026-07-06)',
                  link: '/blog/2026/2026-07-06-stablevla',
                },
                {
                  text: 'ABot (2026-07-05)',
                  link: '/blog/2026/2026-07-05-abot',
                },
                {
                  text: 'LiteVLA (2026-07-05)',
                  link: '/blog/2026/2026-07-05-litevla',
                },
                {
                  text: 'VLX-Go (2026-07-05)',
                  link: '/blog/2026/2026-07-05-vlxgo',
                },
                {
                  text: 'Jetson Orin 部署实战：算力、显存与模型切分的决策 (2026-07-04)',
                  link: '/blog/2026/2026-07-04-jetsonorindeployment',
                },
                {
                  text: 'Agent、Skill 与 Tool：三者的关系与协作边界 (2026-07-02)',
                  link: '/blog/2026/2026-07-02-agentskilltool',
                },
                {
                  text: '模型部署中的 Engine：从推理后端到生产落地 (2026-07-02)',
                  link: '/blog/2026/2026-07-02-modeldeploymentengine',
                },
                {
                  text: '高德地图语音导航逻辑与纯 Python Demo (2026-07-01)',
                  link: '/blog/2026/2026-07-01-gaodevoicenavdemo',
                },
                {
                  text: 'Agent 全景解析：从概念、架构到具身智能的工程实践 (2026-06-28)',
                  link: '/blog/2026/2026-06-28-agentcomprehensiveguide',
                },
                {
                  text: '关节限位 (2026-06-28)',
                  link: '/blog/2026/2026-06-28-jointlimits',
                },
                {
                  text: '运动原语：机器人动作世界的"字母表 (2026-06-28)',
                  link: '/blog/2026/2026-06-28-motionprimitives',
                },
                {
                  text: 'Qwen-RobotNav (2026-06-17)',
                  link: '/blog/2026/2026-06-17-qwenrobotnav',
                },
                {
                  text: 'EmbodiedNav (2026-06-15)',
                  link: '/blog/2026/2026-06-15-embodiednav',
                },
                {
                  text: 'MotionWAM (2026-06-08)',
                  link: '/blog/2026/2026-06-08-MotionWAM',
                },
                {
                  text: 'RLNeeds (2026-06-02)',
                  link: '/blog/2026/2026-06-02-rlneeds',
                },
                {
                  text: 'Uni-LaViRA (2026-05-26)',
                  link: '/blog/2026/2026-05-26-Uni-LaViRA',
                },
                {
                  text: 'LAPose (2026-05-05)',
                  link: '/blog/2026/2026-05-05-lapose',
                },
                {
                  text: 'Physical Intelligence π 系列 VLA 技术演进报告（π0 → π0.7） (2026-04-16)',
                  link: '/blog/2026/2026-04-16-pi07',
                },
                {
                  text: 'CoMaTrack (2026-03-24)',
                  link: '/blog/2026/2026-03-24-comatrack',
                },
                {
                  text: 'GigaWorldPolicy (2026-03-18)',
                  link: '/blog/2026/2026-03-18-gigaworldpolicy',
                },
                {
                  text: 'FastWAM (2026-03-17)',
                  link: '/blog/2026/2026-03-17-fastwam',
                },
                {
                  text: 'VJEPA2.1 (2026-03-15)',
                  link: '/blog/2026/2026-03-15-vjepa21',
                },
                {
                  text: 'RAENWM (2026-03-10)',
                  link: '/blog/2026/2026-03-10-raenwm',
                },
                {
                  text: 'SimVLA (2026-02-20)',
                  link: '/blog/2026/2026-02-20-simvla',
                },
                {
                  text: 'DreamZero (2026-02-17)',
                  link: '/blog/2026/2026-02-17-dreamzero',
                },
                {
                  text: 'ABot-N0 (2026-02-12)',
                  link: '/blog/2026/2026-02-12-abot-n0',
                },
                {
                  text: 'RISE (2026-02-11)',
                  link: '/blog/2026/2026-02-11-rise',
                },
                {
                  text: 'Muon (2026-02-10)',
                  link: '/blog/2026/2026-02-10-muon',
                },
                {
                  text: 'DriveWorldVLA (2026-02-06)',
                  link: '/blog/2026/2026-02-06-driveworldvla',
                },
                {
                  text: 'DriveJEPA (2026-01-29)',
                  link: '/blog/2026/2026-01-29-drivejepa',
                },
                {
                  text: 'LingBotVA (2026-01-29)',
                  link: '/blog/2026/2026-01-29-lingbotva',
                },
                {
                  text: 'C_RADIOv4 (2026-01-24)',
                  link: '/blog/2026/2026-01-24-c-radiov4',
                },
                {
                  text: '具身智能机器狗 Agent 架构设计 (2026-01-22)',
                  link: '/blog/2026/2026-01-22-embodied-robot-dog-architecture',
                },
                {
                  text: 'GeRo (2026-01-16)',
                  link: '/blog/2026/2026-01-16-gero',
                },
                {
                  text: '多 Agent 数据工程流水线架构分析 (2026-01-15)',
                  link: '/blog/2026/2026-01-15-multi-agent-data-pipeline',
                },
                {
                  text: 'FineGrainedAlignmentedVLN (2026-01-10)',
                  link: '/blog/2026/2026-01-10-finegrainedalignmentedvln',
                },
                {
                  text: 'AI Agent 实践案例集 (2026-01-08)',
                  link: '/blog/2026/2026-01-08-agent-practice-cases',
                },
                {
                  text: 'LearningLatentActionWM (2026-01-08)',
                  link: '/blog/2026/2026-01-08-learninglatentactionwm',
                },
                {
                  text: 'VLM4VLA (2026-01-06)',
                  link: '/blog/2026/2026-01-06-vlm4vla',
                },
                {
                  text: 'AI Agent 全面学习指南 (2026-01-01)',
                  link: '/blog/2026/2026-01-01-agent-comprehensive-guide',
                },
                {
                  text: 'LowerMemory (2026-01-01)',
                  link: '/blog/2026/2026-01-01-lowermemory',
                },
                {
                  text: 'ManyForcing (2026-01-01)',
                  link: '/blog/2026/2026-01-01-manyforcing',
                },
                {
                  text: 'ModelSpeed (2026-01-01)',
                  link: '/blog/2026/2026-01-01-modelspeed',
                },
                {
                  text: 'TrainingSpeed (2026-01-01)',
                  link: '/blog/2026/2026-01-01-trainingspeed',
                },
              ],
            },
            {
              text: '📅 2025 年',
              collapsed: true,
              items: [
                {
                  text: 'JEPA_WM (2025-12-30)',
                  link: '/blog/2025/2025-12-30-jepa-wm',
                },
                {
                  text: 'VLAAN (2025-12-17)',
                  link: '/blog/2025/2025-12-17-vlaan',
                },
                {
                  text: 'Motus (2025-12-15)',
                  link: '/blog/2025/2025-12-15-motus',
                },
                {
                  text: 'VLJEPA (2025-12-11)',
                  link: '/blog/2025/2025-12-11-vljepa',
                },
                {
                  text: 'DualVLN：慢思考、快执行——迈向通用视觉-语言导航的双系统基础模型 (2025-12-09)',
                  link: '/blog/2025/2025-12-09-DualVLN',
                },
                {
                  text: 'RoboScapeR (2025-12-03)',
                  link: '/blog/2025/2025-12-03-roboscaper',
                },
                {
                  text: 'ImprovedMeanFlows (2025-12-01)',
                  link: '/blog/2025/2025-12-01-improvedmeanflows',
                },
                {
                  text: 'SocialNav (2025-11-26)',
                  link: '/blog/2025/2025-11-26-SocialNav',
                },
                {
                  text: 'MobileVLA-R1：基于强化学习的视觉-语言-行动框架——迈向可解释的移动机器人连续控制 (2025-11-22)',
                  link: '/blog/2025/2025-11-22-MobileVLA-R1',
                },
                {
                  text: 'pi0.6 (2025-11-18)',
                  link: '/blog/2025/2025-11-18-pi06',
                },
                {
                  text: 'SurveyOnWorldModelsForEmbodiedAI (2025-10-19)',
                  link: '/blog/2025/2025-10-19-surveyonworldmodelsforembodiedai',
                },
                {
                  text: '从 PyTorch InternVL2.5-1B 到 Jetson Orin NX 16GB 的 VLA/VLM 嵌入式部署实战 (2025-10-04)',
                  link: '/blog/2025/2025-10-04-VLA_deploy',
                },
                {
                  text: 'VLACompare (2025-10-01)',
                  link: '/blog/2025/2025-10-01-vlacompare',
                },
                {
                  text: 'WMcompare (2025-10-01)',
                  link: '/blog/2025/2025-10-01-wmcompare',
                },
                {
                  text: 'LongScape (2025-09-26)',
                  link: '/blog/2025/2025-09-26-longscape',
                },
                {
                  text: 'OmniVLA (2025-09-23)',
                  link: '/blog/2025/2025-09-23-OmniVLA',
                },
                {
                  text: 'NavFoM (2025-09-15)',
                  link: '/blog/2025/2025-09-15-NavFoM',
                },
                {
                  text: 'VLA_embodedAI (2025-09-01)',
                  link: '/blog/2025/2025-09-01-vla-embodedai',
                },
                {
                  text: 'MatrixGame2.0 (2025-08-18)',
                  link: '/blog/2025/2025-08-18-matrixgame20',
                },
                {
                  text: 'DinoWorld (2025-07-25)',
                  link: '/blog/2025/2025-07-25-dinoworld',
                },
                {
                  text: 'StreamVLN (2025-07-07)',
                  link: '/blog/2025/2025-07-07-StreamVLN',
                },
                {
                  text: 'GoalVLA (2025-06-30)',
                  link: '/blog/2025/2025-06-30-goalvla',
                },
                {
                  text: 'RoboScape (2025-06-29)',
                  link: '/blog/2025/2025-06-29-roboscape',
                },
                {
                  text: 'VLA0S (2025-06-21)',
                  link: '/blog/2025/2025-06-21-vla0s',
                },
                {
                  text: 'AutoVLA (2025-06-16)',
                  link: '/blog/2025/2025-06-16-autovla',
                },
                {
                  text: 'VJEPA2 (2025-06-11)',
                  link: '/blog/2025/2025-06-11-vjepa2',
                },
                {
                  text: 'LiteVLM：面向嵌入式设备的低延迟 VLM 推理流水线 (2025-06-09)',
                  link: '/blog/2025/2025-06-09-litevlm',
                },
                {
                  text: 'SmolVLM：小而强的端侧视觉语言模型 (2025-06-01)',
                  link: '/blog/2025/2025-06-01-smolvlm',
                },
                {
                  text: 'TrackVLA (2025-05-29)',
                  link: '/blog/2025/2025-05-29-trackvla',
                },
                {
                  text: 'BAGEL (2025-05-20)',
                  link: '/blog/2025/2025-05-20-bagel',
                },
                {
                  text: 'VLM 模型部署实战：推理流水线、预处理与 Token 拼接 (2025-05-20)',
                  link: '/blog/2025/2025-05-20-vlmdeployment',
                },
                {
                  text: 'MeanFlows (2025-05-19 19:52)',
                  link: '/blog/2025/2025-05-19-meanflows',
                },
                {
                  text: 'Pi05 (2025-04-22)',
                  link: '/blog/2025/2025-04-22-pi05',
                },
                {
                  text: 'RaceVLA (2025-03-04)',
                  link: '/blog/2025/2025-03-04-RaceVLA',
                },
                {
                  text: 'ChatVLA (2025-02-20)',
                  link: '/blog/2025/2025-02-20-chatvla',
                },
                {
                  text: 'RAD (2025-02-18)',
                  link: '/blog/2025/2025-02-18-rad',
                },
                {
                  text: 'ROPE (2025-01-04)',
                  link: '/blog/2025/2025-01-04-rope',
                },
                {
                  text: 'DeepSpeed (2025-01-03)',
                  link: '/blog/2025/2025-01-03-deepspeed',
                },
                {
                  text: 'KVCACHE (2025-01-02)',
                  link: '/blog/2025/2025-01-02-kvcache',
                },
                {
                  text: 'FSDP (2025-01-01)',
                  link: '/blog/2025/2025-01-01-fsdp',
                },
              ],
            },
            {
              text: '📅 2024 年',
              collapsed: true,
              items: [
                {
                  text: 'pytorch_weights_datasets (2024-12-26)',
                  link: '/blog/2024/2024-12-26-pytorch-weights-datasets',
                },
                {
                  text: 'FLIP (2024-12-11)',
                  link: '/blog/2024/2024-12-11-flip',
                },
                {
                  text: 'Uni-NaVid (2024-12-09)',
                  link: '/blog/2024/2024-12-09-uni-navid',
                },
                {
                  text: 'NaVILA (2024-12-05)',
                  link: '/blog/2024/2024-12-05-NaVILA',
                },
                {
                  text: 'NWM (2024-12-04)',
                  link: '/blog/2024/2024-12-04-nwm',
                },
                {
                  text: 'pytorch_bug (2024-12-02)',
                  link: '/blog/2024/2024-12-02-pytorch-bug',
                },
                {
                  text: 'CityWalker (2024-11-26)',
                  link: '/blog/2024/2024-11-26-CityWalker',
                },
                {
                  text: 'DinoWM (2024-11-07)',
                  link: '/blog/2024/2024-11-07-dinowm',
                },
                {
                  text: 'Pi0 (2024-10-31)',
                  link: '/blog/2024/2024-10-31-pi0',
                },
                {
                  text: 'RDT1B (2024-10-10)',
                  link: '/blog/2024/2024-10-10-rdt1b',
                },
                {
                  text: 'TinyVLM (2024-10-04)',
                  link: '/blog/2024/2024-10-04-tinyvlm',
                },
                {
                  text: 'E2E (2024-10-03)',
                  link: '/blog/2024/2024-10-03-e2e',
                },
                {
                  text: 'SelfForcing (2024-10-02)',
                  link: '/blog/2024/2024-10-02-selfforcing',
                },
                {
                  text: 'PredictiveWM_VS_GenerativeWM (2024-10-01)',
                  link: '/blog/2024/2024-10-01-predictivewm-vs-generativewm',
                },
                {
                  text: 'MAR (2024-06-17)',
                  link: '/blog/2024/2024-06-17-mar',
                },
                {
                  text: 'OpenVLA (2024-06-13)',
                  link: '/blog/2024/2024-06-13-openvla',
                },
                {
                  text: 'SparseDrive (2024-05-30)',
                  link: '/blog/2024/2024-05-30-sparsedrive',
                },
                {
                  text: 'Survey_Occupancy (2024-05-08)',
                  link: '/blog/2024/2024-05-08-survey-occupancy',
                },
                {
                  text: 'SparseAD (2024-04-10)',
                  link: '/blog/2024/2024-04-10-sparsead',
                },
                {
                  text: 'VAR (2024-04-03)',
                  link: '/blog/2024/2024-04-03-var',
                },
                {
                  text: 'DROID (2024-03-19)',
                  link: '/blog/2024/2024-03-19-droid',
                },
                {
                  text: 'NaVid (2024-02-24)',
                  link: '/blog/2024/2024-02-24-NaVid',
                },
                {
                  text: 'VJEPA (2024-02-15)',
                  link: '/blog/2024/2024-02-15-vjepa',
                },
              ],
            },
            {
              text: '📅 2023 年',
              collapsed: true,
              items: [
                {
                  text: 'Vidar (2023-12-29)',
                  link: '/blog/2023/2023-12-29-vidar',
                },
                {
                  text: 'SparseOcc (2023-12-28)',
                  link: '/blog/2023/2023-12-28-sparseocc',
                },
                {
                  text: 'QUAR-VLA (2023-12-22)',
                  link: '/blog/2023/2023-12-22-quar-vla',
                },
                {
                  text: 'Emu2 (2023-12-20)',
                  link: '/blog/2023/2023-12-20-emu2',
                },
                {
                  text: 'Sparse4DV3 (2023-11-20)',
                  link: '/blog/2023/2023-11-20-sparse4dv3',
                },
                {
                  text: 'NoMaD (2023-10-11)',
                  link: '/blog/2023/2023-10-11-NoMaD',
                },
                {
                  text: '3DpointCloudGenerative (2023-10-10)',
                  link: '/blog/2023/2023-10-10-3dpointcloudgenerative',
                },
                {
                  text: 'MAGVITV2 (2023-10-09)',
                  link: '/blog/2023/2023-10-09-magvitv2',
                },
                {
                  text: 'WM (2023-10-03)',
                  link: '/blog/2023/2023-10-03-wm',
                },
                {
                  text: 'verl (2023-10-01)',
                  link: '/blog/2023/2023-10-01-verl',
                },
                {
                  text: 'DE_VS_SDE (2023-09-18)',
                  link: '/blog/2023/2023-09-18-de-vs-sde',
                },
                {
                  text: 'RenderOcc (2023-09-18)',
                  link: '/blog/2023/2023-09-18-renderocc',
                },
                {
                  text: 'SparseBEV (2023-8-18)',
                  link: '/blog/2023/2023-08-18-sparsebev',
                },
                {
                  text: 'RT2 (2023-07-28)',
                  link: '/blog/2023/2023-07-28-rt2',
                },
                {
                  text: 'Emu1 (2023-07-11)',
                  link: '/blog/2023/2023-07-11-emu1',
                },
                {
                  text: 'ViNT (2023-06-26)',
                  link: '/blog/2023/2023-06-26-ViNT',
                },
                {
                  text: 'Sparse4DV2 (2023-5-23)',
                  link: '/blog/2023/2023-05-23-sparse4dv2',
                },
                {
                  text: 'PrepareData (2023-05-02)',
                  link: '/blog/2023/2023-05-02-PrepareData',
                },
                {
                  text: 'ACT (2023-04-23)',
                  link: '/blog/2023/2023-04-23-act',
                },
                {
                  text: 'StreamPETR (2023-03-21)',
                  link: '/blog/2023/2023-03-21-streampetr',
                },
                {
                  text: 'DiffusionPolicy (2023-03-07)',
                  link: '/blog/2023/2023-03-07-diffusionpolicy',
                },
                {
                  text: 'IJEPA (2023-01-19)',
                  link: '/blog/2023/2023-01-19-ijepa',
                },
                {
                  text: 'DeramerV3 (2023-01-10)',
                  link: '/blog/2023/2023-01-10-deramerv3',
                },
                {
                  text: 'offlineonlineworldmodel (2023-01-01)',
                  link: '/blog/2023/2023-01-01-offlineonlineworldmodel',
                },
              ],
            },
            {
              text: '📅 2022 年',
              collapsed: true,
              items: [
                {
                  text: 'RT1 (2022-12-13)',
                  link: '/blog/2022/2022-12-13-rt1',
                },
                {
                  text: 'Sparse4D (2022-11-19)',
                  link: '/blog/2022/2022-11-19-sparse4d',
                },
                {
                  text: 'MILE (2022-10-14)',
                  link: '/blog/2022/2022-10-14-mile',
                },
                {
                  text: 'GNM (2022-10-07)',
                  link: '/blog/2022/2022-10-07-GNM',
                },
                {
                  text: 'FlowMatching (2022-10-06)',
                  link: '/blog/2022/2022-10-06-flowmatching',
                },
                {
                  text: 'RectifiedFlow (2022-09-07)',
                  link: '/blog/2022/2022-09-07-rectifiedflow',
                },
                {
                  text: 'FlowAndDataGeneration (2022-09-05)',
                  link: '/blog/2022/2022-09-05-flowanddatageneration',
                },
                {
                  text: 'IRIS (2022-09-01)',
                  link: '/blog/2022/2022-09-01-iris',
                },
                {
                  text: 'Wayformer (2022-07-12)',
                  link: '/blog/2022/2022-07-12-wayformer',
                },
                {
                  text: 'LM-Nav (2022-07-10)',
                  link: '/blog/2022/2022-07-10-LM-Nav',
                },
                {
                  text: 'DayDreamer (2022-06-28)',
                  link: '/blog/2022/2022-06-28-daydreamer',
                },
                {
                  text: 'VPT (2022-06-23)',
                  link: '/blog/2022/2022-06-23-vpt',
                },
                {
                  text: 'PETRV2 (2022-06-02)',
                  link: '/blog/2022/2022-06-02-petrv2',
                },
                {
                  text: 'Gato (2022-05-12)',
                  link: '/blog/2022/2022-05-12-gato',
                },
                {
                  text: 'mu_law (2022-05-12)',
                  link: '/blog/2022/2022-05-12-mu-law',
                },
                {
                  text: 'PETR (2022-03-10)',
                  link: '/blog/2022/2022-03-10-petr',
                },
                {
                  text: 'TransDreamer (2022-02-19)',
                  link: '/blog/2022/2022-02-19-transdreamer',
                },
              ],
            },
            {
              text: '📅 2021 年',
              collapsed: true,
              items: [
                {
                  text: 'Mask2Former (2021-12-02)',
                  link: '/blog/2021/2021-12-02-mask2former',
                },
                {
                  text: 'DETR3D (2021-10-13)',
                  link: '/blog/2021/2021-10-13-detr3d',
                },
                {
                  text: 'MaskFormer (2021-07-13)',
                  link: '/blog/2021/2021-07-13-maskformer',
                },
                {
                  text: 'CaDDN (2021-03-01)',
                  link: '/blog/2021/2021-03-01-caddn',
                },
                {
                  text: 'IntentNet (2021-01-20)',
                  link: '/blog/2021/2021-01-20-intentnet',
                },
              ],
            },
            {
              text: '📅 2020 年',
              collapsed: true,
              items: [
                {
                  text: 'SparseRCNN (2020-11-25)',
                  link: '/blog/2020/2020-11-25-sparsercnn',
                },
                {
                  text: 'DreamerV2 (2020-10-05)',
                  link: '/blog/2020/2020-10-05-dreamerv2',
                },
                {
                  text: 'TNT (2020-08-19)',
                  link: '/blog/2020/2020-08-19-tnt',
                },
                {
                  text: 'CVAE_VS_VAE (2020-07-06)',
                  link: '/blog/2020/2020-07-06-cvae-vs-vae',
                },
                {
                  text: 'DETR (2020-05-26)',
                  link: '/blog/2020/2020-05-26-detr',
                },
                {
                  text: 'VectorNet (2020-05-08)',
                  link: '/blog/2020/2020-05-08-vectornet',
                },
              ],
            },
            {
              text: '📅 2019 年',
              collapsed: true,
              items: [
                {
                  text: 'DreamerV1 (2019-12-03)',
                  link: '/blog/2019/2019-12-03-dreamerv1',
                },
                {
                  text: 'MultiPath (2019-10-12)',
                  link: '/blog/2019/2019-10-12-multipath',
                },
                {
                  text: 'Talk2Nav (2019-10-04)',
                  link: '/blog/2019/2019-10-04-Talk2Nav',
                },
                {
                  text: '从 PyTorch YOLO 到 Jetson Orin 的完整嵌入式部署实战 (2019-10-04)',
                  link: '/blog/2019/2019-10-04-Yolo_deploy',
                },
                {
                  text: 'l1l2 (2019-01-01)',
                  link: '/blog/2019/2019-01-01-l1l2',
                },
              ],
            },
            {
              text: '📅 2018 年',
              collapsed: true,
              items: [
                {
                  text: 'PlaNet (2018-11-12)',
                  link: '/blog/2018/2018-11-12-planet',
                },
                {
                  text: 'WorldModels (2018-03-27)',
                  link: '/blog/2018/2018-03-27-worldmodels',
                },
              ],
            },
          ],
        },
        {
          text: '📂 按分类查看',
          collapsed: false,
          items: [
            {
              text: '📂 VLA',
              collapsed: true,
              items: [
                {
                  text: 'DA-Nav (2026-07-16)',
                  link: '/blog/2026/2026-07-16-da-nav',
                },
                {
                  text: '世界模型与VLA对比 (2026-07-14)',
                  link: '/blog/2026/2026-07-14-WorldModels-vs-VLA',
                },
                {
                  text: 'ABot-C0 (2026-07-13)',
                  link: '/blog/2026/2026-07-13-abot-c0',
                },
                {
                  text: 'ABot-N1 (2026-07-11)',
                  link: '/blog/2026/2026-07-11-ABot-N1',
                },
                {
                  text: 'ACE-Brain-0.5：面向 Physical Agentic AI 的统一具身基模型 (2026-07-06)',
                  link: '/blog/2026/2026-07-06-ace-brain-05',
                },
                {
                  text: 'StableVLA：不增加数据，把 VLA 的视觉鲁棒性提升 30% (2026-07-06)',
                  link: '/blog/2026/2026-07-06-stablevla',
                },
                {
                  text: 'ABot (2026-07-05)',
                  link: '/blog/2026/2026-07-05-abot',
                },
                {
                  text: 'LiteVLA (2026-07-05)',
                  link: '/blog/2026/2026-07-05-litevla',
                },
                {
                  text: 'VLX-Go (2026-07-05)',
                  link: '/blog/2026/2026-07-05-vlxgo',
                },
                {
                  text: 'Qwen-RobotNav (2026-06-17)',
                  link: '/blog/2026/2026-06-17-qwenrobotnav',
                },
                {
                  text: 'EmbodiedNav (2026-06-15)',
                  link: '/blog/2026/2026-06-15-embodiednav',
                },
                {
                  text: 'Physical Intelligence π 系列 VLA 技术演进报告（π0 → π0.7） (2026-04-16)',
                  link: '/blog/2026/2026-04-16-pi07',
                },
                {
                  text: 'CoMaTrack (2026-03-24)',
                  link: '/blog/2026/2026-03-24-comatrack',
                },
                {
                  text: 'ABot-N0 (2026-02-12)',
                  link: '/blog/2026/2026-02-12-abot-n0',
                },
                {
                  text: 'VLM4VLA (2026-01-06)',
                  link: '/blog/2026/2026-01-06-vlm4vla',
                },
                {
                  text: 'GoalVLA (2025-06-30)',
                  link: '/blog/2025/2025-06-30-goalvla',
                },
                {
                  text: 'AutoVLA (2025-06-16)',
                  link: '/blog/2025/2025-06-16-autovla',
                },
                {
                  text: 'Pi05 (2025-04-22)',
                  link: '/blog/2025/2025-04-22-pi05',
                },
                {
                  text: 'ChatVLA (2025-02-20)',
                  link: '/blog/2025/2025-02-20-chatvla',
                },
                {
                  text: 'Uni-NaVid (2024-12-09)',
                  link: '/blog/2024/2024-12-09-uni-navid',
                },
                {
                  text: 'Pi0 (2024-10-31)',
                  link: '/blog/2024/2024-10-31-pi0',
                },
                {
                  text: 'RDT1B (2024-10-10)',
                  link: '/blog/2024/2024-10-10-rdt1b',
                },
                {
                  text: 'OpenVLA (2024-06-13)',
                  link: '/blog/2024/2024-06-13-openvla',
                },
                {
                  text: 'DROID (2024-03-19)',
                  link: '/blog/2024/2024-03-19-droid',
                },
                {
                  text: 'QUAR-VLA (2023-12-22)',
                  link: '/blog/2023/2023-12-22-quar-vla',
                },
                {
                  text: 'RT2 (2023-07-28)',
                  link: '/blog/2023/2023-07-28-rt2',
                },
                {
                  text: 'DiffusionPolicy (2023-03-07)',
                  link: '/blog/2023/2023-03-07-diffusionpolicy',
                },
                {
                  text: 'RT1 (2022-12-13)',
                  link: '/blog/2022/2022-12-13-rt1',
                },
                {
                  text: 'Gato (2022-05-12)',
                  link: '/blog/2022/2022-05-12-gato',
                },
                {
                  text: 'Talk2Nav (2019-10-04)',
                  link: '/blog/2019/2019-10-04-Talk2Nav',
                },
              ],
            },
            {
              text: '📂 Understandings',
              collapsed: true,
              items: [
                {
                  text: '仿真数据的两种命运 (2026-07-09)',
                  link: '/blog/2026/2026-07-09-simrealitygap',
                },
                {
                  text: 'Muon (2026-02-10)',
                  link: '/blog/2026/2026-02-10-muon',
                },
                {
                  text: 'LowerMemory (2026-01-01)',
                  link: '/blog/2026/2026-01-01-lowermemory',
                },
                {
                  text: 'ManyForcing (2026-01-01)',
                  link: '/blog/2026/2026-01-01-manyforcing',
                },
                {
                  text: 'ModelSpeed (2026-01-01)',
                  link: '/blog/2026/2026-01-01-modelspeed',
                },
                {
                  text: 'TrainingSpeed (2026-01-01)',
                  link: '/blog/2026/2026-01-01-trainingspeed',
                },
                {
                  text: 'VLACompare (2025-10-01)',
                  link: '/blog/2025/2025-10-01-vlacompare',
                },
                {
                  text: 'WMcompare (2025-10-01)',
                  link: '/blog/2025/2025-10-01-wmcompare',
                },
                {
                  text: 'ROPE (2025-01-04)',
                  link: '/blog/2025/2025-01-04-rope',
                },
                {
                  text: 'DeepSpeed (2025-01-03)',
                  link: '/blog/2025/2025-01-03-deepspeed',
                },
                {
                  text: 'KVCACHE (2025-01-02)',
                  link: '/blog/2025/2025-01-02-kvcache',
                },
                {
                  text: 'FSDP (2025-01-01)',
                  link: '/blog/2025/2025-01-01-fsdp',
                },
                {
                  text: 'E2E (2024-10-03)',
                  link: '/blog/2024/2024-10-03-e2e',
                },
                {
                  text: 'SelfForcing (2024-10-02)',
                  link: '/blog/2024/2024-10-02-selfforcing',
                },
                {
                  text: 'PredictiveWM_VS_GenerativeWM (2024-10-01)',
                  link: '/blog/2024/2024-10-01-predictivewm-vs-generativewm',
                },
                {
                  text: '3DpointCloudGenerative (2023-10-10)',
                  link: '/blog/2023/2023-10-10-3dpointcloudgenerative',
                },
                {
                  text: 'WM (2023-10-03)',
                  link: '/blog/2023/2023-10-03-wm',
                },
                {
                  text: 'DE_VS_SDE (2023-09-18)',
                  link: '/blog/2023/2023-09-18-de-vs-sde',
                },
                {
                  text: 'offlineonlineworldmodel (2023-01-01)',
                  link: '/blog/2023/2023-01-01-offlineonlineworldmodel',
                },
                {
                  text: 'CVAE_VS_VAE (2020-07-06)',
                  link: '/blog/2020/2020-07-06-cvae-vs-vae',
                },
                {
                  text: 'l1l2 (2019-01-01)',
                  link: '/blog/2019/2019-01-01-l1l2',
                },
              ],
            },
            {
              text: '📂 VLN',
              collapsed: true,
              items: [
                {
                  text: 'Uni-LaViRA (2026-05-26)',
                  link: '/blog/2026/2026-05-26-Uni-LaViRA',
                },
                {
                  text: 'FineGrainedAlignmentedVLN (2026-01-10)',
                  link: '/blog/2026/2026-01-10-finegrainedalignmentedvln',
                },
                {
                  text: 'SocialNav (2025-11-26)',
                  link: '/blog/2025/2025-11-26-SocialNav',
                },
                {
                  text: 'OmniVLA (2025-09-23)',
                  link: '/blog/2025/2025-09-23-OmniVLA',
                },
                {
                  text: 'NavFoM (2025-09-15)',
                  link: '/blog/2025/2025-09-15-NavFoM',
                },
                {
                  text: 'StreamVLN (2025-07-07)',
                  link: '/blog/2025/2025-07-07-StreamVLN',
                },
                {
                  text: 'RaceVLA (2025-03-04)',
                  link: '/blog/2025/2025-03-04-RaceVLA',
                },
                {
                  text: 'NaVILA (2024-12-05)',
                  link: '/blog/2024/2024-12-05-NaVILA',
                },
                {
                  text: 'CityWalker (2024-11-26)',
                  link: '/blog/2024/2024-11-26-CityWalker',
                },
                {
                  text: 'NaVid (2024-02-24)',
                  link: '/blog/2024/2024-02-24-NaVid',
                },
                {
                  text: 'NoMaD (2023-10-11)',
                  link: '/blog/2023/2023-10-11-NoMaD',
                },
                {
                  text: 'ViNT (2023-06-26)',
                  link: '/blog/2023/2023-06-26-ViNT',
                },
                {
                  text: 'GNM (2022-10-07)',
                  link: '/blog/2022/2022-10-07-GNM',
                },
                {
                  text: 'LM-Nav (2022-07-10)',
                  link: '/blog/2022/2022-07-10-LM-Nav',
                },
              ],
            },
            {
              text: '📂 WorldModels',
              collapsed: true,
              items: [
                {
                  text: 'DriveJEPA (2026-01-29)',
                  link: '/blog/2026/2026-01-29-drivejepa',
                },
                {
                  text: 'DinoWorld (2025-07-25)',
                  link: '/blog/2025/2025-07-25-dinoworld',
                },
                {
                  text: 'RoboScape (2025-06-29)',
                  link: '/blog/2025/2025-06-29-roboscape',
                },
                {
                  text: 'DinoWM (2024-11-07)',
                  link: '/blog/2024/2024-11-07-dinowm',
                },
                {
                  text: 'IJEPA (2023-01-19)',
                  link: '/blog/2023/2023-01-19-ijepa',
                },
                {
                  text: 'DeramerV3 (2023-01-10)',
                  link: '/blog/2023/2023-01-10-deramerv3',
                },
                {
                  text: 'IRIS (2022-09-01)',
                  link: '/blog/2022/2022-09-01-iris',
                },
                {
                  text: 'TransDreamer (2022-02-19)',
                  link: '/blog/2022/2022-02-19-transdreamer',
                },
                {
                  text: 'DreamerV2 (2020-10-05)',
                  link: '/blog/2020/2020-10-05-dreamerv2',
                },
                {
                  text: 'DreamerV1 (2019-12-03)',
                  link: '/blog/2019/2019-12-03-dreamerv1',
                },
                {
                  text: 'WorldModels (2018-03-27)',
                  link: '/blog/2018/2018-03-27-worldmodels',
                },
              ],
            },
            {
              text: '📂 AIGC',
              collapsed: true,
              items: [
                {
                  text: 'ImprovedMeanFlows (2025-12-01)',
                  link: '/blog/2025/2025-12-01-improvedmeanflows',
                },
                {
                  text: 'BAGEL (2025-05-20)',
                  link: '/blog/2025/2025-05-20-bagel',
                },
                {
                  text: 'MeanFlows (2025-05-19 19:52)',
                  link: '/blog/2025/2025-05-19-meanflows',
                },
                {
                  text: 'MAR (2024-06-17)',
                  link: '/blog/2024/2024-06-17-mar',
                },
                {
                  text: 'VAR (2024-04-03)',
                  link: '/blog/2024/2024-04-03-var',
                },
                {
                  text: 'MAGVITV2 (2023-10-09)',
                  link: '/blog/2023/2023-10-09-magvitv2',
                },
                {
                  text: 'FlowMatching (2022-10-06)',
                  link: '/blog/2022/2022-10-06-flowmatching',
                },
                {
                  text: 'RectifiedFlow (2022-09-07)',
                  link: '/blog/2022/2022-09-07-rectifiedflow',
                },
                {
                  text: 'FlowAndDataGeneration (2022-09-05)',
                  link: '/blog/2022/2022-09-05-flowanddatageneration',
                },
              ],
            },
            {
              text: '📂 Perception',
              collapsed: true,
              items: [
                {
                  text: 'SparseBEV (2023-8-18)',
                  link: '/blog/2023/2023-08-18-sparsebev',
                },
                {
                  text: 'StreamPETR (2023-03-21)',
                  link: '/blog/2023/2023-03-21-streampetr',
                },
                {
                  text: 'Sparse4D (2022-11-19)',
                  link: '/blog/2022/2022-11-19-sparse4d',
                },
                {
                  text: 'PETRV2 (2022-06-02)',
                  link: '/blog/2022/2022-06-02-petrv2',
                },
                {
                  text: 'PETR (2022-03-10)',
                  link: '/blog/2022/2022-03-10-petr',
                },
                {
                  text: 'DETR3D (2021-10-13)',
                  link: '/blog/2021/2021-10-13-detr3d',
                },
                {
                  text: 'CaDDN (2021-03-01)',
                  link: '/blog/2021/2021-03-01-caddn',
                },
              ],
            },
            {
              text: '📂 Deploy',
              collapsed: true,
              items: [
                {
                  text: 'NVIDIA Sparse TOPS 与 Dense TOPS：部署时必须看懂的算力数字游戏 (2026-07-08)',
                  link: '/blog/2026/2026-07-08-nvidiasparsetopsvsdensetops',
                },
                {
                  text: 'Jetson Orin 部署实战：算力、显存与模型切分的决策 (2026-07-04)',
                  link: '/blog/2026/2026-07-04-jetsonorindeployment',
                },
                {
                  text: '模型部署中的 Engine：从推理后端到生产落地 (2026-07-02)',
                  link: '/blog/2026/2026-07-02-modeldeploymentengine',
                },
                {
                  text: '从 PyTorch InternVL2.5-1B 到 Jetson Orin NX 16GB 的 VLA/VLM 嵌入式部署实战 (2025-10-04)',
                  link: '/blog/2025/2025-10-04-VLA_deploy',
                },
                {
                  text: 'VLM 模型部署实战：推理流水线、预处理与 Token 拼接 (2025-05-20)',
                  link: '/blog/2025/2025-05-20-vlmdeployment',
                },
                {
                  text: '从 PyTorch YOLO 到 Jetson Orin 的完整嵌入式部署实战 (2019-10-04)',
                  link: '/blog/2019/2019-10-04-Yolo_deploy',
                },
              ],
            },
            {
              text: '📂 Agents',
              collapsed: true,
              items: [
                {
                  text: 'Agent、Skill 与 Tool：三者的关系与协作边界 (2026-07-02)',
                  link: '/blog/2026/2026-07-02-agentskilltool',
                },
                {
                  text: 'Agent 全景解析：从概念、架构到具身智能的工程实践 (2026-06-28)',
                  link: '/blog/2026/2026-06-28-agentcomprehensiveguide',
                },
                {
                  text: '具身智能机器狗 Agent 架构设计 (2026-01-22)',
                  link: '/blog/2026/2026-01-22-embodied-robot-dog-architecture',
                },
                {
                  text: '多 Agent 数据工程流水线架构分析 (2026-01-15)',
                  link: '/blog/2026/2026-01-15-multi-agent-data-pipeline',
                },
                {
                  text: 'AI Agent 实践案例集 (2026-01-08)',
                  link: '/blog/2026/2026-01-08-agent-practice-cases',
                },
                {
                  text: 'AI Agent 全面学习指南 (2026-01-01)',
                  link: '/blog/2026/2026-01-01-agent-comprehensive-guide',
                },
              ],
            },
            {
              text: '📂 Prediction',
              collapsed: true,
              items: [
                {
                  text: 'Wayformer (2022-07-12)',
                  link: '/blog/2022/2022-07-12-wayformer',
                },
                {
                  text: 'IntentNet (2021-01-20)',
                  link: '/blog/2021/2021-01-20-intentnet',
                },
                {
                  text: 'TNT (2020-08-19)',
                  link: '/blog/2020/2020-08-19-tnt',
                },
                {
                  text: 'VectorNet (2020-05-08)',
                  link: '/blog/2020/2020-05-08-vectornet',
                },
                {
                  text: 'MultiPath (2019-10-12)',
                  link: '/blog/2019/2019-10-12-multipath',
                },
              ],
            },
            {
              text: '📂 WAM',
              collapsed: true,
              items: [
                {
                  text: 'MotionWAM (2026-06-08)',
                  link: '/blog/2026/2026-06-08-MotionWAM',
                },
                {
                  text: 'FastWAM (2026-03-17)',
                  link: '/blog/2026/2026-03-17-fastwam',
                },
                {
                  text: 'RAENWM (2026-03-10)',
                  link: '/blog/2026/2026-03-10-raenwm',
                },
                {
                  text: 'LingBotVA (2026-01-29)',
                  link: '/blog/2026/2026-01-29-lingbotva',
                },
                {
                  text: 'NWM (2024-12-04)',
                  link: '/blog/2024/2024-12-04-nwm',
                },
              ],
            },
            {
              text: '📂 reading',
              collapsed: true,
              items: [
                {
                  text: 'pytorch_weights_datasets (2024-12-26)',
                  link: '/blog/2024/2024-12-26-pytorch-weights-datasets',
                },
                {
                  text: 'Sparse4DV3 (2023-11-20)',
                  link: '/blog/2023/2023-11-20-sparse4dv3',
                },
                {
                  text: 'Sparse4DV2 (2023-5-23)',
                  link: '/blog/2023/2023-05-23-sparse4dv2',
                },
                {
                  text: 'MaskFormer (2021-07-13)',
                  link: '/blog/2021/2021-07-13-maskformer',
                },
              ],
            },
            {
              text: '📂 Occupancy',
              collapsed: true,
              items: [
                {
                  text: 'Survey_Occupancy (2024-05-08)',
                  link: '/blog/2024/2024-05-08-survey-occupancy',
                },
                {
                  text: 'SparseOcc (2023-12-28)',
                  link: '/blog/2023/2023-12-28-sparseocc',
                },
                {
                  text: 'RenderOcc (2023-09-18)',
                  link: '/blog/2023/2023-09-18-renderocc',
                },
              ],
            },
            {
              text: '📂 e2e',
              collapsed: true,
              items: [
                {
                  text: 'SparseDrive (2024-05-30)',
                  link: '/blog/2024/2024-05-30-sparsedrive',
                },
                {
                  text: 'SparseAD (2024-04-10)',
                  link: '/blog/2024/2024-04-10-sparsead',
                },
                {
                  text: 'Vidar (2023-12-29)',
                  link: '/blog/2023/2023-12-29-vidar',
                },
              ],
            },
            {
              text: '📂 VLM',
              collapsed: true,
              items: [
                {
                  text: '原生多模：从 Janus 到 Janus-Pro，一个 Transformer 同时理解与生成 (2026-07-07)',
                  link: '/blog/2026/2026-07-07-原生多模',
                },
                {
                  text: 'LiteVLM：面向嵌入式设备的低延迟 VLM 推理流水线 (2025-06-09)',
                  link: '/blog/2025/2025-06-09-litevlm',
                },
                {
                  text: 'SmolVLM：小而强的端侧视觉语言模型 (2025-06-01)',
                  link: '/blog/2025/2025-06-01-smolvlm',
                },
              ],
            },
            {
              text: '📂 Detection',
              collapsed: true,
              items: [
                {
                  text: 'SparseRCNN (2020-11-25)',
                  link: '/blog/2020/2020-11-25-sparsercnn',
                },
                {
                  text: 'DETR (2020-05-26)',
                  link: '/blog/2020/2020-05-26-detr',
                },
              ],
            },
            {
              text: '📂 EmbodiedAI',
              collapsed: true,
              items: [
                {
                  text: 'ACT (2023-04-23)',
                  link: '/blog/2023/2023-04-23-act',
                },
                {
                  text: 'Gato (2022-05-12)',
                  link: '/blog/2022/2022-05-12-gato',
                },
              ],
            },
            {
              text: '📂 机器人',
              collapsed: true,
              items: [
                {
                  text: '关节限位 (2026-06-28)',
                  link: '/blog/2026/2026-06-28-jointlimits',
                },
                {
                  text: '运动原语：机器人动作世界的"字母表 (2026-06-28)',
                  link: '/blog/2026/2026-06-28-motionprimitives',
                },
              ],
            },
            {
              text: '📂 工具与约定',
              collapsed: true,
              items: [
                {
                  text: 'Neovim + iTerm2 工作流 (2026-06-12)',
                  link: '/blog/notes/2026-06-12-neovim-workflow',
                },
                {
                  text: '写作约定 (2026-06-07)',
                  link: '/blog/notes/2026-06-07-writing-guide',
                },
              ],
            },
            {
              text: '📂 Segmentation',
              collapsed: true,
              items: [
                {
                  text: 'Mask2Former (2021-12-02)',
                  link: '/blog/2021/2021-12-02-mask2former',
                },
              ],
            },
            {
              text: '📂 Tricks',
              collapsed: true,
              items: [
                {
                  text: 'mu_law (2022-05-12)',
                  link: '/blog/2022/2022-05-12-mu-law',
                },
              ],
            },
            {
              text: '📂 自动驾驶',
              collapsed: true,
              items: [
                {
                  text: 'PrepareData (2023-05-02)',
                  link: '/blog/2023/2023-05-02-PrepareData',
                },
              ],
            },
            {
              text: '📂 pytorch',
              collapsed: true,
              items: [
                {
                  text: 'pytorch_bug (2024-12-02)',
                  link: '/blog/2024/2024-12-02-pytorch-bug',
                },
              ],
            },
            {
              text: '📂 Thinking',
              collapsed: true,
              items: [
                {
                  text: 'VLA_embodedAI (2025-09-01)',
                  link: '/blog/2025/2025-09-01-vla-embodedai',
                },
              ],
            },
            {
              text: '📂 Vision',
              collapsed: true,
              items: [
                {
                  text: 'LingBot-Vision 与 LingBot-Depth 2.0：具身智能的空间视觉基座再升级 (2026-07-07)',
                  link: '/blog/2026/2026-07-07-lingbot-vision-depth2',
                },
              ],
            },
            {
              text: '📂 仿真',
              collapsed: true,
              items: [
                {
                  text: '如何构建一个 VLA 导航仿真数据引擎 (2026-07-15)',
                  link: '/blog/2026/2026-07-15-simulation-data-engine',
                },
              ],
            },
            {
              text: '📂 others',
              collapsed: true,
              items: [
                {
                  text: 'NVIDIA主流AI显卡价格与算力对比 (2026-07-15)',
                  link: '/blog/2026/2026-07-15-NVIDIA-GPU-comparison',
                },
                {
                  text: 'InternData-N1：面向通用视觉-语言导航的大规模统一数据集 (2026-07-10)',
                  link: '/blog/2026/2026-07-10-InternData-N1',
                },
                {
                  text: 'Qwen3.6-Plus：面向 Agentic 系统的高层规划基座模型 (2026-07-10)',
                  link: '/blog/2026/2026-07-10-Qwen3.6-Plus',
                },
                {
                  text: '高德地图语音导航逻辑与纯 Python Demo (2026-07-01)',
                  link: '/blog/2026/2026-07-01-gaodevoicenavdemo',
                },
                {
                  text: 'RLNeeds (2026-06-02)',
                  link: '/blog/2026/2026-06-02-rlneeds',
                },
                {
                  text: 'LAPose (2026-05-05)',
                  link: '/blog/2026/2026-05-05-lapose',
                },
                {
                  text: 'GigaWorldPolicy (2026-03-18)',
                  link: '/blog/2026/2026-03-18-gigaworldpolicy',
                },
                {
                  text: 'VJEPA2.1 (2026-03-15)',
                  link: '/blog/2026/2026-03-15-vjepa21',
                },
                {
                  text: 'SimVLA (2026-02-20)',
                  link: '/blog/2026/2026-02-20-simvla',
                },
                {
                  text: 'DreamZero (2026-02-17)',
                  link: '/blog/2026/2026-02-17-dreamzero',
                },
                {
                  text: 'RISE (2026-02-11)',
                  link: '/blog/2026/2026-02-11-rise',
                },
                {
                  text: 'DriveWorldVLA (2026-02-06)',
                  link: '/blog/2026/2026-02-06-driveworldvla',
                },
                {
                  text: 'C_RADIOv4 (2026-01-24)',
                  link: '/blog/2026/2026-01-24-c-radiov4',
                },
                {
                  text: 'GeRo (2026-01-16)',
                  link: '/blog/2026/2026-01-16-gero',
                },
                {
                  text: 'LearningLatentActionWM (2026-01-08)',
                  link: '/blog/2026/2026-01-08-learninglatentactionwm',
                },
                {
                  text: 'JEPA_WM (2025-12-30)',
                  link: '/blog/2025/2025-12-30-jepa-wm',
                },
                {
                  text: 'VLAAN (2025-12-17)',
                  link: '/blog/2025/2025-12-17-vlaan',
                },
                {
                  text: 'Motus (2025-12-15)',
                  link: '/blog/2025/2025-12-15-motus',
                },
                {
                  text: 'VLJEPA (2025-12-11)',
                  link: '/blog/2025/2025-12-11-vljepa',
                },
                {
                  text: 'DualVLN：慢思考、快执行——迈向通用视觉-语言导航的双系统基础模型 (2025-12-09)',
                  link: '/blog/2025/2025-12-09-DualVLN',
                },
                {
                  text: 'RoboScapeR (2025-12-03)',
                  link: '/blog/2025/2025-12-03-roboscaper',
                },
                {
                  text: 'MobileVLA-R1：基于强化学习的视觉-语言-行动框架——迈向可解释的移动机器人连续控制 (2025-11-22)',
                  link: '/blog/2025/2025-11-22-MobileVLA-R1',
                },
                {
                  text: 'pi0.6 (2025-11-18)',
                  link: '/blog/2025/2025-11-18-pi06',
                },
                {
                  text: 'SurveyOnWorldModelsForEmbodiedAI (2025-10-19)',
                  link: '/blog/2025/2025-10-19-surveyonworldmodelsforembodiedai',
                },
                {
                  text: 'LongScape (2025-09-26)',
                  link: '/blog/2025/2025-09-26-longscape',
                },
                {
                  text: 'MatrixGame2.0 (2025-08-18)',
                  link: '/blog/2025/2025-08-18-matrixgame20',
                },
                {
                  text: 'VLA0S (2025-06-21)',
                  link: '/blog/2025/2025-06-21-vla0s',
                },
                {
                  text: 'VJEPA2 (2025-06-11)',
                  link: '/blog/2025/2025-06-11-vjepa2',
                },
                {
                  text: 'TrackVLA (2025-05-29)',
                  link: '/blog/2025/2025-05-29-trackvla',
                },
                {
                  text: 'RAD (2025-02-18)',
                  link: '/blog/2025/2025-02-18-rad',
                },
                {
                  text: 'FLIP (2024-12-11)',
                  link: '/blog/2024/2024-12-11-flip',
                },
                {
                  text: 'TinyVLM (2024-10-04)',
                  link: '/blog/2024/2024-10-04-tinyvlm',
                },
                {
                  text: 'VJEPA (2024-02-15)',
                  link: '/blog/2024/2024-02-15-vjepa',
                },
                {
                  text: 'Emu2 (2023-12-20)',
                  link: '/blog/2023/2023-12-20-emu2',
                },
                {
                  text: 'verl (2023-10-01)',
                  link: '/blog/2023/2023-10-01-verl',
                },
                {
                  text: 'Emu1 (2023-07-11)',
                  link: '/blog/2023/2023-07-11-emu1',
                },
                {
                  text: 'MILE (2022-10-14)',
                  link: '/blog/2022/2022-10-14-mile',
                },
                {
                  text: 'DayDreamer (2022-06-28)',
                  link: '/blog/2022/2022-06-28-daydreamer',
                },
                {
                  text: 'VPT (2022-06-23)',
                  link: '/blog/2022/2022-06-23-vpt',
                },
                {
                  text: 'PlaNet (2018-11-12)',
                  link: '/blog/2018/2018-11-12-planet',
                },
              ],
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
        timeZone: 'Asia/Shanghai',
      },
    },
  },

  // 根据 frontmatter.date 覆盖最后更新时间，使日期与文档保持一致
  transformPageData(pageData) {
    const lastUpdated = computeLastUpdated(pageData.frontmatter, pageData.filePath)
    if (lastUpdated !== undefined) {
      return { lastUpdated }
    }
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
