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
          text: '📅 按年份查看',
          collapsed: false,
          items: [
            {
              text: '📅 2026 年',
              collapsed: true,
              items: [
                {
                  text: 'LiteVLA (2026-07-05)',
                  link: '/blog/2026/LiteVLA',
                },
                {
                  text: 'VLX-Go (2026-07-05)',
                  link: '/blog/2026/VLXGo',
                },
                {
                  text: 'Jetson Orin 部署实战：算力、显存与模型切分的决策 (2026-07-04)',
                  link: '/blog/2026/JetsonOrinDeployment',
                },
                {
                  text: 'Agent、Skill 与 Tool：三者的关系与协作边界 (2026-07-02)',
                  link: '/blog/2026/AgentSkillTool',
                },
                {
                  text: '模型部署中的 Engine：从推理后端到生产落地 (2026-07-02)',
                  link: '/blog/2026/ModelDeploymentEngine',
                },
                {
                  text: '高德地图语音导航逻辑与纯 Python Demo (2026-07-01)',
                  link: '/blog/2026/GaodeVoiceNavDemo',
                },
                {
                  text: 'Agent 全景解析：从概念、架构到具身智能的工程实践 (2026-06-28)',
                  link: '/blog/2026/AgentComprehensiveGuide',
                },
                {
                  text: '关节限位 (2026-06-28)',
                  link: '/blog/2026/JointLimits',
                },
                {
                  text: '运动原语：机器人动作世界的"字母表 (2026-06-28)',
                  link: '/blog/2026/MotionPrimitives',
                },
                {
                  text: 'Qwen-RobotNav (2026-06-17)',
                  link: '/blog/2026/QwenRobotNav',
                },
                {
                  text: 'RLNeeds (2026-06-02)',
                  link: '/blog/2026/RLNeeds',
                },
                {
                  text: 'LAPose (2026-05-05)',
                  link: '/blog/2026/LAPose',
                },
                {
                  text: 'Physical Intelligence π 系列 VLA 技术演进报告（π0 → π0.7） (2026-04-16)',
                  link: '/blog/2026/pi0.7',
                },
                {
                  text: 'GigaWorldPolicy (2026-03-18)',
                  link: '/blog/2026/GigaWorldPolicy',
                },
                {
                  text: 'FastWAM (2026-03-17)',
                  link: '/blog/2026/FastWAM',
                },
                {
                  text: 'VJEPA2.1 (2026-03-15)',
                  link: '/blog/2026/VJEPA2.1',
                },
                {
                  text: 'RAENWM (2026-03-10)',
                  link: '/blog/2026/RAENWM',
                },
                {
                  text: 'SimVLA (2026-02-20)',
                  link: '/blog/2026/SimVLA',
                },
                {
                  text: 'DreamZero (2026-02-17)',
                  link: '/blog/2026/DreamZero',
                },
                {
                  text: 'RISE (2026-02-11)',
                  link: '/blog/2026/RISE',
                },
                {
                  text: 'Muon (2026-02-10)',
                  link: '/blog/2026/Muon',
                },
                {
                  text: 'DriveWorldVLA (2026-02-06)',
                  link: '/blog/2026/DriveWorldVLA',
                },
                {
                  text: 'DriveJEPA (2026-01-29)',
                  link: '/blog/2026/DriveJEPA',
                },
                {
                  text: 'LingBotVA (2026-01-29)',
                  link: '/blog/2026/LingBotVA',
                },
                {
                  text: 'C_RADIOv4 (2026-01-24)',
                  link: '/blog/2026/C_RADIOv4',
                },
                {
                  text: 'GeRo (2026-01-16)',
                  link: '/blog/2026/GeRo',
                },
                {
                  text: 'FineGrainedAlignmentedVLN (2026-01-10)',
                  link: '/blog/2026/FineGrainedAlignmentedVLN',
                },
                {
                  text: 'LearningLatentActionWM (2026-01-08)',
                  link: '/blog/2026/LearningLatentActionWM',
                },
                {
                  text: 'VLM4VLA (2026-01-06)',
                  link: '/blog/2026/VLM4VLA',
                },
                {
                  text: 'LowerMemory (2026-01-01)',
                  link: '/blog/2026/LowerMemory',
                },
                {
                  text: 'ManyForcing (2026-01-01)',
                  link: '/blog/2026/ManyForcing',
                },
                {
                  text: 'ModelSpeed (2026-01-01)',
                  link: '/blog/2026/ModelSpeed',
                },
                {
                  text: 'TrainingSpeed (2026-01-01)',
                  link: '/blog/2026/TrainingSpeed',
                },
              ],
            },
            {
              text: '📅 2025 年',
              collapsed: true,
              items: [
                {
                  text: 'JEPA_WM (2025-12-30)',
                  link: '/blog/2025/JEPA_WM',
                },
                {
                  text: 'VLAAN (2025-12-17)',
                  link: '/blog/2025/VLAAN',
                },
                {
                  text: 'Motus (2025-12-15)',
                  link: '/blog/2025/Motus',
                },
                {
                  text: 'VLJEPA (2025-12-11)',
                  link: '/blog/2025/VLJEPA',
                },
                {
                  text: 'RoboScapeR (2025-12-03)',
                  link: '/blog/2025/RoboScapeR',
                },
                {
                  text: 'ImprovedMeanFlows (2025-12-01)',
                  link: '/blog/2025/ImprovedMeanFlows',
                },
                {
                  text: 'pi0.6 (2025-11-18)',
                  link: '/blog/2025/pi0.6',
                },
                {
                  text: 'SurveyOnWorldModelsForEmbodiedAI (2025-10-19)',
                  link: '/blog/2025/SurveyOnWorldModelsForEmbodiedAI',
                },
                {
                  text: '从 PyTorch InternVL2.5-1B 到 Jetson Orin NX 16GB 的 VLA/VLM 嵌入式部署实战 (2025-10-04)',
                  link: '/blog/2025/2025-10-04-VLA_deploy',
                },
                {
                  text: 'VLACompare (2025-10-01)',
                  link: '/blog/2025/VLACompare',
                },
                {
                  text: 'WMcompare (2025-10-01)',
                  link: '/blog/2025/WMcompare',
                },
                {
                  text: 'LongScape (2025-09-26)',
                  link: '/blog/2025/LongScape',
                },
                {
                  text: 'VLA_embodedAI (2025-09-01)',
                  link: '/blog/2025/VLA_embodedAI',
                },
                {
                  text: 'MatrixGame2.0 (2025-08-18)',
                  link: '/blog/2025/MatrixGame2.0',
                },
                {
                  text: 'DinoWorld (2025-07-25)',
                  link: '/blog/2025/DinoWorld',
                },
                {
                  text: 'GoalVLA (2025-06-30)',
                  link: '/blog/2025/GoalVLA',
                },
                {
                  text: 'RoboScape (2025-06-29)',
                  link: '/blog/2025/RoboScape',
                },
                {
                  text: 'VLA0S (2025-06-21)',
                  link: '/blog/2025/VLA0S',
                },
                {
                  text: 'AutoVLA (2025-06-16)',
                  link: '/blog/2025/AutoVLA',
                },
                {
                  text: 'VJEPA2 (2025-06-11)',
                  link: '/blog/2025/VJEPA2',
                },
                {
                  text: 'TrackVLA (2025-05-29)',
                  link: '/blog/2025/TrackVLA',
                },
                {
                  text: 'BAGEL (2025-05-20)',
                  link: '/blog/2025/BAGEL',
                },
                {
                  text: 'MeanFlows (2025-05-19 19:52)',
                  link: '/blog/2025/MeanFlows',
                },
                {
                  text: 'Pi05 (2025-04-22)',
                  link: '/blog/2025/Pi05',
                },
                {
                  text: 'ChatVLA (2025-02-20)',
                  link: '/blog/2025/ChatVLA',
                },
                {
                  text: 'RAD (2025-02-18)',
                  link: '/blog/2025/RAD',
                },
                {
                  text: 'ROPE (2025-01-04)',
                  link: '/blog/2025/ROPE',
                },
                {
                  text: 'DeepSpeed (2025-01-03)',
                  link: '/blog/2025/DeepSpeed',
                },
                {
                  text: 'KVCACHE (2025-01-02)',
                  link: '/blog/2025/KVCACHE',
                },
                {
                  text: 'FSDP (2025-01-01)',
                  link: '/blog/2025/FSDP',
                },
              ],
            },
            {
              text: '📅 2024 年',
              collapsed: true,
              items: [
                {
                  text: 'pytorch_weights_datasets (2024-12-26)',
                  link: '/blog/2024/pytorch_weights_datasets',
                },
                {
                  text: 'FLIP (2024-12-11)',
                  link: '/blog/2024/FLIP',
                },
                {
                  text: 'NWM (2024-12-04)',
                  link: '/blog/2024/NWM',
                },
                {
                  text: 'pytorch_bug (2024-12-02)',
                  link: '/blog/2024/pytorch_bug',
                },
                {
                  text: 'DinoWM (2024-11-07)',
                  link: '/blog/2024/DinoWM',
                },
                {
                  text: 'Pi0 (2024-10-31)',
                  link: '/blog/2024/Pi0',
                },
                {
                  text: 'RDT1B (2024-10-10)',
                  link: '/blog/2024/RDT1B',
                },
                {
                  text: 'TinyVLM (2024-10-04)',
                  link: '/blog/2024/TinyVLM',
                },
                {
                  text: 'E2E (2024-10-03)',
                  link: '/blog/2024/E2E',
                },
                {
                  text: 'SelfForcing (2024-10-02)',
                  link: '/blog/2024/SelfForcing',
                },
                {
                  text: 'PredictiveWM_VS_GenerativeWM (2024-10-01)',
                  link: '/blog/2024/PredictiveWM_VS_GenerativeWM',
                },
                {
                  text: 'MAR (2024-06-17)',
                  link: '/blog/2024/MAR',
                },
                {
                  text: 'OpenVLA (2024-06-13)',
                  link: '/blog/2024/OpenVLA',
                },
                {
                  text: 'SparseDrive (2024-05-30)',
                  link: '/blog/2024/SparseDrive',
                },
                {
                  text: 'Survey_Occupancy (2024-05-08)',
                  link: '/blog/2024/Survey_Occupancy',
                },
                {
                  text: 'SparseAD (2024-04-10)',
                  link: '/blog/2024/SparseAD',
                },
                {
                  text: 'VAR (2024-04-03)',
                  link: '/blog/2024/VAR',
                },
                {
                  text: 'DROID (2024-03-19)',
                  link: '/blog/2024/DROID',
                },
                {
                  text: 'VJEPA (2024-02-15)',
                  link: '/blog/2024/VJEPA',
                },
              ],
            },
            {
              text: '📅 2023 年',
              collapsed: true,
              items: [
                {
                  text: 'Vidar (2023-12-29)',
                  link: '/blog/2023/Vidar',
                },
                {
                  text: 'SparseOcc (2023-12-28)',
                  link: '/blog/2023/SparseOcc',
                },
                {
                  text: 'QUAR-VLA (2023-12-22)',
                  link: '/blog/2023/QUAR-VLA',
                },
                {
                  text: 'Emu2 (2023-12-20)',
                  link: '/blog/2023/Emu2',
                },
                {
                  text: 'Sparse4DV3 (2023-11-20)',
                  link: '/blog/2023/Sparse4DV3',
                },
                {
                  text: '3DpointCloudGenerative (2023-10-10)',
                  link: '/blog/2023/3DpointCloudGenerative',
                },
                {
                  text: 'MAGVITV2 (2023-10-09)',
                  link: '/blog/2023/MAGVITV2',
                },
                {
                  text: 'WM (2023-10-03)',
                  link: '/blog/2023/WM',
                },
                {
                  text: 'verl (2023-10-01)',
                  link: '/blog/2023/verl',
                },
                {
                  text: 'DE_VS_SDE (2023-09-18)',
                  link: '/blog/2023/DE_VS_SDE',
                },
                {
                  text: 'RenderOcc (2023-09-18)',
                  link: '/blog/2023/RenderOcc',
                },
                {
                  text: 'SparseBEV (2023-8-18)',
                  link: '/blog/2023/2023-8-18-SparseBEV',
                },
                {
                  text: 'RT2 (2023-07-28)',
                  link: '/blog/2023/RT2',
                },
                {
                  text: 'Emu1 (2023-07-11)',
                  link: '/blog/2023/Emu1',
                },
                {
                  text: 'Sparse4DV2 (2023-5-23)',
                  link: '/blog/2023/2023-5-23-Sparse4DV2',
                },
                {
                  text: 'PrepareData (2023-05-02)',
                  link: '/blog/2023/2023-05-02-PrepareData',
                },
                {
                  text: 'ACT (2023-04-23)',
                  link: '/blog/2023/ACT',
                },
                {
                  text: 'StreamPETR (2023-03-21)',
                  link: '/blog/2023/StreamPETR',
                },
                {
                  text: 'DiffusionPolicy (2023-03-07)',
                  link: '/blog/2023/DiffusionPolicy',
                },
                {
                  text: 'IJEPA (2023-01-19)',
                  link: '/blog/2023/Ijepa',
                },
                {
                  text: 'DeramerV3 (2023-01-10)',
                  link: '/blog/2023/DeramerV3',
                },
                {
                  text: 'offlineonlineworldmodel (2023-01-01)',
                  link: '/blog/2023/offlineonlineworldmodel',
                },
              ],
            },
            {
              text: '📅 2022 年',
              collapsed: true,
              items: [
                {
                  text: 'RT1 (2022-12-13)',
                  link: '/blog/2022/RT1',
                },
                {
                  text: 'Sparse4D (2022-11-19)',
                  link: '/blog/2022/Sparse4D',
                },
                {
                  text: 'MILE (2022-10-14)',
                  link: '/blog/2022/MILE',
                },
                {
                  text: 'FlowMatching (2022-10-06)',
                  link: '/blog/2022/FlowMatching',
                },
                {
                  text: 'RectifiedFlow (2022-09-07)',
                  link: '/blog/2022/RectifiedFlow',
                },
                {
                  text: 'FlowAndDataGeneration (2022-09-05)',
                  link: '/blog/2022/FlowAndDataGeneration',
                },
                {
                  text: 'IRIS (2022-09-01)',
                  link: '/blog/2022/IRIS',
                },
                {
                  text: 'Wayformer (2022-07-12)',
                  link: '/blog/2022/Wayformer',
                },
                {
                  text: 'DayDreamer (2022-06-28)',
                  link: '/blog/2022/DayDreamer',
                },
                {
                  text: 'VPT (2022-06-23)',
                  link: '/blog/2022/VPT',
                },
                {
                  text: 'PETRV2 (2022-06-02)',
                  link: '/blog/2022/PETRV2',
                },
                {
                  text: 'Gato (2022-05-12)',
                  link: '/blog/2022/Gato',
                },
                {
                  text: 'mu_law (2022-05-12)',
                  link: '/blog/2022/mu_law',
                },
                {
                  text: 'PETR (2022-03-10)',
                  link: '/blog/2022/PETR',
                },
                {
                  text: 'TransDreamer (2022-02-19)',
                  link: '/blog/2022/TransDreamer',
                },
              ],
            },
            {
              text: '📅 2021 年',
              collapsed: true,
              items: [
                {
                  text: 'Mask2Former (2021-12-02)',
                  link: '/blog/2021/Mask2Former',
                },
                {
                  text: 'DETR3D (2021-10-13)',
                  link: '/blog/2021/DETR3D',
                },
                {
                  text: 'MaskFormer (2021-07-13)',
                  link: '/blog/2021/MaskFormer',
                },
                {
                  text: 'CaDDN (2021-03-01)',
                  link: '/blog/2021/CaDDN',
                },
                {
                  text: 'IntentNet (2021-01-20)',
                  link: '/blog/2021/IntentNet',
                },
              ],
            },
            {
              text: '📅 2020 年',
              collapsed: true,
              items: [
                {
                  text: 'SparseRCNN (2020-11-25)',
                  link: '/blog/2020/SparseRCNN',
                },
                {
                  text: 'DreamerV2 (2020-10-05)',
                  link: '/blog/2020/DreamerV2',
                },
                {
                  text: 'TNT (2020-08-19)',
                  link: '/blog/2020/TNT',
                },
                {
                  text: 'CVAE_VS_VAE (2020-07-06)',
                  link: '/blog/2020/CVAE_VS_VAE',
                },
                {
                  text: 'DETR (2020-05-26)',
                  link: '/blog/2020/DETR',
                },
                {
                  text: 'VectorNet (2020-05-08)',
                  link: '/blog/2020/VectorNet',
                },
              ],
            },
            {
              text: '📅 2019 年',
              collapsed: true,
              items: [
                {
                  text: 'DreamerV1 (2019-12-03)',
                  link: '/blog/2019/DreamerV1',
                },
                {
                  text: 'MultiPath (2019-10-12)',
                  link: '/blog/2019/MultiPath',
                },
                {
                  text: '从 PyTorch YOLO 到 Jetson Orin 的完整嵌入式部署实战 (2019-10-04)',
                  link: '/blog/2019/2019-10-04-Yolo_deploy',
                },
                {
                  text: 'l1l2 (2019-01-01)',
                  link: '/blog/2019/l1l2',
                },
              ],
            },
            {
              text: '📅 2018 年',
              collapsed: true,
              items: [
                {
                  text: 'PlaNet (2018-11-12)',
                  link: '/blog/2018/PlaNet',
                },
                {
                  text: 'WorldModels (2018-03-27)',
                  link: '/blog/2018/WorldModels',
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
              text: '📂 Understandings',
              collapsed: true,
              items: [
                {
                  text: 'Muon (2026-02-10)',
                  link: '/blog/2026/Muon',
                },
                {
                  text: 'LowerMemory (2026-01-01)',
                  link: '/blog/2026/LowerMemory',
                },
                {
                  text: 'ManyForcing (2026-01-01)',
                  link: '/blog/2026/ManyForcing',
                },
                {
                  text: 'ModelSpeed (2026-01-01)',
                  link: '/blog/2026/ModelSpeed',
                },
                {
                  text: 'TrainingSpeed (2026-01-01)',
                  link: '/blog/2026/TrainingSpeed',
                },
                {
                  text: 'VLACompare (2025-10-01)',
                  link: '/blog/2025/VLACompare',
                },
                {
                  text: 'WMcompare (2025-10-01)',
                  link: '/blog/2025/WMcompare',
                },
                {
                  text: 'ROPE (2025-01-04)',
                  link: '/blog/2025/ROPE',
                },
                {
                  text: 'DeepSpeed (2025-01-03)',
                  link: '/blog/2025/DeepSpeed',
                },
                {
                  text: 'KVCACHE (2025-01-02)',
                  link: '/blog/2025/KVCACHE',
                },
                {
                  text: 'FSDP (2025-01-01)',
                  link: '/blog/2025/FSDP',
                },
                {
                  text: 'E2E (2024-10-03)',
                  link: '/blog/2024/E2E',
                },
                {
                  text: 'SelfForcing (2024-10-02)',
                  link: '/blog/2024/SelfForcing',
                },
                {
                  text: 'PredictiveWM_VS_GenerativeWM (2024-10-01)',
                  link: '/blog/2024/PredictiveWM_VS_GenerativeWM',
                },
                {
                  text: '3DpointCloudGenerative (2023-10-10)',
                  link: '/blog/2023/3DpointCloudGenerative',
                },
                {
                  text: 'WM (2023-10-03)',
                  link: '/blog/2023/WM',
                },
                {
                  text: 'DE_VS_SDE (2023-09-18)',
                  link: '/blog/2023/DE_VS_SDE',
                },
                {
                  text: 'offlineonlineworldmodel (2023-01-01)',
                  link: '/blog/2023/offlineonlineworldmodel',
                },
                {
                  text: 'CVAE_VS_VAE (2020-07-06)',
                  link: '/blog/2020/CVAE_VS_VAE',
                },
                {
                  text: 'l1l2 (2019-01-01)',
                  link: '/blog/2019/l1l2',
                },
              ],
            },
            {
              text: '📂 VLA',
              collapsed: true,
              items: [
                {
                  text: 'LiteVLA (2026-07-05)',
                  link: '/blog/2026/LiteVLA',
                },
                {
                  text: 'VLX-Go (2026-07-05)',
                  link: '/blog/2026/VLXGo',
                },
                {
                  text: 'Qwen-RobotNav (2026-06-17)',
                  link: '/blog/2026/QwenRobotNav',
                },
                {
                  text: 'Physical Intelligence π 系列 VLA 技术演进报告（π0 → π0.7） (2026-04-16)',
                  link: '/blog/2026/pi0.7',
                },
                {
                  text: 'VLM4VLA (2026-01-06)',
                  link: '/blog/2026/VLM4VLA',
                },
                {
                  text: 'GoalVLA (2025-06-30)',
                  link: '/blog/2025/GoalVLA',
                },
                {
                  text: 'AutoVLA (2025-06-16)',
                  link: '/blog/2025/AutoVLA',
                },
                {
                  text: 'Pi05 (2025-04-22)',
                  link: '/blog/2025/Pi05',
                },
                {
                  text: 'ChatVLA (2025-02-20)',
                  link: '/blog/2025/ChatVLA',
                },
                {
                  text: 'Pi0 (2024-10-31)',
                  link: '/blog/2024/Pi0',
                },
                {
                  text: 'RDT1B (2024-10-10)',
                  link: '/blog/2024/RDT1B',
                },
                {
                  text: 'OpenVLA (2024-06-13)',
                  link: '/blog/2024/OpenVLA',
                },
                {
                  text: 'DROID (2024-03-19)',
                  link: '/blog/2024/DROID',
                },
                {
                  text: 'QUAR-VLA (2023-12-22)',
                  link: '/blog/2023/QUAR-VLA',
                },
                {
                  text: 'RT2 (2023-07-28)',
                  link: '/blog/2023/RT2',
                },
                {
                  text: 'DiffusionPolicy (2023-03-07)',
                  link: '/blog/2023/DiffusionPolicy',
                },
                {
                  text: 'RT1 (2022-12-13)',
                  link: '/blog/2022/RT1',
                },
                {
                  text: 'Gato (2022-05-12)',
                  link: '/blog/2022/Gato',
                },
              ],
            },
            {
              text: '📂 WorldModels',
              collapsed: true,
              items: [
                {
                  text: 'DriveJEPA (2026-01-29)',
                  link: '/blog/2026/DriveJEPA',
                },
                {
                  text: 'DinoWorld (2025-07-25)',
                  link: '/blog/2025/DinoWorld',
                },
                {
                  text: 'RoboScape (2025-06-29)',
                  link: '/blog/2025/RoboScape',
                },
                {
                  text: 'DinoWM (2024-11-07)',
                  link: '/blog/2024/DinoWM',
                },
                {
                  text: 'IJEPA (2023-01-19)',
                  link: '/blog/2023/Ijepa',
                },
                {
                  text: 'DeramerV3 (2023-01-10)',
                  link: '/blog/2023/DeramerV3',
                },
                {
                  text: 'IRIS (2022-09-01)',
                  link: '/blog/2022/IRIS',
                },
                {
                  text: 'TransDreamer (2022-02-19)',
                  link: '/blog/2022/TransDreamer',
                },
                {
                  text: 'DreamerV2 (2020-10-05)',
                  link: '/blog/2020/DreamerV2',
                },
                {
                  text: 'DreamerV1 (2019-12-03)',
                  link: '/blog/2019/DreamerV1',
                },
                {
                  text: 'WorldModels (2018-03-27)',
                  link: '/blog/2018/WorldModels',
                },
              ],
            },
            {
              text: '📂 AIGC',
              collapsed: true,
              items: [
                {
                  text: 'ImprovedMeanFlows (2025-12-01)',
                  link: '/blog/2025/ImprovedMeanFlows',
                },
                {
                  text: 'BAGEL (2025-05-20)',
                  link: '/blog/2025/BAGEL',
                },
                {
                  text: 'MeanFlows (2025-05-19 19:52)',
                  link: '/blog/2025/MeanFlows',
                },
                {
                  text: 'MAR (2024-06-17)',
                  link: '/blog/2024/MAR',
                },
                {
                  text: 'VAR (2024-04-03)',
                  link: '/blog/2024/VAR',
                },
                {
                  text: 'MAGVITV2 (2023-10-09)',
                  link: '/blog/2023/MAGVITV2',
                },
                {
                  text: 'FlowMatching (2022-10-06)',
                  link: '/blog/2022/FlowMatching',
                },
                {
                  text: 'RectifiedFlow (2022-09-07)',
                  link: '/blog/2022/RectifiedFlow',
                },
                {
                  text: 'FlowAndDataGeneration (2022-09-05)',
                  link: '/blog/2022/FlowAndDataGeneration',
                },
              ],
            },
            {
              text: '📂 Perception',
              collapsed: true,
              items: [
                {
                  text: 'SparseBEV (2023-8-18)',
                  link: '/blog/2023/2023-8-18-SparseBEV',
                },
                {
                  text: 'StreamPETR (2023-03-21)',
                  link: '/blog/2023/StreamPETR',
                },
                {
                  text: 'Sparse4D (2022-11-19)',
                  link: '/blog/2022/Sparse4D',
                },
                {
                  text: 'PETRV2 (2022-06-02)',
                  link: '/blog/2022/PETRV2',
                },
                {
                  text: 'PETR (2022-03-10)',
                  link: '/blog/2022/PETR',
                },
                {
                  text: 'DETR3D (2021-10-13)',
                  link: '/blog/2021/DETR3D',
                },
                {
                  text: 'CaDDN (2021-03-01)',
                  link: '/blog/2021/CaDDN',
                },
              ],
            },
            {
              text: '📂 Prediction',
              collapsed: true,
              items: [
                {
                  text: 'Wayformer (2022-07-12)',
                  link: '/blog/2022/Wayformer',
                },
                {
                  text: 'IntentNet (2021-01-20)',
                  link: '/blog/2021/IntentNet',
                },
                {
                  text: 'TNT (2020-08-19)',
                  link: '/blog/2020/TNT',
                },
                {
                  text: 'VectorNet (2020-05-08)',
                  link: '/blog/2020/VectorNet',
                },
                {
                  text: 'MultiPath (2019-10-12)',
                  link: '/blog/2019/MultiPath',
                },
              ],
            },
            {
              text: '📂 Deploy',
              collapsed: true,
              items: [
                {
                  text: 'Jetson Orin 部署实战：算力、显存与模型切分的决策 (2026-07-04)',
                  link: '/blog/2026/JetsonOrinDeployment',
                },
                {
                  text: '模型部署中的 Engine：从推理后端到生产落地 (2026-07-02)',
                  link: '/blog/2026/ModelDeploymentEngine',
                },
                {
                  text: '从 PyTorch InternVL2.5-1B 到 Jetson Orin NX 16GB 的 VLA/VLM 嵌入式部署实战 (2025-10-04)',
                  link: '/blog/2025/2025-10-04-VLA_deploy',
                },
                {
                  text: '从 PyTorch YOLO 到 Jetson Orin 的完整嵌入式部署实战 (2019-10-04)',
                  link: '/blog/2019/2019-10-04-Yolo_deploy',
                },
              ],
            },
            {
              text: '📂 reading',
              collapsed: true,
              items: [
                {
                  text: 'pytorch_weights_datasets (2024-12-26)',
                  link: '/blog/2024/pytorch_weights_datasets',
                },
                {
                  text: 'Sparse4DV3 (2023-11-20)',
                  link: '/blog/2023/Sparse4DV3',
                },
                {
                  text: 'Sparse4DV2 (2023-5-23)',
                  link: '/blog/2023/2023-5-23-Sparse4DV2',
                },
                {
                  text: 'MaskFormer (2021-07-13)',
                  link: '/blog/2021/MaskFormer',
                },
              ],
            },
            {
              text: '📂 WAM',
              collapsed: true,
              items: [
                {
                  text: 'FastWAM (2026-03-17)',
                  link: '/blog/2026/FastWAM',
                },
                {
                  text: 'RAENWM (2026-03-10)',
                  link: '/blog/2026/RAENWM',
                },
                {
                  text: 'LingBotVA (2026-01-29)',
                  link: '/blog/2026/LingBotVA',
                },
                {
                  text: 'NWM (2024-12-04)',
                  link: '/blog/2024/NWM',
                },
              ],
            },
            {
              text: '📂 Occupancy',
              collapsed: true,
              items: [
                {
                  text: 'Survey_Occupancy (2024-05-08)',
                  link: '/blog/2024/Survey_Occupancy',
                },
                {
                  text: 'SparseOcc (2023-12-28)',
                  link: '/blog/2023/SparseOcc',
                },
                {
                  text: 'RenderOcc (2023-09-18)',
                  link: '/blog/2023/RenderOcc',
                },
              ],
            },
            {
              text: '📂 e2e',
              collapsed: true,
              items: [
                {
                  text: 'SparseDrive (2024-05-30)',
                  link: '/blog/2024/SparseDrive',
                },
                {
                  text: 'SparseAD (2024-04-10)',
                  link: '/blog/2024/SparseAD',
                },
                {
                  text: 'Vidar (2023-12-29)',
                  link: '/blog/2023/Vidar',
                },
              ],
            },
            {
              text: '📂 Detection',
              collapsed: true,
              items: [
                {
                  text: 'SparseRCNN (2020-11-25)',
                  link: '/blog/2020/SparseRCNN',
                },
                {
                  text: 'DETR (2020-05-26)',
                  link: '/blog/2020/DETR',
                },
              ],
            },
            {
              text: '📂 EmbodiedAI',
              collapsed: true,
              items: [
                {
                  text: 'ACT (2023-04-23)',
                  link: '/blog/2023/ACT',
                },
                {
                  text: 'Gato (2022-05-12)',
                  link: '/blog/2022/Gato',
                },
              ],
            },
            {
              text: '📂 Agent',
              collapsed: true,
              items: [
                {
                  text: 'Agent、Skill 与 Tool：三者的关系与协作边界 (2026-07-02)',
                  link: '/blog/2026/AgentSkillTool',
                },
                {
                  text: 'Agent 全景解析：从概念、架构到具身智能的工程实践 (2026-06-28)',
                  link: '/blog/2026/AgentComprehensiveGuide',
                },
              ],
            },
            {
              text: '📂 机器人',
              collapsed: true,
              items: [
                {
                  text: '关节限位 (2026-06-28)',
                  link: '/blog/2026/JointLimits',
                },
                {
                  text: '运动原语：机器人动作世界的"字母表 (2026-06-28)',
                  link: '/blog/2026/MotionPrimitives',
                },
              ],
            },
            {
              text: '📂 工具与约定',
              collapsed: true,
              items: [
                {
                  text: 'Neovim + iTerm2 工作流 (2026-06-12)',
                  link: '/blog/notes/neovim-workflow',
                },
                {
                  text: '写作约定 (2026-06-07)',
                  link: '/blog/notes/writing-guide',
                },
              ],
            },
            {
              text: '📂 Segmentation',
              collapsed: true,
              items: [
                {
                  text: 'Mask2Former (2021-12-02)',
                  link: '/blog/2021/Mask2Former',
                },
              ],
            },
            {
              text: '📂 Tricks',
              collapsed: true,
              items: [
                {
                  text: 'mu_law (2022-05-12)',
                  link: '/blog/2022/mu_law',
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
                  link: '/blog/2024/pytorch_bug',
                },
              ],
            },
            {
              text: '📂 Thinking',
              collapsed: true,
              items: [
                {
                  text: 'VLA_embodedAI (2025-09-01)',
                  link: '/blog/2025/VLA_embodedAI',
                },
              ],
            },
            {
              text: '📂 VLN',
              collapsed: true,
              items: [
                {
                  text: 'FineGrainedAlignmentedVLN (2026-01-10)',
                  link: '/blog/2026/FineGrainedAlignmentedVLN',
                },
              ],
            },
            {
              text: '📂 others',
              collapsed: true,
              items: [
                {
                  text: '高德地图语音导航逻辑与纯 Python Demo (2026-07-01)',
                  link: '/blog/2026/GaodeVoiceNavDemo',
                },
                {
                  text: 'RLNeeds (2026-06-02)',
                  link: '/blog/2026/RLNeeds',
                },
                {
                  text: 'LAPose (2026-05-05)',
                  link: '/blog/2026/LAPose',
                },
                {
                  text: 'GigaWorldPolicy (2026-03-18)',
                  link: '/blog/2026/GigaWorldPolicy',
                },
                {
                  text: 'VJEPA2.1 (2026-03-15)',
                  link: '/blog/2026/VJEPA2.1',
                },
                {
                  text: 'SimVLA (2026-02-20)',
                  link: '/blog/2026/SimVLA',
                },
                {
                  text: 'DreamZero (2026-02-17)',
                  link: '/blog/2026/DreamZero',
                },
                {
                  text: 'RISE (2026-02-11)',
                  link: '/blog/2026/RISE',
                },
                {
                  text: 'DriveWorldVLA (2026-02-06)',
                  link: '/blog/2026/DriveWorldVLA',
                },
                {
                  text: 'C_RADIOv4 (2026-01-24)',
                  link: '/blog/2026/C_RADIOv4',
                },
                {
                  text: 'GeRo (2026-01-16)',
                  link: '/blog/2026/GeRo',
                },
                {
                  text: 'LearningLatentActionWM (2026-01-08)',
                  link: '/blog/2026/LearningLatentActionWM',
                },
                {
                  text: 'JEPA_WM (2025-12-30)',
                  link: '/blog/2025/JEPA_WM',
                },
                {
                  text: 'VLAAN (2025-12-17)',
                  link: '/blog/2025/VLAAN',
                },
                {
                  text: 'Motus (2025-12-15)',
                  link: '/blog/2025/Motus',
                },
                {
                  text: 'VLJEPA (2025-12-11)',
                  link: '/blog/2025/VLJEPA',
                },
                {
                  text: 'RoboScapeR (2025-12-03)',
                  link: '/blog/2025/RoboScapeR',
                },
                {
                  text: 'pi0.6 (2025-11-18)',
                  link: '/blog/2025/pi0.6',
                },
                {
                  text: 'SurveyOnWorldModelsForEmbodiedAI (2025-10-19)',
                  link: '/blog/2025/SurveyOnWorldModelsForEmbodiedAI',
                },
                {
                  text: 'LongScape (2025-09-26)',
                  link: '/blog/2025/LongScape',
                },
                {
                  text: 'MatrixGame2.0 (2025-08-18)',
                  link: '/blog/2025/MatrixGame2.0',
                },
                {
                  text: 'VLA0S (2025-06-21)',
                  link: '/blog/2025/VLA0S',
                },
                {
                  text: 'VJEPA2 (2025-06-11)',
                  link: '/blog/2025/VJEPA2',
                },
                {
                  text: 'TrackVLA (2025-05-29)',
                  link: '/blog/2025/TrackVLA',
                },
                {
                  text: 'RAD (2025-02-18)',
                  link: '/blog/2025/RAD',
                },
                {
                  text: 'FLIP (2024-12-11)',
                  link: '/blog/2024/FLIP',
                },
                {
                  text: 'TinyVLM (2024-10-04)',
                  link: '/blog/2024/TinyVLM',
                },
                {
                  text: 'VJEPA (2024-02-15)',
                  link: '/blog/2024/VJEPA',
                },
                {
                  text: 'Emu2 (2023-12-20)',
                  link: '/blog/2023/Emu2',
                },
                {
                  text: 'verl (2023-10-01)',
                  link: '/blog/2023/verl',
                },
                {
                  text: 'Emu1 (2023-07-11)',
                  link: '/blog/2023/Emu1',
                },
                {
                  text: 'MILE (2022-10-14)',
                  link: '/blog/2022/MILE',
                },
                {
                  text: 'DayDreamer (2022-06-28)',
                  link: '/blog/2022/DayDreamer',
                },
                {
                  text: 'VPT (2022-06-23)',
                  link: '/blog/2022/VPT',
                },
                {
                  text: 'PlaNet (2018-11-12)',
                  link: '/blog/2018/PlaNet',
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
