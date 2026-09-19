export const topics = [
  {
    id: 'thinking',
    name: 'Thinking · 思考',
    description: '个人思考、原理理解、技术判断与跨领域观察。',
  },
  {
    id: 'world-models',
    name: '世界模型',
    description: '从潜在动力学到预测式表征，理解模型如何学习环境。',
  },
  {
    id: 'navigation',
    name: '机器人导航',
    description: '视觉导航、空间理解与跨本体决策。',
  },
  {
    id: 'vla',
    name: 'VLA 与机器人学习',
    description: '让视觉、语言和动作连接起来。',
  },
  {
    id: 'driving',
    name: '自动驾驶',
    description: '端到端规划、闭环学习与驾驶系统。',
  },
  {
    id: 'deployment',
    name: '部署与性能优化',
    description: '量化、推理加速、显存与硬件实践。',
  },
  {
    id: 'perception',
    name: '视觉感知',
    description: '检测、分割、三维表示与视觉基础模型。',
  },
  {
    id: 'generation',
    name: '生成模型',
    description: '扩散、流匹配与多模态生成。',
  },
  {
    id: 'agents',
    name: '智能体',
    description: '工具调用、任务规划与具身智能体。',
  },
  {
    id: 'foundations',
    name: '基础与工程',
    description: '数学原理、训练方法和研究工具。',
  },
]
export const articleTypes = ['论文精读', '技术分析', '工程实践']
const categoryTopics = {
  WorldModels: 'world-models',
  WAM: 'world-models',
  VLN: 'navigation',
  VLA: 'vla',
  EmbodiedAI: 'vla',
  机器人: 'vla',
  自动驾驶: 'driving',
  e2e: 'driving',
  Deploy: 'deployment',
  Perception: 'perception',
  Prediction: 'driving',
  Detection: 'perception',
  Segmentation: 'perception',
  Occupancy: 'perception',
  Vision: 'perception',
  AIGC: 'generation',
  Agents: 'agents',
  仿真: 'world-models',
}
const overrides = {
  planet: 'world-models',
  vpt: 'vla',
  daydreamer: 'world-models',
  mile: 'driving',
  emu1: 'generation',
  emu2: 'generation',
  verl: 'foundations',
  vjepa: 'world-models',
  tinyvlm: 'deployment',
  flip: 'world-models',
  rad: 'driving',
  trackvla: 'navigation',
  vjepa2: 'world-models',
  vla0s: 'vla',
  matrixgame20: 'world-models',
  longscape: 'world-models',
  surveyonworldmodelsforembodiedai: 'world-models',
  pi06: 'vla',
  roboscaper: 'world-models',
  vljepa: 'perception',
  motus: 'vla',
  vlaan: 'vla',
  'jepa-wm': 'world-models',
  learninglatentactionwm: 'world-models',
  gero: 'driving',
  'c-radiov4': 'perception',
  driveworldvla: 'driving',
  rise: 'world-models',
  dreamzero: 'world-models',
  simvla: 'vla',
  vjepa21: 'world-models',
  gigaworldpolicy: 'world-models',
  lapose: 'perception',
  rlneeds: 'thinking',
  gaodevoicenavdemo: 'navigation',
  'nvidia-gpu-comparison': 'deployment',
  fsdp: 'deployment',
  kvcache: 'deployment',
  deepspeed: 'deployment',
  lowermemory: 'deployment',
  modelspeed: 'deployment',
  trainingspeed: 'deployment',
  'pytorch-weights-datasets': 'foundations',
  offlineonlineworldmodel: 'world-models',
  'predictivewm-vs-generativewm': 'world-models',
  wmcompare: 'world-models',
  wm: 'world-models',
  nwm: 'navigation',
  'vla-embodedai': 'vla',
  vlacompare: 'vla',
  act: 'vla',
  gato: 'vla',
  '3dpointcloudgenerative': 'generation',
  selfforcing: 'generation',
  manyforcing: 'world-models',
  e2e: 'driving',
  sparse4dv2: 'perception',
  sparse4dv3: 'perception',
  maskformer: 'perception',
  simrealitygap: 'world-models',
  motionprimitives: 'navigation',
  jointlimits: 'vla',
}
export function asList(value) {
  if (Array.isArray(value))
    return value
      .map(String)
      .map((x) => x.trim())
      .filter(Boolean)
  return typeof value === 'string'
    ? value
        .replace(/^\[|\]$/g, '')
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean)
    : []
}
export function inferTopic(frontmatter, file) {
  if (frontmatter.topic) return frontmatter.topic
  if (
    asList(frontmatter.categories).some((category) =>
      /^(thinkings?|understandings?)$/i.test(category),
    )
  )
    return 'thinking'
  const slug = file
    .replace(/^.*\//, '')
    .replace(/^\d{4}-\d{2}-\d{2}-/, '')
    .replace(/\.md$/, '')
    .toLowerCase()
  return (
    overrides[slug] ||
    asList(frontmatter.categories)
      .map((c) => categoryTopics[c])
      .find(Boolean) ||
    'foundations'
  )
}
export function inferType(frontmatter, body, file) {
  if (frontmatter.type) return frontmatter.type
  if (
    /demo|guide|workflow|deployment|方案|工作流|实战|教程/i.test(
      `${file} ${frontmatter.title}`,
    )
  )
    return '工程实践'
  return /arxiv\.org|论文链接|论文题目|paper link|论文标题/i.test(body)
    ? '论文精读'
    : '技术分析'
}
export function inferTags(frontmatter, body) {
  if (frontmatter.tags) return asList(frontmatter.tags)
  const text = body.slice(0, 8000)
  const tags = [
    'PPO',
    'LoRA',
    'Transformer',
    'Diffusion',
    'Flow Matching',
    'JEPA',
    'TensorRT',
    'CUDA',
    'INT8',
    'INT4',
    'PyTorch',
    'Sim-to-Real',
  ]
  return tags
    .filter((tag) =>
      new RegExp(`\\b${tag.replaceAll(' ', '[- ]?')}\\b`, 'i').test(text),
    )
    .slice(0, 5)
}
