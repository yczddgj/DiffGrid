// 基础颜色调色板 - 循环使用的10种颜色
const BASE_COLORS = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
];

export const COLORS = {
  // 散点图颜色
  scatter: {
    selected: '#ff6b6b',
    baseClasses: BASE_COLORS,
    current: '#1f77b4',
    reference: '#ef5350',
  },

  // 热力图颜色
  heatmap: {
    classHeatmap: 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)',
    densityDifference: 'linear-gradient(to right, #1f77b4, #87ceeb, #f9fafb, #ffa500, #ff7f00)',
    default: 'linear-gradient(to right, #000000, #ffffff)',
  },

  // 按钮 & UI 元素颜色
  ui: {
    // 1. current/reference 基本色（用于下拉框填充、采样图片轮廓等）
    source: {
      current: '#00a3a3',   // 青绿色，区别于 density 蓝
      reference: '#a36f00', // 棕金色，区别于 density 橙
    },

    // 2. 黑灰白中性配色（普通按钮状态）
    neutral: {
      normal: {
        bg: '#f5f5f5',      // 浅灰背景
        text: '#333333',    // 深灰文字
        border: '#cccccc',  // 灰色边框
      },
      hover: {
        bg: '#e0e0e0',
        text: '#111111',
        border: '#999999',
      },
    },

    // 3. 开关型按钮颜色（非 current/reference）
    toggle: {
      on: {
        bg: '#2f2f2f',     // 深灰，现代感
        text: '#ffffff',   // 纯白字体
        border: '#2f2f2f', // 与背景一致，减少边框感
      },
      onHover: {
        bg: '#3a3a3a',     // 稍微提亮，体现悬停
        text: '#ffffff',
        border: '#3a3a3a',
      },
      off: {
        bg: '#fafafa',     // 轻柔白灰
        text: '#444444',   // 柔和深灰字
        border: '#e5e5e5', // 细淡边框
      },
      offHover: {
        bg: '#f0f0f0',     // 稍微加深，悬停感
        text: '#222222',
        border: '#d5d5d5',
      },
    },

  },
};

// ===== 辅助函数 =====
export const getClassColor = (classIndex: number): string =>
  COLORS.scatter.baseClasses[classIndex % COLORS.scatter.baseClasses.length];

export const getClassColors = (classCount: number): string[] =>
  Array.from({ length: classCount }, (_, i) =>
    COLORS.scatter.baseClasses[i % COLORS.scatter.baseClasses.length]
  );

export const getSourceColor = (source: 'current' | 'reference'): string =>
  source === 'current' ? COLORS.scatter.current : COLORS.scatter.reference;

export const getSourceUIFill = (source: 'current' | 'reference'): string =>
  source === 'current' ? COLORS.ui.source.current : COLORS.ui.source.reference;

export const getHeatmapGradient = (backgroundEncoding: string): string => {
  switch (backgroundEncoding) {
    case 'Class-Heatmap':
      return COLORS.heatmap.classHeatmap;
    case 'Density-Difference':
      return COLORS.heatmap.densityDifference;
    default:
      return COLORS.heatmap.default;
  }
};
