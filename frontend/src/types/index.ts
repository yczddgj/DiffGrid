// 视图配置接口
export interface ViewConfig {
  id: number;
  title: string;
  currentSlice: string;
  referenceSlice: string;
  dimensionReduction: {
    method: 'umap' | 'pca' | 'tsne';
    fitMode: 'joint' | 's1' | 's2';
  };
  visualEncoding: {
    background: 'None' | 'Class-Heatmap' | 'Density-Difference';
    scatter: 'None' | 'Class' | 'Current or Reference';
  };
  option: {
    scatterMode: 'Current' | 'Reference' | 'Current & Reference';
    showImages: boolean;
  };
  operation: {
    fraction: number;
    saveSliceName: string;
  };
  lassoShape: number[][];
  lassoSelection: number[];
  zoom: number;
  pan: { x: number; y: number };
  embedding: number[][];
  heatmap: HeatmapData[];
  scatterData: ScatterData[];
  sampleImages: SampleImage[];
  sampleImagesReference: SampleImage[];
  info: ViewInfo;
  toolMode: 'lasso' | 'move';
  classCount: number;
}

// 热力图数据
export interface HeatmapData {
  type: string; // 'back' | 'center' | 'none'
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  value: number;
  id: number;
}

// 散点数据
export interface ScatterData {
  id: number;
  x: number;
  y: number;
  sliceName: string;
  label: number;
  value: number;
  imageUrl: string | null;
  imageX: number | null;
  imageY: number | null;
  thumbUrl: string | null;
  source: 'current' | 'reference';
  selected: boolean;
  title: string | null; // 图表标题
  key_word: string | null; // 图表关键词
}

// 采样图片
export interface SampleImage {
  id: number;
  imageUrl: string;
  thumbUrl: string;
}

// 视图信息
export interface ViewInfo {
  currentSliceShown: number;
  currentSliceTotal: number;
  referenceSliceShown: number;
  referenceSliceTotal: number;
  selectionShown: number;
  selectionTotal: number;
}

// 核心集选择配置
export interface CoresetSelection {
  method: string;
  fraction: number;
  sourceSlice: string;
  targetSlice: string;
}

// 训练配置
export interface Training {
  model: string;
  seed: string | number;
  trainSlice: string;
  evalSlice: string;
  status: 'idle' | 'running' | 'finished' | 'error';
  result: TrainingResult[];
}

// 训练结果
export interface TrainingResult {
  model: string;
  trainSlice: string;
  evalSlice: string;
  accuracy: number;
  loss: number;
  time: string;
}

// 通知
export interface Notification {
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
  duration?: number;
}
