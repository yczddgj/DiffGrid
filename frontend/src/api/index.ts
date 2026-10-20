import axios from 'axios';
import type {
  ViewConfig,
  CoresetSelection,
  Training
} from '../types';
import { API_BASE_URL } from '../config';

// 创建 axios 实例
const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 100000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// 切片管理
export const getSlices = () => api.get('/slices');

// 方法列表获取
export const getCoresetMethods = () => api.get('/coreset_methods');
export const getTrainingModels = () => api.get('/training_models');

// 核心集选择
export const selectCoreset = (data: CoresetSelection) =>
  api.post('/select_coreset', data);

// 训练
export const startTraining = (data: Training) =>
  api.post('/train', data);

export const getTrainingResults = () => api.get('/train_results');

// 可视化数据
export const getHeatmap = (view: number, type: string) =>
  api.get('/heatmap', {
    params: { view, type }
  });

export const getScatter = (view: number, x1: number, x2: number, y1: number, y2: number) =>
  api.get('/scatter', {
    params: { view, x1, x2, y1, y2 }
  });

export const getLassoSampleImages = (view: number) =>
  api.get('/lasso_sample_images', {
    params: { view }
  });

export const getInfo = (view: number) =>
  api.get('/info', {
    params: { view }
  });

// 交互
export const lassoSelectPoints = (view: number, lassoShape: number[][]) =>
  api.post('/lasso_select_points', { view, lassoShape });

export const copyLasso = (view: number, lassoShape: number[][]) =>
  api.post('/copy_lasso', { view, lassoShape });



export const copyPoints = (view: number) =>
  api.post('/copy_points', { view });

export const pastePoints = (view: number) =>
  api.post('/paste_points', { view });

export const deleteLassoPoints = (view: number) =>
  api.post('/delete_lasso_points', { view });

export const upsampleLassoPoints = (view: number) =>
  api.post('/upsample_lasso_points', { view });

export const downsampleLassoPoints = (view: number) =>
  api.post('/downsample_lasso_points', { view });

// 切片操作
export const cutSlice = (view: number, fraction: number) =>
  api.post('/cut_slice', { view, fraction });

export const saveSlice = (view: number, sliceName: string) =>
  api.post('/save_slice', { view, sliceName });

// 模式设置
export const setMode = (mode: string) => api.post('/set_mode', { mode });

// 视图管理
export const addView = () => api.post('/add_view');

export const deleteView = (viewId: number) =>
  api.post('/delete_view', { viewId });

export const getViews = () => api.get('/views');

export const getView = (viewId: number) => api.get(`/view/${viewId}`);

export const updateView = (viewId: number, data: Partial<ViewConfig>) =>
  api.put(`/view/${viewId}`, data);

export default api;
