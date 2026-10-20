import { defineStore } from 'pinia';
import type {
  ViewConfig,
  CoresetSelection,
  Training,
  Notification
} from '../types';
import * as api from '../api';

export const useMainStore = defineStore('main', {
  state: () => ({
    views: [] as ViewConfig[],
    activeViewId: 1,
    mode: null as string | null,
    sliceList: [] as string[],
    coresetSelection: {
      method: '',
      fraction: 0.1,
      sourceSlice: '',
      targetSlice: ''
    } as CoresetSelection,
    coresetMethodList: [] as string[],
    training: {
      model: '',
      seed: 42,
      trainSlice: '',
      evalSlice: '',
      status: 'idle'
    } as Training,
    trainingModelList: [] as string[],
    trainingResults: [] as Array<{
      model: string;
      train_slice: string;
      eval_slice: string;
      acc: number;
      fraction: number;
      method: string;
      time: string;
    }>,
    resultLoading: false,
    notification: null as Notification | null,
    loading: false,
    // 全局复制状态
    globalCopiedLasso: null as number[][] | null,
    globalCopiedPoints: false
  }),

  actions: {
    // 初始化应用数据
    async initializeApp() {
      try {
        this.setLoading(true);

        // 并行获取所有基础数据
        const [
          slicesRes,
          coresetMethodsRes,
          trainingModelsRes,
          trainingResultsRes
        ] = await Promise.all([
          api.getSlices(),
          api.getCoresetMethods(),
          api.getTrainingModels(),
          api.getTrainingResults()
        ]);

        // 更新数据列表
        this.sliceList = slicesRes.data.data || [];
        this.coresetMethodList = coresetMethodsRes.data.data || [];
        this.trainingModelList = trainingModelsRes.data.data || [];
        this.trainingResults = trainingResultsRes.data.data || [];

        // 设置默认值
        if (this.sliceList.length > 0) {
          this.coresetSelection.sourceSlice = this.sliceList[0];
          this.training.trainSlice = this.sliceList[0];
          if (this.sliceList.length > 1) {
            this.training.evalSlice = this.sliceList[1];
          }
        }

        if (this.coresetMethodList.length > 0) {
          this.coresetSelection.method = this.coresetMethodList[0];
        }

        if (this.trainingModelList.length > 0) {
          this.training.model = this.trainingModelList[0];
          console.log(this.trainingModelList);
        }

        console.log('应用初始化完成', {
          slices: this.sliceList.length,
          coresetMethods: this.coresetMethodList.length,
          trainingModels: this.trainingModelList.length
        });
      } catch (error) {
        console.error('初始化应用失败:', error);
        this.setNotification({
          type: 'error',
          message: '初始化应用失败，请检查后端服务是否启动',
          duration: 5000
        });
      } finally {
        this.setLoading(false);
      }
    },

    // 视图管理
    async addView() {
      try {
        const response = await api.addView();
        if (response.data.code === 0) {
          // 从后端获取新创建的视图配置
          const viewId = this.views.length + 1;
          const viewResponse = await api.getView(viewId);

          if (viewResponse.data.code === 0 && viewResponse.data.data) {
            // 使用后端返回的完整配置，确保类别数正确
            const backendConfig = viewResponse.data.data;
            console.log('back',backendConfig)
            const newView: ViewConfig = {
              id: viewId,
              title: `View ${viewId}`,
              currentSlice: backendConfig.currentSlice || '',
              referenceSlice: backendConfig.referenceSlice || '',
              dimensionReduction: {
                method: 'umap',
                fitMode: 'joint'
              },
              visualEncoding: {
                background: backendConfig.visualEncoding?.background,
                scatter: backendConfig.visualEncoding?.scatter
              },
              option: {
                scatterMode: backendConfig.option?.scatterMode,
                showImages: false
              },
              operation: {
                fraction:5000,
                saveSliceName: ''
              },
              lassoShape: [],
              lassoSelection: [],
              zoom: 1.0,
              pan: { x: 0, y: 0 },
              toolMode: 'lasso',
              embedding: [],
              heatmap: [],
              scatterData: [],
              sampleImages: [],
              sampleImagesReference: [],
              info: {
                currentSliceShown: 0,
                currentSliceTotal: 0,
                referenceSliceShown: 0,
                referenceSliceTotal: 0,
                selectionShown: 0,
                selectionTotal: 0
              },
              // 使用后端检测的类别数，如果没有则使用默认值
              classCount: backendConfig.classCount || 10
            };
            this.views.push(newView);
          }
        }
      } catch (error) {
        console.error('添加视图失败:', error);
        throw error;
      }
    },

    deleteView(viewId: number) {
      const index = this.views.findIndex(view => view.id === viewId);
      if (index !== -1) {
        this.views.splice(index, 1);
        // 重新编号
        this.views.forEach((view, i) => {
          view.id = i + 1;
          view.title = `View ${i + 1}`;
        });
      }
      // 如果视图为空，mode设为none
      if (this.views.length === 0) {
        this.mode = null;
      }
    },

    // 刷新切片列表
    async refreshSlices() {
      try {
        const response = await api.getSlices();
        console.log('切片列表:', response.data); //这里结果正常
        if (response.data.code === 0) {
          const newSliceList = response.data.data || [];
          console.log('更新前 sliceList:', this.sliceList);
          console.log('新的 sliceList:', newSliceList);
          this.sliceList = newSliceList;
          console.log('更新后 sliceList:', this.sliceList);
        }
        else{
          console.error('获取切片列表失败，后端返回错误代码:', response.data.code);
        }
      } catch (error) {
        console.error('刷新切片列表失败:', error);
      }
    },

    // 刷新训练结果
    async refreshTrainingResults() {
      try {
        this.resultLoading = true;
        const response = await api.getTrainingResults();
        if (response.data.code === 0) {
          this.trainingResults = response.data.data || [];
        }
      } catch (error) {
        console.error('刷新训练结果失败:', error);
        this.setNotification({
          type: 'error',
          message: '获取训练结果失败',
          duration: 3000
        });
      } finally {
        this.resultLoading = false;
      }
    },

    // UI 状态管理
    setLoading(loading: boolean) {
      this.loading = loading;
    },

    async setMode(mode: string) {
      try {
      await api.setMode(mode);
      console.log(`模式已设置为: ${mode}`);
      // 设置成功后自动创建第一个 view（仅当 views 为空时）
      if (this.views.length === 0) {
        await this.addView();
        this.mode = mode;
      }
      } catch (error) {
      console.error('设置模式失败:', error);
      throw error;
      }
    },

    setNotification(notification: Notification) {
      this.notification = notification;
      if (notification.duration && notification.duration > 0) {
        setTimeout(() => {
          this.notification = null;
        }, notification.duration);
      }
    },

    clearNotification() {
      this.notification = null;
    },

    // 全局复制粘贴状态管理
    setGlobalCopiedLasso(lassoShape: number[][] | null) {
      this.globalCopiedLasso = lassoShape;
    },

    setGlobalCopiedPoints(copied: boolean) {
      this.globalCopiedPoints = copied;
    },

    // 根据数据集名称检测类别数（后端会从实际数据中检测）
    detectClassCount(_sliceName: string): number {
      // 前端只提供默认值，实际的类别数由后端从数据中检测
      // 后端会在设置切片时自动更新正确的类别数
      return 10; // 默认值，实际值由后端检测
    },

    // 更新视图的类别数
    updateViewClassCount(viewId: number, sliceName: string) {
      const view = this.views.find(v => v.id === viewId);
      if (view) {
        view.classCount = this.detectClassCount(sliceName);
      }
    }
  }
});
