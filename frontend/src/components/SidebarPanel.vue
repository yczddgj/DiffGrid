<template>
  <aside class="w-auto bg-white h-auto flex flex-col rounded-3xl shadow-sm m-4 mr-0">
    <!-- 系统标题 -->
    <div class="p-6 text-center rounded-t-lg">
      <h2 class="text-2xl font-bold text-gray-900">Vis System</h2>
    </div>

    <!-- 滚动内容区域 -->
    <div class="flex-1 overflow-y-auto px-6">
      <!-- View Control -->
      <section class="mb-8">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">View Control</h3>
        <div class="bg-gray-200 rounded-lg p-4">
          <div class="grid grid-cols-3 gap-3">
            <!-- 现有视图卡片 -->
            <div v-for="view in views" :key="view.id"
              class="bg-white rounded-lg p-4 relative h-20 flex items-center justify-center border border-gray-300">
              <span class="text-sm font-medium text-gray-700">{{ view.title }}</span>
              <button @click="deleteView(view.id)" class="absolute top-2 right-2 text-gray-400 hover:text-gray-600">
                <XMarkIcon class="h-4 w-4" />
              </button>
            </div>

            <!-- 添加视图按钮 -->
            <button @click="addView"
              class="bg-gray-300 rounded-lg h-20 flex items-center justify-center hover:bg-gray-400 transition-colors">
              <PlusIcon class="h-6 w-6 text-gray-600" />
            </button>
          </div>
        </div>
      </section>

      <!-- Coreset Selection -->
      <section class="mb-8" style="display: none;">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">Coreset Selection</h3>
          <button @click="applyCoreset"
            class="px-4 py-2 rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2 bg-gray-800 text-white hover:bg-gray-900 focus:ring-gray-500">
            Apply
          </button>
        </div>
        <div class="space-y-4">
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Selection Method</label>
              <select v-model="coresetSelection.method"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option v-for="method in coresetMethodList" :key="method" :value="method">
                  {{ method }}
                </option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Fraction</label>
              <input v-model.number="coresetSelection.fraction" type="number" min="0" max="1" step="0.1"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="0.1" />
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Source Slice</label>
              <select v-model="coresetSelection.sourceSlice"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option v-for="slice in sliceList" :key="slice" :value="slice">
                  {{ slice }}
                </option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Target Slice</label>
              <input v-model="coresetSelection.targetSlice" type="text"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="EAGLE-train-SES-0.1" />
            </div>
          </div>


        </div>
      </section>

      <!-- Model Training -->
      <section class="mb-8" v-if="mode == 'coreset'" style="display: none;">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Model Training</h3>
        <div class="space-y-4">
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Model</label>
              <select v-model="training.model"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option v-for="model in trainingModelList" :key="model" :value="model">
                  {{ model }}
                </option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Seed</label>
              <input v-model="training.seed" type="number"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="0" />
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Training Slice</label>
              <select v-model="training.trainSlice"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option value="" disabled selected>training set</option>
                <option v-for="slice in sliceList" :key="slice" :value="slice">
                  {{ slice }}
                </option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Evaluating Slice</label>
              <select v-model="training.evalSlice"
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option value="" disabled selected>validation set</option>
                <option v-for="slice in sliceList" :key="slice" :value="slice">
                  {{ slice }}
                </option>
              </select>
            </div>
          </div>

          <button @click="startTraining" :disabled="training.status === 'running'" :class="['w-full px-4 py-2 rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2',
            training.status === 'running'
              ? 'bg-gray-400 text-white cursor-not-allowed'
              : 'bg-gray-800 text-white hover:bg-gray-900 focus:ring-gray-500']">
            {{ training.status === 'running' ? 'Training...' : 'Start Training' }}
          </button>
        </div>
      </section>

      <!-- Training Results -->
      <section class="mb-6" v-if="mode == 'coreset'">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">Training Results</h3>
          <button @click="updateResults" :disabled="resultLoading" :class="['px-4 py-2 rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2',
            resultLoading
              ? 'bg-gray-400 text-white cursor-not-allowed'
              : 'bg-gray-800 text-white hover:bg-gray-900 focus:ring-gray-500']">
            {{ resultLoading ? 'Updating...' : 'Update' }}
          </button>
        </div>

        <div class="bg-gray-100 rounded-lg p-4 h-60 flex items-center justify-center">
          <div v-if="trainingResults.length === 0" class="text-center">
            <div class="text-6xl text-gray-300 mb-2">📊</div>
            <p class="text-gray-500 text-lg font-medium">Results</p>
          </div>
          <div v-else class="w-full">
            <div class="max-h-52 overflow-y-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-gray-300">
                    <!-- <th class="text-left py-2 font-medium text-gray-700">Model</th> -->
                    <th class="text-left py-2 font-medium text-gray-700">Method</th>
                    <th class="text-left py-2 font-medium text-gray-700">Acc</th>
                    <th class="text-left py-2 font-medium text-gray-700">fraction</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="result in trainingResults" :key="result.time" class="border-b border-gray-200">
                    <!-- <td class="py-2 text-gray-900">{{ result.model }}</td> -->
                    <td class="py-2 text-gray-900">{{ result.method }}</td>
                    <td class="py-2 text-gray-900">{{ result.acc.toFixed(3) }}</td>
                    <td class="py-2 text-gray-900">{{ result.fraction.toFixed(2) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia';
import { useMainStore } from '../stores';
import * as api from '../api';

const store = useMainStore();
const {
  mode,
  views,
  sliceList,
  coresetSelection,
  coresetMethodList,
  training,
  trainingModelList,
  trainingResults,
  resultLoading
} = storeToRefs(store);

// 视图管理
const addView = async () => {
  try {
    // await api.addView();
    store.addView();
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '添加视图失败',
      duration: 3000
    });
  }
};

const deleteView = async (viewId: number) => {
  try {
    await api.deleteView(viewId);
    store.deleteView(viewId);
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '删除视图失败',
      duration: 3000
    });
  }
};

// 核心集选择
const applyCoreset = async () => {
  try {
    store.setLoading(true);
    await api.selectCoreset(coresetSelection.value);
    // 核心集选择完成后刷新切片列表
    await store.refreshSlices();
    store.setNotification({
      type: 'success',
      message: '核心集选择成功',
      duration: 3000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '核心集选择失败',
      duration: 3000
    });
  } finally {
    store.setLoading(false);
  }
};


// 训练
const startTraining = async () => {
  try {
    store.setLoading(true);
    await api.startTraining(training.value);
    store.setNotification({
      type: 'success',
      message: '训练启动成功',
      duration: 3000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '训练启动失败',
      duration: 3000
    });
  } finally {
    store.setLoading(false);
  }
};

// 更新结果
const updateResults = async () => {
  await store.refreshTrainingResults();
};
</script>
