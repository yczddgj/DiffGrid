<script setup lang="ts">
import { onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import { useMainStore } from './stores';
import SidebarPanel from './components/SidebarPanel.vue';
import SingleView from './components/SingleView.vue';
import ModeSelect from './components/ModeSelect.vue';

const store = useMainStore();
const { views, notification, mode, loading } = storeToRefs(store);


// 应用启动时初始化
onMounted(async () => {
  if (store.mode) {
    store.initializeApp();
  }
});

// 创建第一个视图
const createFirstView = async () => {
  try {
    await store.addView();
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '创建视图失败',
      duration: 3000
    });
  }
};

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

// 清除通知
const clearNotification = () => {
  store.clearNotification();
};
</script>

<template>
  <div v-if="!mode">
    <ModeSelect />
  </div>
  <div v-else class="h-screen bg-gray-100 flex">
    <!-- 左侧边栏 -->
    <div v-if="mode == 'coreset'" class="flex-shrink-0 h-screen flex flex-row" style="width: 450px;">
      <SidebarPanel />
    </div>

    <!-- 主内容区域 -->
    <div class="flex-1 overflow-hidden flex flex-row">
      <!-- 加载状态 -->
      <div v-if="loading" class="h-full flex items-center justify-center flex-1">
        <div class="text-center">
          <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p class="text-gray-600">正在加载应用数据...</p>
        </div>
      </div>

      <!-- 主视图区域 - 水平滚动 -->
      <div v-else-if="views.length > 0" class="h-full flex overflow-x-auto p-4" style="align-items: stretch;">
        <SingleView v-for="view in views" :key="view.id" :view="view" />

        <!-- 添加视图按钮 -->
        <div class="flex-shrink-0 flex items-center justify-center ml-4">
          <button @click="addView"
            class="px-4 py-2 bg-white text-gray-600 border border-gray-400 shadow-lg rounded-md hover:bg-gray-400 hover:text-gray-900 transition-colors flex flex-col items-center gap-1">
            <span class="text-xl font-bold">+</span>
          </button>
        </div>
      </div>

      <!-- 欢迎界面 -->
      <div v-else class="h-full flex items-center justify-center flex-1">
        <div class="text-center">
          <div class="text-6xl mb-4">📊</div>
          <h2 class="text-2xl font-semibold text-gray-800 mb-2">Welcome to DiffGrid</h2>
          <p class="text-gray-600 mb-6">Create your first view to start exploring data</p>
          <button @click="createFirstView"
            class="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900 transition-colors font-medium">
            Create First View
          </button>
        </div>
      </div>
    </div>

    <!-- 全局通知 -->
    <div v-if="notification" :class="[
      'fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-md',
      {
        'bg-green-100 border border-green-400 text-green-700': notification.type === 'success',
        'bg-red-100 border border-red-400 text-red-700': notification.type === 'error',
        'bg-blue-100 border border-blue-400 text-blue-700': notification.type === 'info',
        'bg-yellow-100 border border-yellow-400 text-yellow-700': notification.type === 'warning'
      }
    ]">
      <div class="flex justify-between items-start">
        <p class="flex-1">{{ notification.message }}</p>
        <button @click="clearNotification" class="ml-2 text-gray-500 hover:text-gray-700">
          ×
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}

.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}

.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
