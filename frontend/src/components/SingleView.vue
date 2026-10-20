<template>
  <div class="flex-shrink-0 bg-white rounded-3xl flex flex-col mx-2"
    style="aspect-ratio: 1 / 1; width: auto; height: auto; max-height: 100%;">

    <!-- Main View Area -->
    <div class="flex-1 relative min-h-0">
      <!-- 悬浮工具栏 -->
      <div
        class="absolute top-3 left-3 flex flex-wrap items-center gap-4 backdrop-blur-sm bg-white/50 rounded-md shadow-lg p-2 z-10"
        style="max-width: calc(100% - 2rem); width: auto;">
        <!-- View1 + 删除 -->
        <div class="flex items-center gap-2 min-w-0">
          <h2 class="text-lg font-semibold text-gray-800 truncate">{{ view.title }}</h2>
          <button @click="deleteView" class="text-gray-400 hover:text-red-500">
            <XMarkIcon class="h-4 w-4" />
          </button>
        </div>

        <!-- Current Slice -->
        <div class="flex items-center border border-gray-300 rounded-xl shadow-lg h-10 overflow-hidden min-w-[8rem]">
          <!-- 左侧色块（图例） -->
          <div class="w-4 h-full" style="background-color: #42a5f5;"></div>
          <!-- 下拉框 -->
          <select v-model="view.currentSlice" class="flex-1 px-2 py-2 text-base focus:outline-none bg-transparent"
            style="min-width: 0;">
            <option disabled value="">Current Slice</option>
            <option v-for="slice in sliceList" :key="slice" :value="slice">{{ slice }}</option>
          </select>
        </div>

        <!-- Reference Slice -->
        <div class="flex items-center border border-gray-300 rounded-xl shadow-lg h-10 overflow-hidden min-w-[8rem]">
          <!-- 左侧色块（图例） -->
          <div class="w-4 h-full" style="background-color: #ef5350;"></div>
          <!-- 下拉框 -->
          <select v-model="view.referenceSlice" class="flex-1 px-2 py-2 text-base focus:outline-none bg-transparent"
            style="min-width: 0;">
            <option disabled value="">Reference Slice</option>
            <option v-for="slice in sliceList" :key="slice" :value="slice">{{ slice }}</option>
          </select>
        </div>

        <!-- Apply -->
        <button @click="regenerateEmbedding" class="px-4 py-2 font-medium rounded-xl shadow-lg text-base
         hover:bg-gray-700 hover:text-white bg-white/50 text-gray-800
         backdrop-blur-sm border border-gray-300 transition-all duration-150 h-10">
          Apply
        </button>

        <!-- Labels -->
        <button @click="view.option.showImages = !view.option.showImages" :class="[
          'hidden px-4 py-2 text-base rounded-xl border shadow-lg h-10',
          view.option.showImages
            ? 'bg-gray-300 border-gray-400 text-black hover:bg-gray-350 font-medium'
            : 'bg-white text-gray-500 border-gray-300 hover:bg-gray-200 font-medium'
        ]">
          Label
        </button>

        <!-- Lasso 操作工具栏，与 Labels 等放在一起 -->
        <div v-if="mode == 'data-crafter'" class="flex flex-wrap gap-1 items-center">
          <button @click="copyLasso" :disabled="!hasLasso"
            class="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200 disabled:opacity-50">Copy</button>
          <button @click="pasteLasso" :disabled="!globalCopiedLasso"
            class="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200 disabled:opacity-50">Paste</button>
          <button @click="deleteLassoPoints" :disabled="!hasLasso"
            class="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200 disabled:opacity-50">Delete</button>
          <button @click=" upsampleLassoPoints" :disabled="!hasLasso"
            class="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200 disabled:opacity-50">Upsample</button>
          <button @click="downsampleLassoPoints" :disabled="!hasLasso"
            class="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200 disabled:opacity-50">Downsample</button>
        </div>

        <!-- 核心集调整工具栏 -->
        <div v-if="mode == 'data-crafter'" class="flex items-center gap-2">
          <!-- Fraction -->
          <div class="flex items-center border border-gray-300 rounded-xl shadow-lg h-10 overflow-hidden">
            <input v-model.number="view.operation.fraction" type="number" min="0" max="1000000" step="1"
              class="px-2 py-1 text-sm bg-transparent focus:outline-none flex-1 min-w-0" placeholder="Fraction"
              style="width: auto;" />
            <button @click="cutSlice"
              class="px-3 bg-gray-800 text-white text-sm hover:bg-gray-900 flex-shrink-0">Cut</button>
          </div>

          <!-- Save Slice -->
          <div class="flex items-center border border-gray-300 rounded-xl shadow-lg h-10 overflow-hidden">
            <input v-model="view.operation.saveSliceName" type="text"
              class="px-2 py-1 text-sm bg-transparent focus:outline-none flex-1 min-w-0" placeholder="Slice name"
              style="width: auto;" />
            <button @click="saveSlice"
              class="px-3 bg-gray-800 text-white text-sm hover:bg-gray-900 flex-shrink-0">Save</button>
          </div>
        </div>

      </div>




      <!-- 图例 -->
      <div class="hidden absolute right-3 top-3 flex space-x-3 flex-row">
        <!-- 背景热力图图例 -->
        <div v-if="view.visualEncoding.background !== 'None'" class="bg-white rounded-md shadow-sm p-2 z-10 h-fit">
          <div class="flex flex-col space-y-2">
            <h4 class="text-sm font-semibold text-gray-700">Background</h4>

            <!-- Class-Heatmap 类别图例 -->
            <div v-if="view.visualEncoding.background === 'Class-Heatmap'" :class="[
              'gap-x-3 gap-y-1',
              view.classCount <= 20 ? 'grid grid-cols-2' : 'grid grid-cols-2 max-h-32 overflow-y-auto'
            ]">
              <div v-for="item in backgroundLegendItems" :key="item.label" class="flex items-center space-x-2">
                <div class="w-3 h-3 border border-gray-300 flex-shrink-0 rounded-sm"
                  :style="{ backgroundColor: item.color }"></div>
                <span class="text-xs text-gray-600">{{ item.label }}</span>
              </div>
            </div>

            <!-- 其他类型的连续图例 -->
            <div v-else class="flex flex-col space-y-1.5">
              <div class="w-32 h-3 rounded" :style="heatmapGradientStyle"></div>
              <div class="flex justify-between text-xs text-gray-600">
                <span>{{ heatmapMin }}</span>
                <span>{{ heatmapMax }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 散点图图例 -->
        <div v-if="view.visualEncoding.scatter !== 'None'" class="hidden bg-white rounded-md shadow-sm p-2 z-10">
          <div class="flex flex-col space-y-2">
            <h4 class="text-sm font-semibold text-gray-700">Scatterplot</h4>
            <div v-if="view.visualEncoding.scatter === 'Class'" :class="[
              'gap-x-3 gap-y-1',
              view.classCount <= 20 ? 'grid grid-cols-2' : 'grid grid-cols-2 max-h-32 overflow-y-auto'
            ]">
              <div v-for="item in scatterLegendItems" :key="item.label" class="flex items-center space-x-2">
                <div class="w-3 h-3 rounded-full border border-gray-300 flex-shrink-0"
                  :style="{ backgroundColor: item.color }"></div>
                <span class="text-xs text-gray-600">{{ item.label }}</span>
              </div>
            </div>
            <div v-else class="flex flex-col space-y-1.5">
              <div v-for="item in scatterLegendItems" :key="item.label" class="flex items-center space-x-2">
                <div class="w-3 h-3 rounded-full border border-gray-300 flex-shrink-0"
                  :style="{ backgroundColor: item.color }"></div>
                <span class="text-xs text-gray-600">{{ item.label }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 主图区域 -->
      <div class="w-full h-full">
        <ScatterPlot :view-id="view.id" :heatmap="view.heatmap" :scatter-data="view.scatterData" :zoom="view.zoom"
          :pan="view.pan" :lasso-shape="view.lassoShape" :show-images="true"
          :scatter-mode="view.option.scatterMode" :scatter-encoding="view.visualEncoding.scatter"
          :background-encoding="view.visualEncoding.background" @lasso-select="onLassoSelect"
          @zoom-change="onZoomChange" @pan-change="onPanChange" @viewport-change="onViewportChange" />
      </div>



      <!-- Current Samples - 左半屏水平居中 -->
      <div v-if="mode==='tabular' && parsedTabularCurrent.length>0"
        class="absolute bottom-10 bg-white/70 border backdrop-blur-sm rounded-lg shadow-lg p-3 pt-1 z-20 overflow-hidden"
        style="left: 25%; transform: translateX(-50%); width: 47%; max-height: 40%;">
        <h4 class="text-lg font-semibold text-gray-700 mb-2">Current Samples</h4>
        <div class="overflow-auto max-h-[calc(100%-2rem)]">
          <table class="w-full text-xs text-left border-collapse">
            <thead>
              <tr>
                <th v-for="col in tabularColumns" :key="'cur-th-'+col" class="px-2 py-1 font-medium bg-gray-100 border border-gray-200 sticky top-0 z-10">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, rIdx) in parsedTabularCurrent.slice(0,8)" :key="'cur-row-'+rIdx" class="hover:bg-gray-50">
                <td v-for="col in tabularColumns" :key="'cur-td-'+col" class="px-2 py-1 border border-gray-200 whitespace-nowrap" :title="formatCell(row[col])">{{ shortCell(row[col]) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-else-if="mode==='text' && textTitlesCurrent.length>0"
        class="absolute bottom-10 bg-white/70 border backdrop-blur-sm rounded-lg shadow-lg p-3 pt-1 z-20 overflow-hidden"
        style="left: 25%; transform: translateX(-50%); width: 47%; max-height: 40%;">
        <h4 class="text-lg font-semibold text-gray-700 mb-2">Current Samples</h4>
        <div class="overflow-auto max-h-[calc(100%-2rem)]">
          <div class="grid grid-cols-3 gap-2 text-xs">
            <div v-for="(t, idx) in textTitlesCurrent.slice(0,9)" :key="'cur-text-'+idx"
                   class="p-2 border border-gray-200 rounded-md bg-white/80 shadow-sm hover:bg-gray-50 transition" :title="t">
                <div class="clamp-3 leading-snug break-words">{{ t }}</div>
              </div>
            </div>
        </div>
      </div>

      <div v-else-if="mode!=='tabular' && mode!=='text' && view.sampleImages.length > 0"
        class="absolute bottom-10 bg-white/50 border backdrop-blur-sm rounded-lg shadow-lg p-3 pt-1 z-20"
        style="left: 25%; transform: translateX(-50%); width: 47%;">
        <h4 class="text-lg font-semibold text-gray-700 mb-2">Current Samples</h4>
        <div class="grid grid-cols-7 gap-1">
          <div v-for="image in view.sampleImages.slice(0, 14)" :key="image.id"
            class="aspect-square bg-gray-200 rounded overflow-hidden cursor-pointer hover:ring-2 hover:ring-blue-500"
            @click="showLargeImage(image)">
            <img v-if="image.thumbUrl" :src="image.thumbUrl" class="w-full h-full object-cover" />
            <div v-else class="w-full h-full flex items-center justify-center text-gray-400 text-xs">No Image</div>
          </div>
          <div v-for="n in Math.max(0, 14 - view.sampleImages.length)" :key="'placeholder-imgcur-' + n" class="aspect-square bg-transparent rounded"></div>
        </div>
      </div>


      <!-- Reference Samples - 右半屏水平居中 -->
      <div v-if="mode==='tabular' && parsedTabularRef.length>0"
        class="absolute bottom-10 bg-white/70 border backdrop-blur-sm rounded-lg shadow-lg p-3 pt-1 z-20 overflow-hidden"
        style="left: 75%; transform: translateX(-50%); width: 47%; max-height: 40%;">
        <h4 class="text-lg font-semibold text-gray-700 mb-2">Reference Samples</h4>
        <div class="overflow-auto max-h-[calc(100%-2rem)]">
          <table class="w-full text-xs text-left border-collapse">
            <thead>
              <tr>
                <th v-for="col in tabularColumns" :key="'ref-th-'+col" class="px-2 py-1 font-medium bg-gray-100 border border-gray-200 sticky top-0 z-10">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, rIdx) in parsedTabularRef.slice(0,8)" :key="'ref-row-'+rIdx" class="hover:bg-gray-50">
                <td v-for="col in tabularColumns" :key="'ref-td-'+col" class="px-2 py-1 border border-gray-200 whitespace-nowrap" :title="formatCell(row[col])">{{ shortCell(row[col]) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-else-if="mode==='text' && textTitlesRef.length>0"
        class="absolute bottom-10 bg-white/70 border backdrop-blur-sm rounded-lg shadow-lg p-3 pt-1 z-20 overflow-hidden"
        style="left: 75%; transform: translateX(-50%); width: 47%; max-height: 40%;">
        <h4 class="text-lg font-semibold text-gray-700 mb-2">Reference Samples</h4>
        <div class="overflow-auto max-h-[calc(100%-2rem)]">
          <div class="grid grid-cols-3 gap-2 text-xs">
            <div v-for="(t, idx) in textTitlesRef.slice(0,9)" :key="'ref-text-'+idx"
                   class="p-2 border border-gray-200 rounded-md bg-white/80 shadow-sm hover:bg-gray-50 transition" :title="t">
                <div class="clamp-3 leading-snug break-words">{{ t }}</div>
              </div>
            </div>
        </div>
      </div>

      <div v-else-if="mode!=='tabular' && mode!=='text' && view.sampleImagesReference && view.sampleImagesReference.length > 0"
        class="absolute bottom-10 bg-white/50 border backdrop-blur-sm rounded-lg shadow-lg p-3 pt-1 z-20"
        style="left: 75%; transform: translateX(-50%); width: 47%;">
        <h4 class="text-lg font-semibold text-gray-700 mb-2">Reference Samples</h4>
        <div class="grid grid-cols-7 gap-1">
          <div v-for="image in view.sampleImagesReference.slice(0, 14)" :key="image.id"
            class="aspect-square bg-gray-200 rounded overflow-hidden cursor-pointer hover:ring-2 hover:ring-red-500"
            @click="showLargeImage(image)">
            <img v-if="image.thumbUrl" :src="image.thumbUrl" class="w-full h-full object-cover" />
            <div v-else class="w-full h-full flex items-center justify-center text-gray-400 text-xs">No Image</div>
          </div>
          <div v-for="n in Math.max(0, 14 - view.sampleImagesReference.length)" :key="'placeholder-imgref-' + n" class="aspect-square bg-transparent rounded"></div>
        </div>
      </div>


      <!-- 单行页脚,半透明毛玻璃 -->
      <div
        class="absolute bottom-1 right-3 backdrop-blur-sm bg-white/50 border border-gray-300/50 text-xs text-gray-700 px-3 py-1 rounded-lg shadow-sm flex items-center">
        <span>
          Current: {{ view.info.currentSliceShown }}/{{ view.info.currentSliceTotal }}
          · Ref: {{ view.info.referenceSliceShown }}/{{ view.info.referenceSliceTotal }}
          · Sel: {{ view.info.selectionShown }}/{{ view.info.selectionTotal }}
        </span>
      </div>
    </div>
  </div>


  <!-- 大图预览模态窗口 -->
  <div v-if="showImageModal" class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75"
    @click="closeImageModal">
    <div class="relative" @click.stop>
      <!-- 关闭按钮 -->
      <button @click="closeImageModal"
        class="absolute -top-2 -right-2 z-20 text-white hover:text-gray-300 bg-red-600 hover:bg-red-700 rounded-full p-2 shadow-lg transition-colors">
        <XMarkIcon class="h-5 w-5" />
      </button>

      <!-- 图片 -->
      <img v-if="selectedImage" :src="selectedImage.imageUrl" :alt="`Sample ${selectedImage.id}`"
        class="min-w-[400px] min-h-[400px] max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl bg-white"
        @load="() => console.log('Image loaded')" @error="() => console.error('Image failed to load')" />
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  XMarkIcon,
} from '@heroicons/vue/24/outline';
import { computed, onMounted, onBeforeUnmount, watch, ref } from 'vue';
import { storeToRefs } from 'pinia';
import type { ViewConfig } from '../types';
import { useMainStore } from '../stores';
import * as api from '../api';
import ScatterPlot from './ScatterPlot.vue';
import { COLORS, getClassColor, getHeatmapGradient } from '../constants/colors';
const props = defineProps<{
  view: ViewConfig;
}>();

const store = useMainStore();
const { mode, sliceList, globalCopiedLasso } = storeToRefs(store);

// 数据获取状态
const isLoadingData = ref(false);
const viewportBounds = ref({
  x1: 0, x2: 1, y1: 0, y2: 1
});



// 计算属性
const hasLasso = computed(() => props.view.lassoShape.length > 0);

// Legend相关计算属性
const scatterLegendItems = computed(() => {
  switch (props.view.visualEncoding.scatter) {
    case 'Class':
      return Array.from({ length: props.view.classCount }, (_, i) => ({
        label: `Class ${i}`,
        color: getClassColor(i)
      }));
    case 'Current or Reference':
      return [
        { label: 'Current', color: COLORS.scatter.current },
        { label: 'Reference', color: COLORS.scatter.reference }
      ];
    default:
      return [];
  }
});

const backgroundLegendItems = computed(() => {
  switch (props.view.visualEncoding.background) {
    case 'Class-Heatmap':
      return Array.from({ length: props.view.classCount }, (_, i) => ({
        label: `Class ${i}`,
        color: getClassColor(i)
      }));
    default:
      return [];
  }
});

const heatmapMin = computed(() => {
  switch (props.view.visualEncoding.background) {
    case 'Class-Heatmap':
      return 'Low';
    case 'Density-Difference':
      return 'Negative';
    default:
      return 'Min';
  }
});

const heatmapMax = computed(() => {
  switch (props.view.visualEncoding.background) {
    case 'Class-Heatmap':
      return 'High';
    case 'Density-Difference':
      return 'Positive';
    default:
      return 'Max';
  }
});

const heatmapGradientStyle = computed(() => {
  const gradient = getHeatmapGradient(props.view.visualEncoding.background);
  return {
    background: gradient,
    border: '1px solid #d1d5db'
  };
});

// 数据获取函数
const loadScatterData = async () => {
  if (isLoadingData.value) return;

  try {
    console.log("viewportBounds", viewportBounds.value);
    isLoadingData.value = true;
    const response = await api.getScatter(
      props.view.id,
      viewportBounds.value.x1,
      viewportBounds.value.x2,
      viewportBounds.value.y1,
      viewportBounds.value.y2
    );

    if (response.data.code === 0) {
      props.view.scatterData = response.data.data || [];
    }
  } catch (error) {
    console.error('加载散点数据失败:', error);
    store.setNotification({
      type: 'error',
      message: '加载散点数据失败',
      duration: 3000
    });
  } finally {
    isLoadingData.value = false;
  }
};

const loadHeatmapData = async () => {
  // 暂时留空，用户要求热力图先不实现
  try {
    const response = await api.getHeatmap(props.view.id, props.view.visualEncoding.background);
    if (response.data.code === 0) {
      props.view.heatmap = response.data.data || [];
    }
  } catch (error) {
    console.error('加载热力图数据失败:', error);
    // 为兼容性提供空数据
    props.view.heatmap = [];
  }
};

const loadSampleImages = async () => {
  try {
    const response = await api.getLassoSampleImages(props.view.id);
    console.log('Sample Images Response:', response);
    if (response.data.code === 0) {
      const data = response.data.data || [[], []];
      // data: [ [thumbUrls, urls], [thumbUrlsRef, urlsRef] ]
      const [current, reference] = data;
      const [thumbUrls, urls] = current || [[], []];
      const [thumbUrlsRef, urlsRef] = reference || [[], []];

      props.view.sampleImages = thumbUrls.map((thumbUrl: string, index: number) => ({
        id: index,
        imageUrl: (mode.value==='tabular' || mode.value==='text') ? null : (urls[index] || null),
        // 在 tabular/text 模式下不渲染图片，避免错误的 img src
        thumbUrl: (mode.value==='tabular' || mode.value==='text') ? null : (thumbUrl || urls[index] || null),
        title: (mode.value==='tabular' || mode.value==='text') ? (thumbUrl || urls[index] || '') : undefined,
      }));

      props.view.sampleImagesReference = thumbUrlsRef.map((thumbUrl: string, index: number) => ({
        id: index,
        imageUrl: (mode.value==='tabular' || mode.value==='text') ? null : (urlsRef[index] || null),
        // 在 tabular/text 模式下不渲染图片，避免错误的 img src
        thumbUrl: (mode.value==='tabular' || mode.value==='text') ? null : (thumbUrl || urlsRef[index] || null),
        title: (mode.value==='tabular' || mode.value==='text') ? (thumbUrl || urlsRef[index] || '') : undefined,
      }));
    }
  } catch (error) {
    console.error('加载样本图片失败:', error);
    props.view.sampleImages = [];
    props.view.sampleImagesReference = [];
  }
};

const loadViewInfo = async () => {
  try {
    const response = await api.getInfo(props.view.id);
    if (response.data.code === 0) {
      props.view.info = response.data.data;
    }
  } catch (error) {
    console.error('加载视图信息失败:', error);
  }
};

const loadAllData = async () => {
  await Promise.all([
    await loadScatterData(),
    await loadHeatmapData(),
    await loadSampleImages(),
    await loadViewInfo()
  ]);
};

// 删除视图
const deleteView = async () => {
  try {
    await api.deleteView(props.view.id);
    store.deleteView(props.view.id);
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '删除视图失败',
      duration: 3000
    });
  }
};



// 事件处理
const onLassoSelect = async (shape: number[][]) => {
  try {
    // 先保存 Lasso 形状
    props.view.lassoShape = shape;

    // 立即进行本地选择检测，快速更新视图
    updateLocalSelection(shape);
    // if (shape.length < 2) {
    //   // store.setNotification({
    //   //   type: 'warning',
    //   //   message: 'Lasso shape must have at least 2 points',
    //   //   duration: 2000
    //   // });

    //   shape = [];
    // }
    // 然后异步同步到服务器
    const response = await api.lassoSelectPoints(props.view.id, shape);
    props.view.lassoSelection = response.data.data.lassoSelection;

    // 只更新必要的数据，不重新加载散点数据
    await Promise.all([
      await loadViewInfo(),
      await loadSampleImages()
    ]);
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '选择点失败',
      duration: 3000
    });
  }
};

// 本地选择检测函数 - 立即更新散点的选中状态
const updateLocalSelection = (shape: number[][]) => {
  if (shape.length < 3) return; // 至少需要3个点形成多边形

  // shape 传过来的是数据坐标，可以直接使用
  const polygon = shape.map(point => ({ x: point[0], y: point[1] }));

  // 遍历所有散点，检查是否在多边形内部
  // 使用 splice 确保 Vue 能检测到数组变化
  const newScatterData = props.view.scatterData.map(point => {
    const isInside = pointInPolygon({ x: point.x, y: point.y }, polygon);
    return { ...point, selected: isInside };
  });

  // 替换整个数组以触发响应式更新
  props.view.scatterData.splice(0, props.view.scatterData.length, ...newScatterData);
};

// 点在多边形内检测算法（射线法）
const pointInPolygon = (point: { x: number, y: number }, polygon: { x: number, y: number }[]) => {
  let inside = false;
  const { x, y } = point;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const { x: xi, y: yi } = polygon[i];
    const { x: xj, y: yj } = polygon[j];

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
};

const onZoomChange = (zoom: number) => {
  // 只有当zoom值真正发生变化时才更新，避免递归更新
  if (props.view.zoom !== zoom) {
    props.view.zoom = zoom;
  }
};

const onPanChange = (pan: { x: number; y: number }) => {
  // 只有当pan值真正发生变化时才更新，避免递归更新
  if (props.view.pan.x !== pan.x || props.view.pan.y !== pan.y) {
    props.view.pan = pan;
  }
};

// 视口变化处理 - 当用户缩放或平移结束时更新数据
const onViewportChange = async (bounds: { x1: number, x2: number, y1: number, y2: number }) => {
  viewportBounds.value = bounds;

  try {
    // 同时更新后端的zoom和pan状态
    const updateData = {
      visualEncoding: props.view.visualEncoding,
      option: props.view.option,
      zoom: props.view.zoom,
      pan: props.view.pan
    };

    // 顺序执行：先更新配置，再依次加载数据
    await api.updateView(props.view.id, updateData);
    await loadScatterData();
    await loadHeatmapData();
    await loadViewInfo();
  } catch (error) {
    console.error('视口变化处理失败:', error);
    store.setNotification({
      type: 'error',
      message: '更新视图失败',
      duration: 3000
    });
  }
};

// Lasso操作
const copyLasso = async () => {
  try {
    await api.copyLasso(props.view.id, props.view.lassoShape);
    // 更新全局状态，使其他视图也能访问
    store.setGlobalCopiedLasso([...props.view.lassoShape]);
    store.setNotification({
      type: 'success',
      message: 'Lasso shape copied',
      duration: 2000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '复制 Lasso 失败',
      duration: 3000
    });
  }
};

const pasteLasso = async () => {
  try {
    // 直接从全局状态获取复制的lasso形状
    if (globalCopiedLasso.value && globalCopiedLasso.value.length > 0) {
      const copiedShape = [...globalCopiedLasso.value];
      props.view.lassoShape = copiedShape;
      // 触发lasso选择，使粘贴的形状立即生效
      await onLassoSelect(copiedShape);
      store.setNotification({
        type: 'success',
        message: 'Lasso shape pasted',
        duration: 2000
      });
    } else {
      store.setNotification({
        type: 'warning',
        message: '没有可粘贴的Lasso形状',
        duration: 2000
      });
    }
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '粘贴 Lasso 失败',
      duration: 3000
    });
  }
};

// 点操作
const deleteLassoPoints = async () => {
  try {
    // 删除选中的点（lassoSelection），而不是清空整个lasso
    await api.deleteLassoPoints(props.view.id);
    // 本地更新选择
    updateLocalSelection(props.view.lassoShape);
    // 同步到服务器
    const lassoResp = await api.lassoSelectPoints(props.view.id, props.view.lassoShape);
    props.view.lassoSelection = lassoResp.data.data.lassoSelection;
    // 删除点后更新所有相关数据，包括热力图
    await Promise.all([
      loadScatterData(),
      loadHeatmapData(),
      loadViewInfo(),
      loadSampleImages()
    ]);
    store.setNotification({
      type: 'success',
      message: 'Points deleted',
      duration: 2000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '删除点失败',
      duration: 3000
    });
  }
};

const upsampleLassoPoints = async () => {
  try {
    await api.upsampleLassoPoints(props.view.id);
    // 本地更新选择
    updateLocalSelection(props.view.lassoShape);
    // 同步到服务器
    const lassoResp = await api.lassoSelectPoints(props.view.id, props.view.lassoShape);
    props.view.lassoSelection = lassoResp.data.data.lassoSelection;
    // 上采样后更新所有相关数据，包括热力图
    await Promise.all([
      loadScatterData(),
      loadHeatmapData(),
      loadViewInfo(),
      loadSampleImages()
    ]);
    store.setNotification({
      type: 'success',
      message: 'Points upsampled',
      duration: 2000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '上采样失败',
      duration: 3000
    });
  }
};

const downsampleLassoPoints = async () => {
  try {
    await api.downsampleLassoPoints(props.view.id);
    // 本地更新选择
    updateLocalSelection(props.view.lassoShape);
    // 同步到服务器
    const lassoResp = await api.lassoSelectPoints(props.view.id, props.view.lassoShape);
    props.view.lassoSelection = lassoResp.data.data.lassoSelection;
    // 下采样后更新所有相关数据，包括热力图
    await Promise.all([
      loadScatterData(),
      loadHeatmapData(),
      loadViewInfo(),
      loadSampleImages()
    ]);
    store.setNotification({
      type: 'success',
      message: 'Points downsampled',
      duration: 2000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '下采样失败',
      duration: 3000
    });
  }
};

// 切片操作
const cutSlice = async () => {
  try {
    await api.cutSlice(props.view.id, props.view.operation.fraction);
    // 切片后更新所有数据
    await loadAllData();
    // lasso相关更新
    updateLocalSelection(props.view.lassoShape);
    const lassoResp = await api.lassoSelectPoints(props.view.id, props.view.lassoShape);
    props.view.lassoSelection = lassoResp.data.data.lassoSelection;
    await Promise.all([
      loadViewInfo(),
      loadSampleImages()
    ]);
    store.setNotification({
      type: 'success',
      message: `Slice cut with fraction ${props.view.operation.fraction}`,
      duration: 2000
    });
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '切片失败',
      duration: 3000
    });
  }
};

const saveSlice = async () => {
  try {
    await api.saveSlice(props.view.id, props.view.operation.saveSliceName);
    // 刷新切片列表
    await store.refreshSlices();
    store.setNotification({
      type: 'success',
      message: `Slice saved as '${props.view.operation.saveSliceName}'`,
      duration: 2000
    });
    // 清空输入框
    props.view.operation.saveSliceName = '';
  } catch (error) {
    store.setNotification({
      type: 'error',
      message: '保存切片失败',
      duration: 3000
    });
  }
};

// 图片预览
const showImageModal = ref(false);
const selectedImage = ref<{ id: number; imageUrl: string } | null>(null);

const showLargeImage = (image: { id: number; imageUrl: string }) => {
  selectedImage.value = image;
  showImageModal.value = true;
};

const closeImageModal = () => {
  showImageModal.value = false;
  selectedImage.value = null;
};

// 键盘事件处理
const handleKeyDown = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && showImageModal.value) {
    closeImageModal();
  }
};

// 生命周期钩子
onMounted(() => {
  // 组件挂载后立即加载数据
  loadAllData();
  // 添加键盘事件监听
  window.addEventListener('keydown', handleKeyDown);
});

onBeforeUnmount(() => {
  // 移除键盘事件监听
  window.removeEventListener('keydown', handleKeyDown);
});

// 重新生成embedding和重新加载数据
const regenerateEmbedding = async () => {
  try {
    console.log("切片或降维配置变化，重新加载数据...");

    // 先更新后端配置
    const updateData = {
      currentSlice: props.view.currentSlice,
      referenceSlice: props.view.referenceSlice,
      dimensionReduction: props.view.dimensionReduction
    };
    await api.updateView(props.view.id, updateData);

    // 然后重新加载所有数据（因为embedding会变化）
    await loadAllData();
  } catch (error) {
    console.error('更新切片/降维配置失败:', error);
    store.setNotification({
      type: 'error',
      message: '更新切片/降维配置失败',
      duration: 3000
    });
  }
};

// 监听只需要更新视图但不需要重新加载数据的配置变化
watch([
  () => props.view.visualEncoding,
  () => props.view.option
], async () => {
  try {
    console.log("视觉编码或选项变化，仅更新配置...");

    const updateData = {
      visualEncoding: props.view.visualEncoding,
      option: props.view.option,
      zoom: props.view.zoom,
      pan: props.view.pan
    };

    await api.updateView(props.view.id, updateData);

    // 注意：热力图会在单独的watch中处理，这里不需要重复加载
  } catch (error) {
    console.error('更新视图配置失败:', error);
    store.setNotification({
      type: 'error',
      message: '更新视图配置失败',
      duration: 3000
    });
  }
}, { deep: true });

// showImages 变化时不需要重新加载数据，只需要重新渲染ScatterPlot组件即可

// 单独监听背景编码变化，重新加载热力图
watch(() => props.view.visualEncoding.background, async () => {
  console.log("Background encoding changed, reloading heatmap...");
  await loadHeatmapData();
});

// 监听 sliceList 变化，用于调试
watch(sliceList, (newSliceList, oldSliceList) => {
  console.log('sliceList 发生变化:', {
    old: oldSliceList,
    new: newSliceList,
    length: newSliceList?.length || 0
  });
}, { deep: true });

// 模式切换（image/tabular/text）时，重新获取样本，以便后端返回对应模式的数据载体
watch(mode, async () => {
  try {
    await loadSampleImages();
  } catch (e) {
    console.error('切换模式后刷新样本失败:', e);
  }
});

// ===== Tabular / Text 模式辅助计算与格式化 =====
// 解析 sampleImages / sampleImagesReference 的 title 字段（若为整行 JSON）

const parsedTabularCurrent = computed(() => {
  if (mode.value !== 'tabular') return [] as Record<string, any>[];
  return props.view.sampleImages.map((it: any) => {
    const raw = it?.title;
    const parsed = safeParseJSON(raw);
    if (parsed && typeof parsed === 'object') return parsed as Record<string, any>;
    // 兜底：非JSON时，以单列 title 展示
    return { title: (raw ?? '').toString() } as Record<string, any>;
  });
});
const parsedTabularRef = computed(() => {
  if (mode.value !== 'tabular') return [] as Record<string, any>[];
  return (props.view.sampleImagesReference || []).map((it: any) => {
    const raw = it?.title;
    const parsed = safeParseJSON(raw);
    if (parsed && typeof parsed === 'object') return parsed as Record<string, any>;
    // 兜底：非JSON时，以单列 title 展示
    return { title: (raw ?? '').toString() } as Record<string, any>;
  });
});
const tabularColumns = computed(() => {
  if (mode.value !== 'tabular') return [] as string[];
  const cols = new Set<string>();
  const cur = parsedTabularCurrent.value;
  const ref = parsedTabularRef.value;
  cur.forEach(r => Object.keys(r).forEach(k => cols.add(k)));
  ref.forEach(r => Object.keys(r).forEach(k => cols.add(k)));

  const isNonEmpty = (v: any) => {
    if (v === null || v === undefined) return false;
    if (typeof v === 'string') return v.trim().length > 0;
    if (typeof v === 'number') return !Number.isNaN(v);
    // 对于布尔/对象/数组，认为存在即非空
    return true;
  };

  const filtered = Array.from(cols).filter(col =>
    cur.some(r => isNonEmpty((r as any)[col])) || ref.some(r => isNonEmpty((r as any)[col]))
  );
  return filtered.slice(0, 60); // 最多 60 列
});
const textTitlesCurrent = computed(() => mode.value === 'text'
  ? props.view.sampleImages.map(it => ((it as any).title || '').toString())
  : [] as string[]);
const textTitlesRef = computed(() => mode.value === 'text'
  ? (props.view.sampleImagesReference || []).map(it => ((it as any).title || '').toString())
  : [] as string[]);

function safeParseJSON(str: string | undefined) {
  if (!str || typeof str !== 'string') return null;
  let s = str.trim();

  // 尝试截取首尾大括号内的主体，避免日志里带有前后噪声
  const firstBrace = s.indexOf('{');
  const lastBrace = s.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    s = s.slice(firstBrace, lastBrace + 1).trim();
  }

  // 1) 直接按 JSON 解析
  try { return JSON.parse(s); } catch {}

  // 2) 宽松解析：从文本中提取 "key": value 或 'key': value 的键值对
  const obj: Record<string, any> = {};
  let matched = false;

  // a) 双引号键
  const reDouble = /"([^"\\]+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)/g;
  let m: RegExpExecArray | null;
  while ((m = reDouble.exec(s)) !== null) {
    matched = true;
    const k = m[1];
    const rawV = m[2];
    obj[k] = parseLooseValue(rawV);
  }

  // b) 单引号键（例如 Python dict 风格）
  const reSingle = /'([^'\\]+)'\s*:\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|true|false|null|-?\d+(?:\.\d+)?)/g;
  while ((m = reSingle.exec(s)) !== null) {
    matched = true;
    const k = m[1];
    const rawV = m[2];
    obj[k] = parseLooseValue(rawV);
  }

  return matched ? obj : null;
}

function parseLooseValue(raw: string) {
  const v = raw.trim();
  if (v === 'true') return true;
  if (v === 'false') return false;
  if (v === 'null' || v === 'None') return null;
  // 数字
  if (/^-?\d+(?:\.\d+)?$/.test(v)) return Number(v);
  // 去掉包裹引号
  if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
    return v.slice(1, -1);
  }
  return v;
}
function shortCell(v: any) {
  if (v === null || v === undefined) return 'null';
  const s = (typeof v === 'string') ? (v.trim().length ? v : 'null') : String(v);
  return s.length > 20 ? s.slice(0, 17) + '…' : s;
}
function formatCell(v: any) {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'string') return v.trim().length ? v : 'null';
  return String(v);
}
</script>

<style scoped>
.clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
