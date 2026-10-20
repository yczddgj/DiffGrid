<template>
  <div ref="container" class="w-full h-full rounded-3xl shadow-lg overflow-hidden relative">
    <button
      @click="downloadSVG"
      class="absolute top-4 right-4 bg-white/80 hover:bg-white p-2 rounded-full shadow-md transition-colors z-50 cursor-pointer"
      title="Download SVG"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="7 10 12 15 17 10"></polyline>
        <line x1="12" y1="15" x2="12" y2="3"></line>
      </svg>
    </button>
    <svg ref="svg" class="w-full h-full" :style="{
      userSelect: 'none',
      WebkitUserSelect: 'none',
      MozUserSelect: 'none',
      msUserSelect: 'none'
    }"></svg>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, computed, type PropType } from 'vue';
import * as d3 from 'd3';
import type { HeatmapData, ScatterData } from '../types';
import { getClassColor, getSourceColor } from '../constants/colors';

import { useMainStore } from '../stores';

const store = useMainStore();
const { mode, } = storeToRefs(store);

import { storeToRefs } from 'pinia';
const props = defineProps({
  viewId: {
    type: Number,
    required: true
  },
  heatmap: {
    type: Array as PropType<HeatmapData[]>,
    required: true
  },
  scatterData: {
    type: Array as PropType<ScatterData[]>,
    required: true
  },
  zoom: {
    type: Number,
    required: true
  },
  pan: {
    type: Object as PropType<{ x: number; y: number }>,
    required: true
  },
  lassoShape: {
    type: Array as PropType<number[][]>,
    required: true
  },
  showImages: {
    type: Boolean,
    required: true
  },
  scatterMode: {
    type: String as PropType<'Current' | 'Reference' | 'Current & Reference'>,
    required: true
  },
  scatterEncoding: {
    type: String as PropType<'None' | 'Class' | 'Current or Reference'>,
    required: true
  },
  backgroundEncoding: {
    type: String as PropType<'Class-Heatmap' | 'Density-Difference' | 'None'>,
    required: true
  }
});

const emit = defineEmits<{
  (e: 'lasso-select', shape: number[][]): void;
  (e: 'zoom-change', zoom: number): void;
  (e: 'pan-change', pan: { x: number; y: number }): void;
  (e: 'viewport-change', bounds: { x1: number, x2: number, y1: number, y2: number }): void;
}>();

// 下载 SVG
const downloadSVG = () => {
  if (!svg.value) return;

  // 克隆 SVG 节点以避免影响显示
  const svgClone = svg.value.cloneNode(true) as SVGSVGElement;

  // 确保有 xmlns
  svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');

  // 序列化
  const serializer = new XMLSerializer();
  let svgString = serializer.serializeToString(svgClone);

  // 添加 XML 声明
  svgString = '<?xml version="1.0" standalone="no"?>\r\n' + svgString;

  // 创建 Blob 和下载链接
  const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `heatmap-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.svg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// DOM 引用
const container = ref<HTMLDivElement>();
const svg = ref<SVGSVGElement>();

// D3 选择器
let svgSelection: d3.Selection<SVGSVGElement, unknown, null, undefined>;
let heatmapGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
let lassoPolygonGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
let scatterGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
let circleGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
let imageGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
let lassoGroup: d3.Selection<SVGGElement, unknown, null, undefined>;

// 等高线图层
let contourGroup: d3.Selection<SVGGElement, unknown, null, undefined>;

// 比例尺
let xScale: d3.ScaleLinear<number, number>;
let yScale: d3.ScaleLinear<number, number>;
let colorScale: d3.ScaleSequential<string>;

// 缩放行为
let zoom: d3.ZoomBehavior<SVGSVGElement, unknown>;
let currentZoomScale = 1;
let tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>;

// 根据scatterMode筛选散点数据的计算属性
const filteredScatterData = computed(() => {
  switch (props.scatterMode) {
    case 'Current & Reference':
      return props.scatterData;
    default:
      return props.scatterData;
  }
});

// 初始化
onMounted(() => {
  if (!container.value || !svg.value) return;

  // 获取容器尺寸
  const { width, height } = container.value.getBoundingClientRect();

  // 初始化 SVG
  svgSelection = d3.select(svg.value)
    .attr('width', width)
    .attr('height', height);

  svgSelection
    .on('mousedown.preventMiddle', (event) => {
      if (event.button === 1) event.preventDefault();
    })
    .on('auxclick.preventMiddle', (event) => {
      if (event.button === 1) event.preventDefault();
    });


  // 创建图层（按层级顺序：热力图 -> lasso多边形 -> 散点圆圈 -> 图片 -> lasso线条）
  heatmapGroup = svgSelection.append('g').attr('class', 'heatmap-layer');
  lassoPolygonGroup = svgSelection.append('g').attr('class', 'lasso-polygon-layer');
  scatterGroup = svgSelection.append('g').attr('class', 'scatter-layer');
  circleGroup = scatterGroup.append('g').attr('class', 'circle-group');
  imageGroup = scatterGroup.append('g').attr('class', 'image-group');
  lassoGroup = svgSelection.append('g').attr('class', 'lasso-layer');

  // 在 onMounted 中初始化 contourGroup（放在热力图图层之后）
  contourGroup = svgSelection.append('g').attr('class', 'contour-layer');

  // 初始化比例尺 - 保持纵横比
  const size = Math.max(width, height);
  const xOffset = (width - size) / 2;
  const yOffset = (height - size) / 2;

  xScale = d3.scaleLinear().domain([0, 1]).range([xOffset, xOffset + size]);
  yScale = d3.scaleLinear().domain([0, 1]).range([yOffset + size, yOffset]);
  colorScale = d3.scaleSequential(d3.interpolateViridis);

  // 初始化缩放
  zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.2, 3])
    .filter((event) => {
      // 滚轮缩放
      if (event.type === 'wheel') return true;
      // 中键拖动平移
      if (event.type === 'mousedown' && (event.button === 1)) return true;
      return false; // 左键不交给zoom
    })
    .on('zoom', (event) => {
      console.log('on zoom event');
      const { transform } = event;
      currentZoomScale = transform.k;
      heatmapGroup.attr('transform', transform.toString());
      lassoPolygonGroup.attr('transform', transform.toString());
      scatterGroup.attr('transform', transform.toString());
      contourGroup.attr('transform', transform.toString());

      // 重新绘制热力图以调整大小
      drawHeatmap();

      // 重新绘制散点以调整大小
      drawScatter();

      // 只有不是内部变换时才发射事件，避免循环
      if (!isInternalTransform) {
        emit('zoom-change', transform.k);
        emit('pan-change', { x: transform.x, y: transform.y });
      }
    })
    .on('end', (event) => {
      console.log('on end event, isInternalTransform:', isInternalTransform);
      // 只在缩放/平移结束后发射视口边界变化
      const { transform } = event;
      const { width, height } = container.value!.getBoundingClientRect();

      const bounds = {
        x1: xScale.invert(-transform.x / transform.k),
        x2: xScale.invert((width - transform.x) / transform.k),
        y1: yScale.invert((height - transform.y) / transform.k), // 底部（小值）
        y2: yScale.invert(-transform.y / transform.k)           // 顶部（大值）
      };

      emit('viewport-change', bounds);
    });

  svgSelection.call(zoom);
  // 左键 Lasso


  svgSelection
    .on('mousedown', (event) => {
      if (event.button === 0) startLasso(event);
    })
    .on('mousemove', (event) => {
      if (isLassoing) updateLasso(event);
    })
    .on('mouseup', (event) => {
      if (isLassoing && event.button === 0) endLasso();
    });

  // 用于标题显示
  tooltip = d3.select('body')
    .append('div')
    .attr('class', 'image-tooltip')
    .style('position', 'absolute')
    .style('padding', '4px 8px')
    .style('background', 'rgba(0,0,0,0.75)')
    .style('color', '#fff')
    .style('font-size', '12px')
    .style('border-radius', '4px')
    .style('pointer-events', 'none')
    .style('z-index', '10')
    .style('visibility', 'hidden');

  // 初始化渲染
  updateScales();
  drawHeatmap();
  drawLassoPolygon();
  drawScatter();
});


// 更新比例尺
const updateScales = () => {
  if (!container.value) return;

  const { width, height } = container.value.getBoundingClientRect();
  const size = Math.max(width, height);
  const xOffset = (width - size) / 2;
  const yOffset = (height - size) / 2;

  // 使用固定的[0,1] x [0,1]范围，但保持纵横比
  xScale.domain([0, 1]).range([xOffset, xOffset + size]);
  yScale.domain([0, 1]).range([yOffset + size, yOffset]);

  // 更新颜色比例尺
  if (props.heatmap.length > 0) {
    const valueExtent = d3.extent(props.heatmap, d => d.value) as [number, number];
    colorScale.domain(valueExtent);
  }
};

// 绘制热力图
const drawHeatmap = () => {
  if (!heatmapGroup) return;

  const rects = heatmapGroup
    .selectAll('rect')
    .data(props.heatmap);

  // 计算与缩放比例成反比的边框宽度

  // 更新现有矩形
  rects
    .attr('x', d => xScale(d.x0))
    .attr('y', d => yScale(d.y1))
    .attr('width', d => xScale(d.x1) - xScale(d.x0))
    .attr('height', d => yScale(d.y0) - yScale(d.y1))
    .attr('fill', d => getHeatmapColor(d.value))
    .attr('opacity', () => props.backgroundEncoding === 'Density-Difference' ? 1 : 0.5)
    // .attr('stroke', '#f9fafb')
    // .attr('stroke-width', d => d.type === 'center' ? 0 : adjustedStrokeWidth);

  // 添加新矩形
  rects.enter()
    .append('rect')
    .attr('x', d => xScale(d.x0))
    .attr('y', d => yScale(d.y1))
    .attr('width', d => xScale(d.x1) - xScale(d.x0))
    .attr('height', d => yScale(d.y0) - yScale(d.y1))
    .attr('fill', d => getHeatmapColor(d.value))
    .attr('opacity', () => props.backgroundEncoding === 'Density-Difference' ? 1 : 0.5)
    // .attr('stroke', '#f9fafb')
    // .attr('stroke-width', d => d.type === 'center' ? 0 : adjustedStrokeWidth);

  // 移除多余矩形
  rects.exit().remove();

  // 绘制等高线
  // drawContours();
};

// 绘制lasso多边形
const drawLassoPolygon = () => {
  if (!lassoPolygonGroup) return;

  // 只有当lassoShape有数据时才绘制多边形
  if (props.lassoShape && props.lassoShape.length > 2) {
    const polygon = lassoPolygonGroup
      .selectAll('polygon')
      .data([props.lassoShape]);

    // 将数据坐标转换为屏幕坐标并生成points字符串
    const pointsString = props.lassoShape
      .map(point => `${xScale(point[0])},${yScale(point[1])}`)
      .join(' ');

    // 更新现有多边形
    polygon
      .attr('points', pointsString)
      .attr('fill', 'white')
      .attr('opacity', 0.3)
      .attr('stroke', 'grey')
      .attr('stroke-width', 2 / currentZoomScale);

    // 添加新多边形
    polygon.enter()
      .append('polygon')
      .attr('points', pointsString)
      .attr('fill', 'white')
      .attr('opacity', 0.3)
      .attr('stroke', 'grey')
      .attr('stroke-width', 2 / currentZoomScale);

    // 移除多余多边形
    polygon.exit().remove();
  } else {
    // 如果没有lassoShape，清除所有多边形
    lassoPolygonGroup.selectAll('polygon').remove();
  }
};

// 获取热力图颜色的函数
const getHeatmapColor = (value: number): string => {
  switch (props.backgroundEncoding) {
    case 'Class-Heatmap':
      // 对于类别热力图，使用与散点图相同的类别颜色
      return getClassColor(value);
    case 'Density-Difference':
      // 修改为自定义插值函数，确保0.5处为纯白色
      return customRdBuInterpolator(value);
    default:
      // 对于其他类型，使用连续颜色标尺
      return colorScale(value);
  }
};

// 自定义RdBu插值器，确保0.5处为纯白色
const customRdBuInterpolator = (t: number): string => {
  const color = d3.rgb(d3.interpolateRdBu(t));
  const pureWhite = d3.rgb('#ffffff');
  // const pureWhite = d3.rgb('#f9f9ff');
  const fakeWhite = d3.rgb(d3.interpolateRdBu(0.5));
  const diffRatio = 1 - Math.abs(t - 0.5) * 2;

  // 修复颜色计算：使用RGB通道值进行计算
  const r = color.r + diffRatio * (pureWhite.r - fakeWhite.r);
  const g = color.g + diffRatio * (pureWhite.g - fakeWhite.g);
  const b = color.b + diffRatio * (pureWhite.b - fakeWhite.b);

  return d3.rgb(r, g, b).toString();
};

// 获取散点颜色的函数
const getScatterColor = (d: ScatterData) => {
  // if (d.selected) return COLORS.scatter.selected; // 选中的点始终是红色

  switch (props.scatterEncoding) {
    case 'None':
      return 'transparent'; // 不显示散点
    case 'Class':
      return getClassColor(d.label); // 基于类别的彩色编码
    case 'Current or Reference':
      return getSourceColor(d.source); // current: 蓝色, reference: 绿色
    default:
      return getClassColor(d.label);
  }
};


function splitIntoTwoLines(text: string) {
  const cleanText = text.trim();
  if (cleanText.length <= 12) return [cleanText]; // 字符数不超过12，不换行

  const words = cleanText.split(/\s+/);
  if (words.length <= 1) return [cleanText]; // 没空格，也不换行

  // 贪婪拆成两行，使长度差最小
  let bestSplit = [cleanText];
  let minDiff = Infinity;
  for (let i = 1; i < words.length; i++) {
    const line1 = words.slice(0, i).join(' ');
    const line2 = words.slice(i).join(' ');
    const diff = Math.abs(line1.length - line2.length);
    if (diff < minDiff) {
      bestSplit = [line1, line2];
      minDiff = diff;
    }
  }
  return bestSplit;
}

function truncateKeyword(text: string, maxLen = 10): string {
  if (!text) return '';
  if (text.length <= maxLen) return text;
  // 预留一位给省略号，保持整体不超过 maxLen
  if (maxLen <= 1) return '…';
  return text.slice(0, maxLen - 1) + '…';
}

function getNearestMidpoint(anchor: [number, number], imageCenter: [number, number], halfSize: number): [number, number] {
  const [ax, ay] = anchor;
  const [cx, cy] = imageCenter;

  // 计算四个边界中点
  const top: [number, number] = [cx, cy - halfSize];
  const bottom: [number, number] = [cx, cy + halfSize];
  const left: [number, number] = [cx - halfSize, cy];
  const right: [number, number] = [cx + halfSize, cy];

  // 计算到锚点的距离
  const points = [top, bottom, left, right];
  let minDist = Infinity;
  let nearestPoint: [number, number] = top;

  for (const point of points) {
    const dx = point[0] - ax;
    const dy = point[1] - ay;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < minDist) {
      minDist = dist;
      nearestPoint = point;
    }
  }

  return nearestPoint;
}


// 绘制散点
const drawScatter = () => {
  if (!circleGroup || !imageGroup) return;

  // 如果scatter encoding是None，则清空所有散点
  if (props.scatterEncoding === 'None') {
    circleGroup.selectAll('circle').remove();
  }
  else {
    console.log(props.scatterData);

    const circles = circleGroup
      .selectAll('circle')
      .data(filteredScatterData.value, (d: any) => d.id);

    // 计算固定大小的半径（与缩放比例成反比）
    const baseRadius = 4;
    const selectedRadius = 4;
    const hoverRadius = 6;
    const adjustedRadius = baseRadius / currentZoomScale;
    const adjustedSelectedRadius = selectedRadius / currentZoomScale;
    const adjustedHoverRadius = hoverRadius / currentZoomScale;
    const adjustedStrokeWidth = 1.5 / currentZoomScale;

    // 用于追踪hover的点
    let hoveredId: string | number | null = null;

    // 更新现有圆点
    circles
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => {
        if (hoveredId === d.id) return adjustedHoverRadius;
        return d.selected ? adjustedSelectedRadius : adjustedRadius;
      })
      .attr('fill', d => getScatterColor(d))
      .attr('stroke', d => hoveredId === d.id ? '#ff9800' : (d.selected ? '#f9fafb' : '#333'))
      .attr('stroke-width', d => hoveredId === d.id ? adjustedStrokeWidth * 2 : adjustedStrokeWidth)
      .attr('opacity', 1.0);

    // 添加新圆点
    circles.enter()
      .append('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => d.selected ? adjustedSelectedRadius : adjustedRadius)
      .attr('fill', d => getScatterColor(d))
      .attr('stroke', d => d.selected ? '#f9fafb' : '#333')
      .attr('stroke-width', adjustedStrokeWidth)
      .attr('opacity', 1.0)
      // .on('mouseover', function (event, d) {
      //   hoveredId = d.id;
      //   d3.select(this)
      //     .attr('r', adjustedHoverRadius)
      //     .attr('stroke', '#444444')
      //     .attr('stroke-width', adjustedStrokeWidth * 2);
      //   // 将当前圆点提到最上方
      //   this.parentNode?.appendChild(this);

      //   // 在circleGroup中添加图片
      //   circleGroup.selectAll('image.scatter-hover-image').remove();
      //   if (d.imageUrl) {
      //     circleGroup.append('image')
      //       .attr('class', 'scatter-hover-image')
      //       .attr('x', xScale(d.x) - 32 / currentZoomScale)
      //       .attr('y', yScale(d.y) - 80 / currentZoomScale)
      //       .attr('width', 80 / currentZoomScale)
      //       .attr('height', 80 / currentZoomScale)
      //       .attr('href', d.imageUrl)
      //       .attr('style', 'pointer-events:none;')
      //       .attr('opacity', 1.0);
      //   }
      // })
      // .on('mouseout', function (event, d) {
      //   hoveredId = null;
      //   d3.select(this)
      //     .attr('r', d.selected ? adjustedSelectedRadius : adjustedRadius)
      //     .attr('stroke', d.selected ? '#f9fafb' : '#333')
      //     .attr('stroke-width', adjustedStrokeWidth);
      //   circleGroup.selectAll('image.scatter-hover-image').remove();
      // })
      ;

    // 移除多余圆点
    circles.exit().remove();
  }

  // 连线、图片和标题图层
  const lineGroup: d3.Selection<SVGGElement, unknown, null, undefined> =
    imageGroup.select<SVGGElement>('.line-layer').empty()
      ? imageGroup.append('g').attr('class', 'line-layer')
      : imageGroup.select<SVGGElement>('.line-layer');

  const imageBorderGroup: d3.Selection<SVGGElement, unknown, null, undefined> =
    imageGroup.select<SVGGElement>('.image-border-layer').empty()
      ? imageGroup.append('g').attr('class', 'image-border-layer')
      : imageGroup.select<SVGGElement>('.image-border-layer');

  const imageLayer: d3.Selection<SVGGElement, unknown, null, undefined> =
    imageGroup.select<SVGGElement>('.image-layer').empty()
      ? imageGroup.append('g').attr('class', 'image-layer')
      : imageGroup.select<SVGGElement>('.image-layer');

  const titleLayer: d3.Selection<SVGGElement, unknown, null, undefined> =
    imageGroup.select<SVGGElement>('.title-layer').empty()
      ? imageGroup.append('g').attr('class', 'title-layer')
      : imageGroup.select<SVGGElement>('.title-layer');



  // 过滤数据
  if (props.showImages) {
    // 判断 mode 是否为 'default'
    if (mode.value == 'kde-scatter') {
      const imagesWithUrl = filteredScatterData.value.filter((d: ScatterData) =>d.thumbUrl);

      // 尺寸设置
      const baseImageSize = 1224 / 80 * 3;
      const adjustedImageSize = baseImageSize / currentZoomScale;
      const halfImageSize = adjustedImageSize / 2;
      const borderStrokeWidth = 2 / currentZoomScale;
      const textYOffset = adjustedImageSize / 2 + 14 / currentZoomScale;
      const fontSize = 20 / currentZoomScale;
      const strokeWidth = 2 / currentZoomScale;

      // ==== 边框 ====
      const borders = imageBorderGroup
        .selectAll<SVGRectElement, ScatterData>('rect.image-border')
        .data(imagesWithUrl, (d: ScatterData) => `${d.x}-${d.y}-${d.thumbUrl}`);

      borders.exit().remove();

      borders
        .attr('x', (d: ScatterData) => xScale(d.x) - halfImageSize - borderStrokeWidth / 2)
        .attr('y', (d: ScatterData) => yScale(d.y) - halfImageSize - borderStrokeWidth / 2)
        .attr('width', adjustedImageSize + borderStrokeWidth)
        .attr('height', adjustedImageSize + borderStrokeWidth)
        .attr('stroke-width', borderStrokeWidth);

      borders.enter()
        .append('rect')
        .attr('class', 'image-border')
        .attr('x', (d: ScatterData) => xScale(d.x) - halfImageSize - borderStrokeWidth * 1.5)
        .attr('y', (d: ScatterData) => yScale(d.y) - halfImageSize - borderStrokeWidth * 1.5)
        .attr('width', adjustedImageSize + borderStrokeWidth * 3)
        .attr('height', adjustedImageSize + borderStrokeWidth * 3)
        .attr('fill', '#888')
        .attr('stroke', 'white')
        .attr('stroke-width', borderStrokeWidth);

      // ==== 图片 ====
      const images = imageLayer
        .selectAll<SVGImageElement, ScatterData>('image.scatter-image')
        .data(imagesWithUrl, d => d.id);

      images.exit().remove();

      images
        .attr('x', (d: ScatterData) => xScale(d.x) - halfImageSize)
        .attr('y', (d: ScatterData) => yScale(d.y) - halfImageSize)
        .attr('width', adjustedImageSize)
        .attr('height', adjustedImageSize)
        .style('opacity', 0.9)
        .style('border-radius', `${3 / currentZoomScale}px`)
        .style('box-shadow', `0 0 ${6 / currentZoomScale}px rgba(0,0,0,0.3)`);

      const imagesEnter = images.enter()
        .append('image')
        .attr('class', 'scatter-image')
        .attr('x', (d: ScatterData) => xScale(d.x) - halfImageSize)
        .attr('y', (d: ScatterData) => yScale(d.y) - halfImageSize)
        .attr('width', adjustedImageSize)
        .attr('height', adjustedImageSize)
        .attr('href', d => d.thumbUrl || d.imageUrl || '')
        .style('opacity', 0.9)
        .style('border-radius', `${3 / currentZoomScale}px`)
        .style('box-shadow', `0 0 ${6 / currentZoomScale}px rgba(0,0,0,0.3)`)
        .style('cursor', 'pointer');

      // 添加事件监听器
      imagesEnter
        .on('mouseover', (_event, d) => {
          tooltip
            .text(d.title)
            .style('visibility', 'visible');
        })
        .on('mousemove', (event) => {
          tooltip
            .style('top', (event.pageY - 30) + 'px')
            .style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', () => {
          tooltip.style('visibility', 'hidden');
        });

      // ==== 绘制标题 ====
      const titleGroups = titleLayer
        .selectAll<SVGGElement, ScatterData>('g.image-title-group')
        .data(imagesWithUrl, d => `${d.x}-${d.y}-${d.thumbUrl}`);

      titleGroups.exit().remove();
      titleGroups.remove();

      const allTitleGroups = titleLayer
        .selectAll<SVGGElement, ScatterData>('g.image-title-group')
        .data(imagesWithUrl, d => `${d.x}-${d.y}-${d.thumbUrl}`)
        .enter()
        .append('g')
        .attr('class', 'image-title-group')
        .attr('transform', (d: ScatterData) => {
          // 固定在图片下方
          const imageX = xScale(d.x);
          const imageY = yScale(d.y);
          return `translate(${imageX}, ${imageY + textYOffset})`;
        });

      allTitleGroups.each(function (d) {
        const g = d3.select(this);
        const lines = splitIntoTwoLines(d.key_word || '');
        lines.forEach((line, i) => {
          g.append('text')
        .attr('class', 'title-stroke')
        .attr('text-anchor', 'middle')
        .attr('font-size', `${fontSize}px`)
        .attr('stroke', 'white')
        .attr('stroke-width', `${strokeWidth}px`)
        .attr('fill', 'none')
        .attr('font-weight', 'bold')
        .style('pointer-events', 'none')
        .attr('dy', `${i * 12.0 / currentZoomScale}`)
        .text(line);
        });
        lines.forEach((line, i) => {
          g.append('text')
        .attr('class', 'title-fill')
        .attr('text-anchor', 'middle')
        .attr('font-size', `${fontSize}px`)
        .attr('stroke', 'none')
        .attr('fill', '#666')
        .attr('font-weight', 'bold')
        .style('pointer-events', 'none')
        .attr('dy', `${i * 12.0 / currentZoomScale}`)
        .text(line);
        });
      });

      borders.raise();
      images.raise();
    } else {
      // Tabular / Text 模式：显示关键词与连线
      if (mode.value === 'tabular' || mode.value === 'text') {
        const dataItems = filteredScatterData.value.filter((d: ScatterData) => d.key_word || d.title);

        // 尺寸设置（用于连线终点的虚拟“图片”方框大小）
        const baseImageSize = 1224 / 80 * 3;
        const adjustedImageSize = baseImageSize / currentZoomScale;
        const halfImageSize = adjustedImageSize / 2;
        const fontSize = 12 / currentZoomScale;
        const strokeWidth = 2 / currentZoomScale;

        // 计算连线所需数据（依旧使用 imageX/imageY，若无则回退到锚点坐标）
        const lineData = dataItems.map(d => ({
          ...d,
          anchor: [xScale(d.x), yScale(d.y)],
          imageCenter: [xScale(d.imageX ?? d.x), yScale(d.imageY ?? d.y)],
          nearest: getNearestMidpoint(
            [xScale(d.x), yScale(d.y)],
            [xScale(d.imageX ?? d.x), yScale(d.imageY ?? d.y)],
            halfImageSize
          )
        }));

        // 计算连线路径函数（复用默认逻辑）
        const calculateLinePath = (d: typeof lineData[0]) => {
          const { anchor, imageCenter, nearest } = d;
          let cornerPoint: [number, number];
          let cornerLength = 100 / currentZoomScale;
          let cornerPoint1: [number, number];
          let cornerPoint2: [number, number];
          if (Math.abs(imageCenter[0] - anchor[0]) > Math.abs(imageCenter[1] - anchor[1])) {
            cornerPoint = [anchor[0], imageCenter[1]];
            cornerLength = Math.min(cornerLength, Math.abs(nearest[0] - cornerPoint[0]));
            cornerLength = Math.min(cornerLength, Math.abs(cornerPoint[1] - anchor[1]));
            cornerPoint2 = [cornerPoint[0] + Math.sign(nearest[0] - cornerPoint[0]) * cornerLength, cornerPoint[1]];
            cornerPoint1 = [cornerPoint[0], cornerPoint[1] + Math.sign(anchor[1] - cornerPoint[1]) * cornerLength];
          } else {
            cornerPoint = [imageCenter[0], anchor[1]];
            cornerLength = Math.min(cornerLength, Math.abs(nearest[1] - cornerPoint[1]));
            cornerLength = Math.min(cornerLength, Math.abs(cornerPoint[0] - anchor[0]));
            cornerPoint2 = [cornerPoint[0], cornerPoint[1] + Math.sign(nearest[1] - cornerPoint[1]) * cornerLength];
            cornerPoint1 = [cornerPoint[0] + Math.sign(anchor[0] - cornerPoint[0]) * cornerLength, cornerPoint[1]];
          }
          const pointList = [anchor, cornerPoint1, cornerPoint2, nearest];
          const eps = 3 / currentZoomScale;
          const filteredPointList = [pointList[0]];
          for (let i = 1; i < pointList.length; i++) {
            const [x1, y1] = filteredPointList[filteredPointList.length - 1];
            const [x2, y2] = pointList[i];
            const dist = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
            if (dist >= eps) filteredPointList.push(pointList[i]);
          }
          const firstDir = [filteredPointList[1][0] - filteredPointList[0][0], filteredPointList[1][1] - filteredPointList[0][1]];
            const l = Math.sqrt(firstDir[0] ** 2 + firstDir[1] ** 2);
            const firstEmptyLength = 3 / currentZoomScale;
            filteredPointList[0] = [filteredPointList[0][0] + firstDir[0] / l * firstEmptyLength, filteredPointList[0][1] + firstDir[1] / l * firstEmptyLength];
          return filteredPointList.reduce((acc, pt, idx) => idx === 0 ? `M ${pt[0]},${pt[1]}` : acc + ` L ${pt[0]},${pt[1]}`,'');
        };

        // 连线背景
        const bgLines = lineGroup
          .selectAll<SVGPathElement, typeof lineData[0]>('path.image-line-bg')
          .data(lineData, d => `${d.x}-${d.y}-${d.id}`);
        bgLines.exit().remove();
        bgLines
          .attr('d', calculateLinePath)
          .attr('stroke', 'white')
          .attr('stroke-width', 4 / currentZoomScale);
        bgLines.enter()
          .append('path')
          .attr('class', 'image-line-bg')
          .attr('d', calculateLinePath)
          .attr('stroke', 'white')
          .attr('stroke-width', 4 / currentZoomScale)
          .attr('opacity', 1)
          .attr('fill', 'none');

        // 主连线
        const lines = lineGroup
          .selectAll<SVGPathElement, typeof lineData[0]>('path.image-line')
          .data(lineData, d => `${d.x}-${d.y}-${d.id}`);
        lines.exit().remove();
        lines
          .attr('d', calculateLinePath)
          .attr('stroke', '#888')
          .attr('stroke-width', 2 / currentZoomScale);
        lines.enter()
          .append('path')
          .attr('class', 'image-line')
          .attr('d', calculateLinePath)
          .attr('stroke', '#888')
          .attr('stroke-width', 2 / currentZoomScale)
          .attr('opacity', 0.9)
          .attr('fill', 'none');

        // 终点圆点
        const dots = lineGroup
          .selectAll<SVGCircleElement, typeof lineData[0]>('circle.line-dot')
          .data(lineData, d => `${d.x}-${d.y}-${d.id}`);
        dots.exit().remove();
        dots
          .attr('r', 5 / currentZoomScale)
          .attr('cx', d => d.anchor[0])
          .attr('cy', d => d.anchor[1]);
        dots.enter()
          .append('circle')
          .attr('class', 'line-dot')
          .attr('r', 5 / currentZoomScale)
          .attr('fill', 'transparent')
          .attr('opacity', 0.9)
          .attr('cx', d => d.anchor[0])
          .attr('cy', d => d.anchor[1])
          .attr('stroke', 'white')
          .attr('stroke-width', 2 / currentZoomScale);

        const bgDots = lineGroup
          .selectAll<SVGCircleElement, typeof lineData[0]>('circle.line-dot-bg')
          .data(lineData, d => `${d.x}-${d.y}-${d.id}`);
        bgDots.exit().remove();
        bgDots
          .attr('r', 3.5 / currentZoomScale)
          .attr('cx', d => d.anchor[0])
          .attr('cy', d => d.anchor[1]);
        bgDots.enter()
          .append('circle')
          .attr('class', 'line-dot-bg')
          .attr('r', 3.5 / currentZoomScale)
          .attr('fill', 'transparent')
          .attr('opacity', 0.9)
          .attr('cx', d => d.anchor[0])
          .attr('cy', d => d.anchor[1])
          .attr('stroke', '#888')
          .attr('stroke-width', 2 / currentZoomScale);
        bgDots.raise();

        // 清除可能已有的图片与边框内容
        imageBorderGroup.selectAll('rect.image-border').remove();
        imageLayer.selectAll('image.scatter-image').remove();

        // 标题（单文本，截断10字符）
        const labelGroups = titleLayer
          .selectAll<SVGGElement, ScatterData>('g.tabular-label-group')
          .data(dataItems, d => `${d.x}-${d.y}-${d.id}`);
        labelGroups.exit().remove();
        labelGroups.remove();
        const allLabelGroups = titleLayer
          .selectAll<SVGGElement, ScatterData>('g.tabular-label-group')
          .data(dataItems, d => `${d.x}-${d.y}-${d.id}`)
          .enter()
          .append('g')
          .attr('class', 'tabular-label-group')
          .attr('transform', (d: ScatterData) => {
            const anchorX = xScale(d.x);
            const anchorY = yScale(d.y);
            const imageX = xScale(d.imageX ?? d.x);
            const imageY = yScale(d.imageY ?? d.y);
            // 计算连线在虚拟边框上的最近顶点
            const nearest = getNearestMidpoint([anchorX, anchorY], [imageX, imageY], halfImageSize);
            // 沿连线方向稍微外推，避免与终点圆点重叠
            const dirX = nearest[0] - anchorX;
            const dirY = nearest[1] - anchorY;
            const len = Math.sqrt(dirX * dirX + dirY * dirY) || 1;
            const offsetDist = 6 / currentZoomScale; // 偏移距离
            const labelX = nearest[0] + dirX / len * offsetDist;
            const labelY = nearest[1] + dirY / len * offsetDist;
            return `translate(${labelX}, ${labelY})`;
          });
        allLabelGroups.each(function(d){
          const g = d3.select(this);
          const truncated = truncateKeyword(d.key_word || d.title || '', 15);
          // stroke
          g.append('text')
            .attr('text-anchor','middle')
            .attr('font-size', `${fontSize}px`)
            .attr('stroke','white')
            .attr('stroke-width', `${strokeWidth}px`)
            .attr('fill','none')
            .attr('font-weight','bold')
            .style('pointer-events','none')
            .text(truncated);
          // fill (hover target)
          g.append('text')
            .attr('class','tabular-label-fill')
            .attr('text-anchor','middle')
            .attr('font-size', `${fontSize}px`)
            .attr('stroke','none')
            .attr('fill','#666')
            .attr('font-weight','bold')
            .style('cursor','pointer')
            .text(truncated)
            .on('mouseover',()=>{
              tooltip.text(d.title).style('visibility','visible');
            })
            .on('mousemove',(event)=>{
              tooltip
                .style('top', (event.pageY - 30) + 'px')
                .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout',()=>{
              tooltip.style('visibility','hidden');
            });
        });
        // 提升层级
        lines.raise();
        bgLines.raise();
        labelGroups.raise();
        return; // 结束此分支，不执行默认图片逻辑
      }

      // ===== 原有 default 模式代码（带连线和标记 + 图片） =====
      const imagesWithUrl = filteredScatterData.value.filter((d: ScatterData) => d.title || d.thumbUrl);

      // 尺寸设置
      const baseImageSize = 1224 / 80 * 3;
      const adjustedImageSize = baseImageSize / currentZoomScale;
      const halfImageSize = adjustedImageSize / 2;
      const borderStrokeWidth = 2 / currentZoomScale;
      const textYOffset = adjustedImageSize / 2 + 14 / currentZoomScale;
      const fontSize = 12 / currentZoomScale;
      const strokeWidth = 2 / currentZoomScale;

      // ==== 绘制连线 ====
      const lineData = imagesWithUrl.map(d => ({
        ...d,
        anchor: [xScale(d.x), yScale(d.y)],
        imageCenter: [xScale(d.imageX ?? 0), yScale(d.imageY ?? 0)],
        nearest: getNearestMidpoint(
          [xScale(d.x), yScale(d.y)],
          [xScale(d.imageX ?? 0), yScale(d.imageY ?? 0)],
          halfImageSize
        )
      }));

      // 计算连线路径的通用函数
      const calculateLinePath = (d: typeof lineData[0]) => {
        const { anchor, imageCenter, nearest } = d;
        let cornerPoint: [number, number];
        let cornerLength = 100 / currentZoomScale;
        let cornerPoint1: [number, number];
        let cornerPoint2: [number, number];

        if (Math.abs(imageCenter[0] - anchor[0]) > Math.abs(imageCenter[1] - anchor[1])) {
          cornerPoint = [anchor[0], imageCenter[1]];
          cornerLength = Math.min(cornerLength, Math.abs(nearest[0] - cornerPoint[0]));
          cornerLength = Math.min(cornerLength, Math.abs(cornerPoint[1] - anchor[1]));
          cornerPoint2 = [cornerPoint[0] + Math.sign(nearest[0] - cornerPoint[0]) * cornerLength, cornerPoint[1]];
          cornerPoint1 = [cornerPoint[0], cornerPoint[1] + Math.sign(anchor[1] - cornerPoint[1]) * cornerLength];
        } else {
          cornerPoint = [imageCenter[0], anchor[1]];
          cornerLength = Math.min(cornerLength, Math.abs(nearest[1] - cornerPoint[1]));
          cornerLength = Math.min(cornerLength, Math.abs(cornerPoint[0] - anchor[0]));
          cornerPoint2 = [cornerPoint[0], cornerPoint[1] + Math.sign(nearest[1] - cornerPoint[1]) * cornerLength];
          cornerPoint1 = [cornerPoint[0] + Math.sign(anchor[0] - cornerPoint[0]) * cornerLength, cornerPoint[1]];
        }

        const pointList = [anchor, cornerPoint1, cornerPoint2, nearest];
        const eps = 3 / currentZoomScale;
        const filteredPointList = [pointList[0]];
        for (let i = 1; i < pointList.length; i++) {
          const [x1, y1] = filteredPointList[filteredPointList.length - 1];
          const [x2, y2] = pointList[i];
          const dist = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
          if (dist >= eps) {
            filteredPointList.push(pointList[i]);
          }
        }
        const firstDir = [filteredPointList[1][0] - filteredPointList[0][0], filteredPointList[1][1] - filteredPointList[0][1]];
        const l = Math.sqrt(firstDir[0] ** 2 + firstDir[1] ** 2);
        const firstEmtpyLength = 3 / currentZoomScale;
        filteredPointList[0] = [filteredPointList[0][0] + firstDir[0] / l * firstEmtpyLength, filteredPointList[0][1] + firstDir[1] / l * firstEmtpyLength];

        const svgPath = filteredPointList.reduce((acc, pt, idx) => {
          if (idx === 0) {
            return `M ${pt[0]},${pt[1]}`;
          } else {
            return acc + ` L ${pt[0]},${pt[1]}`;
          }
        }, '');
        return svgPath;
      };

      // 连线背景（白色描边）
      const bgLines = lineGroup
        .selectAll<SVGPathElement, typeof lineData[0]>('path.image-line-bg')
        .data(lineData, d => `${d.x}-${d.y}-${d.thumbUrl}`);

      bgLines.exit().remove();

      bgLines
        .attr('d', calculateLinePath)
        .attr('stroke', 'white')
        .attr('stroke-width', 4 / currentZoomScale);

      bgLines.enter()
        .append('path')
        .attr('class', 'image-line-bg')
        .attr('d', calculateLinePath)
        .attr('stroke', 'white')
        .attr('stroke-width', 4 / currentZoomScale)
        .attr('opacity', 1)
        .attr('fill', 'none');

      // 主连线
      const lines = lineGroup
        .selectAll<SVGPathElement, typeof lineData[0]>('path.image-line')
        .data(lineData, d => `${d.x}-${d.y}-${d.thumbUrl}`);

      lines.exit().remove();

      lines
        .attr('d', calculateLinePath)
        .attr('stroke', '#888')
        .attr('stroke-width', 2 / currentZoomScale);

      lines.enter()
        .append('path')
        .attr('class', 'image-line')
        .attr('d', calculateLinePath)
        .attr('stroke', '#888')
        .attr('stroke-width', 2 / currentZoomScale)
        .attr('opacity', 0.9)
        .attr('fill', 'none');

      // 连线终点圆点
      const dots = lineGroup
        .selectAll<SVGCircleElement, typeof lineData[0]>('circle.line-dot')
        .data(lineData, d => `${d.x}-${d.y}-${d.thumbUrl}`);

      dots.exit().remove();

      dots
        .attr('r', 5 / currentZoomScale)
        .attr('cx', d => d.anchor[0])
        .attr('cy', d => d.anchor[1]);

      dots.enter()
        .append('circle')
        .attr('class', 'line-dot')
        .attr('data-id', d => d.id)
        .attr('r', 5 / currentZoomScale)
        .attr('fill', 'transparent')
        .attr('opacity', 0.9)
        .attr('cx', d => d.anchor[0])
        .attr('cy', d => d.anchor[1])
        .attr('stroke', 'white')
        .attr('stroke-width', 2 / currentZoomScale);

      const bgDots = lineGroup
        .selectAll<SVGCircleElement, typeof lineData[0]>('circle.line-dot-bg')
        .data(lineData, d => `${d.x}-${d.y}-${d.thumbUrl}`);

      bgDots.exit().remove();
      bgDots
        .attr('r', 3.5 / currentZoomScale)
        .attr('cx', d => d.anchor[0])
        .attr('cy', d => d.anchor[1]);
      bgDots.enter()
        .append('circle')
        .attr('class', 'line-dot-bg')
        .attr('data-id', d => d.id)
        .attr('r', 3.5 / currentZoomScale)
        .attr('fill', 'transparent')
        .attr('opacity', 0.9)
        .attr('cx', d => d.anchor[0])
        .attr('cy', d => d.anchor[1])
        .attr('stroke', '#888')
        .attr('stroke-width', 2 / currentZoomScale);
      bgDots.raise();

      // ==== 边框 ====
      const borders = imageBorderGroup
        .selectAll<SVGRectElement, ScatterData>('rect.image-border')
        .data(imagesWithUrl, (d: ScatterData) => `${d.x}-${d.y}-${d.thumbUrl}`);

      borders.exit().remove();

      borders
        .attr('x', (d: ScatterData) => xScale(d.imageX ?? 0) - halfImageSize - borderStrokeWidth / 2)
        .attr('y', (d: ScatterData) => yScale(d.imageY ?? 0) - halfImageSize - borderStrokeWidth / 2)
        .attr('width', adjustedImageSize + borderStrokeWidth)
        .attr('height', adjustedImageSize + borderStrokeWidth)
        .attr('stroke-width', borderStrokeWidth);

      borders.enter()
        .append('rect')
        .attr('class', 'image-border')
        .attr('x', (d: ScatterData) => xScale(d.imageX ?? 0) - halfImageSize - borderStrokeWidth * 1.5)
        .attr('y', (d: ScatterData) => yScale(d.imageY ?? 0) - halfImageSize - borderStrokeWidth * 1.5)
        .attr('width', adjustedImageSize + borderStrokeWidth * 3)
        .attr('height', adjustedImageSize + borderStrokeWidth * 3)
        .attr('fill', '#888')
        .attr('stroke', 'white')
        .attr('stroke-width', borderStrokeWidth);

      // ==== 图片 ====
      const images = imageLayer
        .selectAll<SVGImageElement, ScatterData>('image.scatter-image')
        .data(imagesWithUrl, d => d.id);

      images.exit().remove();

      images
        .attr('x', (d: ScatterData) => xScale(d.imageX ?? 0) - halfImageSize)
        .attr('y', (d: ScatterData) => yScale(d.imageY ?? 0) - halfImageSize)
        .attr('width', adjustedImageSize)
        .attr('height', adjustedImageSize)
        .style('opacity', 0.9)
        .style('border-radius', `${3 / currentZoomScale}px`)
        .style('box-shadow', `0 0 ${6 / currentZoomScale}px rgba(0,0,0,0.3)`);

      const imagesEnter = images.enter()
        .append('image')
        .attr('class', 'scatter-image')
        .attr('x', (d: ScatterData) => xScale(d.imageX ?? 0) - halfImageSize)
        .attr('y', (d: ScatterData) => yScale(d.imageY ?? 0) - halfImageSize)
        .attr('width', adjustedImageSize)
        .attr('height', adjustedImageSize)
        .attr('href', d => d.thumbUrl || d.imageUrl || '')
        .style('opacity', 0.9)
        .style('border-radius', `${3 / currentZoomScale}px`)
        .style('box-shadow', `0 0 ${6 / currentZoomScale}px rgba(0,0,0,0.3)`)
        .style('cursor', 'pointer');

      imagesEnter
        .on('mouseover', (_event, d) => {
          tooltip
            .text(d.title)
            .style('visibility', 'visible');
        })
        .on('mousemove', (event) => {
          tooltip
            .style('top', (event.pageY - 30) + 'px')
            .style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', () => {
          tooltip.style('visibility', 'hidden');
        });

      // ==== 绘制标题 ====
      const titleGroups = titleLayer
        .selectAll<SVGGElement, ScatterData>('g.image-title-group')
        .data(imagesWithUrl, d => `${d.x}-${d.y}-${d.thumbUrl}`);

      titleGroups.exit().remove();
      titleGroups.remove();

      const allTitleGroups = titleLayer
        .selectAll<SVGGElement, ScatterData>('g.image-title-group')
        .data(imagesWithUrl, d => `${d.x}-${d.y}-${d.thumbUrl}`)
        .enter()
        .append('g')
        .attr('class', 'image-title-group')
        .attr('transform', (d: ScatterData) => {
          const imageY = yScale(d.imageY ?? 0);
          const imageX = xScale(d.imageX ?? 0);
          const anchorY = yScale(d.y);
          const anchorX = xScale(d.x);
          let offsetY = 0;
          if (imageY < anchorY && Math.abs(imageY - anchorY) > Math.abs(imageX - anchorX)) {
            offsetY = -textYOffset + 10 / currentZoomScale;
            const titleLines = splitIntoTwoLines(d.key_word || '');
            if (titleLines.length > 1) {
              offsetY -= 10 / currentZoomScale;
            }
          } else {
            offsetY = textYOffset;
          }
          return `translate(${imageX}, ${imageY + offsetY})`;
        });

      allTitleGroups.each(function (d) {
        const g = d3.select(this);
        const lines = splitIntoTwoLines(d.key_word || '');
        lines.forEach((line, i) => {
          g.append('text')
            .attr('class', 'title-stroke')
            .attr('text-anchor', 'middle')
            .attr('font-size', `${fontSize}px`)
            .attr('stroke', 'white')
            .attr('stroke-width', `${strokeWidth}px`)
            .attr('fill', 'none')
            .attr('font-weight', 'bold')
            .style('pointer-events', 'none')
            .attr('dy', `${i * 12.0 / currentZoomScale}`)
            .text(line);
        });
        lines.forEach((line, i) => {
          g.append('text')
            .attr('class', 'title-fill')
            .attr('text-anchor', 'middle')
            .attr('font-size', `${fontSize}px`)
            .attr('stroke', 'none')
            .attr('fill', '#666')
            .attr('font-weight', 'bold')
            .style('pointer-events', 'none')
            .attr('dy', `${i * 12.0 / currentZoomScale}`)
            .text(line);
        });
      });

      borders.raise();
      images.raise();
    }
  }

  else {
    // 如果不显示图片，清除所有相关内容，包括标题和连线
    imageGroup.selectAll('rect.image-border').remove();
    imageGroup.selectAll('image').remove();
    imageGroup.selectAll('text.image-title').remove();
    imageGroup.selectAll('g.image-title-group').remove();
    imageGroup.selectAll('path.image-line').remove(); // 改为清除 path 元素
    imageGroup.selectAll('path.image-line-bg').remove();
    imageGroup.selectAll('circle.line-dot').remove();
    imageGroup.selectAll('circle.line-dot-bg').remove();
  }
};


// Lasso 选择
let isLassoing = false;
let lassoPoints: number[][] = [];
let lassoStart: [number, number] | null = null; // 记录起始点
const CLICK_DISTANCE_THRESHOLD = 0.5; // 单击判定阈值（像素）

const startLasso = (event: MouseEvent) => {
  if (event.button !== 0) return; // 只允许左键
  isLassoing = true;
  lassoPoints = [];
  const point = d3.pointer(event);
  lassoStart = point as [number, number];
  lassoPoints.push(point);
  drawLasso();
};

const updateLasso = (event: MouseEvent) => {
  if (!isLassoing) return;
  const point = d3.pointer(event);
  lassoPoints.push(point);
  drawLasso();
};

const endLasso = () => {
  if (!isLassoing) return;
  isLassoing = false;

  if (!lassoStart) return;

  // 判断是否为点击而非拖拽
  const endPoint = lassoPoints[lassoPoints.length - 1];
  const dx = endPoint[0] - lassoStart[0];
  const dy = endPoint[1] - lassoStart[1];
  const distance = Math.sqrt(dx * dx + dy * dy);

  if (distance < CLICK_DISTANCE_THRESHOLD) {
    // 左键点击 → 清空 lasso
    console.log('Lasso click detected, clearing lasso.');
    lassoPoints = [];
    drawLasso();
    drawLassoPolygon();
    emit('lasso-select', []);
    return;
  }

  // 否则就是有效的 Lasso
  if (lassoPoints.length > 2) {
    const currentTransform = d3.zoomTransform(svgSelection.node()!);
    const dataCoords = lassoPoints.map(point => {
      const transformedX = (point[0] - currentTransform.x) / currentTransform.k;
      const transformedY = (point[1] - currentTransform.y) / currentTransform.k;
      return [
        xScale.invert(transformedX),
        yScale.invert(transformedY)
      ];
    });
    emit('lasso-select', dataCoords);
  }

  lassoPoints = [];
  drawLasso();
};


const drawLasso = () => {
  if (!lassoGroup) return;

  if (lassoPoints.length === 0) {
    lassoGroup.selectAll('path').remove(); // 清掉所有路径
    return;
  }

  const path = lassoGroup
    .selectAll('path')
    .data([lassoPoints]);

  const lineGenerator = d3.line().curve(d3.curveCardinal.tension(0.3));

  path
    .attr('d', d => lineGenerator(d as [number, number][])!)
    .attr('stroke', '#000000')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '10,10')
    .attr('stroke-linecap', 'round')
    .attr('fill', 'none');

  path.enter()
    .append('path')
    .attr('d', d => lineGenerator(d as [number, number][])!)
    .attr('stroke', '#000000')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '10,10')
    .attr('stroke-linecap', 'round')
    .attr('fill', 'none');

  path.exit().remove();
};


// 监听属性变化
watch(() => props.heatmap, () => {
  updateScales();
  drawHeatmap();
  drawLassoPolygon();
}, { deep: true });

watch(() => props.scatterData, () => {
  updateScales();
  drawScatter();
}, { deep: true });

watch(() => props.showImages, () => {
  drawScatter();
});

watch(() => props.scatterMode, () => {
  drawScatter();
});

watch(() => props.scatterEncoding, () => {
  drawScatter();
});


// 监听缩放和平移，避免重复触发
let isInternalTransform = false;

watch(() => props.zoom, (newZoom) => {
  if (!svgSelection || isInternalTransform) return;
  isInternalTransform = true;
  currentZoomScale = newZoom;
  svgSelection.call(
    zoom.transform,
    d3.zoomIdentity
      .translate(props.pan.x, props.pan.y)
      .scale(newZoom)
  );
  setTimeout(() => { isInternalTransform = false; }, 0);
});

watch(() => props.pan, (newPan) => {
  if (!svgSelection || isInternalTransform) return;
  isInternalTransform = true;
  currentZoomScale = props.zoom;
  svgSelection.call(
    zoom.transform,
    d3.zoomIdentity
      .translate(newPan.x, newPan.y)
      .scale(props.zoom)
  );
  setTimeout(() => { isInternalTransform = false; }, 0);
}, { deep: true });

// 监听 Lasso 形状
watch(() => props.lassoShape, (newShape) => {
  lassoPoints = newShape;
  drawLasso();
  drawLassoPolygon();
}, { deep: true });

</script>
