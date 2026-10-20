import time
import numpy as np
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import random
from datetime import datetime
from datatools.slice import Slice
from heatmap_calculator import HeatmapCalculator
from datatools.reducer.runner import get_slice_different_embedding
from config import feature_type
import config
from image_placement_optimizer import ImagePlacementOptimizer


@dataclass
class ViewConfig:
    """View的配置信息"""
    id: int
    title: str
    current_slice: str
    reference_slice: str = ""
    # 系统模式，区分coreset和普通default
    mode: str = "default"
    dimension_reduction: Dict[str, str] = field(default_factory=lambda: {
        "method": "umap",
        "fitMode": "joint"
    })
    visual_encoding: Dict[str, str] = field(default_factory=lambda: {
        "background": "Density-Difference",
        "scatter": "Current or Reference"
    })
    option: Dict[str, Any] = field(default_factory=lambda: {
        "scatterMode": "Current & Reference",
        "showImages": False
    })
    operation: Dict[str, Any] = field(default_factory=lambda: {
        "fraction": 0.1,
        "saveSliceName": "",
        "samplingRatio": 0.1  # 上采样和下采样的操作比例
    })
    lasso_shape: List[List[float]] = field(default_factory=list)
    lasso_selection: List[int] = field(default_factory=list)
    lasso_selection_reference: List[int] = field(default_factory=list)
    zoom: float = 1.0
    pan: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    tool_mode: str = "lasso"
    copied_points: bool = False
    class_count: int = 10


class ViewManager:
    """管理单个View的类，包含该View的所有状态和数据处理逻辑"""

    def __init__(self, view_id: int, mode: str, max_scatter_points: int = 1000):
        """
        初始化ViewManager

        Args:
            view_id: View的唯一标识
            max_scatter_points: 散点图最大显示点数
        """
        self.view_id = view_id
        self.max_scatter_points = max_scatter_points

        # 初始化热力图计算器
        if mode == 'kde':
            self.heatmap_calculator = HeatmapCalculator(kde_sigma=0.015)
        elif mode == 'lsdd':
            self.heatmap_calculator = HeatmapCalculator(lsdd_sigma=0.015, lsdd_lambda=0.015)
        else:
            self.heatmap_calculator = HeatmapCalculator()

        self.heatmap_cache = None

        # 初始化图片位置优化器
        self.image_optimizer = ImagePlacementOptimizer()
        self.mode = mode
        # 不同mode的配置
        if mode == 'scatter':
            # 初始化View配置
            self.config = ViewConfig(
                mode=mode,
                id=view_id,
                title=f"View {view_id}",
                current_slice="",
                reference_slice=""
            )
            self.config.visual_encoding["background"] = 'None'

        elif mode == 'kde-scatter':
            self.config = ViewConfig(
                mode=mode,
                id=view_id,
                title=f"View {view_id}",
                current_slice="",
                reference_slice=""
            )
        else:
            self.config = ViewConfig(
                mode=mode,
                id=view_id,
                title=f"View {view_id}",
                current_slice="",
                reference_slice=""
            )
            self.config.visual_encoding["scatter"] = 'None'
            if mode in ['kde', 'lsdd']:
                self.config.dimension_reduction["fitMode"] = 'coreset'


        # 数据缓存 - 使用Slice对象
        self._current_slice: Optional[Slice] = None
        self._reference_slice: Optional[Slice] = None
        self._current_slice_embedding: Optional[np.ndarray] = None
        self._reference_slice_embedding: Optional[np.ndarray] = None

        # 核心集相关，全集的切片和embedding
        self._whole_slice: Optional[Slice] = None
        self._whole_slice_embedding: Optional[np.ndarray] = None

        # 采样相关
        self._current_thumb_indices: List[int] = []  # 缩略图不为空的索引
        self._reference_thumb_indices: List[int] = []
        self._current_sampled_indices: List[int] = []
        self._reference_sampled_indices: List[int] = []
        self._viewport_bounds: Dict[str, float] = {
            "x_min": 0, "x_max": 1,
            "y_min": 0, "y_max": 1
        }

        # 视口采样状态
        self._previous_viewport_bounds: Dict[str, float] = {
            "x_min": 0, "x_max": 1,
            "y_min": 0, "y_max": 1
        }
        # 分别为current和reference切片设置基础采样比例
        self._current_base_sampling_ratio: float = 0.0  # Current切片基础采样比例
        self._reference_base_sampling_ratio: float = 0.0  # Reference切片基础采样比例
        self._reference_sampled_points: Dict[int, tuple] = {}  # 参考切片采样点的坐标缓存

        # 统计信息
        self._info = {
            "currentSliceShown": 0,
            "currentSliceTotal": 0,
            "referenceSliceShown": 0,
            "referenceSliceTotal": 0,
            "selectionShown": 0,
            "selectionTotal": 0
        }

    def load_slice(self, slice_name: str) -> Optional[Slice]:
        """加载切片对象"""
        if not slice_name:
            return None

        try:
            # 使用Slice类加载
            slice_obj = Slice(slice_name)
            return slice_obj
        except (FileNotFoundError, Exception) as e:
            print(f"Error loading slice {slice_name}: {e}")
            return None

    def get_slice_embedding(self, slice_obj: Slice = None, father=False) -> np.ndarray:
        """
        获取切片的二维嵌入坐标并归一化到[0,1] x [0,1]
        根据降维配置使用不同的降维方法和模式
        """

        # 使用降维函数，slice1固定为current，slice2固定为reference
        raw_embedding1, raw_embedding2, combined = get_slice_different_embedding(
            feature_type=feature_type,
            slice1=self._current_slice,  # 固定为current
            slice2=self._reference_slice,  # 固定为reference
            reducer_name=self.config.dimension_reduction["method"],
            fit_mode=self.config.dimension_reduction["fitMode"],
            use_regularization=config.USE_REGULARIZATION and self.config.mode not in ["kde-scatter", "kde", "lsdd"]
        )
        if father:
            return combined
        if slice_obj is not None:
            return raw_embedding1 if slice_obj.name == self._current_slice.name else raw_embedding2
        return raw_embedding1, raw_embedding2

    def _generate_random_embedding(self, n_samples: int) -> np.ndarray:
        """生成随机嵌入坐标作为fallback"""
        np.random.seed(42)  # 确保可重复性
        return np.random.rand(n_samples, 2)

    def _normalize_embedding(self, embedding: np.ndarray, padding: float = 0.1) -> np.ndarray:
        """
        将embedding归一化到[padding, 1-padding] x [padding, 1-padding]范围，保持长宽比，中心在(0.5, 0.5)

        Args:
            embedding: 原始二维嵌入坐标 (N, 2)
            padding: 边界padding，默认0.1，数据将归一化到[padding, 1-padding]范围

        Returns:
            归一化后的坐标 (N, 2)
        """
        if len(embedding) == 0:
            return embedding

        # 计算当前数据的边界
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

        # 计算当前范围大小
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 避免除零错误
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0

        # 计算可用范围（减去两边的padding）
        available_range = 1.0 - 2 * padding

        # 计算缩放因子，保持长宽比（以长边为准），并考虑padding
        scale_factor = available_range / max(x_range, y_range)

        # 计算当前数据的中心
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # 居中并缩放到[padding, 1-padding]范围
        normalized_embedding = embedding.copy()
        normalized_embedding[:, 0] = (
            embedding[:, 0] - x_center) * scale_factor + 0.5
        normalized_embedding[:, 1] = (
            embedding[:, 1] - y_center) * scale_factor + 0.5

        return normalized_embedding

    def set_current_slice(self, slice_name: str) -> bool:
        """设置当前切片"""
        slice_obj = self.load_slice(slice_name)
        if slice_obj is None:
            return False

        self.config.current_slice = slice_name
        self._current_slice = slice_obj

        # 自动检测类别数
        self.config.class_count = self._detect_class_count(slice_name)

        # 全集设置
        self._whole_slice = slice_obj.get_father_slice()

        return True

    def _detect_class_count(self, slice_name: str) -> int:
        """从Slice的label字段中检测实际的类别数"""
        try:
            # 如果当前slice已经加载，直接从中获取
            if self._current_slice is not None:
                labels = self._current_slice.get_field('label')
                if labels is not None and len(labels) > 0:
                    # 获取label的最大值 + 1（假设label从0开始）
                    max_label = max(labels)
                    return int(max_label) + 1

            return 10

        except Exception as e:
            print(f"Error detecting class count from slice {slice_name}: {e}")

        # 如果无法从数据中获取，则使用默认值
        return 10

    def set_reference_slice(self, slice_name: str) -> bool:
        """设置参考切片"""
        if not slice_name:
            self.config.reference_slice = ""
            self._reference_slice = None
            self._reference_slice_embedding = None
            self._reference_sampled_indices = []
            self._update_info()
            return True

        slice_obj = self.load_slice(slice_name)
        if slice_obj is None:
            return False

        self.config.reference_slice = slice_name
        self._reference_slice = slice_obj

        return True

    def _reload_all(self):
        if self._current_slice is None or self._reference_slice is None:
            return

        """重载所有数据"""
        print('reloading')
        self._get_thumb_indices_in_view()
        # 重置视口采样状态
        self._current_sampled_indices = []
        self._reference_sampled_indices = []

        # 强制重新初始化采样比例
        self._current_base_sampling_ratio = 0.0
        self._reference_base_sampling_ratio = 0.0

        """使用log尺度初始化基础采样比例"""
        self._calculate_log_scale_sampling_ratios()

        self._resample_data()

    def _calculate_log_scale_sampling_ratios(self):
        """
        使用log尺度计算Current和Reference切片的基础采样比例
        确保数量级差距大的两个切片都能有合理的可见性
        """
        import math

        # 获取两个切片的点数
        current_count = 0
        reference_count = 0

        if self._current_slice_embedding is not None and len(self._current_slice_embedding) > 0:
            current_count = len(self._current_slice_embedding)
        if self._reference_slice_embedding is not None and len(self._reference_slice_embedding) > 0:
            reference_count = len(self._reference_slice_embedding)

        print(
            f"cur name: {self._current_slice.name}, ref name: {self._reference_slice.name}")
        print(
            f"current_count: {current_count}, reference_count: {reference_count}")

        if current_count == 0 and reference_count == 0:
            self._current_base_sampling_ratio = 1.0
            self._reference_base_sampling_ratio = 1.0
            print("No data points, using default sampling ratios")
            return

        if current_count == 0:
            self._current_base_sampling_ratio = 0.0
            self._reference_base_sampling_ratio = min(
                1.0, self.max_scatter_points / reference_count)
            print(
                f"Only reference slice: ratio={self._reference_base_sampling_ratio:.4f}")
            return

        if reference_count == 0:
            self._current_base_sampling_ratio = min(
                1.0, self.max_scatter_points / current_count)
            self._reference_base_sampling_ratio = 0.0
            print(
                f"Only current slice: ratio={self._current_base_sampling_ratio:.4f}")
            return

        # 使用log尺度来计算权重，避免数量级差距过大的问题
        # 使用log10来计算，为较小的数据集提供更多权重
        log_current = math.log10(max(1, current_count))
        log_reference = math.log10(max(1, reference_count))
        total_log_weight = log_current + log_reference

        # 根据log权重分配最大显示点数
        # 给数量少的切片更多的相对权重
        current_weight = log_current / total_log_weight
        reference_weight = log_reference / total_log_weight

        # 分配点数：确保每个切片至少能分配到一定比例的点数
        min_allocation_ratio = 0.0  # 最少分配20%的点数

        current_allocation = max(min_allocation_ratio,
                                 current_weight) * self.max_scatter_points
        reference_allocation = max(
            min_allocation_ratio, reference_weight) * self.max_scatter_points

        # 如果总分配超过最大点数，按比例缩放
        total_allocation = current_allocation + reference_allocation
        if total_allocation > self.max_scatter_points:
            scale_factor = self.max_scatter_points / total_allocation
            current_allocation *= scale_factor
            reference_allocation *= scale_factor

        # 计算最终的采样比例
        self._current_base_sampling_ratio = min(
            1.0, current_allocation / current_count)
        self._reference_base_sampling_ratio = min(
            1.0, reference_allocation / reference_count)

        print(f"Log-scale sampling ratios calculated:")
        print(
            f"  Current: {current_count} points → ratio={self._current_base_sampling_ratio:.4f} (target: {int(current_allocation)})")
        print(
            f"  Reference: {reference_count} points → ratio={self._reference_base_sampling_ratio:.4f} (target: {int(reference_allocation)})")
        print(
            f"  Log weights: current={current_weight:.3f}, reference={reference_weight:.3f}")

    def update_viewport(self, viewport: Dict[str, float]):
        """
        更新当前视图的视口范围

        Args:
            x_min: 视口x最小值
            x_max: 视口x最大值
            y_min: 视口y最小值
            y_max: 视口y最大值
        """
        self._viewport_bounds = viewport
        # 视口变化后，重新采样点
        self._resample_data()

    def _resample_data(self):
        """根据当前视口更新采样点数（只统计视口内的点数）"""
        self.heatmap_cache = None

        print("Simple resampling...")
        t0 = time.time()

        # 当前切片视口内点数
        if self._current_slice_embedding is not None:
            in_viewport = (
                (self._current_slice_embedding[:, 0] >= self._viewport_bounds["x_min"]) &
                (self._current_slice_embedding[:, 0] <= self._viewport_bounds["x_max"]) &
                (self._current_slice_embedding[:, 1] >= self._viewport_bounds["y_min"]) &
                (self._current_slice_embedding[:, 1]
                 <= self._viewport_bounds["y_max"])
            )
            self._current_sampled_indices = np.where(in_viewport)[0].tolist()
        else:
            self._current_sampled_indices = []

        # 参考切片视口内点数
        if self._reference_slice_embedding is not None:
            in_viewport = (
                (self._reference_slice_embedding[:, 0] >= self._viewport_bounds["x_min"]) &
                (self._reference_slice_embedding[:, 0] <= self._viewport_bounds["x_max"]) &
                (self._reference_slice_embedding[:, 1] >= self._viewport_bounds["y_min"]) &
                (self._reference_slice_embedding[:, 1]
                 <= self._viewport_bounds["y_max"])
            )
            self._reference_sampled_indices = np.where(in_viewport)[0].tolist()
        else:
            self._reference_sampled_indices = []

        t1 = time.time()
        print(
            f"[resample_data] 采样点数: 当前 {len(self._current_sampled_indices)}, 参考 {len(self._reference_sampled_indices)}, 耗时: {(t1-t0)*1000:.2f}ms")

        self._update_info()

    def _intelligent_sample_viewport(self, embedding: Optional[np.ndarray],
                                     current_indices: List[int],
                                     slice_type: str) -> tuple[List[int], Dict[int, tuple]]:
        """
        视口采样：保留重叠区域内的采样点

        Args:
            embedding: 嵌入坐标
            current_indices: 当前采样的点索引
            current_points: 当前采样点的坐标缓存
            slice_type: 切片类型("current" 或 "reference")

        Returns:
            (new_indices, new_points): 新的采样点索引和坐标缓存
        """
        if embedding is None or len(embedding) == 0:
            return [], {}

        if len(current_indices) == 0:
            old_viewport = {
                "x_min": 0,
                "x_max": 0,
                "y_min": 0,
                "y_max": 0
            }
        else:
            old_viewport = self._previous_viewport_bounds

        # 计算视口区域
        new_viewport = self._viewport_bounds

        # print(f'old_viewport: {old_viewport}, new_viewport: {new_viewport}')

        # 根据视口面积变化调整采样比例
        initial_area = 1.0  # 初始视口[0,1]x[0,1]的面积
        old_viewport_area = (old_viewport["x_max"] - old_viewport["x_min"]) * (
            old_viewport["y_max"] - old_viewport["y_min"])
        new_viewport_area = (new_viewport["x_max"] - new_viewport["x_min"]) * (
            new_viewport["y_max"] - new_viewport["y_min"])
        old_area_ratio = old_viewport_area / initial_area
        new_area_ratio = new_viewport_area / initial_area

        # 根据切片类型选择对应的基础采样比例
        if slice_type == "current":
            base_sampling_ratio = self._current_base_sampling_ratio
        else:  # reference
            base_sampling_ratio = self._reference_base_sampling_ratio

        # 调整后的采样比例
        if len(current_indices) == 0:
            old_sampling_ratio = 0.0
        else:
            old_sampling_ratio = base_sampling_ratio / old_area_ratio

        new_sampling_ratio = base_sampling_ratio / new_area_ratio
        new_sampling_ratio = min(1.0, new_sampling_ratio)  # 不超过100%

        # print(f'{slice_type}_base_sampling_ratio: {base_sampling_ratio}, old_sampling_ratio: {old_sampling_ratio}, new_sampling_ratio: {new_sampling_ratio}')

        in_old_viewport = (
            (embedding[:, 0] >= old_viewport["x_min"]) & (embedding[:, 0] <= old_viewport["x_max"]) &
            (embedding[:, 1] >= old_viewport["y_min"]) & (
                embedding[:, 1] <= old_viewport["y_max"])
        )
        in_new_viewport = (
            (embedding[:, 0] >= new_viewport["x_min"]) & (embedding[:, 0] <= new_viewport["x_max"]) &
            (embedding[:, 1] >= new_viewport["y_min"]) & (
                embedding[:, 1] <= new_viewport["y_max"])
        )

        in_extra_area_total = in_new_viewport & ~in_old_viewport
        in_extra_area_total_indices = np.where(in_extra_area_total)[0]
        in_remain_area_total = in_old_viewport & in_new_viewport
        in_remain_area_total_indices = np.where(in_remain_area_total)[0]
        # print(f'in_remain_area_total_indices: {in_extra_area_total_indices}')

        extra_area_target_count = int(
            np.sum(in_extra_area_total) * new_sampling_ratio)
        remain_area_target_count = int(
            np.sum(in_remain_area_total) * new_sampling_ratio)

        # print(f'extra_area_total_count: {len(in_extra_area_total_indices)}, remain_area_total_count: {len(in_remain_area_total_indices)}')
        # print(f'extra_area_target_count: {extra_area_target_count}, remain_area_target_count: {remain_area_target_count}')

        new_indices = []

        # 从新增区域随机采样
        extra_area_total_count = np.sum(in_extra_area_total)
        if extra_area_total_count > 0:
            sample_size = min(int(extra_area_target_count),
                              extra_area_total_count)
            if sample_size > 0:
                sampled_indices = np.random.choice(
                    in_extra_area_total_indices, size=sample_size, replace=False)
                new_indices.extend(sampled_indices.tolist())

        # 处理重叠区域
        if len(in_remain_area_total_indices) > 0:
            remain_area_indices = list(
                set(in_remain_area_total_indices) & set(current_indices))
            if len(remain_area_indices) > remain_area_target_count:
                remain_area_indices = np.random.choice(
                    remain_area_indices, size=remain_area_target_count, replace=False)
            else:
                extra_sampling_ratio = new_sampling_ratio - old_sampling_ratio
                remain_area_extra_indices = list(
                    set(in_remain_area_total_indices) - set(current_indices))
                extra_sample_size = int(
                    len(remain_area_extra_indices) * extra_sampling_ratio)
                if extra_sample_size > 0:
                    remain_area_extra_indices = np.random.choice(
                        remain_area_extra_indices, size=extra_sample_size, replace=False)
                    remain_area_indices.extend(remain_area_extra_indices)

            new_indices.extend(remain_area_indices)

        return new_indices

    def _poisson_disk_sampling_for_images(self, embedding: np.ndarray, all_indices: List[int], image_size_in_data_coords: float = 0.1) -> List[int]:
        """
        对要显示图片的点进行泊松圆盘采样

        Args:
            embedding: 嵌入坐标
            all_indices: 所有候选点的索引
            image_size_in_data_coords: 图片大小在数据坐标系中的尺寸（用作圆盘半径）

        Returns:
            List[int]: 采样后的点索引列表
        """
        import time
        total_start = time.perf_counter()
        step_start = time.perf_counter()

        if len(all_indices) == 0:
            return []

        # 步骤1: 随机选择候选点
        if len(all_indices) > 1000:
            np.random.seed(42)
            all_indices = np.random.choice(
                all_indices, 1000, replace=False).tolist()
        step_end = time.perf_counter()
        print(f"步骤1-随机选择候选点: {(step_end - step_start)*1000:.2f}ms")

        # 确保embedding是numpy数组
        step_start = time.perf_counter()
        embedding = np.array(embedding)
        step_end = time.perf_counter()
        print(f"步骤2-转换嵌入坐标: {(step_end - step_start)*1000:.2f}ms")

        # 步骤3: 获取热力图缓存
        step_start = time.perf_counter()
        heatmap_cache = self.heatmap_cache
        if heatmap_cache is None:
            # 如果热力图缓存不存在，则生成它
            heatmap_cache = self.get_heatmap_data()
            self.heatmap_cache = heatmap_cache
        step_end = time.perf_counter()
        print(f"步骤3-获取热力图缓存: {(step_end - step_start)*1000:.2f}ms")

        # 步骤4: 构建热力图网格索引
        step_start = time.perf_counter()
        back_rects = [
            rect for rect in heatmap_cache if rect.get('type') == 'back']
        if not back_rects:
            point_values = [0.0] * len(all_indices)
            return point_values
        step_end = time.perf_counter()
        print(f"步骤4-构建热力图网格索引: {(step_end - step_start)*1000:.2f}ms")

        # 步骤5: 创建KDTree索引
        step_start = time.perf_counter()
        from scipy.spatial import KDTree

        # 准备网格中心点和值
        centers = []
        values = []
        for rect in back_rects:
            center_x = (rect['x0'] + rect['x1']) / 2.0
            center_y = (rect['y0'] + rect['y1']) / 2.0
            centers.append([center_x, center_y])
            values.append(rect.get('value', 0.0))

        # 创建KDTree索引
        kdtree = KDTree(centers)
        step_end = time.perf_counter()
        print(f"步骤5-创建KDTree索引: {(step_end - step_start)*1000:.2f}ms")

        # 步骤6: 准备点坐标并批量查询
        step_start = time.perf_counter()
        points_to_query = embedding[all_indices]
        distances, indices = kdtree.query(points_to_query, k=1)
        step_end = time.perf_counter()
        print(f"步骤6-批量查询最近网格: {(step_end - step_start)*1000:.2f}ms")

        # 步骤7: 计算点价值
        step_start = time.perf_counter()
        point_values = []
        for i, idx in enumerate(indices):
            # 获取最近网格的值
            value = values[idx]
            # 计算价值（密度差异绝对值）
            point_values.append(abs(value - 0.5))
        step_end = time.perf_counter()
        print(f"步骤7-计算点价值: {(step_end - step_start)*1000:.2f}ms")

        # 步骤8: 根据价值排序
        step_start = time.perf_counter()
        sorted_indices = sorted(
            range(len(all_indices)),
            key=lambda i: point_values[i],
            reverse=True
        )
        sorted_indices_list = [all_indices[i] for i in sorted_indices]
        step_end = time.perf_counter()
        print(f"步骤8-根据价值排序: {(step_end - step_start)*1000:.2f}ms")

        # 步骤9: 泊松圆盘采样
        step_start = time.perf_counter()
        sampled_indices = []
        for idx in sorted_indices_list:
            x, y = embedding[idx]
            too_close = False

            # 检查与已选点的距离
            for selected_idx in sampled_indices:
                selected_x, selected_y = embedding[selected_idx]
                distance = np.sqrt((x - selected_x)**2 + (y - selected_y)**2)
                if distance < image_size_in_data_coords:
                    too_close = True
                    break

            if not too_close:
                sampled_indices.append(idx)
        step_end = time.perf_counter()
        print(f"步骤9-泊松圆盘采样: {(step_end - step_start)*1000:.2f}ms")

        # 总耗时
        total_end = time.perf_counter()
        print(f"泊松圆盘采样总耗时: {(total_end - total_start)*1000:.2f}ms")

        return sampled_indices

    def _optimize_image_positions(self, merged_metadata, sampled_set, image_size_ratio):
        """
        优化需要显示图片的点的位置，将它们放置在密度差异较小的区域

        Args:
            merged_metadata: 合并的元数据列表
            sampled_set: 需要显示图片的索引集合
            image_size_ratio: 图片大小比例

        Returns:
            Dict[int, Tuple[float, float]]: 优化后的图片位置字典
        """
        try:
            # 获取需要显示图片的点
            image_indices = []
            image_positions = []

            for i, (source, idx) in enumerate(merged_metadata):
                if i in sampled_set:
                    image_indices.append(idx)
                    if source == "current":
                        x, y = self._current_slice_embedding[idx]
                    else:
                        x, y = self._reference_slice_embedding[idx]
                    image_positions.append((x, y))

            if not image_positions:
                return {}

            # 获取热力图数据用于计算密度差异
            heatmap_data = self.get_heatmap_data()

            # 计算图片大小比例
            viewport_width = self._viewport_bounds["x_max"] - \
                self._viewport_bounds["x_min"]
            viewport_height = self._viewport_bounds["y_max"] - \
                self._viewport_bounds["y_min"]

            # 调用优化器
            placements = self.image_optimizer.optimize_image_positions(
                image_positions=image_positions,
                heatmap_data=heatmap_data,
                viewport=self._viewport_bounds,
                image_size_ratio=image_size_ratio
            )

            # 构建优化后的位置字典
            optimized_pos = {}
            for placement in placements:
                if placement.image_id < len(image_indices):
                    idx = image_indices[placement.image_id]
                    optimized_pos[idx] = (
                        placement.target_x, placement.target_y)

            # 为没有优化位置的图片使用原始位置
            for i, (source, idx) in enumerate(merged_metadata):
                if i in sampled_set and idx not in optimized_pos:
                    if source == "current":
                        x, y = self._current_slice_embedding[idx]
                    else:
                        x, y = self._reference_slice_embedding[idx]
                    optimized_pos[idx] = (x, y)

            print(f"成功优化了 {len(placements)} 张图片的位置")
            return optimized_pos

        except Exception as e:
            print(f"图片位置优化失败: {e}")
            return {}

    def _get_thumb_indices_in_view(self):
        """获取当前视图中缩略图不为空且在视口范围内的索引，使用向量化加速"""
        # 获取缩略图不为空的索引
        self._current_thumb_indices = self._current_slice.get_title_indices()
        self._reference_thumb_indices = self._reference_slice.get_title_indices()

    def align_points_to_heatmap_grid(self, points: np.ndarray) -> np.ndarray:
        """
        将输入的一组点对齐到heatmap cache中的grid的中心点。
        Args:
            points (np.ndarray): 形状为 (N, 2) 的点坐标数组
        Returns:
            np.ndarray: 对齐后的点坐标数组，形状为 (N, 2)
        """
        # 检查heatmap_cache是否存在且有效
        if not hasattr(self, "heatmap_cache") or not self.heatmap_cache:
            raise ValueError("heatmap cache中没有数据，请先生成热力图。")

        # 从热力图缓存中提取所有背景网格的中心点
        grid_centers = []
        for rect in self.heatmap_cache:
            if rect.get('type') == 'back':  # 只处理背景网格
                x0, y0 = rect['x0'], rect['y0']
                x1, y1 = rect['x1'], rect['y1']
                center_x = (x0 + x1) / 2.0
                center_y = (y0 + y1) / 2.0
                grid_centers.append([center_x, center_y])

        if not grid_centers:
            raise ValueError("heatmap cache中没有找到网格数据。")

        grid_centers = np.array(grid_centers)  # 转换为numpy数组

        # 对每个点，找到最近的grid center
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_centers)
        _, idxs = tree.query(points)
        aligned_points = grid_centers[idxs]
        return aligned_points

    def get_scatter_data(self) -> List[Dict[str, Any]]:
        import time
        total_start = time.time()
        print(f"[get_scatter_data] 开始处理散点数据")

        poisson_points = []
        poisson_indices = []
        scatter_data = []
        merged_points = []
        merged_metadata = []  # 保存 (source, original_idx) 的元组
        current_selected_set = set(self.config.lasso_selection)

        viewport_width = self._viewport_bounds["x_max"] - \
            self._viewport_bounds["x_min"]
        viewport_height = self._viewport_bounds["y_max"] - \
            self._viewport_bounds["y_min"]

        image_size_ratio = 4 / 80
        image_size_in_data = max(
            viewport_width, viewport_height) * image_size_ratio

        # 如果模式不为'default', 先随机采样，再在这些点里做 poisson 采样决定 thumbUrl
        if self.config.mode == 'kde-scatter':
            max_points = 100

            scatter_data = []
            current_selected_set = set(self.config.lasso_selection)

            all_points = []       # 保存采样点的坐标
            all_metadata = []     # 保存 (source, idx, sliceName, selected) 信息

            # 统计current和reference视窗内的点数
            current_in_viewport = []
            reference_in_viewport = []
            current_total = 0
            reference_total = 0

            if self._current_slice_embedding is not None:
                in_viewport = (
                    (self._current_slice_embedding[:, 0] >= self._viewport_bounds["x_min"]) &
                    (self._current_slice_embedding[:, 0] <= self._viewport_bounds["x_max"]) &
                    (self._current_slice_embedding[:, 1] >= self._viewport_bounds["y_min"]) &
                    (self._current_slice_embedding[:, 1] <= self._viewport_bounds["y_max"])
                )
                current_in_viewport = np.where(in_viewport)[0]
                current_total = len(current_in_viewport)
            if self._reference_slice_embedding is not None:
                in_viewport = (
                    (self._reference_slice_embedding[:, 0] >= self._viewport_bounds["x_min"]) &
                    (self._reference_slice_embedding[:, 0] <= self._viewport_bounds["x_max"]) &
                    (self._reference_slice_embedding[:, 1] >= self._viewport_bounds["y_min"]) &
                    (self._reference_slice_embedding[:, 1] <= self._viewport_bounds["y_max"])
                )
                reference_in_viewport = np.where(in_viewport)[0]
                reference_total = len(reference_in_viewport)

            total = current_total + reference_total
            # 按比例分配采样数
            if total == 0:
                current_sample_num = 0
                reference_sample_num = 0
            else:
                current_sample_num = int(round(max_points * current_total / total))
                reference_sample_num = max_points - current_sample_num  # 保证总数为max_points

            np.random.seed(42)
            # 采样current
            if current_total > 0 and current_sample_num > 0:
                if current_total <= current_sample_num:
                    sampled_current_indices = current_in_viewport
                else:
                    sampled_current_indices = np.random.choice(
                        current_in_viewport, current_sample_num, replace=False)
                for idx in sampled_current_indices:
                    all_points.append(self._current_slice_embedding[idx])
                    all_metadata.append(
                        ("current", idx, self.config.current_slice, (idx in current_selected_set))
                    )
            # 采样reference
            if reference_total > 0 and reference_sample_num > 0:
                if reference_total <= reference_sample_num:
                    sampled_reference_indices = reference_in_viewport
                else:
                    sampled_reference_indices = np.random.choice(
                        reference_in_viewport, reference_sample_num, replace=False)
                for idx in sampled_reference_indices:
                    all_points.append(self._reference_slice_embedding[idx])
                    all_metadata.append(
                        ("reference", idx, self.config.reference_slice, (idx in current_selected_set))
                    )

            # --- 在合并的采样点上跑一次 Poisson ---
            if all_points:
                poisson_keep = self._poisson_disk_sampling_for_images(
                    np.array(all_points),
                    list(range(len(all_points))),
                    image_size_in_data * 3
                )
                poisson_keep_set = set(poisson_keep)
            else:
                poisson_keep_set = set()

            # --- 构建返回数据 ---
            for i, (source, idx, slice_name, selected) in enumerate(all_metadata):
                scatter_data.append({
                    "id": f"{'ref_' if source=='reference' else ''}{idx}",
                    "x": float(all_points[i][0]),
                    "y": float(all_points[i][1]),
                    "sliceName": slice_name,
                    "imageUrl": self._get_image_url_for_point(idx, source),
                    "label": self._get_label_for_point(idx, source),
                    "value": self._get_value_for_point(idx, source),
                    "thumbUrl": self._get_image_url_for_point(idx, source, thumb=True) if i in poisson_keep_set else None,
                    "source": source,
                    "title": self._get_title_for_point(idx, source),
                    "key_word": self._get_key_word_for_point(idx, source),
                    "selected": selected
                })

            print(
                f"[get_scatter_data] 普通模式返回点数: {len(scatter_data)}, 其中带缩略图 {len(poisson_keep_set)}")
            return scatter_data



        # 获取视窗内的所有点
        t0 = time.time()
        if self._current_slice_embedding is not None:
            # 筛选视窗内的点
            in_viewport = (
                (self._current_slice_embedding[:, 0] >= self._viewport_bounds["x_min"]) &
                (self._current_slice_embedding[:, 0] <= self._viewport_bounds["x_max"]) &
                (self._current_slice_embedding[:, 1] >= self._viewport_bounds["y_min"]) &
                (self._current_slice_embedding[:, 1] <= self._viewport_bounds["y_max"])
            )
            current_indices = np.where(in_viewport)[0]

            # 找到视窗内缩略图不为空的点
            current_thumb_mask = np.isin(
                current_indices, self._current_thumb_indices)
            merged_points.extend(self._current_slice_embedding[current_indices])
            merged_metadata.extend([("current", idx)
                                   for idx in current_indices])
            poisson_points.extend(
                self._current_slice_embedding[current_indices[current_thumb_mask]])
            poisson_indices.extend((np.where(current_thumb_mask)[
                                   0] + len(merged_points) - len(current_indices)).tolist())

        if self._reference_slice_embedding is not None:
            # 筛选视窗内的点
            in_viewport = (
                (self._reference_slice_embedding[:, 0] >= self._viewport_bounds["x_min"]) &
                (self._reference_slice_embedding[:, 0] <= self._viewport_bounds["x_max"]) &
                (self._reference_slice_embedding[:, 1] >= self._viewport_bounds["y_min"]) &
                (self._reference_slice_embedding[:, 1] <= self._viewport_bounds["y_max"])
            )
            reference_indices = np.where(in_viewport)[0]

            reference_thumb_mask = np.isin(
                reference_indices, self._reference_thumb_indices)
            merged_points.extend(
                self._reference_slice_embedding[reference_indices])
            merged_metadata.extend([("reference", idx)
                                   for idx in reference_indices])
            poisson_points.extend(
                self._reference_slice_embedding[reference_indices[reference_thumb_mask]])
            poisson_indices.extend((np.where(reference_thumb_mask)[
                                   0] + len(merged_points) - len(reference_indices)).tolist())

        t1 = time.time()
        print(f"[get_scatter_data] 获取视窗内点耗时: {(t1 - t0)*1000:.2f}ms, "
              f"当前切片点数: {len(poisson_points)//2 if poisson_points else 0}, "
              f"参考切片点数: {len(poisson_points)//2 if poisson_points else 0}")

        t0 = time.time()
        sampled_indices = []
        sampled_indices = self._poisson_disk_sampling_for_images(
            poisson_points,
            list(range(len(poisson_points))),
            image_size_in_data * 3
        )
        t1 = time.time()
        print(
            f"[get_scatter_data] 泊松圆盘采样耗时: {(t1 - t0)*1000:.2f}ms, 采样点数: {len(sampled_indices)}")

        # 加速：直接用 numpy 索引和 set 构造
        t0 = time.time()
        if poisson_indices and sampled_indices:
            sampled_set = set(np.array(poisson_indices)[
                              np.array(sampled_indices, dtype=int)])
        else:
            sampled_set = set()
        t1 = time.time()
        print(f"[get_scatter_data] 构建采样集合耗时: {(t1 - t0)*1000:.2f}ms")

        # 优化图片位置
        t0 = time.time()
        optimized_image_pos = self._optimize_image_positions(
            merged_metadata, sampled_set, image_size_ratio)
        t1 = time.time()
        print(f"[get_scatter_data] 优化图片位置耗时: {(t1 - t0)*1000:.2f}ms")

        # 如果优化失败，使用原始位置
        t0 = time.time()
        if not optimized_image_pos:
            image_pos = {}
            for i, (source, idx) in enumerate(merged_metadata):
                if source == "current":
                    x, y = self._current_slice_embedding[idx]
                else:
                    x, y = self._reference_slice_embedding[idx]
                image_pos[idx] = (x, y)
        else:
            image_pos = optimized_image_pos
        t1 = time.time()
        print(f"[get_scatter_data] 准备图片位置耗时: {(t1 - t0)*1000:.2f}ms")

        # 对齐sampled_set中的点到热力图网格中心
        aligned_scatter_coords = {}  # 初始化字典
        t0 = time.time()
        if sampled_set:
            # 收集需要对齐的点
            points_to_align = []
            for i in sampled_set:
                source, idx = merged_metadata[i]
                if source == "current":
                    point = self._current_slice_embedding[idx]
                else:
                    point = self._reference_slice_embedding[idx]
                points_to_align.append(point)

            # 对齐点到热力图网格中心
            if points_to_align:
                try:
                    aligned = self.align_points_to_heatmap_grid(
                        np.array(points_to_align))
                    for i, point in zip(sampled_set, aligned):
                        aligned_scatter_coords[i] = point
                except Exception as e:
                    print(f"对齐点到热力图网格失败: {e}")
                    # 失败时使用原始位置
                    for i in sampled_set:
                        source, idx = merged_metadata[i]
                        if source == "current":
                            aligned_scatter_coords[i] = self._current_slice_embedding[idx]
                        else:
                            aligned_scatter_coords[i] = self._reference_slice_embedding[idx]
        t1 = time.time()
        print(
            f"[get_scatter_data] 对齐点到热力图网格耗时: {(t1 - t0)*1000:.2f}ms, 对齐点数: {len(aligned_scatter_coords)}")

        # 构建散点数据
        t0 = time.time()
        scatter_data = []
        for i in sorted(sampled_set):
            source, idx = merged_metadata[i]

            # 如果点在sampled_set中，使用对齐后的位置
            if i in aligned_scatter_coords:
                x, y = aligned_scatter_coords[i]
            else:
                # 否则使用原始位置
                if source == "current":
                    x, y = self._current_slice_embedding[idx]
                else:
                    x, y = self._reference_slice_embedding[idx]

            # 构建数据项
            if source == "current":
                scatter_data.append({
                    "id": int(idx),
                    "x": float(x),
                    "y": float(y),
                    "sliceName": self.config.current_slice,
                    "label": self._get_label_for_point(idx, "current"),
                    "value": self._get_value_for_point(idx, "current"),
                    "imageUrl": self._get_image_url_for_point(idx, "current") if i in sampled_set else None,
                    "imageX": float(image_pos[idx][0]) if i in sampled_set else None,
                    "imageY": float(image_pos[idx][1]) if i in sampled_set else None,
                    "thumbUrl": self._get_image_url_for_point(idx, "current", thumb=True) if i in sampled_set else None,
                    "source": "current",
                    "title": self._get_title_for_point(idx, "current") if i in sampled_set else None,
                    "key_word": self._get_key_word_for_point(idx, "current") if i in sampled_set else None,
                    "selected": idx in current_selected_set
                })
            else:
                scatter_data.append({
                    "id": f"ref_{idx}",
                    "x": float(x),
                    "y": float(y),
                    "sliceName": self.config.reference_slice,
                    "label": self._get_label_for_point(idx, "reference"),
                    "value": self._get_value_for_point(idx, "reference"),
                    "imageUrl": self._get_image_url_for_point(idx, "reference") if i in sampled_set else None,
                    "imageX": float(image_pos[idx][0]) if i in sampled_set else None,
                    "imageY": float(image_pos[idx][1]) if i in sampled_set else None,
                    "thumbUrl": self._get_image_url_for_point(idx, "reference", thumb=True) if i in sampled_set else None,
                    "source": "reference",
                    "title": self._get_title_for_point(idx, "reference") if i in sampled_set else None,
                    "key_word": self._get_key_word_for_point(idx, "reference") if i in sampled_set else None,
                    "selected": False
                })
        t1 = time.time()
        print(
            f"[get_scatter_data] 构建散点数据耗时: {(t1 - t0)*1000:.2f}ms, 总点数: {len(scatter_data)}")

        total_end = time.time()
        print(f"[get_scatter_data] 总耗时: {(total_end - total_start)*1000:.2f}ms")
        return scatter_data

    def get_heatmap_data(self) -> List[Dict[str, Any]]:
        """获取热力图数据"""
        # 如果提供了heat_type参数，使用该类型；否则使用配置中的类型
        background_type = self.config.visual_encoding["background"]

        if self.heatmap_cache is not None:
            return self.heatmap_cache

        if background_type == "Class-Heatmap":
            self.heatmap_cache = self._get_class_heatmap_data()
        elif background_type == "Density-Difference":

            t0 = time.time()
            self.heatmap_cache = self._get_density_difference_heatmap_data()
            t1 = time.time()
            print(f"[get_heatmap_data] 生成密度差异热力图耗时: {(t1 - t0)*1000:.2f}ms")

        else:
            # 返回空热力图
            return []

        return self.heatmap_cache

    def _get_class_heatmap_data(self) -> List[Dict[str, Any]]:
        """生成类别热力图数据"""
        if self._current_slice is None or self._current_slice_embedding is None:
            return []

        try:
            # 获取类别标签
            labels = []
            for i in range(len(self._current_slice)):
                label = self._get_label_for_point(i, "current")
                labels.append(label)

            labels = np.array(labels)

            # 使用热力图计算器生成类别热力图
            return self.heatmap_calculator.get_class_heatmap(
                self._current_slice_embedding,
                labels,
                normalize=False
            )
        except Exception as e:
            print(f"Error generating class heatmap: {e}")
            return []

    def _get_density_difference_heatmap_data(self) -> List[Dict[str, Any]]:
        """生成密度差异热力图数据"""
        if self._current_slice is None or self._current_slice_embedding is None:
            return []
        if self._reference_slice is None or self._reference_slice_embedding is None:
            return []
        if self._viewport_bounds is None:
            print("视口尚未设置，无法生成密度差异热力图")
            return []

        try:
            # 获取当前视图的视口范围
            viewport = self._viewport_bounds  # 直接使用字典形式

            # 使用热力图计算器生成密度差异热力图
            method = 'mix'
            sigma = None
            if self.config.mode in ["kde-scatter", "kde"]:
                method = 'kde'
            elif self.config.mode == "lsdd":
                method = 'lsdd'
                sigma = 0.01

            return self.heatmap_calculator.get_density_difference_heatmap(
                self._current_slice_embedding,
                self._reference_slice_embedding,
                viewport,
                method=method,
                sigma=sigma
            )
        except Exception as e:
            print(f"生成密度差异热力图时出错: {e}")
            return []

    def lasso_select_points(self, lasso_shape: List[List[float]]) -> List[int]:
        """
        Lasso选择点 - 使用NumPy向量化操作优化性能

        Args:
            lasso_shape: Lasso多边形的顶点坐标

        Returns:
            List[int]: 被选中的点在切片中的索引列表
        """
        self.config.lasso_shape = lasso_shape

        # 如果lasso_shape为None或空，则视为采样为空
        if not lasso_shape:
            self.config.lasso_selection = []
            self.config.lasso_selection_reference = []
            self._update_info()
            return []

        # 计算lasso_shape的包围盒
        x_coords = [p[0] for p in lasso_shape]
        y_coords = [p[1] for p in lasso_shape]
        bbox_min_x, bbox_max_x = min(x_coords), max(x_coords)
        bbox_min_y, bbox_max_y = min(y_coords), max(y_coords)

        selected_indices = []
        if self._current_slice_embedding is not None:
            # 使用NumPy向量化操作筛选包围盒内的点
            points = self._current_slice_embedding
            in_bbox = (points[:, 0] >= bbox_min_x) & (points[:, 0] <= bbox_max_x) & \
                     (points[:, 1] >= bbox_min_y) & (points[:, 1] <= bbox_max_y)
            bbox_indices = np.where(in_bbox)[0]
            bbox_points = points[in_bbox]

            # 对包围盒内的点进行精确选择
            for i, point in enumerate(bbox_points):
                if self._point_in_polygon(point[0], point[1], lasso_shape):
                    # 将numpy.int64转换为Python int类型
                    selected_indices.append(int(bbox_indices[i]))

        selected_indices_reference = []
        if self._reference_slice_embedding is not None:
            # 同样处理参考切片
            points_ref = self._reference_slice_embedding
            in_bbox_ref = (points_ref[:, 0] >= bbox_min_x) & (points_ref[:, 0] <= bbox_max_x) & \
                        (points_ref[:, 1] >= bbox_min_y) & (points_ref[:, 1] <= bbox_max_y)
            bbox_indices_ref = np.where(in_bbox_ref)[0]
            bbox_points_ref = points_ref[in_bbox_ref]

            for i, point in enumerate(bbox_points_ref):
                if self._point_in_polygon(point[0], point[1], lasso_shape):
                    # 将numpy.int64转换为Python int类型
                    selected_indices_reference.append(int(bbox_indices_ref[i]))

        self.config.lasso_selection = selected_indices
        self.config.lasso_selection_reference = selected_indices_reference
        self._update_info()
        return selected_indices

    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """判断点是否在多边形内（简单的射线法）"""
        if len(polygon) < 3:
            return False

        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / \
                                (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_lasso_sample_images(self) -> List[str]:
        """获取Lasso选择的样本图片"""
        if self._current_slice is None:
            return []

        try:
            # 最多显示8张图片
            sample_indices = self.config.lasso_selection[:15]
            sample_indices_reference = self.config.lasso_selection_reference[:15]
            # tabular / text 模式复用图片通道返回 title，避免前端多接口改动
            print(f"Mode: {getattr(self.config, 'mode', None)}, Sample indices: {sample_indices}, Reference indices: {sample_indices_reference}")
            if getattr(self.config, 'mode', None) in ('tabular', 'text'):
                current_titles = []
                reference_titles = []
                for idx in sample_indices:
                    current_titles.append(self._get_title_for_point(idx, 'current') or '')
                if self._reference_slice is not None:
                    for idx in sample_indices_reference:
                        reference_titles.append(self._get_title_for_point(idx, 'reference') or '')
                # 结构保持 [[thumbs, full],[thumbsRef, fullRef]]，此处两者相同都为标题
                return [[current_titles, current_titles], [reference_titles, reference_titles]]
            # 使用Slice的批量获取URL方法（默认图片模式）
            if self._reference_slice is None:
                return [[self._current_slice.get_image_urls(sample_indices, thumbnail=True), self._current_slice.get_image_urls(sample_indices)], [[], []]]
            return [[self._current_slice.get_image_urls(sample_indices, thumbnail=True), self._current_slice.get_image_urls(sample_indices)],
                    [self._reference_slice.get_image_urls(sample_indices_reference, thumbnail=True), self._reference_slice.get_image_urls(sample_indices_reference)]]
        except Exception as e:
            print(f"Error getting lasso sample images: {e}")
            return []

    def delete_lasso_points(self) -> bool:
        """删除Lasso选中的点"""
        if not self.config.lasso_selection or self._current_slice is None:
            print("No points selected for deletion or current slice is None")
            return False

        try:
            # 获取当前全部global_ids
            current_global_ids = self._current_slice.global_ids.copy()

            # 创建要保留的mask（删除选中的点）
            keep_mask = np.ones(len(current_global_ids), dtype=bool)
            keep_mask[self.config.lasso_selection] = False

            # 更新global_ids
            new_global_ids = current_global_ids[keep_mask]
            self._current_slice.global_ids = new_global_ids

            # 重新生成embedding和重置采样
            self.refresh_embedding()

            print(
                f"Deleted {len(self.config.lasso_selection)} points from current slice '{self._current_slice.name}' and saved to disk")

            # 触发重新采样
            self._resample_data()

            # 重置选择
            self.config.lasso_selection = []
            self.config.lasso_shape = []
            self._update_info()
            return True
        except Exception as e:
            print(f"Error deleting lasso points: {e}")
            return False

    def refresh_embedding(self):
        """
        刷新当前和参考切片的embedding
        """
        if self._current_slice is not None and self._reference_slice is not None:
            self._current_slice_embedding, self._reference_slice_embedding = self.get_slice_embedding()
            self._whole_slice_embedding = self.get_slice_embedding(
                father=True)
        # 联合归一化
        self._normalize_embeddings_jointly()
        self._reload_all()

    def upsample_lasso_points(self, operation_ratio: Optional[float] = None) -> bool:
        """
        上采样：在lasso范围内找到Whole有但Current没有的点，按比例添加到Current中

        Args:
            operation_ratio: 操作比例，如果为None则使用配置中的值
        """
        if not self.config.lasso_shape or self._current_slice is None or self._whole_slice is None or self._whole_slice_embedding is None:
            return False

        # try:
        # 使用配置中的操作比例
        if operation_ratio is None:
            operation_ratio = self.config.operation.get(
                "samplingRatio", 0.01)
        # 1. 在Whole切片中找到lasso范围内的所有点
        whole_lasso_indices = []
        for idx in range(len(self._whole_slice_embedding)):
            x, y = self._whole_slice_embedding[idx]
            if self._point_in_polygon(x, y, self.config.lasso_shape):
                whole_lasso_indices.append(idx)

        if not whole_lasso_indices:
            print("No whole points found in lasso area")
            return False

        # 2. 获取Whole中lasso范围内的global_ids
        whole_global_ids_in_lasso = self._whole_slice.global_ids[whole_lasso_indices]

        # 3. 找到Current中没有的点（Whole有但Current没有的点）
        current_global_ids_set = set(self._current_slice.global_ids)
        missing_global_ids = [gid for gid in whole_global_ids_in_lasso
                                if gid not in current_global_ids_set]

        if not missing_global_ids:
            print("No missing points found to upsample")
            return False

        # 4. 按操作比例选择要添加的点
        num_to_add = max(1, int(len(missing_global_ids) * operation_ratio))
        print("上采样增加点数:", num_to_add)
        np.random.seed(42)  # 确保可重复性
        selected_indices = np.random.choice(len(missing_global_ids),
                                            min(num_to_add, len(
                                                missing_global_ids)),
                                            replace=False)
        selected_global_ids = [missing_global_ids[i]
                                for i in selected_indices]

        # 5. 添加到Current切片
        new_global_ids = np.concatenate(
            [self._current_slice.global_ids, selected_global_ids])
        self._current_slice.global_ids = new_global_ids

        # 6. 重新生成embedding和重置采样
        self.refresh_embedding()
        self._current_sampled_indices = []
        self._reference_sampled_indices = []

        print(
            f"Upsampled: added {len(selected_global_ids)} points from whole to current slice")

        # 7. 触发重新采样
        self._resample_data()

        # 8. 重置选择
        self.config.lasso_selection = []
        self.config.lasso_shape = []
        self._update_info()
        return True
        # except Exception as e:
        #     print(f"Error upsampling lasso points: {e}")
        #     return False

    def downsample_lasso_points(self, operation_ratio: Optional[float] = None) -> bool:
        """
        下采样：在lasso选中的区域内，随机删除固定比例的点

        Args:
            operation_ratio: 操作比例，如果为None则使用配置中的值
        """
        if not self.config.lasso_selection or self._current_slice is None:
            return False

        try:
            # 使用配置中的操作比例
            if operation_ratio is None:
                operation_ratio = self.config.operation.get(
                    "samplingRatio", 0.01)
            # 1. 获取lasso选中的点数
            selected_count = len(self.config.lasso_selection)

            # 2. 计算要删除的点数
            num_to_remove = max(1, int(selected_count * operation_ratio))

            if num_to_remove >= selected_count:
                print(
                    f"Cannot remove {num_to_remove} points from {selected_count} selected points")
                return False

            # 3. 从选中的点中随机选择要删除的点
            np.random.seed(42)  # 确保可重复性
            indices_to_remove = np.random.choice(len(self.config.lasso_selection),
                                                 num_to_remove,
                                                 replace=False)

            # 4. 获取要删除的实际索引
            actual_indices_to_remove = [
                self.config.lasso_selection[i] for i in indices_to_remove]

            # 5. 创建保留mask
            current_global_ids = self._current_slice.global_ids.copy()
            keep_mask = np.ones(len(current_global_ids), dtype=bool)
            keep_mask[actual_indices_to_remove] = False

            # 6. 更新global_ids
            new_global_ids = current_global_ids[keep_mask]
            self._current_slice.global_ids = new_global_ids

            # 7. 重新生成embedding和重置采样
            self.refresh_embedding()
            self._current_sampled_indices = []
            self._reference_sampled_indices = []

            print(
                f"Downsampled: removed {num_to_remove} points from {selected_count} selected points in lasso area")

            # 9. 触发重新采样
            self._resample_data()

            # 10. 重置选择
            self.config.lasso_selection = []
            self.config.lasso_shape = []
            self._update_info()
            return True
        except Exception as e:
            print(f"Error downsampling lasso points: {e}")
            return False

    def save_slice(self, slice_name: str) -> bool:
        """保存当前状态为新切片"""
        if self._current_slice is None:
            return False

        try:
            # 获取当前切片的global_ids
            current_global_ids = self._current_slice.global_ids

            # 如果有lasso选择，只保存选中的点
            selected_global_ids = current_global_ids

            # 创建新的Slice对象并保存
            new_slice = Slice(
                name=slice_name,
                global_ids=selected_global_ids,
                overwrite=True  # 允许覆盖
            )
            new_slice.save()

            print(
                f"Slice saved as {slice_name} with {len(selected_global_ids)} samples")
            return True
        except Exception as e:
            print(f"Error saving slice {slice_name}: {e}")
            return False

    def cut_slice(self, number: float) -> bool:
        """按比例裁剪切片"""
        if self._current_slice is None:
            return False

        try:
            # 计算要保留的样本数量
            total_samples = len(self._current_slice)
            keep_samples = number

            if keep_samples <= 0:
                return False

            # 随机选择要保留的样本索引
            np.random.seed(42)  # 确保可重复性
            keep_indices = np.random.choice(
                total_samples, keep_samples, replace=False)
            keep_indices = np.sort(keep_indices)

            # 获取要保留的global_ids
            original_global_ids = self._current_slice.global_ids
            new_global_ids = original_global_ids[keep_indices]

            # 创建新的切片来替换当前切片
            # 这里我们直接修改当前切片的数据
            self._current_slice.global_ids = new_global_ids

            # 重新生成embedding并重置采样
            self.refresh_embedding()
            self._current_sampled_indices = []

            # 清除lasso选择（因为索引已经改变）
            self.config.lasso_selection = []
            self.config.lasso_shape = []

            print(
                f"Cut slice with fraction {number}: {total_samples} -> {len(new_global_ids)} samples")
            self._update_info()
            return True
        except Exception as e:
            print(f"Error cutting slice: {e}")
            return False

    def copy_lasso(self) -> bool:
        """复制Lasso形状"""
        if not self.config.lasso_shape:
            return False

        # 只需要验证有lasso形状可以复制，实际复制由前端处理
        return True

    def copy_points(self) -> bool:
        """复制点"""
        if not self.config.lasso_selection:
            return False

        self.config.copied_points = True
        # TODO: 实现实际的点复制逻辑
        return True

    def paste_points(self) -> bool:
        """粘贴点"""
        if not self.config.copied_points:
            return False

        # TODO: 实现实际的点粘贴逻辑
        return True

    def _get_label_for_point(self, index: int, source: str) -> int:
        """获取点的标签（分类）"""
        try:
            if source == "current" and self._current_slice is not None:
                label = self._current_slice.get_label(index)
                return int(label)  # 确保转换为Python int类型
            elif source == "reference" and self._reference_slice is not None:
                label = self._reference_slice.get_label(index)
                return int(label)  # 确保转换为Python int类型
            else:
                return 0  # 默认标签
        except Exception as e:
            print(f"Error getting label for point {index} in {source}: {e}")
            return 0

    def _get_value_for_point(self, index: int, source: str) -> float:
        """获取点的数值"""
        # TODO: 可以根据需要从Slice中获取其他数值字段
        # 这里先返回基于标签的模拟值
        try:
            label = self._get_label_for_point(index, source)
            return float(label) / 10.0  # 将标签转换为0-1的值
        except Exception:
            return 0.0

    def _get_title_for_point(self, index: int, source: str) -> Optional[str]:
        """获取点的标题"""
        try:
            slice_obj = None
            if source == "current" and self._current_slice is not None:
                slice_obj = self._current_slice
            elif source == "reference" and self._reference_slice is not None:
                slice_obj = self._reference_slice

            if slice_obj is None:
                return None

            # 使用Slice类的新方法获取标题
            return slice_obj.get_title(index)

        except Exception as e:
            print(f"Error getting title for point {index} in {source}: {e}")
            return None

    def _get_key_word_for_point(self, index: int, source: str) -> Optional[str]:
        """获取点的标题"""
        try:
            slice_obj = None
            if source == "current" and self._current_slice is not None:
                slice_obj = self._current_slice
            elif source == "reference" and self._reference_slice is not None:
                slice_obj = self._reference_slice

            if slice_obj is None:
                return None

            # 使用Slice类的新方法获取关键字
            return slice_obj.get_key_word(index)

        except Exception as e:
            print(f"Error getting title for point {index} in {source}: {e}")
            return None

    def _get_image_url_for_point(self, index: int, source: str, thumb=False) -> Optional[str]:
        """获取点对应的图片URL"""
        try:
            slice_obj = None
            if source == "current" and self._current_slice is not None:
                slice_obj = self._current_slice
            elif source == "reference" and self._reference_slice is not None:
                slice_obj = self._reference_slice

            if slice_obj is None:
                return None

            # 直接使用Slice类的新方法获取图片URL
            if thumb:
                thumb_url = slice_obj.get_image_url(index, thumbnail=thumb)
                # print('TU',thumb_url)
                if thumb_url != '':
                    return thumb_url
            return slice_obj.get_image_url(index, thumbnail=False)

        except Exception as e:
            print(f"Error getting image URL for point {index} in {source}: {e}")
            return None

    def _get_thumb_url_for_point(self, index: int, source: str) -> Optional[str]:
        """获取点对应的缩略图URL"""
        try:
            slice_obj = None
            if source == "current" and self._current_slice is not None:
                slice_obj = self._current_slice
            elif source == "reference" and self._reference_slice is not None:
                slice_obj = self._reference_slice

            if slice_obj is None:
                return None

            return slice_obj.get_image_url(index, thumbnail=True)

        except Exception as e:
            print(f"Error getting image URL for point {index} in {source}: {e}")
            return None

    def _update_info(self):
        """更新统计信息"""
        self._info["currentSliceTotal"] = len(
            self._current_slice) if self._current_slice is not None else 0
        self._info["currentSliceShown"] = len(self._current_sampled_indices)

        self._info["referenceSliceTotal"] = len(
            self._reference_slice) if self._reference_slice is not None else 0
        self._info["referenceSliceShown"] = len(self._reference_sampled_indices)

        self._info["selectionTotal"] = len(self.config.lasso_selection)
        # 计算当前显示的点中有多少被选中
        if self.config.lasso_selection and self._current_sampled_indices:
            selected_set = set(self.config.lasso_selection)
            sampled_set = set(self._current_sampled_indices)
            self._info["selectionShown"] = len(
                selected_set.intersection(sampled_set))
        else:
            self._info["selectionShown"] = 0

        print(self._info)

    def get_info(self) -> Dict[str, int]:
        """获取统计信息"""
        return self._info.copy()

    def get_config(self) -> Dict[str, Any]:
        """获取View配置"""
        return {
            "id": self.config.id,
            "title": self.config.title,
            "currentSlice": self.config.current_slice,
            "referenceSlice": self.config.reference_slice,
            "dimensionReduction": self.config.dimension_reduction,
            "visualEncoding": self.config.visual_encoding,
            "option": self.config.option,
            "operation": self.config.operation,
            "lassoShape": self.config.lasso_shape,
            "lassoSelection": self.config.lasso_selection,
            "zoom": self.config.zoom,
            "pan": self.config.pan,
            "toolMode": self.config.tool_mode,
            "copiedPoints": self.config.copied_points,
            "classCount": self.config.class_count
        }

    def update_config(self, config_update: Dict[str, Any]) -> bool:
        """更新View配置"""

        print("Updating view configuration...")
        # 检查是否有降维配置的变更
        dimension_reduction_changed = False
        if "dimensionReduction" in config_update:
            old_config = self.config.dimension_reduction.copy()
            self.config.dimension_reduction.update(
                config_update["dimensionReduction"])
            if old_config != self.config.dimension_reduction:
                dimension_reduction_changed = True

        if "currentSlice" in config_update and config_update["currentSlice"] != self.config.current_slice:
            if not self.set_current_slice(config_update["currentSlice"]):
                return False
            dimension_reduction_changed = True

        if "referenceSlice" in config_update and config_update["referenceSlice"] != self.config.reference_slice:
            if not self.set_reference_slice(config_update["referenceSlice"]):
                return False
            dimension_reduction_changed = True

        # 更新其他配置
        for key, value in config_update.items():
            # 跳过已经处理过的字段
            if key in ["currentSlice", "referenceSlice", "dimensionReduction"]:
                continue

            if hasattr(self.config, key):
                # 将驼峰命名转换为下划线命名（如classCount -> class_count）
                if key == "classCount":
                    setattr(self.config, "class_count", value)
                else:
                    setattr(self.config, key, value)
            elif key in ["visualEncoding", "option", "operation", "pan"]:
                current_dict = getattr(
                    self.config, key.lower().replace("encoding", "_encoding"))
                if isinstance(value, dict):
                    current_dict.update(value)

        # 如果降维配置改变，重新生成embedding
        if dimension_reduction_changed:
            self.refresh_embedding()

        # 如果视口相关配置改变，重新采样
        elif any(key in config_update for key in ["zoom", "pan"]):
            self._resample_data()

        return True


    def _normalize_embeddings_jointly(self):
        """
        联合归一化current、reference和whole的embedding
        确保它们在同一个坐标系中，便于比较
        """
        if self._current_slice_embedding is None:
            return

        all_embeddings = [self._current_slice_embedding]
        if self._reference_slice_embedding is not None:
            all_embeddings.append(self._reference_slice_embedding)

        # 合并所有embedding来计算全局边界
        combined_embedding = np.vstack(all_embeddings)

        if self._whole_slice_embedding is not None:
            combined_embedding = self._whole_slice_embedding

        # 计算全局边界
        x_min, x_max = combined_embedding[:, 0].min(
        ), combined_embedding[:, 0].max()
        y_min, y_max = combined_embedding[:, 1].min(
        ), combined_embedding[:, 1].max()

        # 计算当前范围大小
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 避免除零错误
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0

        # 计算可用范围（减去两边的padding）
        padding = 0.1
        available_range = 1.0 - 2 * padding

        # 计算缩放因子，保持长宽比（以长边为准），并考虑padding
        scale_factor = available_range / max(x_range, y_range)

        # 计算当前数据的中心
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # 归一化current embedding
        self._current_slice_embedding[:, 0] = (
            self._current_slice_embedding[:, 0] - x_center) * scale_factor + 0.5
        self._current_slice_embedding[:, 1] = (
            self._current_slice_embedding[:, 1] - y_center) * scale_factor + 0.5

        # 归一化reference embedding（如果存在）
        if self._reference_slice_embedding is not None:
            self._reference_slice_embedding[:, 0] = (
                self._reference_slice_embedding[:, 0] - x_center) * scale_factor + 0.5
            self._reference_slice_embedding[:, 1] = (
                self._reference_slice_embedding[:, 1] - y_center) * scale_factor + 0.5

        # 归一化whole embedding（如果存在）
        if self._whole_slice_embedding is not None:
            self._whole_slice_embedding[:, 0] = (
                self._whole_slice_embedding[:, 0] - x_center) * scale_factor + 0.5
            self._whole_slice_embedding[:, 1] = (
                self._whole_slice_embedding[:, 1] - y_center) * scale_factor + 0.5


class GlobalViewManager:
    """全局视图管理器，管理所有View实例"""

    def __init__(self, max_scatter_points: int = 500):
        """
        初始化全局视图管理器

        Args:
            max_scatter_points: 每个视图散点图最大显示点数
        """
        self.max_scatter_points = max_scatter_points
        self.views: Dict[int, ViewManager] = {}
        self.next_view_id = 1

    def create_view(self, current_slice: str = "", reference_slice: str = "", mode: str = "default") -> Tuple[int, ViewManager]:
        """
        创建新视图

        Args:
            current_slice: 当前切片名称
            reference_slice: 参考切片名称

        Returns:
            (view_id, view_manager): 新创建的视图ID和管理器实例
        """
        view_id = self.next_view_id
        view_manager = ViewManager(view_id, mode, self.max_scatter_points)

        # 设置当前切片
        if current_slice:
            view_manager.set_current_slice(current_slice)

        # 设置参考切片
        if reference_slice:
            view_manager.set_reference_slice(reference_slice)

        if view_manager._current_slice is not None and view_manager._reference_slice is not None:
            view_manager.refresh_embedding()

        self.views[view_id] = view_manager
        self.next_view_id += 1

        return view_id, view_manager

    def delete_view(self, view_id: int) -> bool:
        """
        删除视图

        Args:
            view_id: 要删除的视图ID

        Returns:
            bool: 是否删除成功
        """
        if view_id not in self.views:
            return False

        del self.views[view_id]

        # 重新编号视图（保持连续性）
        self._renumber_views()

        return True

    def get_view(self, view_id: int) -> Optional[ViewManager]:
        """
        获取指定视图

        Args:
            view_id: 视图ID

        Returns:
            ViewManager: 视图管理器实例，如果不存在则返回None
        """
        return self.views.get(view_id)

    def get_all_views(self) -> List[Dict[str, Any]]:
        """
        获取所有视图的配置信息

        Returns:
            List[Dict]: 所有视图的配置列表
        """
        return [view.get_config() for view in self.views.values()]

    def get_view_count(self) -> int:
        """获取视图总数"""
        return len(self.views)

    def _renumber_views(self):
        """重新编号视图，确保ID连续"""
        # 按ID排序
        sorted_views = sorted(self.views.items())

        # 清空并重新分配ID
        self.views.clear()
        self.next_view_id = 1

        for i, (old_id, view_manager) in enumerate(sorted_views, 1):
            # 更新view的配置
            view_manager.config.id = i
            view_manager.config.title = f"View {i}"
            view_manager.view_id = i

            # 重新添加到字典
            self.views[i] = view_manager
            self.next_view_id = i + 1

    def clear_all_views(self):
        """清空所有视图"""
        self.views.clear()
        self.next_view_id = 1

    def update_view_config(self, view_id: int, config_update: Dict[str, Any]) -> bool:
        """
        更新指定视图的配置

        Args:
            view_id: 视图ID
            config_update: 要更新的配置

        Returns:
            bool: 是否更新成功
        """
        view_manager = self.get_view(view_id)
        if view_manager is None:
            return False

        return view_manager.update_config(config_update)

    def get_view_scatter_data(self, view_id: int, viewport_bounds: Optional[Dict[str, float]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        获取指定视图的散点数据

        Args:
            view_id: 视图ID
            viewport_bounds: 视口边界 {"x_min": float, "x_max": float, "y_min": float, "y_max": float}

        Returns:
            List[Dict]: 散点数据列表，如果视图不存在则返回None
        """
        view_manager = self.get_view(view_id)
        if view_manager is None:
            return None

        # 如果提供了视口边界，更新视口
        if viewport_bounds:
            view_manager.update_viewport(
                viewport_bounds
            )

        return view_manager.get_scatter_data()

    def get_view_heatmap_data(self, view_id: int, heat_type: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        获取指定视图的热力图数据

        Args:
            view_id: 视图ID
            heat_type: 热力图类型，如果提供则临时使用该类型

        Returns:
            List[Dict]: 热力图数据列表，如果视图不存在则返回None
        """
        view_manager = self.get_view(view_id)
        if view_manager is None:
            return None

        return view_manager.get_heatmap_data()

    def lasso_select_in_view(self, view_id: int, lasso_shape: List[List[float]]) -> Optional[List[int]]:
        """
        在指定视图中进行Lasso选择

        Args:
            view_id: 视图ID
            lasso_shape: Lasso形状坐标列表

        Returns:
            List[int]: 选中的点索引列表，如果视图不存在则返回None
        """
        view_manager = self.get_view(view_id)
        if view_manager is None:
            return None

        return view_manager.lasso_select_points(lasso_shape)

    def get_view_info(self, view_id: int) -> Optional[Dict[str, int]]:
        """
        获取指定视图的统计信息

        Args:
            view_id: 视图ID

        Returns:
            Dict: 统计信息，如果视图不存在则返回None
        """
        view_manager = self.get_view(view_id)
        if view_manager is None:
            return None

        return view_manager.get_info()

    def get_view_sample_images(self, view_id: int) -> Optional[List[str]]:
        """
        获取指定视图的Lasso选择样本图片

        Args:
            view_id: 视图ID

        Returns:
            List[str]: 图片URL列表，如果视图不存在则返回None
        """
        view_manager = self.get_view(view_id)
        if view_manager is None:
            return None

        return view_manager.get_lasso_sample_images()
