import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class ImagePlacement:
    """图片位置信息"""
    image_id: int
    original_x: float
    original_y: float
    target_x: float
    target_y: float
    cost: float

class ImagePlacementOptimizer:
    """
    图片位置优化器

    使用QAP (Quadratic Assignment Problem) 模型来优化图片在低密度差异区域的位置
    """

    def __init__(self,
                 grid_resolution: int = 80,
                 overlap_penalty: float = 1e10,
                 min_distance_ratio: float = 0.01,
                 max_distance_ratio: float = 0.02):
        """
        初始化图片位置优化器

        Args:
            grid_resolution: 网格分辨率
            overlap_penalty: 重叠惩罚权重
            min_distance_ratio: 最小距离比例（相对于视口尺寸）
            max_distance_ratio: 最大距离比例（相对于视口尺寸）
        """
        self.grid_resolution = grid_resolution
        self.overlap_penalty = overlap_penalty
        self.min_distance_ratio = min_distance_ratio
        self.max_distance_ratio = max_distance_ratio

    def optimize_image_positions(self,
                                image_positions: List[Tuple[float, float]],
                                heatmap_data: List[Dict[str, Any]],
                                viewport: Dict[str, float],
                                image_size_ratio: float = 4 / 80) -> List[ImagePlacement]:
        """
        优化图片位置

        Args:
            image_positions: 图片原始位置列表 [(x, y), ...]
            heatmap_data: 热力图数据，包含密度差异信息
            viewport: 视口范围 {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}
            image_size_ratio: 图片大小比例（相对于视口尺寸）

        Returns:
            优化后的图片位置列表
        """
        if not image_positions:
            return []

        # print(f"开始优化 {len(image_positions)} 张图片的位置...")
        start_time = time.time()

        # 生成候选位置（网格中心点）
        candidate_positions = self._generate_candidate_positions(viewport)

        # 计算密度差异成本矩阵
        density_cost_matrix = self._compute_density_cost_matrix(
            candidate_positions, heatmap_data, viewport
        )

        # 筛选每张图片的有效候选位置
        valid_candidates = self._filter_valid_candidates(
            image_positions, candidate_positions, viewport, image_size_ratio
        )

        # 构建QAP问题并求解
        placements = self._solve_qap(
            image_positions, candidate_positions, valid_candidates,
            density_cost_matrix, viewport, image_size_ratio
        )

        print(f"图片位置优化完成，耗时: {time.time() - start_time:.3f}秒")
        return placements

    def _generate_candidate_positions(self, viewport: Dict[str, float]) -> np.ndarray:
        """生成候选位置（网格中心点）"""
        x_min, x_max = viewport['x_min'], viewport['x_max']
        y_min, y_max = viewport['y_min'], viewport['y_max']

        # 创建网格中心点
        x_centers = np.linspace(
            x_min + (x_max - x_min) / (2 * self.grid_resolution),
            x_max - (x_max - x_min) / (2 * self.grid_resolution),
            self.grid_resolution
        )
        y_centers = np.linspace(
            y_min + (y_max - y_min) / (2 * self.grid_resolution),
            y_max - (y_max - y_min) / (2 * self.grid_resolution),
            self.grid_resolution
        )

        xx, yy = np.meshgrid(x_centers, y_centers)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _compute_density_cost_matrix(self,
                                   candidate_positions: np.ndarray,
                                   heatmap_data: List[Dict[str, Any]],
                                   viewport: Dict[str, float]) -> np.ndarray:
        """
        计算密度差异成本矩阵

        成本是每个候选位置7x7邻域内的密度差异绝对值之和
        """
        n_candidates = len(candidate_positions)
        cost_matrix = np.zeros(n_candidates)

        if not heatmap_data:
            return cost_matrix

        # 创建密度差异网格
        density_grid = self._create_density_grid(heatmap_data, viewport)

        # 为每个候选位置计算7x7邻域的密度差异成本
        for i, (x, y) in enumerate(candidate_positions):
            cost_matrix[i] = self._compute_neighborhood_cost(
                x, y, density_grid, viewport
            )

        return cost_matrix

    def _create_density_grid(self,
                           heatmap_data: List[Dict[str, Any]],
                           viewport: Dict[str, float]) -> np.ndarray:
        """根据热力图数据创建密度差异网格"""
        grid = np.zeros((self.grid_resolution, self.grid_resolution))

        x_min, x_max = viewport['x_min'], viewport['x_max']
        y_min, y_max = viewport['y_min'], viewport['y_max']

        for cell in heatmap_data:
            if cell['type'] == 'center':
                continue
            # 将热力图坐标映射到网格索引
            x0, y0 = cell['x0'], cell['y0']
            x1, y1 = cell['x1'], cell['y1']

            # 计算网格索引
            grid_x = int((x0 - x_min) / (x_max - x_min) * self.grid_resolution)
            grid_y = int((y0 - y_min) / (y_max - y_min) * self.grid_resolution)

            # 确保索引在有效范围内
            grid_x = np.clip(grid_x, 0, self.grid_resolution - 1)
            grid_y = np.clip(grid_y, 0, self.grid_resolution - 1)

            # 使用密度差异值的绝对值作为成本
            grid[grid_y, grid_x] = abs(cell.get('value', 0) - 0.5)

        return grid

    def _compute_neighborhood_cost(self,
                                 x: float, y: float,
                                 density_grid: np.ndarray,
                                 viewport: Dict[str, float]) -> float:
        """计算位置(x,y)周围7x7邻域的密度差异成本"""
        x_min, x_max = viewport['x_min'], viewport['x_max']
        y_min, y_max = viewport['y_min'], viewport['y_max']

        # 将坐标转换为网格索引
        grid_x = int((x - x_min) / (x_max - x_min) * self.grid_resolution)
        grid_y = int((y - y_min) / (y_max - y_min) * self.grid_resolution)

        # 计算7x7邻域的成本
        total_cost = 0.0
        neighbor_radius = 1  # 7x7邻域的半径

        for dx in range(-neighbor_radius, neighbor_radius + 1):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy

                # 检查边界
                if 0 <= nx < self.grid_resolution and 0 <= ny < self.grid_resolution:
                    total_cost += density_grid[ny, nx]

        return total_cost

    def _filter_valid_candidates(self,
                               image_positions: List[Tuple[float, float]],
                               candidate_positions: np.ndarray,
                               viewport: Dict[str, float],
                               image_size_ratio: float) -> List[List[int]]:
        """为每张图片筛选距离原位置合适的候选位置"""
        viewport_width = viewport['x_max'] - viewport['x_min']
        viewport_height = viewport['y_max'] - viewport['y_min']
        viewport_size = min(viewport_width, viewport_height)

        min_distance = self.min_distance_ratio * viewport_size
        max_distance = self.max_distance_ratio * viewport_size

        # 计算图片半宽（半高，因为正方形）
        half_size = (viewport_size * image_size_ratio) / 2

        # print(f'half_size: {half_size}, viewport_size: {viewport_size}, image_size_ratio: {image_size_ratio}')

        valid_candidates = []

        for i, (orig_x, orig_y) in enumerate(image_positions):
            # 计算候选位置到原始图片矩形边界的最小欧氏距离
            dx = np.abs(candidate_positions[:, 0] - orig_x) - half_size
            dy = np.abs(candidate_positions[:, 1] - orig_y) - half_size
            dx = np.maximum(dx, 0)  # 负值表示在图片内部，设为0
            dy = np.maximum(dy, 0)  # 负值表示在图片内部，设为0
            distances = np.maximum(dx, dy)

            # print(f'distances: {distances} {min_distance} {max_distance}')


            # 计算候选位置到所有原始位置的距离，排除当前原位置
            other_positions = image_positions[:i] + image_positions[i+1:]
            if other_positions:
                candidate_to_origins = cdist(candidate_positions, other_positions)
                overlap_mask = (candidate_to_origins > min_distance * 2).all(axis=1)
            else:
                # 没有其他图片，不检查重叠
                overlap_mask = np.ones(len(candidate_positions), dtype=bool)

            # 筛选条件：1) 距离当前原位置合适 2) 不覆盖其他原位置
            valid_mask = (
                (distances >= min_distance) &
                (distances <= max_distance) &
                overlap_mask
            )
            valid_indices = np.where(valid_mask)[0].tolist()

            # 如果没有有效候选位置，选择最近的几个且不覆盖其他原位置的
            if not valid_indices:
                print(f'no valid indices')
                if other_positions:
                    candidate_to_others = cdist(candidate_positions, other_positions)
                    non_overlap_mask = (candidate_to_others > min_distance).all(axis=1)
                else:
                    non_overlap_mask = np.ones(len(candidate_positions), dtype=bool)

                non_overlap_distances = np.where(non_overlap_mask, distances, np.inf)

                # 在剩余候选中选择最近的
                closest_indices = np.argsort(non_overlap_distances)[:10]
                valid_indices = closest_indices.tolist()

            valid_candidates.append(valid_indices)

        return valid_candidates

    def _solve_qap(self,
                  image_positions: List[Tuple[float, float]],
                  candidate_positions: np.ndarray,
                  valid_candidates: List[List[int]],
                  density_cost_matrix: np.ndarray,
                  viewport: Dict[str, float],
                  image_size_ratio: float) -> List[ImagePlacement]:
        """使用匈牙利算法求解图片位置分配问题（近似解）"""
        n_images = len(image_positions)

        # 使用贪心算法快速分配位置
        # if n_images <= 20:
        #     # 对于少量图片，使用匈牙利算法
        #     return self._solve_hungarian(
        #         image_positions, candidate_positions, valid_candidates,
        #         density_cost_matrix, viewport, image_size_ratio
        #     )
        # else:
        # 对于大量图片，使用贪心算法
        return self._solve_greedy(
            image_positions, candidate_positions, valid_candidates,
            density_cost_matrix, viewport, image_size_ratio
        )

    def _solve_hungarian(self,
                        image_positions: List[Tuple[float, float]],
                        candidate_positions: np.ndarray,
                        valid_candidates: List[List[int]],
                        density_cost_matrix: np.ndarray,
                        viewport: Dict[str, float],
                        image_size_ratio: float) -> List[ImagePlacement]:
        """使用匈牙利算法求解（适用于较少图片）"""
        n_images = len(image_positions)

        # 创建一个共同的候选位置集合
        all_candidates = set()
        for candidates in valid_candidates:
            all_candidates.update(candidates)
        all_candidates = sorted(list(all_candidates))
        n_candidates = len(all_candidates)

        # 构建成本矩阵（密度差异 + 重叠惩罚）
        cost_matrix = self._build_combined_cost_matrix(
            image_positions, candidate_positions, all_candidates,
            density_cost_matrix, valid_candidates, viewport, image_size_ratio
        )

        # 使用匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 构建结果
        placements = []
        for i, pos_idx in zip(row_ind, col_ind):
            if i < n_images and pos_idx < n_candidates:
                candidate_idx = all_candidates[pos_idx]
                target_x, target_y = candidate_positions[candidate_idx]
                orig_x, orig_y = image_positions[i]

                placements.append(ImagePlacement(
                    image_id=i,
                    original_x=orig_x,
                    original_y=orig_y,
                    target_x=target_x,
                    target_y=target_y,
                    cost=density_cost_matrix[candidate_idx]
                ))

        return placements

    def _solve_greedy(self,
                     image_positions: List[Tuple[float, float]],
                     candidate_positions: np.ndarray,
                     valid_candidates: List[List[int]],
                     density_cost_matrix: np.ndarray,
                     viewport: Dict[str, float],
                     image_size_ratio: float) -> List[ImagePlacement]:
        """使用贪心算法求解（快速近似解）"""
        n_images = len(image_positions)
        placements = []
        used_candidates = set()

        # 计算图片大小（用于重叠检测）
        viewport_width = viewport['x_max'] - viewport['x_min']
        viewport_height = viewport['y_max'] - viewport['y_min']
        image_size = min(viewport_width, viewport_height) * image_size_ratio

        # 按密度差异成本对图片排序（优先处理成本高的图片）
        image_costs = []
        for i, candidates in enumerate(valid_candidates):
            if candidates:
                min_cost = min(density_cost_matrix[c] for c in candidates)
                image_costs.append((min_cost, i))
            else:
                image_costs.append((float('inf'), i))

        # 按成本升序排序（优先给低成本图片分配好位置）
        image_costs.sort()

        for _, img_idx in image_costs:
            orig_x, orig_y = image_positions[img_idx]
            candidates = valid_candidates[img_idx]

            best_candidate = None
            best_cost = float('inf')

            for candidate_idx in candidates:
                if candidate_idx in used_candidates:
                    continue

                target_x, target_y = candidate_positions[candidate_idx]

                # 检查与已放置图片的重叠
                overlap = False
                for placement in placements:
                    dist = np.sqrt((target_x - placement.target_x)**2 +
                                 (target_y - placement.target_y)**2)
                    if dist < image_size:
                        overlap = True
                        break

                if not overlap:
                    cost = density_cost_matrix[candidate_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = candidate_idx

            # 如果找到合适位置，则分配
            if best_candidate is not None:
                target_x, target_y = candidate_positions[best_candidate]
                placements.append(ImagePlacement(
                    image_id=img_idx,
                    original_x=orig_x,
                    original_y=orig_y,
                    target_x=target_x,
                    target_y=target_y,
                    cost=best_cost
                ))
                used_candidates.add(best_candidate)

        return placements

    def _build_combined_cost_matrix(self,
                                   image_positions: List[Tuple[float, float]],
                                   candidate_positions: np.ndarray,
                                   candidates: List[int],
                                   density_cost_matrix: np.ndarray,
                                   valid_candidates: List[List[int]],
                                   viewport: Dict[str, float],
                                   image_size_ratio: float) -> np.ndarray:
        """构建包含密度差异和重叠惩罚的组合成本矩阵"""
        n_images = len(image_positions)
        n_candidates = len(candidates)

        # 扩展矩阵以处理图片数量与候选位置数量不匹配的情况
        matrix_size = max(n_images, n_candidates)
        cost_matrix = np.full((matrix_size, matrix_size), 1e10)

        candidate_to_idx = {cand: i for i, cand in enumerate(candidates)}

        # 计算图片大小
        viewport_width = viewport['x_max'] - viewport['x_min']
        viewport_height = viewport['y_max'] - viewport['y_min']
        image_size = min(viewport_width, viewport_height) * image_size_ratio

        for i, (orig_x, orig_y) in enumerate(image_positions):
            for cand_idx in valid_candidates[i]:
                if cand_idx in candidate_to_idx:
                    j = candidate_to_idx[cand_idx]

                    # 密度差异成本
                    density_cost = density_cost_matrix[cand_idx]

                    # 简化的重叠惩罚：基于候选位置间的距离
                    overlap_penalty = 0
                    pos1 = candidate_positions[cand_idx]
                    for k, other_cand_idx in enumerate(candidates):
                        if k != j:
                            pos2 = candidate_positions[other_cand_idx]
                            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                            if dist < image_size:
                                overlap_penalty += self.overlap_penalty / len(candidates)

                    cost_matrix[i, j] = density_cost + overlap_penalty

        return cost_matrix
