import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import time
import config


class HeatmapCalculator:
    """热力图计算器，用于生成各种类型的热力图数据"""

    def __init__(self,
                 kde_n_centers: int = 10000,
                 kde_sigma: float = 0.01,
                 lsdd_sigma: float = 0.01,
                 lsdd_lambda: float = 0.012,
                 grid_resolution: int = config.GRID_RESOLUTION,
                 lsdd_max_value: float = 30.0):  # 新增LSDD最大值参数
        """
        初始化热力图计算器

        Args:
            kde_n_centers: KDE中心点数量，默认100
            kde_sigma: KDE的高斯核带宽参数，默认0.05
            lsdd_sigma: LSDD的高斯核带宽参数，默认0.05
            lsdd_lambda: LSDD的正则化参数，默认0.05
            grid_resolution: 网格分辨率，默认80
            lsdd_max_value: LSDD值的最大范围，默认1.0
        """
        # 移除固定grid_size参数
        self.kde_n_centers = kde_n_centers
        self.kde_sigma = kde_sigma
        self.lsdd_sigma = lsdd_sigma
        self.lsdd_lambda = lsdd_lambda
        self.grid_resolution = grid_resolution  # 存储网格分辨率
        self.lsdd_max_value = lsdd_max_value  # 存储LSDD最大值
        self.discretize_mode = 'sqrt'  # 可选 'linear', 'square', 'log', 'sqrt'
        # 设定随机种子，包括np\torch等
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        np.random.seed(42)

    def get_class_heatmap(self, embedding: np.ndarray, labels: np.ndarray,
                          viewport: Dict[str, float],  # 改为字典形式
                          normalize: bool = False) -> List[Dict[str, Any]]:
        """
        生成类别热力图，每个网格显示该区域的主要类别

        Args:
            embedding: 嵌入数据，形状为 (n_samples, 2)
            labels: 类别标签，形状为 (n_samples,)
            viewport: 视口范围字典 {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}
            normalize: 是否归一化值

        Returns:
            热力图网格数据列表
        """
        # 从字典中提取坐标值
        return self._get_heatmap(embedding, labels, viewport, heatmap_type='class', normalize=normalize)

    def get_density_difference_heatmap(self, embedding1: np.ndarray, embedding2: np.ndarray,
                                       viewport: Dict[str, float],  # 改为字典形式
                                       method: str = 'lsdd',
                                       sigma: float = None) -> List[Dict[str, Any]]:
        """
        生成密度差异热力图

        Args:
            embedding1: 第一个分布的嵌入数据，形状为 (n_samples, 2)
            embedding2: 第二个分布的嵌入数据，形状为 (n_samples, 2)
            viewport: 视口范围字典 {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}
            method: 密度差异计算方法，'lsdd' 或 'kde'

        Returns:
            热力图网格数据列表
        """
        # 从字典中提取坐标值

        # 计算动态LSDD核大小（基于视口大小）
        viewport_width = viewport['x_max'] - viewport['x_min']
        viewport_height = viewport['y_max'] - viewport['y_min']
        min_side = min(viewport_width, viewport_height)
        dynamic_lsdd_sigma = self.lsdd_sigma

        if method == 'lsdd':
            return self.get_kde_heatmap_on_lsdd_colorscale(
                embedding1, embedding2, viewport, method='lsdd', sigma=sigma if sigma is not None else dynamic_lsdd_sigma)
        elif method == 'kde':
            return self.get_kde_heatmap_on_lsdd_colorscale(
                embedding1, embedding2, viewport, method='kde')
        elif method == 'mix':
            return self.get_mixed_heatmap_on_lsdd_colorscale(embedding1, embedding2, viewport, sigma=sigma)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return self._get_function_heatmap(diff_func, viewport, normalize=True)

    def get_weighted_density_heatmap(self, embedding: np.ndarray, weights: np.ndarray,
                                     normalize: bool = False) -> List[Dict[str, Any]]:
        """
        生成加权密度热力图

        Args:
            embedding: 嵌入数据，形状为 (n_samples, 2)
            weights: 权重，形状为 (n_samples,)
            normalize: 是否归一化值

        Returns:
            热力图网格数据列表
        """
        return self._get_heatmap(embedding, weights, heatmap_type='weighted', normalize=normalize)

    def _get_heatmap(self, embedding: np.ndarray, weights: np.ndarray,
                     viewport: Dict[str, float],  # 保持列表形式（内部使用）
                     heatmap_type: str = '', normalize: bool = False) -> List[Dict[str, Any]]:
        """
        通用热力图生成函数

        Args:
            embedding: 嵌入数据，形状为 (n_samples, 2)
            weights: 权重/标签，形状为 (n_samples,)
            viewport: 视口范围 {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}
            heatmap_type: 热力图类型 ('', 'max', 'count', 'class', 'weighted')
            normalize: 是否归一化值

        Returns:
            热力图网格数据列表
        """
        # 检查viewport
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']
        grid_size_x = (x1 - x0) / self.grid_resolution  # 使用self.grid_resolution
        grid_size_y = (y1 - y0) / self.grid_resolution  # 使用self.grid_resolution

        # 生成网格坐标（只生成视口内的网格）
        grid_points = self._generate_grid_points(viewport)

        # 初始化存储变量
        n_grids = self.grid_resolution * self.grid_resolution  # 使用self.grid_resolution
        heatmap_values = np.zeros(n_grids)
        counts = np.zeros(n_grids)

        if heatmap_type == 'class':
            # 类别热力图：每个网格显示众数类别
            unique_labels = np.unique(weights)
            unique_labels = unique_labels[unique_labels >= 0]  # 过滤掉负值（如果有的话）
            num_classes = len(unique_labels)

            # 创建标签到索引的映射
            label_to_idx = {label: idx for idx,
                            label in enumerate(unique_labels)}

            grid_samples = [[0 for _ in range(num_classes)]
                            for _ in range(n_grids)]

            for i in range(len(embedding)):
                grid_idx = self._get_grid_index(embedding[i], viewport)
                if 0 <= grid_idx < n_grids:
                    label = int(weights[i])
                    if label in label_to_idx:  # 确保标签是有效的
                        grid_samples[grid_idx][label_to_idx[label]] += 1
                        counts[grid_idx] += 1

            # 计算每个网格的主要类别，保持原始标签值
            avg_values = np.full(n_grids, -1, dtype=int)  # 初始化为-1
            for i in range(n_grids):
                if counts[i] > 0:
                    dominant_idx = np.argmax(grid_samples[i])
                    avg_values[i] = unique_labels[dominant_idx]  # 使用原始标签值

        elif heatmap_type == 'count':
            # 计数热力图：每个网格显示样本数量
            for i in range(len(embedding)):
                grid_idx = self._get_grid_index(embedding[i], viewport)
                if 0 <= grid_idx < n_grids:
                    counts[grid_idx] += 1
            avg_values = counts

        elif heatmap_type == 'max':
            # 最大值热力图：每个网格显示最大权重
            for i in range(len(embedding)):
                grid_idx = self._get_grid_index(embedding[i], viewport)
                if 0 <= grid_idx < n_grids:
                    heatmap_values[grid_idx] = max(
                        heatmap_values[grid_idx], weights[i])
            avg_values = heatmap_values

        else:
            # 默认：平均值热力图
            for i in range(len(embedding)):
                grid_idx = self._get_grid_index(embedding[i], viewport)
                if 0 <= grid_idx < n_grids:
                    heatmap_values[grid_idx] += weights[i]
                    counts[grid_idx] += 1

            avg_values = np.divide(heatmap_values, counts,
                                   out=np.zeros_like(heatmap_values),
                                   where=counts != 0)

        if normalize and heatmap_type != 'class':
            avg_values = self._min_max_normalize(avg_values)

        # 生成结果列表
        return self._format_heatmap_results(grid_points, avg_values, counts, heatmap_type, grid_size_x, grid_size_y)

    def _get_function_heatmap(self, func, viewport: Dict[str, float], normalize: bool = False) -> List[Dict[str, Any]]:
        """
        根据函数生成热力图

        Args:
            func: 计算函数，接受(x, y)坐标并返回值
            viewport: 视口范围 {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}
            normalize: 是否归一化值

        Returns:
            热力图网格数据列表
        """
        # 生成网格坐标（基于视口）
        grid_points = self._generate_grid_points(viewport)

        # 计算网格中心点的值
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']
        grid_size_x = (x1 - x0) / self.grid_resolution
        grid_size_y = (y1 - y0) / self.grid_resolution

        grid_centers = grid_points + np.array([grid_size_x / 2, grid_size_y / 2])
        values = func(grid_centers)

        # 确保values是numpy数组
        if not isinstance(values, np.ndarray):
            values = np.array([values] if np.isscalar(values) else values)

        if normalize:
            values = self._min_max_normalize(values)

        # 生成结果列表（不需要counts参数，所有网格都有值）
        return self._format_heatmap_results(grid_points, values, grid_size_x=grid_size_x, grid_size_y=grid_size_y)

    def _generate_grid_points(self, viewport: Dict[str, float]) -> np.ndarray:
        """生成视口内的网格坐标点，使用self.grid_resolution"""
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']

        # 使用linspace生成均匀的网格点（左上角坐标）
        x_coords = np.linspace(x0, x1, self.grid_resolution, endpoint=False)
        y_coords = np.linspace(y0, y1, self.grid_resolution, endpoint=False)

        # 创建网格
        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

        return np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    def _format_heatmap_results(self, grid_points: np.ndarray, values: np.ndarray,
                                counts: Optional[np.ndarray] = None,
                                heatmap_type: str = '', grid_size_x: float = 1.0, grid_size_y: float = 1.0) -> List[Dict[str, Any]]:
        """格式化热力图结果"""
        result_list = []
        for i in range(len(grid_points)):
            # 跳过没有样本的网格（所有类型都跳过，实现透明效果）
            if counts is not None and counts[i] == 0:
                continue

            # 对于class类型，确保值是有效的类别ID（不是-1）
            if heatmap_type == 'class' and values[i] == -1:
                continue

            result_list.append({
                'x0': float(grid_points[i, 0]),
                'y0': float(grid_points[i, 1]),
                'x1': float(grid_points[i, 0] + grid_size_x),
                'y1': float(grid_points[i, 1] + grid_size_y),
                'value': float(values[i])
            })

        return result_list

    def _get_grid_index(self, point: np.ndarray, viewport: Dict[str, float]) -> int:
        """获取点所在的网格索引（基于视口），视口外返回-1"""
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']
        # 如果点在窗口外，直接返回-1
        if not (x0 <= point[0] < x1 and y0 <= point[1] < y1):
            return -1
        grid_size_x = (x1 - x0) / self.grid_resolution  # 使用self.grid_resolution
        grid_size_y = (y1 - y0) / self.grid_resolution  # 使用self.grid_resolution

        x_idx = int((point[0] - x0) // grid_size_x)
        y_idx = int((point[1] - y0) // grid_size_y)
        return x_idx * self.grid_resolution + y_idx

    def _standardize_input(self, x) -> np.ndarray:
        """标准化输入数据为numpy数组"""
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
        else:
            x = np.array(x).reshape(-1, 2)
        return x

    def _to_torch_tensor(self, x: np.ndarray, device: torch.device) -> torch.Tensor:
        """转换为PyTorch张量并移到指定设备"""
        return torch.tensor(x, dtype=torch.float32).to(device)

    def _setup_kde_centers(self, embedding: np.ndarray, n_centers: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
        """设置KDE中心点和设备"""
        n_samples = embedding.shape[0]
        n_centers = n_centers or min(self.kde_n_centers, n_samples)

        # 随机选择中心点
        center_indices = np.random.choice(n_samples, n_centers, replace=False)
        centers = embedding[center_indices]

        # 转换为PyTorch张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        centers = self._to_torch_tensor(centers, device)
        embedding = self._to_torch_tensor(embedding, device)

        return centers, embedding, device

    def _min_max_normalize(self, x: np.ndarray) -> np.ndarray:
        """最小-最大归一化"""
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min == 0:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    def _clamp_normalize(self, x: np.ndarray,) -> np.ndarray:
        """最小-最大归一化后，截断到0.5，再映射到0-1"""
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min == 0:
            return np.zeros_like(x)
        x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0.1, 0.5)*2

    def _symmetric_normalize(self, x: np.ndarray, clamp_value: float) -> np.ndarray:
        """
        对输入进行对称归一化，将数值映射到 [-clamp_value, clamp_value] 区间，保持0对称。
        使用 max(abs(x)) 作为标准值，正负分别映射到 clamp_value / -clamp_value。

        Args:
            x: 输入数组

        Returns:
            归一化后的数组，范围 [-1, 1]
        """
        max_abs = np.max(np.abs(x))
        if max_abs == 0:
            return np.zeros_like(x)
        return (x * clamp_value) / max_abs

    def _discretize_value(self, x: np.ndarray, num_bins: int = 15, num_padding=0) -> np.ndarray:
        """
        将输入值离散为 num_bins 个区间，**仅0被映射到中间bin**，其他值按绝对值远离0进行划分。
        返回离散索引数组，范围 [0, num_bins-1]，中间的 bin 为 num_bins//2。
        使用 self.lsdd_max_value 作为值的最大范围。

        支持三种模式：
        - 'linear': 线性离散化
        - 'square': 平方变换后离散化（强调较大值）
        - 'log': 对数变换后离散化（强调较小值）
        """
        # 使用成员变量作为值的范围
        x = np.clip(x, -self.lsdd_max_value, self.lsdd_max_value)
        center_bin = num_bins // 2

        # 根据模式进行变换
        if self.discretize_mode == 'square':
            # 平方变换：保持符号，但压缩小值范围
            sign = np.sign(x)
            abs_x = np.abs(x)
            # 避免除以零
            max_val = self.lsdd_max_value if self.lsdd_max_value > 0 else 1.0
            # 平方变换并归一化
            x = sign * (abs_x / max_val) ** 2 * max_val
        elif self.discretize_mode == 'sqrt':
            # 平方根变换：保持符号，但压缩大值范围
            sign = np.sign(x)
            abs_x = np.abs(x)
            # 避免除以零
            max_val = self.lsdd_max_value if self.lsdd_max_value > 0 else 1.0
            # 平方根变换并归一化
            x = sign * np.sqrt(abs_x / max_val) * max_val
        elif self.discretize_mode == 'log':
            # 对数变换：保持符号，但压缩大值范围
            sign = np.sign(x)
            abs_x = np.abs(x)
            # 避免除以零和log(0)
            max_val = self.lsdd_max_value if self.lsdd_max_value > 0 else 1.0
            min_val = max_val * 1e-6  # 避免log(0)
            # 对数变换并归一化
            x = sign * np.log1p(abs_x / min_val) / \
                np.log1p(max_val / min_val) * max_val

        # print(f"max value: {np.max(x)}, min value: {np.min(x)}")
        # 'linear' 模式不需要变换

        # 重新裁剪变换后的值
        x = np.clip(x, -self.lsdd_max_value, self.lsdd_max_value)

        result = np.full_like(x, fill_value=center_bin, dtype=int)

        # 非0部分分别处理负数和正数
        neg_mask = x < 0
        pos_mask = x > 0

        # 左边 bin（负值）：分成 center_bin 个区间 [-max, 0)
        neg_bins = np.linspace(-self.lsdd_max_value, 0, center_bin + 1)
        result[neg_mask] = np.digitize(x[neg_mask], neg_bins[1:], right=True)

        # 右边 bin（正值）：分成 center_bin 个区间 (0, max]
        pos_bins = np.linspace(0, self.lsdd_max_value, center_bin + 1)
        result[pos_mask] = center_bin + \
            np.digitize(x[pos_mask], pos_bins[1:], right=False)

        # 归一化处理
        result = result/num_bins
        result = (result-0.5)/(num_bins+num_padding*2)*num_bins+0.5

        # print(f"max value: {np.max(result)}, min value: {np.min(result)}")
        return result

    def _sample_large_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray,
                                 max_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        对大数据量的embedding进行采样

        Args:
            embedding1: 第一个分布的嵌入数据
            embedding2: 第二个分布的嵌入数据
            max_samples: 最大采样数量

        Returns:
            采样后的(embedding1, embedding2)
        """
        # 对embedding1进行采样
        if embedding1.shape[0] > max_samples:
            print(
                f"Embedding1数据量 {embedding1.shape[0]} 超过限制，随机采样到 {max_samples}")
            np.random.seed(42)  # 确保可重复性
            sample_indices1 = np.random.choice(
                embedding1.shape[0], max_samples, replace=False)
            embedding1 = embedding1[sample_indices1]

        # 对embedding2进行采样
        if embedding2.shape[0] > max_samples:
            print(
                f"Embedding2数据量 {embedding2.shape[0]} 超过限制，随机采样到 {max_samples}")
            np.random.seed(43)  # 使用不同的种子确保两个采样独立
            sample_indices2 = np.random.choice(
                embedding2.shape[0], max_samples, replace=False)
            embedding2 = embedding2[sample_indices2]

        print(
            f"采样后数据量: embedding1={embedding1.shape[0]}, embedding2={embedding2.shape[0]}")
        return embedding1, embedding2

    def _get_lsdd_diff_func(self, embedding1: np.ndarray, embedding2: np.ndarray,
                            viewport: Dict[str, float],  # 新增视口参数
                            sigma: float = None,
                            ours: bool = True,
                            sample: bool = False):   # 可选的动态sigma
        """
        获取LSDD密度差异函数（支持视口和动态sigma）
        """
        if config.MODE in ['default', 'lsdd']:
            sample = True
        # 计算扩展视口（viewport + 0.5倍padding）
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']
        width = x1 - x0
        height = y1 - y0

        alpha = 1  # 控制缩放灵敏度
        # 根据视口大小动态设置sigma，视口越小sigma越小，计算越细致
        if sigma == None:
            min_side = min(width, height)
            lsdd_sigma = (min_side ** alpha) / 50.0
            if lsdd_sigma < 0.001:
                lsdd_sigma = 0.001
            print(f"视口大小: {min_side:.4f}, 动态sigma: {lsdd_sigma:.6f}")
            self.lsdd_sigma = lsdd_sigma
        else:
            lsdd_sigma = sigma
            self.lsdd_sigma = lsdd_sigma
            print(f"使用自定义sigma: {lsdd_sigma:.6f}")

        print(f"计算LSDD密度差异函数... 使用sigma={lsdd_sigma:.6f}")
        start_time = time.time()
        step_start = time.time()  # 步骤计时开始

        if sample:
            # 采样大数据量以提高性能
            embedding1, embedding2 = self._sample_large_embeddings(
                embedding1, embedding2)
            print(f"数据采样完成，耗时: {time.time() - step_start:.4f}秒")
            step_start = time.time()  # 重置步骤计时

        embedding_all = np.vstack([embedding1, embedding2])
        n1, n2 = embedding1.shape[0], embedding2.shape[0]
        n_all = embedding_all.shape[0]

        # 转换为PyTorch张量并移到GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embedding_all = torch.tensor(
            embedding_all, dtype=torch.float32).to(device)
        print(f"数据转换完成，耗时: {time.time() - step_start:.4f}秒")
        step_start = time.time()  # 重置步骤计时

        # 使用视口内的网格点作为中心点
        if ours:
            # grid_points = self._generate_grid_points(
            #     {'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1})
            grid_points = self._generate_grid_points(
                {'x_min': x0, 'y_min': y0, 'x_max': x1, 'y_max': y1})

            # 限制中心点数量以防止OOM
            max_centers = 10000
            grid_size_x = (x1 - x0) / self.grid_resolution
            grid_size_y = (y1 - y0) / self.grid_resolution

            if grid_points.shape[0] > max_centers:
                print(f"LSDD中心点数量 {grid_points.shape[0]} 超过限制 {max_centers}，进行随机采样")
                np.random.seed(42)
                indices = np.random.choice(grid_points.shape[0], max_centers, replace=False)
                grid_points_sampled = grid_points[indices]
                # 加上半个网格大小，使其成为中心点
                grid_points_sampled = grid_points_sampled + np.array([grid_size_x / 2, grid_size_y / 2])

                center_points = torch.tensor(grid_points_sampled, dtype=torch.float32).to(device)
                n_center = max_centers
            else:
                # 加上半个网格大小
                grid_points = grid_points + np.array([grid_size_x / 2, grid_size_y / 2])
                n_center = grid_points.shape[0]
                center_points = torch.tensor(
                    grid_points, dtype=torch.float32).to(device)
        else:
            n_center = n_all
            center_points = embedding_all
        print(f"中心点生成完成，耗时: {time.time() - step_start:.4f}秒")
        step_start = time.time()  # 重置步骤计时

        # 计算H矩阵
        diff_matrix = center_points[:, None, :] - center_points[None, :, :]
        distance_matrix = torch.norm(diff_matrix, dim=2)
        d = 2
        H = (np.pi * lsdd_sigma ** 2) ** (d / 2) * \
            torch.exp(-distance_matrix ** 2 / (4 * lsdd_sigma ** 2))
        print(f"H矩阵计算完成，耗时: {time.time() - step_start:.4f}秒")
        step_start = time.time()  # 重置步骤计时

        # 计算h_hat
        def compute_h_hat(embedding, n):
            embedding = torch.tensor(embedding, dtype=torch.float32).to(device)
            # 使用分批处理避免内存爆炸
            batch_size = 10000  # 根据可用内存调整
            result = torch.zeros(n_center, device=device)

            for i in range(0, embedding.size(0), batch_size):
                batch = embedding[i:i+batch_size]
                diff_matrix = batch[:, None, :] - center_points[None, :, :]
                distance_matrix = torch.norm(diff_matrix, dim=2)
                result += torch.sum(torch.exp(-distance_matrix **
                                    2 / (2 * lsdd_sigma ** 2)), dim=0)

            return result / n

        h_hat1 = compute_h_hat(embedding1, n1)
        h_hat2 = compute_h_hat(embedding2, n2)
        h_hat = h_hat1 - h_hat2
        print(f"h_hat计算完成，耗时: {time.time() - step_start:.4f}秒")
        step_start = time.time()  # 重置步骤计时

        # 使用梯度下降计算theta_hat
        if ours:
            theta_hat = torch.zeros(
                n_center, dtype=torch.float32, device=device, requires_grad=True)

            def get_loss(theta_hat):
                theta_hat_column = theta_hat.unsqueeze(1)
                loss = theta_hat_column.t() @ H @ theta_hat_column - 2 * h_hat @ theta_hat_column + \
                    self.lsdd_lambda * (theta_hat_column.t() @ theta_hat_column)
                return loss.squeeze()

            optimizer = torch.optim.LBFGS([theta_hat])

            def closure():
                optimizer.zero_grad()
                loss = get_loss(theta_hat)
                loss.backward()
                return loss

            # 优化
            for _ in range(10):
                optimizer.step(closure)

            theta_hat = theta_hat.detach()
        else:
            # 使用矩阵求逆计算theta_hat
            # 将张量移动到CPU后再转换为numpy数组
            H_cpu = H.cpu().numpy()
            h_hat_cpu = h_hat.cpu().numpy()

            HaddI = H_cpu + np.eye(n_center)
            HaddI_inv = np.linalg.inv(HaddI)
            theta_hat_np = HaddI_inv @ h_hat_cpu

            # 将结果转换回PyTorch张量并放回原设备
            theta_hat = torch.tensor(
                theta_hat_np, dtype=torch.float32, device=device)
        print(f"theta_hat计算完成，耗时: {time.time() - step_start:.4f}秒")
        step_start = time.time()  # 重置步骤计时

        # 定义差异函数
        def g(x):
            x = self._standardize_input(x)

            # 分批处理以避免OOM
            batch_size = 2000  # 2000 * 10000 * 8 bytes = 160MB
            n_x = x.shape[0]
            result_list = []

            for i in range(0, n_x, batch_size):
                x_batch = x[i:i+batch_size]
                x_batch_torch = self._to_torch_tensor(x_batch, device)

                diff = x_batch_torch[:, None, :] - center_points[None, :, :]
                dist = torch.sum(diff ** 2, dim=2)
                kernel_values = torch.exp(-dist / (2 * lsdd_sigma ** 2))
                res_batch = torch.sum(theta_hat[None, :] * kernel_values, dim=1)
                result_list.append(res_batch.cpu().numpy())

            result = np.concatenate(result_list)

            if result.shape[0] == 1 and n_x == 1:
                return result[0]
            return result

        print(f"LSDD计算完成，总耗时: {time.time() - start_time:.4f}秒")
        return g

    def _get_kde_diff_func(self, embedding1: np.ndarray, embedding2: np.ndarray):
        """
        获取KDE密度差异函数

        Args:
            embedding1: 第一个分布的嵌入数据
            embedding2: 第二个分布的嵌入数据

        Returns:
            密度差异计算函数
        """
        # 采样大数据量以提高KDE计算性能
        embedding1, embedding2 = self._sample_large_embeddings(
            embedding1, embedding2)

        kde1 = self._get_fast_kde_func(embedding1)
        kde2 = self._get_fast_kde_func(embedding2)

        def diff_func(x):
            return kde1(x) - kde2(x)

        return diff_func

    def _get_fast_kde_func(self, embedding: np.ndarray):
        """
        快速KDE函数（统一实现，调用加权KDE函数）

        Args:
            embedding: 嵌入数据，形状为 (n_samples, 2)

        Returns:
            KDE计算函数
        """
        # 直接调用加权KDE函数，不提供权重（使用均匀权重）
        return self.get_weighted_kde_func(embedding, weights=None)

    def get_weighted_kde_func(self, embedding: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        统一的KDE函数实现（支持加权和非加权）

        Args:
            embedding: 嵌入数据，形状为 (n_samples, 2)
            weights: 权重，形状为 (n_samples,)，如果为None则使用均匀权重

        Returns:
            KDE计算函数

        Note:
            这是统一的KDE实现，当weights=None时等价于原来的_get_fast_kde_func
        """
        # 对大数据量进行采样以提高性能
        max_samples = 10000
        if embedding.shape[0] > max_samples:
            print(
                f"单个KDE embedding数据量 {embedding.shape[0]} 超过限制，随机采样到 {max_samples}")
            np.random.seed(42)  # 确保可重复性
            sample_indices = np.random.choice(
                embedding.shape[0], max_samples, replace=False)
            embedding = embedding[sample_indices]
            if weights is not None:
                weights = weights[sample_indices]
            print(f"KDE采样后数据量: {embedding.shape[0]}")

        n_samples = embedding.shape[0]
        centers, embedding_torch, device = self._setup_kde_centers(embedding)

        if weights is not None:
            weights = self._to_torch_tensor(weights, device)
            weights = weights / torch.sum(weights)  # 归一化
        else:
            weights = torch.ones(n_samples, device=device) / n_samples

        # 计算中心点权重
        diff_matrix = embedding_torch[:, None, :] - centers[None, :, :]
        dist_matrix = torch.sum(diff_matrix ** 2, dim=2)
        kernel_matrix = torch.exp(-dist_matrix / (2 * self.kde_sigma ** 2))
        center_weights = torch.sum(weights[:, None] * kernel_matrix, dim=0)
        center_weights = center_weights / torch.sum(center_weights)

        def kde_func(x):
            x = self._standardize_input(x)

            # 分批处理以避免OOM
            batch_size = 2000
            n_x = x.shape[0]
            result_list = []

            for i in range(0, n_x, batch_size):
                x_batch = x[i:i+batch_size]
                x_batch_torch = self._to_torch_tensor(x_batch, device)

                diff = x_batch_torch[:, None, :] - centers[None, :, :]
                dist = torch.sum(diff ** 2, dim=2)
                kernel_values = torch.exp(-dist / (2 * self.kde_sigma ** 2))
                density_batch = torch.sum(center_weights[None, :] * kernel_values, dim=1)
                result_list.append(density_batch.cpu().numpy())

            density = np.concatenate(result_list)

            if density.shape[0] == 1 and n_x == 1:
                return density[0]
            return density

        return kde_func

    def get_mixed_heatmap_on_lsdd_colorscale(self,
                                             embedding1: np.ndarray,
                                             embedding2: np.ndarray,
                                             viewport: Dict[str, float],
                                             sigma: float = None) -> List[Dict[str, Any]]:  # 保持列表形式（内部使用）
        """
        生成 mixed 复合热力图，返回细分四小格的列表，每个格子坐标和value与原heatmap格式一致。
        """
        import time

        t0 = time.time()
        grid_points = self._generate_grid_points(viewport)
        t1 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] 生成网格点耗时: {(t1 - t0)*1000:.2f}ms")

        n_grids = len(grid_points)
        counts1 = np.zeros(n_grids, dtype=int)
        counts2 = np.zeros(n_grids, dtype=int)

        # 计算网格大小
        t2 = time.time()
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']
        grid_size_x = (x1 - x0) / self.grid_resolution  # 使用self.grid_resolution
        grid_size_y = (y1 - y0) / self.grid_resolution  # 使用self.grid_resolution
        centers = grid_points + np.array([grid_size_x/2, grid_size_y/2])  # 计算中心点
        t3 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] 计算网格大小和中心点耗时: {(t3 - t2)*1000:.2f}ms")

        # 使用向量化操作统计embedding数量
        t4 = time.time()

        # 计算网格索引 - 向量化操作
        def get_grid_indices(embedding, viewport):
            # 计算网格坐标
            x_coords = embedding[:, 0]
            y_coords = embedding[:, 1]
            col = ((x_coords - x0) / grid_size_x).astype(int)
            row = ((y_coords - y0) / grid_size_y).astype(int)

            # 计算一维索引
            indices = col * self.grid_resolution + row

            # 过滤无效索引
            valid_mask = (col >= 0) & (col < self.grid_resolution) & (row >= 0) & (row < self.grid_resolution)
            indices[~valid_mask] = -1  # 标记无效索引

            return indices

        # 统计embedding1
        indices1 = get_grid_indices(embedding1, viewport)
        valid_indices1 = indices1[indices1 >= 0]
        counts1 = np.bincount(valid_indices1, minlength=n_grids)

        # 统计embedding2
        indices2 = get_grid_indices(embedding2, viewport)
        valid_indices2 = indices2[indices2 >= 0]
        counts2 = np.bincount(valid_indices2, minlength=n_grids)

        t5 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] 统计embedding数量耗时: {(t5 - t4)*1000:.2f}ms")

        # d1_fun = self._get_fast_kde_func(embedding1)
        # d2_fun = self._get_fast_kde_func(embedding2)
        # d1_raw = d1_fun(centers)
        # d2_raw = d2_fun(centers)
        t6 = time.time()
        d1_raw = counts1
        d2_raw = counts2
        d1_norm = self._clamp_normalize(d1_raw)
        d2_norm = self._clamp_normalize(d2_raw)
        t7 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] 归一化embedding数量耗时: {(t7 - t6)*1000:.2f}ms")
        # print(f"Embedding1数量归一化: {len(np.nonzero(d1_norm)[0])}")
        # print(f"Embedding2数量归一化: {len(np.nonzero(d2_norm)[0])}")

        # 计算LSDD差异
        t8 = time.time()
        diff_func = self._get_lsdd_diff_func(
            embedding1, embedding2, viewport, sigma=sigma)
        lsdd_vals = diff_func(centers)
        lsdd_vals = np.array(lsdd_vals, dtype=float)
        t9 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] 计算LSDD差异耗时: {(t9 - t8)*1000:.2f}ms")

        # 确保LSDD可看清,当值过小时，放缩到一定范围
        # t10 = time.time()
        # if config.MODE == 'default':
        #     least_value = 15
        #     lsdd_vals = self._symmetric_normalize(lsdd_vals, least_value)
        # t11 = time.time()
        # print(f"[get_mixed_heatmap_on_lsdd_colorscale] LSDD归一化耗时: {(t11 - t10)*1000:.2f}ms")

        # 直接对LSDD值进行分bin处理，不再归一化
        t12 = time.time()
        lsdd_discrete = self._discretize_value(
            np.array(lsdd_vals), num_bins=2100, num_padding=1)
        t13 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] LSDD分bin处理耗时: {(t13 - t12)*1000:.2f}ms")

        results = []
        t14 = time.time()
        for i, gp in enumerate(grid_points):
            # if counts1[i] + counts2[i] == 0:
            #     # 可选：无点的格子不返回，透明处理
            #     continue

            # dominant = 1 if d1_norm[i] >= d2_norm[i] else -1

            x0, y0 = gp
            x1, y1 = x0 + grid_size_x, y0 + grid_size_y
            results.append({
                'x0': float(x0),
                'y0': float(y0),
                'x1': float(x1),
                'y1': float(y1),
                'value': float(lsdd_discrete[i]),
                'type': 'back',
                'id': i,
            })

            center_draw = True if lsdd_vals[i] > 0 and d2_raw[i] > 0 else (
                True if lsdd_vals[i] < 0 and d1_raw[i] > 0 else False)
            if not center_draw:
                continue  # 不绘制中心格子

            # center_value = 1 - lsdd_discrete[i]
            if lsdd_vals[i] > 0:
                center_value = 0.2
            else:
                center_value = 0.8
            # 计算中心格子大小,分四档
            s_raw = (d1_norm[i] if lsdd_vals[i] < 0 else d2_norm[i])
            if s_raw <= 0:
                continue  # 忽略不可见 center
            size = int(s_raw*4)+2
            s = size * (min(grid_size_x, grid_size_y) / 12.0)

            cx, cy = x0 + grid_size_x / 2, y0 + grid_size_y / 2
            # print(s)
            results.append({
                'x0': float(cx - s / 2),
                'y0': float(cy - s / 2),
                'x1': float(cx + s / 2),
                'y1': float(cy + s / 2),
                'value': float(center_value),
                'type': 'center',
                'id': i,
            })
        t15 = time.time()
        print(f"[get_mixed_heatmap_on_lsdd_colorscale] 生成最终结果耗时: {(t15 - t14)*1000:.2f}ms")

        return results

    def get_kde_heatmap_on_lsdd_colorscale(self,
                                             embedding1: np.ndarray,
                                             embedding2: np.ndarray,
                                             viewport: Dict[str, float],
                                             method: str = 'kde',
                                             sigma: float = None) -> List[Dict[str, Any]]:  # 保持列表形式（内部使用）
        """
        生成 mixed 复合热力图，返回细分四小格的列表，每个格子坐标和value与原heatmap格式一致。
        """
        grid_points = self._generate_grid_points(viewport)
        n_grids = len(grid_points)
        counts1 = np.zeros(n_grids, dtype=int)
        counts2 = np.zeros(n_grids, dtype=int)

        # 计算网格大小
        x0 = viewport['x_min']
        y0 = viewport['y_min']
        x1 = viewport['x_max']
        y1 = viewport['y_max']
        grid_size_x = (x1 - x0) / self.grid_resolution  # 使用self.grid_resolution
        grid_size_y = (y1 - y0) / self.grid_resolution  # 使用self.grid_resolution
        centers = grid_points + \
            np.array([grid_size_x/2, grid_size_y/2])  # 计算中心点

        # 统计embedding数量 - 向量化操作
        def get_grid_indices(embedding, viewport):
            # 计算网格坐标
            x_coords = embedding[:, 0]
            y_coords = embedding[:, 1]
            col = ((x_coords - x0) / grid_size_x).astype(int)
            row = ((y_coords - y0) / grid_size_y).astype(int)

            # 计算一维索引
            indices = col * self.grid_resolution + row

            # 过滤无效索引
            valid_mask = (col >= 0) & (col < self.grid_resolution) & (row >= 0) & (row < self.grid_resolution)
            indices[~valid_mask] = -1  # 标记无效索引

            return indices

        # 统计embedding1
        indices1 = get_grid_indices(embedding1, viewport)
        valid_indices1 = indices1[indices1 >= 0]
        counts1 = np.bincount(valid_indices1, minlength=n_grids)

        # 统计embedding2
        indices2 = get_grid_indices(embedding2, viewport)
        valid_indices2 = indices2[indices2 >= 0]
        counts2 = np.bincount(valid_indices2, minlength=n_grids)

        # d1_fun = self._get_fast_kde_func(embedding1)
        # d2_fun = self._get_fast_kde_func(embedding2)
        # d1_raw = d1_fun(centers)
        # d2_raw = d2_fun(centers)
        d1_raw = counts1
        d2_raw = counts2
        d1_norm = self._clamp_normalize(d1_raw)
        d2_norm = self._clamp_normalize(d2_raw)
        # print(f"Embedding1数量归一化: {len(np.nonzero(d1_norm)[0])}")
        # print(f"Embedding2数量归一化: {len(np.nonzero(d2_norm)[0])}")

        # 计算LSDD差异
        if method == 'lsdd':
            diff_func = self._get_lsdd_diff_func(embedding1, embedding2, viewport, sigma=sigma)
        else:
            diff_func = self._get_kde_diff_func(embedding1, embedding2)

        lsdd_vals = diff_func(centers)
        lsdd_vals = np.array(lsdd_vals, dtype=float)

        # 确保LSDD可看清,当值过小时，放缩到一定范围
        least_value = 10
        lsdd_vals = self._symmetric_normalize(lsdd_vals, least_value)

        # 直接对LSDD值进行分bin处理，不再归一化
        lsdd_discrete = self._discretize_value(
            np.array(lsdd_vals), num_bins=2100, num_padding=1)
        results = []
        for i, gp in enumerate(grid_points):
            # if counts1[i] + counts2[i] == 0:
            #     # 可选：无点的格子不返回，透明处理
            #     continue

            # dominant = 1 if d1_norm[i] >= d2_norm[i] else -1

            x0, y0 = gp
            x1, y1 = x0 + grid_size_x, y0 + grid_size_y
            results.append({
                'x0': float(x0),
                'y0': float(y0),
                'x1': float(x1),
                'y1': float(y1),
                'value': float(lsdd_discrete[i]),
                'type': 'back',
                'id': i,
            })


        return results
