from datatools.reducer.runner import get_slice_different_embedding_for_evaluation
import numpy as np
from typing import List, Dict, Any
from datatools.slice import Slice
from matplotlib.colors import ListedColormap


def get_coverage_heatmap(embedding1: np.ndarray,
                         embedding2: np.ndarray,
                         name1: str = 'synthetic',
                         name2: str = 'real',
                         grid_size: int = 20,
                         output_path: str = 'coverage_heatmap.png'):
    """
        生成20*20网格的覆盖率热力图数据。
        每格颜色有四种，例如白色、红色、蓝色、紫色分别表示无点、只有embedding1、只有embedding2和两种都有。
        绘制图表，并标注覆盖率=(紫色格子数量)/(红色格子数量+紫色格子数量)。
        保存绘制的图表，图例和标题。
        返回两个列表，分别是只有红色格子的embedding1序号和只有蓝色格子的embedding2序号。
    """
    import matplotlib.pyplot as plt

    # 合并所有点，确定边界，保证边界正方形且中心不变
    all_points = np.vstack([embedding1, embedding2])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    half_size = max(x_max - x_min, y_max - y_min) / 2
    x_min = x_center - half_size
    x_max = x_center + half_size
    y_min = y_center - half_size
    y_max = y_center + half_size

    # 计算每个点所在的格子
    def get_grid_indices(emb, x_min, x_max, y_min, y_max, grid_size):
        x_idx = ((emb[:, 0] - x_min) / (x_max - x_min + 1e-8)
                 * grid_size).astype(int)
        y_idx = ((emb[:, 1] - y_min) / (y_max - y_min + 1e-8)
                 * grid_size).astype(int)
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)
        return x_idx, y_idx

    grid1 = np.zeros((grid_size, grid_size), dtype=bool)
    grid2 = np.zeros((grid_size, grid_size), dtype=bool)

    x1, y1 = get_grid_indices(embedding1, x_min, x_max, y_min, y_max, grid_size)
    x2, y2 = get_grid_indices(embedding2, x_min, x_max, y_min, y_max, grid_size)

    grid1[y1, x1] = True
    grid2[y2, x2] = True

    # 0: 无点，1: 只有embedding1，2: 只有embedding2，3: 两种都有
    heatmap = np.zeros((grid_size, grid_size), dtype=int)
    heatmap += grid1.astype(int)
    heatmap += grid2.astype(int) * 2

    # 统计格子数量
    red_count = np.sum(heatmap == 1)
    purple_count = np.sum(heatmap == 3)
    coverage = purple_count / (red_count+purple_count)

    # 颜色映射
    cmap = ListedColormap(['white', 'red', 'blue', 'purple'])
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, origin='lower', cmap=cmap,
               extent=[x_min, x_max, y_min, y_max])

    # 图例
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='white', label='No points'),
        mpatches.Patch(color='red', label=f'Only {name1}'),
        mpatches.Patch(color='blue', label=f'Only {name2}'),
        mpatches.Patch(color='purple', label='Both')
    ]
    plt.legend(handles=legend_handles, loc='upper left')
    plt.title(
        f'Coverage Heatmap\nCoverage={coverage:.2%} ({name2}/{name1})')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # 计算每个embedding1/embedding2点所在格子的类型
    only1_indices = []
    only2_indices = []
    for idx, (x, y) in enumerate(zip(x1, y1)):
        if heatmap[y, x] == 1:
            only1_indices.append(idx)
    for idx, (x, y) in enumerate(zip(x2, y2)):
        if heatmap[y, x] == 2:
            only2_indices.append(idx)

    return coverage, only1_indices, only2_indices


def calculate_coverage(slice1: Slice, slice2: Slice,name1:str,name2:str) -> float:
    """
    计算覆盖率，给定两个slice，计算它们的embedding覆盖率。
    """
    embedding1, embedding2 = get_slice_different_embedding_for_evaluation(
        slice1=slice1,
        slice2=slice2,
        feature_type='clip',
        reducer_name='umap',
        fit_mode='joint'
    )
    print(f"[✓] {slice1.name} embedding shape: {embedding1.shape}")

    # 计算覆盖率
    coverage, only1_indices, only2_indices = get_coverage_heatmap(
        embedding1=embedding1,
        embedding2=embedding2,
        name1=name1,
        name2=name2,
        grid_size=20,
        output_path=f'coverage_heatmap_{slice1.name}_{slice2.name}.png'
    )
    # 返回两个indices对应的路径
    print(only1_indices[:10], only2_indices[:10])
    slice1_indices = slice1.get_field('image_path')
    slice2_indices = slice2.get_field('image_path')
    # Convert to numpy array for advanced indexing if not already
    slice1_indices = np.array(slice1_indices)
    slice2_indices = np.array(slice2_indices)
    print(f"[✓] {slice1.name} indices: {slice1_indices[only1_indices[:10]]}")
    print(f"[✓] {slice2.name} indices: {slice2_indices[only2_indices[:10]]}")
    # 将两个列表输出为json,路径只保留最后一级目录和文件名
    import os
    import json
    def get_last_dir_and_file(paths):
        result = []
        for p in paths:
            parts = os.path.normpath(p).split(os.sep)
            if len(parts) >= 2:
                result.append(os.path.join(parts[-2], parts[-1]))
            else:
                result.append(parts[-1])
        return result

    only1_paths = get_last_dir_and_file(slice1_indices[only1_indices])
    only2_paths = get_last_dir_and_file(slice2_indices[only2_indices])

    with open(f'coverage_indices_{slice1.name}_{slice2.name}.json', 'w') as f:
        coverage_data = {
            'coverage': coverage,
            f'only_{name1}': only1_paths,
            f'only_{name2}': only2_paths
        }
        json.dump(coverage_data, f, ensure_ascii=False, indent=2)
    return coverage


if __name__ == "__main__":
    # # Infographic
    # slice1 = Slice('JN-real')
    # slice2 = Slice('JN-synthetic')
    # slice3 = Slice('JN-new_real')

    # coverage = calculate_coverage(slice1, slice2,name1=slice1.name,name2=slice2.name)
    # print(f"Coverage for {slice2.name} in {slice1.name}: {coverage:.2%}")

    # coverage = calculate_coverage(slice3, slice2,name1=slice3.name,name2=slice2.name)
    # print(f"Coverage for {slice2.name} in {slice3.name}: {coverage:.2%}")
    slice1 = Slice('JN-real')
    slice2 = Slice('JN-synthetic')

    coverage = calculate_coverage(
        slice1, slice2, name1='real', name2='synthetic')
    print(f"Coverage for {slice2.name} in {slice1.name}: {coverage:.2%}")
