from datatools.slice import Slice
from typing import Literal
import os
import numpy as np
import importlib.util
import sys

from .pca import PCAReducer
from .umap import UMAPReducer
from config import REDUCED_DIR, n_neighbors
from datatools.reducer.registry import get_reducer

def _create_regularizer():
    # Baselines and installations with regularization disabled do not need Vulkan.
    regularization_path = os.path.join(os.path.dirname(__file__), 'de-cluttering', 'regularization.py')
    spec = importlib.util.spec_from_file_location("regularization", regularization_path)
    regularization_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regularization_module)
    return regularization_module.Regularizer()


def reduce_and_save(
    feature_path: str,
    output_path: str = None,
    reducer_name: str = 'umap',
    n_components: int = 2,
    save_model_path: str = None,
    load_model_path: str = None
):
    reducer_cls = get_reducer(reducer_name)
    assert reducer_cls is not None, f"未知降维器: {reducer_name}"

    print(f"[→] 加载特征：{feature_path}")
    features = np.load(feature_path)
    embedding = []
    if load_model_path and os.path.exists(load_model_path):
        print(f"[→] 加载已有降维模型：{load_model_path}")
        reducer = reducer_cls(n_components=n_components)
        reducer = reducer.load(load_model_path)
        embedding = reducer.transform(features)
    else:
        print(f"[→] 使用 {reducer_name.upper()} 降维到 {n_components}D")
        reducer = reducer_cls(n_components=n_components)
        embedding = reducer.fit_transform(features)

        if save_model_path:
            print(f"[✓] 保存降维模型到：{save_model_path}")
            reducer.save(save_model_path)
        else:
            reducer_model_path = os.path.join(
                output_path, f"{reducer_name}_model.joblib")
            print(f"[✓] 保存降维模型到：{reducer_model_path}")
            reducer.save(reducer_model_path)
    if output_path:
        embedding_path = os.path.join(
            output_path, f"embedding_{reducer_name}.npy")
        print(f"[✓] 保存 embedding 到：{embedding_path}")
        np.save(embedding_path, embedding)

    return embedding


def reduce_runner(
    features: np.ndarray,
    output_path: str = None,
    reducer_name: str = 'umap',
    n_components: int = 2,
    save_model_path: str = None,
    load_model_path: str = None
):
    reducer_cls = get_reducer(reducer_name)
    assert reducer_cls is not None, f"未知降维器: {reducer_name}"

    embedding = []
    if load_model_path and os.path.exists(load_model_path):
        print(f"[→] 加载已有降维模型：{load_model_path}")
        reducer = reducer_cls(n_components=n_components)
        reducer = reducer.load(load_model_path)
        embedding = reducer.transform(features)
    else:
        print(f"[→] 使用 {reducer_name.upper()} 降维到 {n_components}D")
        reducer = reducer_cls(n_components=n_components)
        embedding = reducer.fit_transform(features)

        if save_model_path:
            print(f"[✓] 保存降维模型到：{save_model_path}")
            reducer.save(save_model_path)
        else:
            reducer_model_path = os.path.join(
                output_path, f"{reducer_name}_model.joblib")
            print(f"[✓] 保存降维模型到：{reducer_model_path}")
            reducer.save(reducer_model_path)
    if output_path:
        embedding_path = os.path.join(
            output_path, f"embedding_{reducer_name}.npy")
        print(f"[✓] 保存 embedding 到：{embedding_path}")
        np.save(embedding_path, embedding)

    return embedding


def get_slice_different_embedding(
    slice1: Slice,
    slice2: Slice,
    feature_type: str = 'dreamsim',
    reducer_name: str = 'umap',
    fit_mode: Literal['joint', 's1', 's2', 'coreset'] = 'joint',
    output_root: str = REDUCED_DIR,
    use_regularization: bool = True
) -> np.ndarray:
    print(f"[→] 获取不同 slice 的 embedding: {slice1.name}, {slice2.name}")
    print(f"[→] 使用降维器: {reducer_name}, 特征类型: {feature_type}, 模式: {fit_mode}")
    if slice1.name==slice2.name and fit_mode!='coreset':
        fit_mode='s1'

    if fit_mode == 'joint' or fit_mode == 'coreset':
        # 无序组合标识符（用于 joint 情况）
        slice_names = sorted([slice1.name, slice2.name])
        pair_id = f"{slice_names[0]}_{slice_names[1]}"
        base_dir = os.path.join(output_root, pair_id)
    elif fit_mode == 's1':
        # 仅使用 slice1 的标识符
        base_dir = os.path.join(output_root, slice1.name)
    elif fit_mode == 's2':
        # 仅使用 slice2 的标识符
        base_dir = os.path.join(output_root, slice2.name)
    else:
        raise ValueError(f"未知 fit_mode: {fit_mode}")

    suffix = '' if n_neighbors == 15 or reducer_name == 'pca' else f'_n{n_neighbors}'

    base_dir = os.path.join(base_dir, reducer_name)
    if feature_type != None:
        print(f"[→] 使用特征类型: {feature_type}")
        base_dir=os.path.join(base_dir, feature_type)
    os.makedirs(base_dir, exist_ok=True)

    model_path = os.path.join(base_dir, f"model{suffix}.joblib")

    emb_dir = os.path.join(base_dir, "embedding")
    os.makedirs(emb_dir, exist_ok=True)

    emb_dir = os.path.join(base_dir, "embedding")
    os.makedirs(emb_dir, exist_ok=True)

    # === 先检查 embedding 文件是否存在 ===
    emb_path1 = os.path.join(emb_dir, f"{slice1.name}{suffix}.npy")
    emb_path2 = os.path.join(emb_dir, f"{slice2.name}{suffix}.npy")

    embedding1_exists = os.path.exists(emb_path1)
    embedding2_exists = os.path.exists(emb_path2)

    # 只有在需要生成 embedding 时才加载 reducer
    if not embedding1_exists or not embedding2_exists:
        print(f"[→] 需要生成 embedding，准备加载降维器")
        reducer_cls = get_reducer(reducer_name)
        assert reducer_cls is not None, f"未知降维器: {reducer_name}"
        reducer = reducer_cls(n_components=2)

        # 加载特征数据（如果需要训练模型，需要所有特征；如果只生成embedding，需要对应特征）
        feature_type=feature_type.split('-')
        feat1 = slice1.get_field('feature', feature_type=feature_type)
        feat2 = slice2.get_field('feature', feature_type=feature_type)

        if os.path.exists(model_path):
            print(f"[→] 已存在降维模型：{model_path}")
            reducer = reducer.load(model_path)
            print(f"[✓] 加载降维模型完成")
        else:
            # === 准备训练数据 ===
            if fit_mode == 'joint':
                fit_feature = np.concatenate([feat1, feat2], axis=0)
            elif fit_mode == 'coreset':
                # 去除重合点
                ids1 = slice1.global_ids
                ids2 = slice2.global_ids
                mask = ~np.isin(ids2, ids1)
                feat2_unique = feat2[mask]
                fit_feature = np.concatenate([feat1, feat2_unique], axis=0)
                print(f"[→] Coreset模式: 去除 {len(ids2) - np.sum(mask)} 个重合点")
            elif fit_mode == 's1':
                fit_feature = feat1
            elif fit_mode == 's2':
                fit_feature = feat2
            else:
                raise ValueError(f"未知 fit_mode: {fit_mode}")
            print('正在训练降维模型')
            reducer.fit(fit_feature)
            reducer.save(model_path)

        print(f"[→] 降维模型加载完成")
    else:
        print(f"[✓] 所有 embedding 文件已存在，跳过降维器加载")

    # === 加载或生成 embedding ===
    if embedding1_exists:
        print(f"[✓] 已存在 slice1 embedding：{emb_path1}")
        embedding1 = np.load(emb_path1)
    else:
        print(f"[→] 生成 slice1 embedding: {len(feat1)}")
        embedding1 = reducer.transform(feat1)
        np.save(emb_path1, embedding1)

    if embedding2_exists:
        print(f"[✓] 已存在 slice2 embedding：{emb_path2}")
        embedding2 = np.load(emb_path2)
    else:
        print(f"[→] 生成 slice2 embedding: {len(feat2)}")
        embedding2 = reducer.transform(feat2)
        np.save(emb_path2, embedding2)

    print(f"[→] 对合并的 embedding 进行 regularization 处理")

    # 合并两个 embedding
    combined_embedding = np.concatenate([embedding1, embedding2], axis=0)
    print(f"[→] 合并两个 embedding")

    # 转换范围从原始范围到 [0, 1] 进行 regularization
    print(f"原始 embedding 范围: min={combined_embedding.min(axis=0)}, max={combined_embedding.max(axis=0)}")

    # 先归一化到 [0, 1]
    min_vals = combined_embedding.min(axis=0)
    max_vals = combined_embedding.max(axis=0)
    combined_normalized = (combined_embedding - min_vals) / (max_vals - min_vals + 1e-8)

    print(f"归一化后范围: min={combined_normalized.min(axis=0)}, max={combined_normalized.max(axis=0)}")
    print(f"[→] 归一化到 [0, 1] 范围")

    if not use_regularization:
        emb1 = combined_normalized[:len(embedding1)]
        emb2 = combined_normalized[len(embedding1):]
        # 统一返回三元组形式
        return emb1, emb2, combined_normalized

    print(f"[→] 对合并的 embedding 进行 regularization 处理")

    # === 新的基于网格的 regularization 逻辑 ===
    grid_size = 0.01  # 默认网格大小
    print(f"[→] 使用网格大小: {grid_size}")

    # 1. 将数据离散化到网格
    grid_coords = np.floor(combined_normalized / grid_size).astype(int)
    unique_grids = np.unique(grid_coords, axis=0)
    print(f"[→] 发现 {len(unique_grids)} 个有数据的网格")

    # 2. 收集所有有数据的网格的四周格点
    grid_points_set = set()
    for grid in unique_grids:
        # 对于2D数据，添加当前网格的4个角点
        for dx in [0, 1]:
            for dy in [0, 1]:
                grid_point = (grid[0] + dx, grid[1] + dy)
                grid_points_set.add(grid_point)

    # 转换为numpy数组并计算实际坐标
    grid_points_coords = np.array(list(grid_points_set))
    grid_points_real = grid_points_coords * grid_size

    # 确保格点在[0,1]范围内
    grid_points_real = np.clip(grid_points_real, 0, 1)

    # print(f"[→] 生成 {len(grid_points_real)} 个格点用于 regularization")

    # 3. 转换格点到 [-1, 1] 范围进行 regularization
    grid_points_for_reg = grid_points_real * 2 - 1
    grid_points_for_reg = grid_points_for_reg.astype(np.float32)

    # print(f"格点范围: min={grid_points_for_reg.min(axis=0)}, max={grid_points_for_reg.max(axis=0)}")

    # 4. 执行 regularization
    with _create_regularizer() as regularizer:
        regularized_grid_points = regularizer.regularize(grid_points_for_reg, kernel_radius=8, iterations=32)

    # print(f"[→] 执行格点 regularization")

    # 5. 转换回 [0, 1] 范围
    regularized_grid_points = (regularized_grid_points + 1) / 2

    # 6. 对所有原始数据进行向量化双线性插值
    def bilinear_interpolate_vectorized(points, grid_coords_array, regularized_grid_coords, grid_size):
        """
        向量化双线性插值
        """
        n_points = len(points)
        result = np.zeros_like(points)

        # 1. 批量计算所有点的网格坐标
        grid_x = np.floor(points[:, 0] / grid_size).astype(int)
        grid_y = np.floor(points[:, 1] / grid_size).astype(int)

        # 2. 计算插值权重
        x_frac = (points[:, 0] / grid_size) - grid_x
        y_frac = (points[:, 1] / grid_size) - grid_y
        x_frac = np.clip(x_frac, 0, 1)
        y_frac = np.clip(y_frac, 0, 1)

        # 3. 高效的格点网格构建和查找
        # 找到网格的边界
        min_grid_x, max_grid_x = grid_coords_array[:, 0].min(), grid_coords_array[:, 0].max()
        min_grid_y, max_grid_y = grid_coords_array[:, 1].min(), grid_coords_array[:, 1].max()

        # 扩展边界以包含所有可能的角点
        min_grid_x = min(min_grid_x, grid_x.min())
        max_grid_x = max(max_grid_x, grid_x.max() + 1)
        min_grid_y = min(min_grid_y, grid_y.min())
        max_grid_y = max(max_grid_y, grid_y.max() + 1)

        # 创建完整的格点网格
        grid_width = max_grid_x - min_grid_x + 1
        grid_height = max_grid_y - min_grid_y + 1

        # 初始化格点值数组，默认使用原始坐标（向量化）
        grid_values = np.zeros((grid_width, grid_height, 2))

        # 向量化填充网格的默认值
        i_indices, j_indices = np.meshgrid(np.arange(grid_width), np.arange(grid_height), indexing='ij')
        real_x = (min_grid_x + i_indices) * grid_size
        real_y = (min_grid_y + j_indices) * grid_size
        grid_values[:, :, 0] = np.clip(real_x, 0, 1)
        grid_values[:, :, 1] = np.clip(real_y, 0, 1)

        # 向量化填入regularized的值
        grid_i = grid_coords_array[:, 0] - min_grid_x
        grid_j = grid_coords_array[:, 1] - min_grid_y
        grid_values[grid_i, grid_j] = regularized_grid_points

        # 4. 完全向量化的角点值获取
        # 计算所有角点的网格索引（完全向量化）
        base_x = grid_x - min_grid_x
        base_y = grid_y - min_grid_y

        # 一次性获取所有4个角点的值
        idx_x = np.stack([base_x, base_x + 1, base_x, base_x + 1], axis=1)  # (n_points, 4)
        idx_y = np.stack([base_y, base_y, base_y + 1, base_y + 1], axis=1)  # (n_points, 4)

        # 限制索引在有效范围内
        idx_x = np.clip(idx_x, 0, grid_width - 1)
        idx_y = np.clip(idx_y, 0, grid_height - 1)

        # 批量获取所有角点值
        corner_values = grid_values[idx_x, idx_y]  # 利用高级索引一次性获取

        # 5. 向量化双线性插值计算
        # 获取4个角点的值
        v00 = corner_values[:, 0]  # (grid_x, grid_y)
        v10 = corner_values[:, 1]  # (grid_x+1, grid_y)
        v01 = corner_values[:, 2]  # (grid_x, grid_y+1)
        v11 = corner_values[:, 3]  # (grid_x+1, grid_y+1)

        # 双线性插值公式的向量化实现
        x_frac = x_frac[:, np.newaxis]  # 扩展维度用于广播
        y_frac = y_frac[:, np.newaxis]

        # 先在x方向插值
        top = v00 * (1 - x_frac) + v10 * x_frac
        bottom = v01 * (1 - x_frac) + v11 * x_frac

        # 再在y方向插值
        result = top * (1 - y_frac) + bottom * y_frac

        return result

    # print(f"[→] 对所有数据点进行向量化双线性插值")
    regularized_combined = bilinear_interpolate_vectorized(
        combined_normalized,
        grid_points_coords,
        regularized_grid_points,
        grid_size
    )
    np.save(os.path.join(base_dir, f"regularized_combined{suffix}.npy"), regularized_combined)
    print(f"[✓] Regularization 和插值完成，结果保存到 {os.path.join(base_dir, f'regularized_combined{suffix}.npy')}")

    # print(f"[→] 插值完成")


    # 7. 还原到原始范围
    # regularized_combined = regularized_combined * (max_vals - min_vals) + min_vals

    # 分离回两个 slice 的结果
    embedding1_reg = regularized_combined[:len(embedding1)]
    embedding2_reg = regularized_combined[len(embedding1):]

    # 统一返回三元组形式
    return embedding1_reg, embedding2_reg, regularized_combined


def get_slice_different_embedding_for_evaluation(
    slice1: Slice,
    slice2: Slice,
    feature_type: str = 'dreamsim',
    reducer_name: str = 'umap',
    fit_mode: Literal['joint', 's1', 's2'] = 'joint',
    output_root: str = REDUCED_DIR
):
    print(f"[→] 获取不同 slice 的 embedding: {slice1.name}, {slice2.name}")
    print(f"[→] 使用降维器: {reducer_name}, 特征类型: {feature_type}, 模式: {fit_mode}")
    if slice1.name == slice2.name:
        fit_mode = 's1'

    if fit_mode == 'joint':
        # 无序组合标识符（用于 joint 情况）
        slice_names = sorted([slice1.name, slice2.name])
        pair_id = f"{slice_names[0]}_{slice_names[1]}"
        base_dir = os.path.join(output_root, pair_id)
    elif fit_mode == 's1':
        # 仅使用 slice1 的标识符
        base_dir = os.path.join(output_root, slice1.name)
    elif fit_mode == 's2':
        # 仅使用 slice2 的标识符
        base_dir = os.path.join(output_root, slice2.name)

    suffix = '' if n_neighbors == 15 or reducer_name == 'pca' else f'_n{n_neighbors}'

    base_dir = os.path.join(base_dir, reducer_name)
    if feature_type != None:
        base_dir = os.path.join(base_dir, feature_type)
    os.makedirs(base_dir, exist_ok=True)

    model_path = os.path.join(base_dir, f"model{suffix}.joblib")

    emb_dir = os.path.join(base_dir, "embedding")
    os.makedirs(emb_dir, exist_ok=True)

    # === 准备目标 embedding ===
    emb_path1 = os.path.join(emb_dir, f"{slice1.name}{suffix}.npy")
    emb_path2 = os.path.join(emb_dir, f"{slice2.name}{suffix}.npy")

    if os.path.exists(emb_path1) and os.path.exists(emb_path2):
        print(f"[✓] 已存在 slice1 embedding：{emb_path1}")
        print(f"[✓] 已存在 slice2 embedding：{emb_path2}")
        return np.load(emb_path1), np.load(emb_path2)

    # === 准备目标 embedding ===
    reducer_cls = get_reducer(reducer_name)
    assert reducer_cls is not None, f"未知降维器: {reducer_name}"
    reducer = reducer_cls(n_components=2)

    feature_type = feature_type.split('-')
    feat1 = slice1.get_field('feature', feature_type=feature_type)
    feat2 = slice2.get_field('feature', feature_type=feature_type)

    if os.path.exists(model_path):
        print(f"[→] 已存在降维模型：{model_path}")
        reducer = reducer.load(model_path)
    else:
        # === 准备训练数据 ===
        if fit_mode == 'joint':
            fit_feature = np.concatenate([feat1, feat2], axis=0)
        elif fit_mode == 's1':
            fit_feature = feat1
        elif fit_mode == 's2':
            fit_feature = feat2
        else:
            raise ValueError(f"未知 fit_mode: {fit_mode}")
        print('正在训练降维模型')
        reducer.fit(fit_feature)
        reducer.save(model_path)

    # === 准备目标 embedding ===

    if os.path.exists(emb_path1):
        print(f"[✓] 已存在 slice1 embedding：{emb_path1}")
    else:
        print(f"[→] 生成 slice1 embedding: {len(feat1)}")
        embedding1 = reducer.transform(feat1)
        np.save(emb_path1, embedding1)

    if os.path.exists(emb_path2):
        print(f"[✓] 已存在 slice2 embedding：{emb_path2}")
    else:
        print(f"[→] 生成 slice2 embedding: {len(feat2)}")
        embedding2 = reducer.transform(feat2)
        np.save(emb_path2, embedding2)

    return embedding1, embedding2
