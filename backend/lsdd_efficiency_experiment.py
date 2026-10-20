import time
import numpy as np
from heatmap_calculator import HeatmapCalculator

def generate_embeddings(n_points):
    """生成指定点数的混合高斯分布embedding，权重和均值均为随机"""
    n_components = 3  # 混合分布的高斯分量数

    # 随机生成权重并归一化
    raw_weights = np.random.rand(n_components)
    weights = raw_weights / raw_weights.sum()

    # 随机生成均值，均匀分布在[0, 1]区间
    means = np.random.rand(n_components, 2)

    # 协方差矩阵可以固定或微调为较小的正定矩阵
    covariances = [
        np.array([[0.01, 0], [0, 0.01]]),
        np.array([[0.02, 0], [0, 0.01]]),
        np.array([[0.01, 0], [0, 0.02]])
    ]

    # 按权重分配每个分量的点数
    counts = np.random.multinomial(n_points, weights)
    embeddings = []
    for i in range(n_components):
        samples = np.random.multivariate_normal(means[i], covariances[i], counts[i])
        embeddings.append(samples)
    embeddings = np.vstack(embeddings)
    # 如果有浮点误差导致点数不对，随机采样/截断
    if embeddings.shape[0] > n_points:
        embeddings = embeddings[:n_points]
    elif embeddings.shape[0] < n_points:
        extra = np.random.choice(embeddings.shape[0], n_points - embeddings.shape[0])
        embeddings = np.vstack([embeddings, embeddings[extra]])
    return embeddings

def run_experiment(embedding_sizes, repeats=3, grid_resolution=12):
    """运行LSDD效率对比实验"""
    # 初始化HeatmapCalculator
    calculator = HeatmapCalculator(grid_resolution=grid_resolution, lsdd_lambda=0.01)

    # 视口设置为整个单位正方形
    viewport = {'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1}

    # 采样点数量用于精度评估
    n_sample_points = 100

    results = []

    for size in embedding_sizes:
        print(f"\n===== 测试点数: {size} =====")
        ours_times = []
        baseline_times = []
        mse_values = []  # 存储每次实验的均方误差

        # 运行5次实验取平均
        for i in range(repeats):
            print(f"--- 第 {i+1}/{repeats} 次运行 ---")
            emb1 = generate_embeddings(size)
            emb2 = generate_embeddings(size)

            # 测试ours方法
            start = time.time()
            g_ours = calculator._get_lsdd_diff_func(emb1, emb2, viewport, sigma=0.1, ours=True, sample=False)
            ours_time = time.time() - start
            print(f"ours方法耗时: {ours_time:.4f}秒")
            ours_times.append(ours_time)

            # 测试非ours方法
            try:
                start = time.time()
                g_baseline = calculator._get_lsdd_diff_func(emb1, emb2, viewport, sigma=0.1, ours=False, sample=False)
                baseline_time = time.time() - start
                print(f"非ours方法耗时: {baseline_time:.4f}秒")
                baseline_times.append(baseline_time)

                # 计算均方误差
                sample_points = np.random.rand(n_sample_points, 2)
                ours_values = np.array([g_ours(p) for p in sample_points])
                baseline_values = np.array([g_baseline(p) for p in sample_points])

                mse = np.mean((ours_values - baseline_values) ** 2)
                mse_values.append(mse)
                print(f"均方误差(MSE): {mse:.6f}")

            except Exception as e:
                print(f"非ours方法执行失败: {str(e)}")
                baseline_times.append(float('inf'))
                mse_values.append(float('nan'))

        # 计算平均时间和平均MSE
        avg_ours = sum(ours_times) / len(ours_times)
        avg_baseline = sum(baseline_times) / len(baseline_times) if baseline_times else float('inf')
        avg_mse = np.nanmean(mse_values)  # 忽略NaN值计算平均

        results.append({
            'points': size,
            'avg_ours': avg_ours,
            'avg_baseline': avg_baseline,
            'avg_mse': avg_mse,
            'speedup': avg_baseline / avg_ours if avg_ours > 0 else 0
        })

    return results

def save_results_to_csv(results, filename="lsdd_efficiency_results.csv"):
    """将结果保存为CSV文件"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['points', 'avg_ours_time', 'avg_baseline_time', 'speedup', 'avg_mse']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for res in results:
            writer.writerow({
                'points': res['points'],
                'avg_ours_time': res['avg_ours'],
                'avg_baseline_time': res['avg_baseline'],
                'speedup': res['speedup'],
                'avg_mse': res['avg_mse']
            })
    print(f"结果已保存到 {filename}")

def print_results(results):
    """打印实验结果"""
    print("\n===== 实验结果汇总 =====")
    print("点数\tours平均(s)\t非ours平均(s)\t加速比\t平均MSE")
    for res in results:
        print(f"{res['points']}\t{res['avg_ours']:.4f}\t\t{res['avg_baseline']:.4f}\t\t{res['speedup']:.2f}x\t{res['avg_mse']:.6f}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Synthetic LSDD efficiency experiment")
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 300, 1000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--grid-resolution', type=int, default=12)
    parser.add_argument('--output', type=Path, default=Path('outputs/lsdd_efficiency_results.csv'))
    args = parser.parse_args()
    if args.repeats < 1 or args.grid_resolution < 2 or min(args.sizes) < 1:
        parser.error('sizes/repeats must be positive and grid resolution at least 2')
    np.random.seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = run_experiment(args.sizes, args.repeats, args.grid_resolution)
    print_results(results)
    save_results_to_csv(results, str(args.output))
