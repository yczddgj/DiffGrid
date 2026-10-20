import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datatools.slice import Slice
from datatools.reducer.runner import get_slice_different_embedding
import os
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap, TwoSlopeNorm
import config
import numpy.ma as ma

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """将嵌入归一化到[0,1]范围"""
    min_val = emb.min(axis=0)
    max_val = emb.max(axis=0)
    return (emb - min_val) / (max_val - min_val + 1e-8)

def calculate_common_kde_bandwidth(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算两个嵌入的公共KDE带宽"""
    combined = np.vstack([emb1, emb2])
    kde = gaussian_kde(combined.T)
    return kde.factor  # 返回计算出的带宽因子

def draw_two_scatter_plots(
    emb1_norm: np.ndarray,
    emb2_norm: np.ndarray,
    slice1_name: str,
    slice2_name: str,
    output_path: str = 'scatter_comparison.png'
):
    """直接绘制两个散点图（使用预计算的嵌入）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # 第一个散点图
    ax1.scatter(emb1_norm[:, 0], emb1_norm[:, 1], alpha=0.5, label=slice1_name, color='blue')
    ax1.set_title(f'Scatter Plot of {slice1_name}')
    ax1.legend(loc='upper right')  # 指定图例位置
    ax1.axis('equal')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.margins(0, 0)

    # 第二个散点图
    ax2.scatter(emb2_norm[:, 0], emb2_norm[:, 1], alpha=0.5, label=slice2_name, color='red')
    ax2.set_title(f'Scatter Plot of {slice2_name}')
    ax2.legend(loc='upper right')  # 指定图例位置
    ax2.axis('equal')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.margins(0, 0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def draw_joint_reduction_scatter(
    emb1_norm: np.ndarray,
    emb2_norm: np.ndarray,
    slice1_name: str,
    slice2_name: str,
    output_path: str = 'joint_scatter.png'
):
    """两个slice联合降维的散点图（使用预计算的嵌入）"""
    plt.figure(figsize=(12, 12))
    plt.scatter(emb1_norm[:, 0], emb1_norm[:, 1], alpha=0.5, label=slice1_name, color='blue')
    plt.scatter(emb2_norm[:, 0], emb2_norm[:, 1], alpha=0.5, label=slice2_name, color='red')
    plt.title('Joint Dimensionality Reduction Scatter Plot')
    plt.legend(loc='upper right')  # 指定图例位置
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def _white_ramp(base_color='blue'):
    """
    从白色到指定主色的单向渐变 colormap。
    base_color: 'blue' 或 'red' 或 任何 matplotlib 识别的颜色值
    """
    if base_color == 'blue':
        # 你也可以换成更深/更浅的蓝色
        return LinearSegmentedColormap.from_list('white2blue', ['#FFFFFF', '#2C7FB8'])
    if base_color == 'red':
        return LinearSegmentedColormap.from_list('white2red', ['#FFFFFF', '#D7301F'])
    # 兜底
    return LinearSegmentedColormap.from_list('white2color', ['#FFFFFF', base_color])


def draw_overlay_contours(
    emb1_norm: np.ndarray,
    emb2_norm: np.ndarray,
    bw_method: float,
    slice1_name: str,
    slice2_name: str,
    output_path: str = 'contour_overlay.png',
    # 可调参数
    grid_size: int = 100,
    zero_white_ratio: float = 0.01,   # 小于 max(z)*该比例的密度直接变白
    red_alpha: int = 128,             # 叠加第二幅图时的不透明度(0-255)
):
    """绘制两个contour叠加；密度接近0的地方显示纯白。"""
    # 统一网格
    xx, yy = np.mgrid[0:1:complex(grid_size), 0:1:complex(grid_size)]

    # 统一带宽
    kde1 = gaussian_kde(emb1_norm.T, bw_method=bw_method)
    z1 = kde1(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    kde2 = gaussian_kde(emb2_norm.T, bw_method=bw_method)
    z2 = kde2(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # 计算阈值：密度小于该值的地方显示白色
    thr1 = z1.max() * zero_white_ratio if z1.max() > 0 else 0
    thr2 = z2.max() * zero_white_ratio if z2.max() > 0 else 0

    # 做 mask：小于阈值视为“接近0”
    z1m = ma.masked_less(z1, thr1)
    z2m = ma.masked_less(z2, thr2)

    # 自定义“白->蓝”“白->红”的渐变
    cmap_blue = _white_ramp('blue')
    cmap_red  = _white_ramp('red')
    # 对被 mask 的值使用白色
    cmap_blue.set_bad('#FFFFFF')
    cmap_red.set_bad('#FFFFFF')

    # 分别保存两张透明底图
    temp_dir = os.path.dirname(output_path) or '.'
    temp_file1 = os.path.join(temp_dir, f"temp_{slice1_name}.png")
    temp_file2 = os.path.join(temp_dir, f"temp_{slice2_name}.png")

    plt.figure(figsize=(12, 12))
    plt.contourf(xx, yy, z1m, cmap=cmap_blue, levels=10)
    plt.axis('equal'); plt.xlim(0,1); plt.ylim(0,1); plt.margins(0,0)
    plt.savefig(temp_file1, dpi=300, transparent=True)
    plt.clf()

    plt.figure(figsize=(12, 12))
    plt.contourf(xx, yy, z2m, cmap=cmap_red, levels=10)
    plt.axis('equal'); plt.xlim(0,1); plt.ylim(0,1); plt.margins(0,0)
    plt.savefig(temp_file2, dpi=300, transparent=True)
    plt.clf()

    # 叠加
    from PIL import Image
    img1 = Image.open(temp_file1).convert('RGBA')
    img2 = Image.open(temp_file2).convert('RGBA')

    combined = Image.new('RGBA', img1.size, (0,0,0,0))
    combined = Image.alpha_composite(combined, img1)

    img2_with_alpha = img2.copy()
    img2_with_alpha.putalpha(int(red_alpha))  # 默认 50% 透明
    combined = Image.alpha_composite(combined, img2_with_alpha)

    combined.save(output_path, 'PNG')

    # 清理
    os.remove(temp_file1)
    os.remove(temp_file2)
    plt.close('all')


def draw_density_difference(
    emb1_norm: np.ndarray,
    emb2_norm: np.ndarray,
    bw_method: float,
    slice1_name: str,
    slice2_name: str,
    output_path: str = 'density_difference.png',
    grid_size: int = 500,
    white_band_ratio: float = 0.01,     # 在 |diff| < max_abs*ratio 的范围留下纯白带
    use_symlog: bool = True,            # True: 使用 SymLogNorm；False: 使用 TwoSlopeNorm 线性
    linthresh_ratio: float = 0.01,      # SymLog 线性阈值占 max_abs 的比例
    levels: int = 40
):
    """kde 相减绘制密度差；在 0 附近留一档白色带。"""
    # 统一网格
    xx, yy = np.mgrid[0:1:complex(grid_size), 0:1:complex(grid_size)]

    kde1 = gaussian_kde(emb1_norm.T, bw_method=bw_method)
    z1 = kde1(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    kde2 = gaussian_kde(emb2_norm.T, bw_method=bw_method)
    z2 = kde2(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    diff = z2 - z1
    max_abs = float(np.max(np.abs(diff))) if np.isfinite(diff).any() else 1.0

    # 白色带：把 |diff| < 白带阈值 的区域 mask 掉
    white_thr = max_abs * white_band_ratio
    diff_masked = ma.masked_inside(diff, -white_thr, white_thr)

    # 颜色映射：发散色图 + mask 为白
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#FFFFFF')  # mask 后以白色显示

    # 选择归一化
    if use_symlog:
        # symlog：便于同时看清近零和尾部
        norm = SymLogNorm(
            linthresh=max_abs * linthresh_ratio if max_abs > 0 else 1e-12,
            linscale=0.1,
            vmin=-max_abs, vmax=max_abs
        )
    else:
        # 线性：以 0 为中心
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    plt.figure(figsize=(12, 12))
    contour = plt.contourf(xx, yy, diff_masked, cmap=cmap, levels=levels, norm=norm)
    plt.colorbar(contour, label=f'{slice2_name} - {slice1_name}')
    plt.title(f'Density Difference ({slice2_name} - {slice1_name})')  # 与 diff 一致
    plt.axis('equal'); plt.xlim(0,1); plt.ylim(0,1); plt.margins(0,0)
    plt.savefig(output_path, dpi=300)
    plt.close('all')

def main():
    # 硬编码两个slice名称
    # slice_name1 = 'Infographic-real'
    # slice_name2 = 'Infographic-synthetic'
    config.MODE = 'coreset'

    slice_name1 = 'CIFAR10-train'
    slice_name2 = 'CIFAR10-SES_0.1_2025-03-03 14:31:09.345358'

    slice1 = Slice(slice_name1)
    slice2 = Slice(slice_name2)

    # 创建输出文件夹
    output_dir = 'baseline_plots'
    os.makedirs(output_dir, exist_ok=True)

    # 预计算所有需要的嵌入
    # 使用新的三元组返回值形式
    emb1_joint, emb2_joint, _ = get_slice_different_embedding(
        slice1, slice2,
        reducer_name='umap',
        fit_mode='coreset',
        use_regularization=False
    )

    # 保存emb1和emb2的npy文件
    np.save(os.path.join(output_dir, f"{slice_name1}_embedding.npy"), emb1_joint)
    np.save(os.path.join(output_dir, f"{slice_name2}_embedding.npy"), emb2_joint)

    # if len(emb2_joint) > 1000000:
    #     sample_num = 100000
    #     emb2_joint = emb2_joint[np.random.choice(len(emb2_joint), sample_num, replace=False)]

    # 联合降维的归一化
    combined_joint = np.vstack([emb1_joint, emb2_joint])
    combined_joint_norm = normalize_embedding(combined_joint)
    emb1_joint_norm = combined_joint_norm[:len(emb1_joint)]
    emb2_joint_norm = combined_joint_norm[len(emb1_joint):]

    # 计算公共KDE带宽
    common_bandwidth = calculate_common_kde_bandwidth(emb1_joint_norm, emb2_joint_norm)

    # 绘制四个baseline（使用预计算结果）
    draw_two_scatter_plots(
        emb1_joint_norm, emb2_joint_norm,
        slice_name1, slice_name2,
        os.path.join(output_dir, 'scatter_comparison.png')
    )
    draw_joint_reduction_scatter(
        emb1_joint_norm, emb2_joint_norm,
        slice_name1, slice_name2,
        os.path.join(output_dir, 'joint_scatter.png')
    )
    draw_overlay_contours(
        emb1_joint_norm, emb2_joint_norm, common_bandwidth,
        slice_name1, slice_name2,
        os.path.join(output_dir, 'contour_overlay.png')
    )
    draw_density_difference(
        emb1_joint_norm, emb2_joint_norm, common_bandwidth,
        slice_name1, slice_name2,
        os.path.join(output_dir, 'density_difference.png')
    )

if __name__ == '__main__':
    main()
