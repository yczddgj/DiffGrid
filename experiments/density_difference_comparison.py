#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot LSDD and KDE density differences for two subsets of a 2D embedding."""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from typing import Optional

# Use the backend density estimator.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_HERE, '..', 'backend'))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
try:
    from heatmap_calculator import HeatmapCalculator  # type: ignore
except Exception as e:
    HeatmapCalculator = None


def load_indices(path: str, n_total: int, offset: int = 0) -> np.ndarray:
    """Load indices or boolean mask and optionally subtract a global offset."""
    arr = np.load(path)
    if arr.dtype == bool:
        if arr.shape[0] != n_total:
            raise ValueError(
                f"Boolean mask length {arr.shape[0]} != embedding size {n_total}")
        return np.where(arr)[0]
    arr = arr.astype(int)
    if offset:
        arr = arr - int(offset)
    # sanity check
    if np.any(arr < 0) or np.any(arr >= n_total):
        bad_lo = int(np.sum(arr < 0))
        bad_hi = int(np.sum(arr >= n_total))
        raise IndexError(
            f"Indices out of range: {bad_lo} < 0, {bad_hi} >= {n_total}. Check --index-offset.")
    return arr


def symmetric_normalize(x: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return np.zeros_like(x)
    return x / max_abs


def value_transform(x: np.ndarray, mode: str = 'sqrt') -> np.ndarray:
    """Apply per-value transform for visualization.

    Modes:
    - none, sqrt, square, log: legacy diverging transforms keeping sign
    - heatmap: use HeatmapCalculator's symmetric normalize + discretize pipeline, then map to [-1,1]
    """
    if mode == 'none':
        return x
    if mode == 'heatmap':
        if HeatmapCalculator is None:
            raise ImportError("Heatmap discretize mode requires HeatmapCalculator importable")
        # Use backend processing: symmetric normalize with clamp, then discretize to many bins
        calc = HeatmapCalculator()
        vals = calc._symmetric_normalize(x, clamp_value=10)
        disc = calc._discretize_value(vals, num_bins=2100, num_padding=1)  # approx [0,1]
        return 2.0 * disc - 1.0  # map to [-1,1]
    ax = np.abs(x)
    s = np.sign(x)
    if mode == 'sqrt':
        y = np.sqrt(ax)
    elif mode == 'square':
        y = ax ** 2
    elif mode == 'log':
        k = 999.0
        y = np.log1p(k * ax) / np.log1p(k)
    else:
        y = ax
    return s * y


def _bilinear_interpolate_vectorized(points: np.ndarray,
                                     grid_size: float,
                                     grid_values: np.ndarray) -> np.ndarray:
    """Vectorized bilinear interpolation on a regular [0,1]^2 grid.

    points: (N,2) in [0,1]
    grid_size: step between grid nodes
    grid_values: (Gx, Gy, 2) regularized corner positions in [0,1]
    returns: (N,2)
    """
    Gx, Gy, _ = grid_values.shape
    # Clamp to [0,1-eps] to avoid index overflow at the far edge
    eps = 1e-8
    px = np.clip(points[:, 0], 0.0, 1.0 - eps)
    py = np.clip(points[:, 1], 0.0, 1.0 - eps)

    fx = px / grid_size
    fy = py / grid_size
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.clip(x0 + 1, 0, Gx - 1)
    y1 = np.clip(y0 + 1, 0, Gy - 1)

    sx = fx - x0
    sy = fy - y0
    sx = sx[:, None]
    sy = sy[:, None]

    v00 = grid_values[x0, y0]
    v10 = grid_values[x1, y0]
    v01 = grid_values[x0, y1]
    v11 = grid_values[x1, y1]

    top = v00 * (1 - sx) + v10 * sx
    bottom = v01 * (1 - sx) + v11 * sx
    out = top * (1 - sy) + bottom * sy
    return out


def _box_blur(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    """Separable 3x3 box blur applied 'passes' times. Works on 2D arrays."""
    out = arr.copy()
    for _ in range(passes):
        # pad with edge values
        pad = np.pad(out, ((1, 1), (1, 1)), mode='edge')
        # 3x3 average
        out = (
            pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:] +
            pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
        ) / 9.0
    return out


def regularize_embedding_inline(embedding: np.ndarray,
                                grid_size: float = 0.01,
                                iterations: int = 16,
                                step: float = 0.02,
                                density_blur_passes: int = 2,
                                seed: int = 42) -> np.ndarray:
    """Approximate runner.py's regularization pipeline without external imports.

    Steps:
    1) Normalize embedding to [0,1]^2
    2) Build a uniform corner grid of spacing grid_size
    3) Estimate density on a cell grid via histogram2d, smooth with box blur
    4) Compute density gradients; move corner grid opposite to gradient (repulsion)
    5) Bilinear-interpolate new positions for all points

    Returns new embedding in original scale (not re-normalized yet), but here we keep it in [0,1].
    """
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError('embedding must be (N,2)')

    rng = np.random.default_rng(seed)
    # 1) normalize to [0,1]
    min_vals = embedding.min(axis=0)
    max_vals = embedding.max(axis=0)
    span = np.maximum(max_vals - min_vals, 1e-8)
    emb01 = (embedding - min_vals) / span

    # 2) uniform corner grid
    Gx = int(np.floor(1.0 / grid_size)) + 1
    Gy = int(np.floor(1.0 / grid_size)) + 1
    grid_size_real = 1.0 / (Gx - 1)
    xs = np.linspace(0.0, 1.0, Gx)
    ys = np.linspace(0.0, 1.0, Gy)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    corner_positions = np.stack([xv, yv], axis=2)  # (Gx,Gy,2)

    # 3) density on cells (Gx-1 by Gy-1)
    Hx, Hy = Gx - 1, Gy - 1
    H, xedges, yedges = np.histogram2d(emb01[:, 0], emb01[:, 1], bins=[Hx, Hy], range=[[0, 1], [0, 1]])
    H = _box_blur(H, passes=density_blur_passes)

    # upsample density to corner grid by simple padding and averaging
    # create corner density by padding H and averaging neighbors
    H_pad = np.pad(H, ((1, 0), (1, 0)), mode='edge')  # shape (Hx+1, Hy+1) = (Gx, Gy)
    Hd = H_pad.astype(float)

    # 4) compute gradient on corner grid and move corners opposite to gradient
    # central differences with Neumann boundary
    for _ in range(iterations):
        gx = np.zeros_like(Hd)
        gy = np.zeros_like(Hd)
        gx[1:-1, :] = (Hd[2:, :] - Hd[0:-2, :]) * 0.5
        gy[:, 1:-1] = (Hd[:, 2:] - Hd[:, 0:-2]) * 0.5
        # normalize gradient magnitude to avoid huge steps
        gnorm = np.sqrt(gx * gx + gy * gy) + 1e-8
        ux = gx / gnorm
        uy = gy / gnorm
        # move opposite to gradient (repel from high density)
        corner_positions[:, :, 0] = np.clip(corner_positions[:, :, 0] - step * ux, 0.0, 1.0)
        corner_positions[:, :, 1] = np.clip(corner_positions[:, :, 1] - step * uy, 0.0, 1.0)
        # slight diffusion on Hd to emulate multi-pass processing
        Hd = _box_blur(Hd, passes=1)

    # 5) interpolate new point positions
    reg = _bilinear_interpolate_vectorized(emb01, grid_size_real, corner_positions)
    # keep in original scale
    emb_reg = reg * span + min_vals
    return emb_reg


def make_grid(bounds, grid_res: int):
    """Generates meshgrid and centers using the given bounds."""
    (xmin, xmax), (ymin, ymax) = bounds
    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    xv, yv = np.meshgrid(x, y, indexing='ij')  # shape (grid_res, grid_res)
    centers = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (grid_res^2, 2)
    return xv, yv, centers


def get_square_bounds(all_pts: np.ndarray):
    """Calculate the square bounds based on the maximum span of the data."""
    xmin, xmax = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    ymin, ymax = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    x_span = xmax - xmin
    y_span = ymax - ymin
    max_span = max(x_span, y_span)

    # Center the span to make the resulting grid square
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    new_xmin = x_center - max_span / 2
    new_xmax = x_center + max_span / 2
    new_ymin = y_center - max_span / 2
    new_ymax = y_center + max_span / 2

    return ((new_xmin, new_xmax), (new_ymin, new_ymax))


def compute_lsdd(sub1: np.ndarray, sub2: np.ndarray, grid_res: int, sigma: float | None, lam: float):
    """Compute LSDD on a square grid by reusing HeatmapCalculator's implementation.

    This calls HeatmapCalculator._get_lsdd_diff_func to obtain g(x), evaluates it on
    the grid points, and then performs symmetric normalization to [-1, 1].
    """

    if HeatmapCalculator is None:
        raise ImportError("Failed to import HeatmapCalculator from refactor/backend. Check sys.path setup.")

    all_pts = np.vstack([sub1, sub2])
    # 保持与原脚本一致：强制正方形边界
    bounds = get_square_bounds(all_pts)
    xv, yv, centers = make_grid(bounds, grid_res)

    (xmin, xmax), (ymin, ymax) = bounds
    # 若未指定 sigma，交由 HeatmapCalculator 动态估计（与其实现保持一致）
    hc_sigma = None if (sigma is None or sigma == "auto") else float(sigma)

    # 构造 HeatmapCalculator 并设置关键参数
    calc = HeatmapCalculator(
        grid_resolution=grid_res,
        lsdd_sigma=float(hc_sigma) if hc_sigma is not None else 0.01,
        lsdd_lambda=float(lam),
    )

    viewport = {
        'x_min': float(xmin),
        'y_min': float(ymin),
        'x_max': float(xmax),
        'y_max': float(ymax),
    }

    # 获取 LSDD 差异函数并计算网格上的数值
    g_func = calc._get_lsdd_diff_func(sub1, sub2, viewport, sigma=hc_sigma, ours=True, sample=False)
    g_vals = g_func(centers)
    g_vals = np.asarray(g_vals, dtype=float).reshape(grid_res, grid_res)

    # 归一化到 [-1, 1]（仅用于可视化一致性）
    g_grid = symmetric_normalize(g_vals)

    # 返回 HeatmapCalculator 内部实际使用的 sigma，便于日志展示
    used_sigma = float(getattr(calc, 'lsdd_sigma', hc_sigma if hc_sigma is not None else 0.0))
    return xv, yv, g_grid, bounds, used_sigma


def compute_kde_difference(sub1: np.ndarray, sub2: np.ndarray, grid_res: int, bandwidth: float | None):

    all_pts = np.vstack([sub1, sub2])
    # 使用正方形边界
    bounds = get_square_bounds(all_pts)

    xv, yv, centers = make_grid(bounds, grid_res)

    # 自动带宽：基于新的正方形边界的边长
    (xmin, xmax), (ymin, ymax) = bounds
    if bandwidth is None or bandwidth == "auto":
        width = xmax - xmin
        bandwidth = max(width / 50.0, 1e-3)
    bandwidth = float(bandwidth)

    kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde1.fit(sub1)
    kde2.fit(sub2)

    log_d1 = kde1.score_samples(centers)  # log density
    log_d2 = kde2.score_samples(centers)
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)
    diff = d1 - d2
    diff_grid = diff.reshape(grid_res, grid_res)
    diff_grid = symmetric_normalize(diff_grid)
    return xv, yv, diff_grid, bounds, bandwidth


def compute_kde_difference_backend_heatmap(sub1: np.ndarray, sub2: np.ndarray, grid_res: int, bandwidth: float | None):
    """Compute KDE difference using HeatmapCalculator's KDE function only (no heatmap visual mapping).

    We only change the density estimation method to backend's KDE; post-processing remains
    the same as sklearn path: symmetric_normalize to [-1, 1].
    """
    if HeatmapCalculator is None:
        raise ImportError("Failed to import HeatmapCalculator from refactor/backend. Check sys.path setup.")

    all_pts = np.vstack([sub1, sub2])
    bounds = get_square_bounds(all_pts)
    xv, yv, centers = make_grid(bounds, grid_res)

    (xmin, xmax), (ymin, ymax) = bounds
    if bandwidth is None or bandwidth == "auto":
        width = xmax - xmin
        bandwidth = max(width / 50.0, 1e-3)
    bandwidth = float(bandwidth)

    # Construct calculator and compute KDE diff function
    calc = HeatmapCalculator(grid_resolution=grid_res, kde_sigma=float(bandwidth))
    diff_func = calc._get_kde_diff_func(sub1, sub2)
    vals = np.asarray(diff_func(centers), dtype=float)

    # Keep visualization consistent with sklearn path
    diff_grid = vals.reshape(grid_res, grid_res)
    diff_grid = symmetric_normalize(diff_grid)
    return xv, yv, diff_grid, bounds, bandwidth


def plot_comparison(lsdd_grid, kde_grid, bounds, output_path, cmap='RdBu_r',
                    cell_px: int = 10, dpi: int = 200, color_band: float = 0.75,
                    show_grid: bool = False):
    """Plot two grids ensuring each logical cell has identical pixel size.

    cell_px: pixel size per grid cell (square)
    dpi: figure DPI for saving
    color_band: fraction (0<band<=1) we keep from center of diverging colormap (e.g. 0.75 keeps [-0.75,0.75])
    show_grid: optionally overlay faint grid lines
    """
    (xmin, xmax), (ymin, ymax) = bounds
    grid_res = lsdd_grid.shape[0]
    assert lsdd_grid.shape == (grid_res, grid_res)
    # Compute figure pixel dimensions: two panels + colorbar column (~ 0.6 cell width)
    cbar_px = int(cell_px * 0.6)
    fig_width_px = grid_res * cell_px * 2 + cbar_px + 40  # padding
    fig_height_px = grid_res * cell_px + 40
    fig_w_in = fig_width_px / dpi
    fig_h_in = fig_height_px / dpi

    import numpy as _np
    from matplotlib import cm as _cm
    from matplotlib import colors as _mcolors
    from matplotlib import gridspec as _gridspec

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor='white')
    fig.patch.set_facecolor('white')
    gs = _gridspec.GridSpec(1, 3, width_ratios=[grid_res, grid_res, cbar_px / (cell_px * grid_res)],
                            wspace=0.05, left=0.02, right=0.98, top=0.92, bottom=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Truncate colormap central band
    base_cmap = _cm.get_cmap(cmap)
    band = float(color_band)
    band = max(0.05, min(1.0, band))
    # Map desired data band [-band, band] inside full [-1,1]
    minval = ( -band + 1.0 ) / 2.0
    maxval = (  band + 1.0 ) / 2.0
    truncated_colors = base_cmap(_np.linspace(minval, maxval, 256))
    cmap_trunc = _mcolors.LinearSegmentedColormap.from_list(
        f"{cmap}_trunc_{band:.2f}", truncated_colors)

    vmin, vmax = -1.0, 1.0
    im0 = ax0.imshow(lsdd_grid.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
                     cmap=cmap_trunc, vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
    ax0.set_title('LSDD Density Difference')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_facecolor('white')

    im1 = ax1.imshow(kde_grid.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
                     cmap=cmap_trunc, vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
    ax1.set_title('KDE Difference Density')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_facecolor('white')

    if show_grid:
        # Overlay grid lines with alpha
        for ax in (ax0, ax1):
            xs = _np.linspace(xmin, xmax, grid_res + 1)
            ys = _np.linspace(ymin, ymax, grid_res + 1)
            for xg in xs:
                ax.axvline(xg, color='k', linewidth=0.2, alpha=0.2)
            for yg in ys:
                ax.axhline(yg, color='k', linewidth=0.2, alpha=0.2)

    # Colorbar axis
    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label('Normalized Density Difference (-1 ~ 1)')

    fig.suptitle('Density Difference Comparison (Red: subset1 > subset2 / Blue: subset2 > subset1)')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Figure saved: {output_path}  (size: {fig_width_px}x{fig_height_px} px, cell: {cell_px}px)")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize density difference by LSDD and KDE subtraction')
    parser.add_argument('--embedding', required=True,
                        help='2D embedding .npy file')
    parser.add_argument('--idx1', required=True,
                        help='Subset indices .npy file')
    parser.add_argument('--idx2', required=True,
                        help='Subset indices .npy file')
    parser.add_argument('--grid', type=int, default=120,
                        help='Grid resolution (default 120)')
    parser.add_argument('--normalize-embedding', choices=[
                        'none', 'minmax'], default='minmax', help='Coordinate normalization (minmax keeps aspect ratio)')
    parser.add_argument('--kernel-width', default='auto',
                        help='Unified kernel width for BOTH LSDD and KDE (float or auto)')
    parser.add_argument('--lsdd-sigma', default='auto',
                        help='(Optional) LSDD kernel bandwidth (float or auto); ignored if --kernel-width is set')
    parser.add_argument('--kde-bandwidth', default='auto',
                        help='(Optional) KDE bandwidth (float or auto); ignored if --kernel-width is set')
    parser.add_argument('--kde-method', choices=['sklearn', 'backend_heatmap'], default='sklearn',
                        help='KDE implementation to use for the right panel: sklearn (default) or backend_heatmap')
    parser.add_argument('--lambda', dest='lam', type=float,
                        default=0.01, help='LSDD regularization lambda')
    parser.add_argument('--index-offset', type=int, default=0,
                        help='Global index offset to subtract (default 201000 for CIFAR10)')
    parser.add_argument('--discretize-mode', choices=[
                        'none', 'sqrt', 'square', 'log', 'heatmap'], default='sqrt', help='Value transform mode (default sqrt). Use heatmap to apply backend discretization pipeline')
    parser.add_argument('--device', default='auto',
                        help='Device to use (auto|cpu|cuda:<id>). Currently affects torch operations if extended.')
    parser.add_argument('--out-dir', default='outputs/comparison',
                        help='Directory for auto-named outputs')
    parser.add_argument('--output', default=None,
                        help='Explicit output path (overrides auto name)')
    parser.add_argument('--save-npy', action='store_true',
                        help='Save raw (undiscretized) grids as .npy')
    # Regularization options (inline, no imports)
    parser.add_argument('--regularize-embedding', action='store_true',
                        help='Apply inline regularization to the loaded embedding and save a new embedding file')
    parser.add_argument('--reg-grid-size', type=float, default=0.01,
                        help='Regularization grid size in [0,1] space (default 0.01)')
    parser.add_argument('--reg-iterations', type=int, default=16,
                        help='Regularization iterations (default 16)')
    parser.add_argument('--reg-step', type=float, default=0.02,
                        help='Regularization step per iteration (default 0.02)')
    parser.add_argument('--reg-output', default=None,
                        help='Path to save the regularized embedding .npy; defaults to <embedding>_regularized.npy')
    args = parser.parse_args()

    embedding = np.load(args.embedding)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError('embedding must be shape (N, 2)')
    if args.normalize_embedding == 'minmax':
        min_xy = embedding.min(axis=0)
        max_xy = embedding.max(axis=0)
        span_xy = max_xy - min_xy
        max_span = np.max(np.where(span_xy == 0, 1.0, span_xy))
        embedding = (embedding - min_xy) / (max_span if max_span != 0 else 1.0)
        print(
            f"Applied isotropic min-max normalization (keep aspect): x[{min_xy[0]:.4f},{max_xy[0]:.4f}] y[{min_xy[1]:.4f},{max_xy[1]:.4f}] -> largest span scaled to 1")

    # Optional: inline regularization and saving
    if args.regularize_embedding:
        print(f"[*] Regularizing embedding inline (grid={args.reg_grid_size}, iters={args.reg_iterations}, step={args.reg_step}) ...")
        emb_reg = regularize_embedding_inline(
            embedding,
            grid_size=args.reg_grid_size,
            iterations=args.reg_iterations,
            step=args.reg_step,
        )
        # Save new embedding
        if args.reg_output is None:
            base, ext = os.path.splitext(args.embedding)
            reg_path = base + '_regularized.npy'
        else:
            reg_path = args.reg_output
        np.save(reg_path, emb_reg)
        print(f"[✓] Regularized embedding saved: {reg_path}")
        # Use the regularized embedding for downstream computation
        embedding = emb_reg

    idx1 = load_indices(args.idx1, embedding.shape[0], offset=args.index_offset)
    idx2 = load_indices(args.idx2, embedding.shape[0], offset=args.index_offset)

    sub1 = embedding[idx1]
    sub2 = embedding[idx2]

    print(f"subset1 size: {sub1.shape[0]}, subset2 size: {sub2.shape[0]}")
    if args.device != 'auto':
        print(f"Requested device: {args.device}")

    # Unified width precedence: --kernel-width overrides per-method settings
    if args.kernel_width != 'auto':
        kw = float(args.kernel_width)
        lsdd_sigma = kw
        kde_bw = kw
        print(f"[unified] Using kernel width for both methods: {kw}")
    else:
        lsdd_sigma = None if args.lsdd_sigma == 'auto' else float(
            args.lsdd_sigma)
        kde_bw = None if args.kde_bandwidth == 'auto' else float(
            args.kde_bandwidth)

    print('[*] Computing LSDD difference...')
    _, _, lsdd_grid, bounds, used_sigma = compute_lsdd(
        sub1, sub2, args.grid, lsdd_sigma, args.lam)
    print(f'    LSDD sigma = {used_sigma:.5f}')

    print('[*] Computing KDE difference...')
    if args.kde_method == 'backend_heatmap':
        _, _, kde_grid, _, used_bw = compute_kde_difference_backend_heatmap(
            sub1, sub2, args.grid, kde_bw)
        print('    Using backend heatmap-style KDE difference')
    else:
        _, _, kde_grid, _, used_bw = compute_kde_difference(
            sub1, sub2, args.grid, kde_bw)
    print(f'    KDE bandwidth = {used_bw:.5f}')

    # Apply discretize transform for visualization
    lsdd_vis = value_transform(lsdd_grid, args.discretize_mode)
    kde_vis = value_transform(kde_grid, args.discretize_mode)

    # Auto naming if output not provided
    if args.output is None:
        os.makedirs(args.out_dir, exist_ok=True)
        emb_base = os.path.splitext(os.path.basename(args.embedding))[0]
        lam_tag = f"{args.lam:.3f}".rstrip('0').rstrip('.')
        if args.kernel_width != 'auto':
            kw_tag = f"{float(args.kernel_width):.3f}".rstrip('0').rstrip('.')
            fname = f"density_diff_{emb_base}_g{args.grid}_kw{kw_tag}_lam{lam_tag}_kde{args.kde_method}_norm{args.normalize_embedding}_mode{args.discretize_mode}_off{args.index_offset}.png"
        else:
            # use raw user-provided sigma/bw strings for clarity
            sig_tag = ('auto' if args.lsdd_sigma ==
                       'auto' else f"{float(args.lsdd_sigma):.3f}".rstrip('0').rstrip('.'))
            bw_tag = ('auto' if args.kde_bandwidth ==
                      'auto' else f"{float(args.kde_bandwidth):.3f}".rstrip('0').rstrip('.'))
            fname = f"density_diff_{emb_base}_g{args.grid}_sig{sig_tag}_bw{bw_tag}_lam{lam_tag}_kde{args.kde_method}_norm{args.normalize_embedding}_mode{args.discretize_mode}_off{args.index_offset}.png"
        output_path = os.path.join(args.out_dir, fname)
    else:
        output_path = args.output

    plot_comparison(lsdd_vis, kde_vis, bounds, output_path)

    if args.save_npy:
        base, ext = os.path.splitext(output_path)
        np.save(base + '_lsdd.npy', lsdd_grid)
        np.save(base + '_kde.npy', kde_grid)
        print(f"Saved grids: {base}_lsdd.npy, {base}_kde.npy")


if __name__ == '__main__':
    main()
