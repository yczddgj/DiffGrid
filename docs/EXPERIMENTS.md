# Experiments and reproducibility

This repository releases the interactive system and supporting research utilities. It does not bundle the paper's original datasets, pretrained weights, extracted features, selected subsets, participant records, or machine-specific caches. The synthetic demo is a software example, not a reproduction of the paper's reported results.

## Installation and smoke tests

The dependency versions are based on the research environment (Python 3.10, PyTorch 2.6, torchvision 0.21). A CPU PyTorch build can run the portable checks:

```bash
python -m unittest discover -s tests -v
cd frontend
npm ci
npm run build
```

The tests create their own temporary data directory and verify index loading and refresh, static sample serving, PCA comparison, finite density maps, selection and previews, zero difference for identical distributions, and the sign of LSDD on separated distributions. They leave the user's data untouched and do not import optional feature/annotation models.

### Full Vulkan layout

With a compatible Vulkan 1.3 device and driver installed:

```bash
python -m pip install -r backend/requirements-vulkan.txt
python scripts/check_vulkan.py
DIFFGRID_FEATURE_TYPE=demo DIFFGRID_GRID_RESOLUTION=32 python backend/app.py
```

Choose DiffGrid and the two demo slices in the UI. This exercises the bundled regularizer and grid interpolation. The portable smoke tests explicitly disable regularization; passing them does not validate Vulkan on another machine. There is no silent fallback if regularization is enabled but Vulkan is unavailable.

## Synthetic LSDD efficiency experiment

```bash
python backend/lsdd_efficiency_experiment.py --sizes 100 300 --repeats 1 --grid-resolution 12
```

This writes `outputs/lsdd_efficiency_results.csv`. It compares the grid-center optimization path and the all-sample matrix baseline in the bundled calculator. Larger sample counts can make the baseline prohibitively expensive in time and memory. Grid resolution, bandwidth, regularization, device, sampling, and random seed all affect the result. The historical baseline uses its own regularization convention; the MSE column is a diagnostic, not a numerical-equivalence guarantee. Timings include the Python calls and are intended as exploratory measurements rather than a replacement for the paper's benchmark protocol.

## KDE / LSDD comparison on your own embedding

An additional script from the associated experiments is included with explicit input paths:

```bash
python experiments/density_difference_comparison.py \
  --embedding embedding.npy --idx1 current_indices.npy --idx2 reference_indices.npy \
  --grid 32 --index-offset 0 --out-dir outputs/comparison
```

The embedding is an `(N, 2)` NumPy array. Index files contain row indices, boolean masks, or global IDs; for global IDs, set `--index-offset` to the embedding's starting global ID. Use `--help` for bandwidth, display transform, optional regularization, and output-array options. The optional regularization in this standalone plotting script is a separate visualization experiment; the system's Vulkan path lives in `datatools/reducer/runner.py`.

## Dataset-dependent research utilities

| Script | Inputs / role |
| --- | --- |
| `backend/infographic_evaluation.py` | Coverage functions for prepared image datasets; its historical entry point expects `JN-real` and `JN-synthetic` slices |
| `backend/baseline_draw.py` | Scatterplot / contour / KDE comparisons using prepared slices; edit the dataset names in `main()` for your own analysis |
| `backend/datatools/user_study.py` | Class-ratio subset preparation helpers; no participant records are included |
| `backend/datatools/selection/coreset.py` | Historical DeepCore-backed classification training utilities for prepared benchmark subsets |

Install `backend/requirements-experiments.txt` for optional dataset/training dependencies. The legacy `plot_lsdd_efficiency.py` expects a CSV in its working directory; the CSV is generated locally, not distributed as a paper result. Training and annotation helpers require additional data/model setup and are not covered by the portable tests. The web coreset-selection endpoint returns HTTP 501 because the original handler was unfinished.

To compare with the paper, acquire the relevant datasets and model weights from their original providers, prepare the same feature spaces and subsets, and use the paper's settings. Keep original data licenses and access restrictions with those local artifacts.
