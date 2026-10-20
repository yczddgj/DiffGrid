# Experiments

The scripts below use synthetic data or prepared dataset slices. Original experiment datasets and pretrained weights must be obtained separately.

## Checks

```bash
python -m unittest discover -s tests -v
cd frontend
npm run build
```

The backend tests use temporary synthetic data, PCA, and a small grid. To check Vulkan separately:

```bash
python -m pip install -r backend/requirements-vulkan.txt
python scripts/check_vulkan.py
```

## LSDD timing

```bash
python backend/lsdd_efficiency_experiment.py --sizes 100 300 --repeats 1 --grid-resolution 12
```

Results are written to `outputs/lsdd_efficiency_results.csv`. The script compares grid-center optimization with an all-sample matrix baseline. The baseline uses a different regularization convention; its MSE is not an equivalence test. Large baseline runs require substantial memory. These demo settings differ from the paper's benchmark settings.

## KDE / LSDD plots

```bash
python experiments/density_difference_comparison.py \
  --embedding embedding.npy --idx1 current_indices.npy --idx2 reference_indices.npy \
  --grid 32 --index-offset 0 --out-dir outputs/comparison
```

The embedding must have shape `(N, 2)`. Index files contain row indices, boolean masks, or global IDs. For global IDs, set `--index-offset` to the embedding's starting ID. Run with `--help` for bandwidth and output options. The plotting script's optional layout regularization is separate from the system's Vulkan implementation.

## Other scripts

| Script | Use |
| --- | --- |
| `backend/infographic_evaluation.py` | Coverage evaluation; the entry point expects `JN-real` and `JN-synthetic` slices |
| `backend/baseline_draw.py` | Scatterplot, contour, and KDE figures; set dataset names in `main()` |
| `backend/datatools/user_study.py` | Generate subsets with specified class ratios |
| `backend/datatools/selection/coreset.py` | DeepCore classification training on prepared subsets |
| `backend/plot_lsdd_efficiency.py` | Plot `lsdd_efficiency_results.csv` from the working directory |

Install `backend/requirements-experiments.txt` for the dataset and training helpers. The web coreset-selection endpoint is unfinished and returns HTTP 501.

## Configuration

Relative paths resolve from the repository root. The backend uses shared in-memory view state; run one process per analysis session.

| Environment variable | Default |
| --- | --- |
| `DIFFGRID_DATA_DIR` | `database` |
| `DIFFGRID_MODEL_DIR` | `backend/models` |
| `DIFFGRID_HOST` | `127.0.0.1` |
| `DIFFGRID_PORT` | `5000` |
| `DIFFGRID_FEATURE_TYPE` | `dreamsim` |
| `DIFFGRID_N_NEIGHBORS` | `100` |
| `DIFFGRID_REGULARIZATION` | `1` |
| `DIFFGRID_GRID_RESOLUTION` | `400` |

The example uses feature key `demo` and a 32 × 32 grid. Set `DIFFGRID_REGULARIZATION=0` to disable layout regularization. For a different backend port, also set `DIFFGRID_PROXY_TARGET` in `frontend/.env.local`; see `frontend/.env.example`.
