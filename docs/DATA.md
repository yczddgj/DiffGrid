# Data preparation

DiffGrid compares **slices**: named arrays of global sample IDs. A **block** maps a contiguous ID range to dataset files. Raw datasets, cached features, models, and generated files stay in the local data directory and are ignored by Git.

## Synthetic example

```bash
python scripts/prepare_demo.py
```

The generator uses seed 42 and never downloads data. It refuses to overwrite an existing demo. For a separate data root, use `--data-dir ./outputs/another-demo` and set the same `DIFFGRID_DATA_DIR` when starting Flask. A relative backend environment path resolves from the repository root.

## File layout

```text
database/
  index/
    global_index.json
    blocks/00000000.json
  datasets/MyDataset/
    images.json
    images/...
    labels.npy
    features.npy
  slices/
    MyDataset-current.npy
    MyDataset-reference.npy
  reduced/                   Generated projections and reducer caches
  results/                   Optional training results
```

`index/global_index.json` is a sorted list of disjoint sample-ID ranges:

```json
[
  {
    "start_id": 0,
    "length": 1000,
    "discription": "MyDataset",
    "block_path": "blocks/00000000.json"
  }
]
```

The historical field name `discription` is retained for compatibility. `block_path` is relative to `index/`. A block file contains:

```json
{
  "dataset": "MyDataset",
  "type": "images",
  "task": "visual-comparison",
  "image_index_path": "MyDataset/images.json",
  "thumbnail_index_path": "MyDataset/images.json",
  "label": {"type": "npy", "path": "MyDataset/labels.npy"},
  "features": {"custom": "MyDataset/features.npy"}
}
```

- Block paths to features, labels, and index files are relative to `datasets/`.
- `images.json` is a JSON list such as `["images/0000.png", "images/0001.png"]`. Each image path is relative to the directory containing this index file. Use portable relative paths inside `datasets/`.
- `features.npy` is a finite numeric matrix of shape `(N, D)`. Samples compared across blocks must share a feature space and dimension. `labels.npy` is a length-`N` integer vector, typically with zero-based contiguous class IDs.
- The block length, image list, labels, and feature rows must match in order. For ready-to-run comparisons, supply at least two nonempty slices.
- A slice is a one-dimensional `int64` NumPy array of global IDs. Slices can overlap and span multiple blocks. A point at row `i` of a block has global ID `start_id + i`.
- Set `DIFFGRID_FEATURE_TYPE=custom` to use the feature key above. Precomputed features avoid all image-model dependencies. For combined feature keys, the existing runner accepts hyphen-separated keys.

Projection caches are reused. When changing feature values or projection parameters for an existing dataset, use a new data directory or remove the corresponding local `reduced/` cache before running again.

## Register prepared files

Put `images.json`, `features.npy`, and `labels.npy` under `database/datasets/MyDataset/`, then run from the repository root:

```bash
PYTHONPATH=backend python - <<'PY'
from pathlib import Path
import json
import numpy as np
from config import DATASET_DIR, SLICE_DIR
from datatools.block_manager import BlockManager

root = Path(DATASET_DIR) / 'MyDataset'
images = json.loads((root / 'images.json').read_text())
features = np.load(root / 'features.npy', allow_pickle=False)
labels = np.load(root / 'labels.npy', allow_pickle=False)
assert features.ndim == 2 and np.isfinite(features).all()
assert len(images) == len(features) == len(labels) and len(images) >= 2
block = {
    'dataset': 'MyDataset', 'type': 'images', 'task': 'visual-comparison',
    'image_index_path': 'MyDataset/images.json',
    'thumbnail_index_path': 'MyDataset/images.json',
    'features': {'custom': 'MyDataset/features.npy'},
    'label': {'type': 'npy', 'path': 'MyDataset/labels.npy'},
}
# Register once. For meaningful analysis, substitute your own current/reference groups.
start = BlockManager().register_block(len(images), 'MyDataset', block)
ids = np.arange(start, start + len(images), dtype=np.int64)
out = Path(SLICE_DIR)
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'MyDataset-current.npy', ids[:len(ids)//2])
np.save(out / 'MyDataset-reference.npy', ids[len(ids)//2:])
PY
```

Start the backend with `DIFFGRID_FEATURE_TYPE=custom`. Keep the server stopped while registering blocks, then restart it or refresh the slice list.

## Images, tabular data, and text

Optional image extractors live in `backend/datatools/feature_extractor/`. ResNet, CLIP, DINO, and DreamSim require their own pretrained weights and preprocessing. Install `requirements-features.txt` for DreamSim/CLIP. For a reproducible analysis, precompute the intended model's features and register those files explicitly; the generic extraction runner does not infer every model's preprocessing from a block.

`datatools/utils.py` also includes CSV adapters for tabular data and precomputed text embeddings. Install `requirements-experiments.txt` for pandas. Text CSVs must supply embeddings; the adapter does not create language-model embeddings. The **Tabular** and **Text** modes use the same block/slice representation and display titles/keywords in place of image previews.

Optional title metadata in a block is:

```json
{
  "title": {
    "indices_path": "MyDataset/title_indices.npy",
    "titles_path": "MyDataset/titles.json",
    "key_word_path": "MyDataset/keywords.json"
  }
}
```

`title_indices.npy` contains block-local row indices; the JSON lists contain the corresponding strings in the same order. `scripts/prepare_demo.py` generates an example of this metadata.

The Qwen annotation helper (`utils_label.py`) is an optional research utility. It additionally needs `modelscope`, `transformers`, `qwen-vl-utils`, and a compatible Qwen2.5-VL checkpoint under `DIFFGRID_MODEL_DIR/qwen2.5`. These large-model annotation workflows are outside the minimal installation and automated tests.
