#!/usr/bin/env python3
"""Create synthetic images, features, and two slices without model downloads."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from config import DATABASE_DIR


def prepare_demo(data_dir: Path, points: int = 300):
    if points < 10:
        raise ValueError('--points must be at least 10 per slice')
    dataset_dir = data_dir / 'datasets' / 'demo'
    slice_dir = data_dir / 'slices'
    if dataset_dir.exists():
        raise FileExistsError(f'{dataset_dir} already exists. Use a new --data-dir to create another demo.')
    for name in ('Demo-current', 'Demo-reference'):
        if (slice_dir / f'{name}.npy').exists():
            raise FileExistsError(f'Slice {name} already exists.')
    (dataset_dir / 'images').mkdir(parents=True)
    slice_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    centers = np.array([[0.2, 0.3], [0.65, 0.25], [0.6, 0.75]])
    reference_labels = rng.choice(3, points, p=[0.5, 0.3, 0.2])
    current_labels = rng.choice(3, points, p=[0.2, 0.3, 0.5])
    labels = np.concatenate([reference_labels, current_labels]).astype(np.int64)
    xy = np.clip(centers[labels] + rng.normal(0, 0.06, (2 * points, 2)), 0.02, 0.98)
    features = np.column_stack([xy, xy ** 2, np.sin(xy * np.pi),
                                rng.normal(0, 0.025, (2 * points, 2))]).astype(np.float32)
    np.save(dataset_dir / 'features.npy', features)
    np.save(dataset_dir / 'labels.npy', labels)

    colors = ['#73a2c6', '#e5aa70', '#8dbb96']
    image_names = []
    for i, (point, label) in enumerate(zip(xy, labels)):
        img = Image.new('RGB', (96, 96), '#f7f7f5')
        draw = ImageDraw.Draw(img)
        x, y = (point * 64 + 16).tolist()
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=colors[label])
        draw.text((5, 5), f'Class {label}', fill='#303030')
        name = f'images/{i:05d}.png'
        img.save(dataset_dir / name)
        image_names.append(name)
    (dataset_dir / 'images.json').write_text(json.dumps(image_names, indent=2) + '\n')
    titles = [f'Synthetic sample {i}, class {int(label)}' for i, label in enumerate(labels)]
    (dataset_dir / 'titles.json').write_text(json.dumps(titles, indent=2) + '\n')
    np.save(dataset_dir / 'title_indices.npy', np.arange(2 * points))

    index_dir = data_dir / 'index'
    (index_dir / 'blocks').mkdir(parents=True, exist_ok=True)
    index_path = index_dir / 'global_index.json'
    entries = json.loads(index_path.read_text()) if index_path.exists() else []
    start_id = max((entry['start_id'] + entry['length'] for entry in entries), default=0)
    block_path = f'blocks/{start_id:08d}.json'
    block = {
        'dataset': 'Demo', 'type': 'synthetic', 'task': 'visual-comparison',
        'image_index_path': 'demo/images.json',
        'thumbnail_index_path': 'demo/images.json',
        'label': {'type': 'npy', 'path': 'demo/labels.npy'},
        'features': {'demo': 'demo/features.npy'},
        'title': {'indices_path': 'demo/title_indices.npy', 'titles_path': 'demo/titles.json',
                  'key_word_path': 'demo/titles.json'},
    }
    (index_dir / block_path).write_text(json.dumps(block, indent=2) + '\n')
    entries.append({'start_id': start_id, 'length': 2 * points,
                    'discription': 'Synthetic demonstration data', 'block_path': block_path})
    index_path.write_text(json.dumps(entries, indent=2) + '\n')
    np.save(slice_dir / 'Demo-reference.npy', np.arange(start_id, start_id + points, dtype=np.int64))
    np.save(slice_dir / 'Demo-current.npy', np.arange(start_id + points, start_id + 2 * points, dtype=np.int64))
    print(f'Created Demo-current and Demo-reference ({points} samples each) in {data_dir}')
    print('Start the backend with DIFFGRID_FEATURE_TYPE=demo. See README.md for Vulkan and CPU options.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATABASE_DIR)
    parser.add_argument('--points', type=int, default=300, help='samples per slice (default: 300)')
    args = parser.parse_args()
    prepare_demo(args.data_dir.expanduser().resolve(), args.points)


if __name__ == '__main__':
    main()
