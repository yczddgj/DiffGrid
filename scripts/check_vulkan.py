#!/usr/bin/env python3
"""Optional check for Vulkan regularization, including result lifetime after cleanup."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from datatools.reducer.runner import _create_regularizer


def main():
    points = np.random.default_rng(42).uniform(-0.8, 0.8, (128, 2)).astype(np.float32)
    with _create_regularizer() as regularizer:
        first = regularizer.regularize(points, kernel_radius=8, iterations=2)
        second = regularizer.regularize(points, kernel_radius=8, iterations=2)
    assert first.shape == points.shape
    assert np.isfinite(first).all() and np.isfinite(second).all()
    # Results must remain valid after a second call and device cleanup.
    np.testing.assert_allclose(first, second, atol=1e-4)
    print('Vulkan regularization and resource cleanup passed.')


if __name__ == '__main__':
    main()
