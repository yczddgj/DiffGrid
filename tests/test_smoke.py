"""Check a fresh, model-free installation against generated data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class DiffGridSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='diffgrid-test-')
        cls.data = Path(cls.temp.name) / 'example-data'
        os.environ.update({
            'DIFFGRID_DATA_DIR': str(cls.data),
            'DIFFGRID_FEATURE_TYPE': 'demo',
            'DIFFGRID_REGULARIZATION': '0',
            'DIFFGRID_GRID_RESOLUTION': '8',
            'NUMBA_NUM_THREADS': '2',
        })
        # Run outside the repository to check location-independent configuration.
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_demo.py'),
                        '--points', '64'], check=True, cwd=cls.temp.name,
                       capture_output=True, text=True)
        sys.path.insert(0, str(ROOT / 'backend'))
        from app import app, global_view_manager
        cls.app = app
        cls.manager = global_view_manager
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        import config
        config.MODE = 'default'
        self.manager.clear_all_views()

    def assert_success(self, response):
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True)[:500])
        body = response.get_json()
        self.assertEqual(body['code'], 0, body)
        return body.get('data')

    def create_view(self):
        result = self.assert_success(self.client.post('/api/add_view'))
        view_id = result['viewId']
        self.assert_success(self.client.put(f'/api/view/{view_id}', json={
            'currentSlice': 'Demo-current', 'referenceSlice': 'Demo-reference',
            'dimensionReduction': {'method': 'pca', 'fitMode': 'joint'},
            'visualEncoding': {'background': 'Density-Difference', 'scatter': 'Class'},
        }))
        return view_id

    def test_fresh_install_endpoints_and_optional_imports(self):
        self.assertEqual(self.assert_success(self.client.get('/api/slices')),
                         ['Demo-current', 'Demo-reference'])
        self.assertEqual(self.assert_success(self.client.get('/api/train_results')), [])
        self.assertIn('pca', self.assert_success(self.client.get('/api/reduction_methods')))
        for module in ('dreamsim', 'clip', 'modelscope', 'vulkan'):
            self.assertNotIn(module, sys.modules)

    def test_custom_data_root_and_static_images(self):
        from datatools.slice import Slice
        sample = Slice('Demo-current')
        self.assertEqual(len(sample), 64)
        self.assertEqual(np.asarray(sample.get_field('feature', 'demo')).shape, (64, 8))
        image_url = sample.get_image_url(0)
        self.assertTrue(image_url.startswith('/static/datasets/demo/images/'))
        response = self.client.get(image_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/png')
        response.close()
        self.assertEqual(self.client.get('/static/datasets/missing.png').status_code, 404)

    def test_index_reload_discovers_new_blocks(self):
        from datatools.block_manager import BlockManager
        from datatools.global_index import get_global_index
        index = get_global_index()
        before = len(index.global_index)
        block = json.loads((self.data / 'index/blocks/00000000.json').read_text())
        manager = BlockManager(str(self.data / 'index'))
        start = manager.register_block(1, 'Reload test', block)
        index.reload()
        self.assertEqual(len(index.global_index), before + 1)
        self.assertEqual(index.get_info(start)['offset'], 0)

    def test_comparison_map_selection_and_previews(self):
        view_id = self.create_view()
        heatmap = self.assert_success(self.client.get(f'/api/heatmap?view={view_id}'))
        self.assertTrue(heatmap)
        self.assertTrue(all(np.isfinite(cell['value']) for cell in heatmap))
        info = self.assert_success(self.client.get(f'/api/info?view={view_id}'))
        self.assertEqual(info['currentSliceTotal'], 64)
        self.assertEqual(info['referenceSliceTotal'], 64)
        self.assert_success(self.client.post('/api/lasso_select_points', json={
            'view': view_id, 'lassoShape': [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]],
        }))
        self.assertEqual(len(self.manager.get_view(view_id).config.lasso_selection), 64)
        images = self.assert_success(self.client.get(f'/api/lasso_sample_images?view={view_id}'))
        self.assertTrue(images[0][0])

    def test_lsdd_zero_difference_and_sign(self):
        from heatmap_calculator import HeatmapCalculator
        calculator = HeatmapCalculator(grid_resolution=8)
        viewport = {'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1}
        rng = np.random.default_rng(7)
        first = rng.normal([0.2, 0.2], 0.025, (32, 2))
        second = rng.normal([0.8, 0.8], 0.025, (32, 2))
        locations = np.array([[0.2, 0.2], [0.8, 0.8]])
        zero = calculator._get_lsdd_diff_func(first, first, viewport, sigma=0.1)
        np.testing.assert_allclose(zero(locations), 0, atol=1e-6)
        difference = calculator._get_lsdd_diff_func(first, second, viewport, sigma=0.1)
        values = difference(locations)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)


if __name__ == '__main__':
    unittest.main()
