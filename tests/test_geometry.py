import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openpi/src"))
from openpi.spatial.geometry import lift_pixel, project_point, StageSwitch, innermost_pixel
from openpi.spatial.cache import GeometryDataset, metadata_fingerprint


class GeometryTests(unittest.TestCase):
    def test_depth_axes_and_rigid_transform(self):
        k = np.array([[100, 0, 2], [0, 200, 1], [0, 0, 1.]])
        depth = np.full((4, 5), 2.)
        transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.]])
        point = lift_pixel([3, 2], depth, k, transform)
        np.testing.assert_allclose(point, [0.99, 2.02, 5], atol=1e-6)
        np.testing.assert_allclose(project_point(point, k, transform), [3, 2], atol=1e-4)

    def test_invalid_depth_and_pixel_rejected(self):
        for depth, pixel in [(np.zeros((4, 4)), [1, 1]), (np.ones((4, 4)), [8, 0]),
                             (np.full((4, 4), np.nan), [1, 1])]:
            with self.assertRaises(ValueError):
                lift_pixel(pixel, depth, np.eye(3), np.eye(4))

    def test_invalid_extrinsic_rejected(self):
        transform = np.eye(4)
        transform[0, 0] = 2
        with self.assertRaises(ValueError):
            lift_pixel([1, 1], np.ones((4, 4)), np.eye(3), transform)

    def test_object_height_switch_strict_and_latched(self):
        switch = StageSwitch(0.)
        self.assertEqual(switch.update(0.01), 0)
        self.assertEqual(switch.update(0.011), 1)
        self.assertEqual(switch.update(0.), 1)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy unavailable locally; run on simulator host")
    def test_mask_interior(self):
        mask = np.zeros((9, 9), bool)
        mask[2:7, 2:7] = True
        np.testing.assert_array_equal(innermost_pixel(mask), [4, 4])
        with self.assertRaises(ValueError):
            innermost_pixel(np.zeros((9, 9), bool))

    def test_sidecar_alignment_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta").mkdir()
            for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
                (root / "meta" / name).write_text("{}")
            class Dataset:
                def __init__(self):
                    self.root = root
                def __len__(self):
                    return 2
                def __getitem__(self, index):
                    return {"episode_index": 0, "frame_index": index}
            cache = root / "cache.npz"
            np.savez(cache, displacement=np.ones((2, 3)), episode_index=np.zeros(2), frame_index=[0, 1])
            cache.with_suffix(".json").write_text(json.dumps({"repo_id": "local/test",
                "metadata_sha256": metadata_fingerprint(root)}))
            dataset = GeometryDataset(Dataset(), cache, "local/test")
            np.testing.assert_array_equal(dataset[1]["spatial_displacement"], [1, 1, 1])
            dataset.frame[1] = 8
            with self.assertRaises(ValueError):
                dataset[1]
            (root / "meta/info.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                GeometryDataset(Dataset(), cache, "local/test")


if __name__ == "__main__":
    unittest.main()
