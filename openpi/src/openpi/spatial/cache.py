"""Strictly attach geometry to matching LeRobot frames before augmentation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def metadata_fingerprint(dataset_root):
    root = Path(dataset_root) / "meta"
    selected = [root / name for name in ("info.json", "tasks.jsonl", "episodes.jsonl")]
    if not all(path.is_file() for path in selected):
        raise FileNotFoundError("Expected LeRobot v2 meta/info.json, tasks.jsonl, episodes.jsonl")
    digest = hashlib.sha256()
    for path in selected:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class GeometryDataset:
    def __init__(self, dataset, cache_path, repo_id):
        self.dataset = dataset
        path = Path(cache_path)
        manifest = json.loads(path.with_suffix(".json").read_text())
        if manifest["repo_id"] != repo_id:
            raise ValueError("Geometry cache belongs to a different dataset")
        # LeRobot's root is resolved by its own configured cache path.
        if metadata_fingerprint(dataset.root) != manifest["metadata_sha256"]:
            raise ValueError("Dataset metadata changed: rebuild geometry cache")
        with np.load(path, allow_pickle=False) as data:
            self.displacement = data["displacement"].astype(np.float32)
            self.episode = data["episode_index"]
            self.frame = data["frame_index"]
        if self.displacement.shape != (len(dataset), 3) or not np.isfinite(self.displacement).all():
            raise ValueError("Geometry cache must contain exactly one finite 3D displacement per frame")
        if self.episode.shape != (len(dataset),) or self.frame.shape != (len(dataset),):
            raise ValueError("Geometry frame identity arrays have incorrect shape")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        result = dict(self.dataset[index])
        for key, expected in (("episode_index", self.episode[index]), ("frame_index", self.frame[index])):
            if int(result[key]) != int(expected):
                raise ValueError(f"Geometry misalignment at dataset index {index}: {key}")
        result["spatial_displacement"] = self.displacement[index].copy()
        return result
