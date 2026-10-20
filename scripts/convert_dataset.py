"""Convert our replay episodes to LeRobot; sidecar indices are generated in the same pass."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import RUNS, setup


def main():
    setup()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
    from openpi.spatial.cache import metadata_fingerprint
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[20, 69], default=69)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    repo_id = args.repo_id or f"local/pi05_3d_{args.scale}"
    source = RUNS / f"data/replayed_{args.scale}"
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest["task_count"] != args.scale:
        raise ValueError("Partial smoke replay cannot be used as the full training dataset")
    files = sorted(source.glob("*.npz"))
    if not files:
        raise FileNotFoundError(source)
    destination = Path(HF_LEROBOT_HOME) / repo_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {destination}")
    features = {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, robot_type="panda", fps=20, features=features,
                                    image_writer_threads=8, image_writer_processes=0)
    displacements, episodes, frames = [], [], []
    for episode_index, file in enumerate(files):
        with np.load(file, allow_pickle=False) as episode:
            for frame in range(len(episode["actions"])):
                dataset.add_frame({key: episode[key][frame] for key in features} | {"task": str(episode["prompt"])})
                displacements.append(episode["displacement"][frame])
                episodes.append(episode_index)
                frames.append(frame)
        dataset.save_episode()
    cache = RUNS / "data/geometry" / repo_id / "cache.npz"
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, displacement=np.asarray(displacements, dtype=np.float32),
                        episode_index=np.asarray(episodes), frame_index=np.asarray(frames))
    manifest = {"repo_id": repo_id, "metadata_sha256": metadata_fingerprint(destination),
                "frames": len(frames), "episodes": len(files), "task_scale": args.scale,
                "fps": 20, "fps_note": "Implementation choice from LIBERO control_freq=20; paper does not specify data fps",
                "replay_manifest": source_manifest}
    cache.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    (RUNS / "reports/data_conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)
    print(cache)


if __name__ == "__main__":
    main()
