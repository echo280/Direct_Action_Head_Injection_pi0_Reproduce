"""Download weights or original state-bearing LIBERO data; no training starts here."""
import argparse
import json
from pathlib import Path
import zipfile
from common import RUNS, setup

setup()
parser = argparse.ArgumentParser()
parser.add_argument("what", choices=["weights", "data"])
parser.add_argument("--scale", type=int, choices=[20, 69], default=69)
args = parser.parse_args()
if args.what == "weights":
    from openpi.shared.download import maybe_download
    location = maybe_download("gs://openpi-assets/checkpoints/pi05_base")
    report = {"source": "gs://openpi-assets/checkpoints/pi05_base", "local": str(location)}
else:
    from huggingface_hub import HfApi, snapshot_download
    repo = "yifengzhu-hf/LIBERO-datasets"
    revision = HfApi().dataset_info(repo).sha
    suites = ["libero_object", "libero_spatial"]
    if args.scale == 69:
        suites += ["libero_90", "libero_100"]
    patterns = [pattern for suite in suites for pattern in (f"{suite}/**", f"{suite}.zip")]
    destination = RUNS / "data/raw"
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=destination, allow_patterns=patterns)
    for archive in destination.glob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Archive path escapes data directory")
            zipped.extractall(destination)
    report = {"source": repo, "revision": revision, "scale": args.scale,
              "hdf5_files": len(list(destination.rglob("*.hdf5")))}
    if report["hdf5_files"] == 0:
        raise RuntimeError("No HDF5 files downloaded; inspect the repository layout and docs/data_and_weights.md")
(RUNS / f"reports/download_{args.what}.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
