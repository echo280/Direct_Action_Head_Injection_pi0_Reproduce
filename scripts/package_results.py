"""Produce a small return ZIP by default; checkpoints are explicitly optional."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
from common import ROOT, RUNS, setup

setup()
parser = argparse.ArgumentParser()
parser.add_argument("--include-checkpoints", action="store_true")
args = parser.parse_args()
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
archive = RUNS / f"results_{timestamp}.zip"
folders = ["reports", "logs", "evaluation"]
if args.include_checkpoints:
    folders.append("checkpoints")
manifest = []
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
    for folder in folders:
        for path in sorted((RUNS / folder).rglob("*")):
            if path.is_file() and not path.is_symlink():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                relative = path.relative_to(ROOT).as_posix()
                zipped.write(path, relative)
                manifest.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    zipped.writestr("RESULTS_MANIFEST.json", json.dumps({"checkpoints_included": args.include_checkpoints,
                     "files": manifest}, indent=2))
print(archive)
print("Checkpoints included:", args.include_checkpoints)
