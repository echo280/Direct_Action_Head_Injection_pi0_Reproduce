"""Fetch the exact official PRO BDDL/state files; never generate fresh perturbations."""
import argparse
import json

from common import RUNS, setup
from evaluation_protocol import expected_pro_files, load_protocol, sha256, verify_ood_assets, verify_pro_checkout


def main():
    setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source = load_protocol()["libero_pro"]
    verify_pro_checkout()
    directory = RUNS / "data/libero_pro"
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        files = expected_pro_files()
        snapshot_download(source["dataset"], repo_type="dataset", revision=source["dataset_revision"],
                          local_dir=directory, allow_patterns=files)
        manifest = {"source": source, "files": {relative: sha256(directory / relative) for relative in files}}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest = verify_ood_assets()
    report = {"status": "assets_verified", "source": source, "files": len(manifest["files"]),
              "manifest_sha256": sha256(directory / "manifest.json"),
              "scope": "download/code integrity only; simulator rollouts not validated"}
    (RUNS / "reports/ood_assets.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
