"""Evaluation provenance and semantic checks; usable without JAX or a simulator."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from common import ROOT, RUNS


def load_protocol():
    return json.loads((ROOT / "configs/evaluation.json").read_text())


def effective_suite(suite, condition):
    protocol = load_protocol()
    if suite not in protocol["suites"] or condition not in protocol["conditions"]:
        raise ValueError(f"Unsupported suite/condition: {suite}/{condition}")
    return suite + protocol["conditions"][condition]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pro_root():
    return RUNS / "cache/LIBERO-PRO"


def verify_pro_checkout():
    expected = load_protocol()["libero_pro"]["commit"]
    checkout = pro_root()
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"LIBERO-PRO commit mismatch: {actual}, expected {expected}")
    # Runtime results belong outside this checkout; modifications to code/assets change the protocol.
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("LIBERO-PRO tracked files changed; preserve changes and restore the pinned checkout before evaluation")
    return actual


def expected_pro_files(checkout=None):
    """Read the pinned official task registry without importing torch/MuJoCo."""
    checkout = checkout or pro_root()
    path = checkout / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mapping = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets):
            mapping = ast.literal_eval(node.value)
            break
    if mapping is None:
        raise ValueError("Official task registry not found")
    protocol = load_protocol()
    files = []
    for suite, spec in protocol["suites"].items():
        for condition in ("task", "position"):
            name = effective_suite(suite, condition)
            tasks = mapping[name]
            if len(tasks) != spec["tasks"] or len(set(tasks)) != len(tasks):
                raise ValueError(f"Unexpected task coverage: {name}")
            for task in tasks:
                if Path(task).name != task or "/" in task or "\\" in task or task in (".", ".."):
                    raise ValueError(f"Invalid task name: {task}")
                files.extend([f"bddl_files/{name}/{task}.bddl", f"init_files/{name}/{task}.pruned_init"])
    return sorted(files)


def verify_asset_manifest(directory, expected_files, manifest, source):
    if manifest.get("source") != source or set(manifest.get("files", {})) != set(expected_files):
        raise ValueError("OOD download manifest has a different source or incomplete task coverage")
    for relative in expected_files:
        path = Path(directory) / relative
        if not path.is_file() or path.stat().st_size == 0 or sha256(path) != manifest["files"][relative]:
            raise ValueError(f"Missing or changed OOD asset: {relative}; rerun scripts/setup_ood.sh")


def verify_ood_assets():
    verify_pro_checkout()
    directory = RUNS / "data/libero_pro"
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_asset_manifest(directory, expected_pro_files(), manifest, load_protocol()["libero_pro"])
    return manifest


def task_semantics(env):
    """The file name may still name the old target after a task perturbation."""
    instruction = env.language_instruction
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("BDDL environment did not provide a language instruction")
    goals = env.env.parsed_problem["goal_state"]
    pairs = [g for g in goals if len(g) == 3 and g[0].lower() in ("on", "in")]
    if len(pairs) != 1:
        raise ValueError(f"Expected one grasp/place goal, got {goals}")
    return instruction, pairs[0][1], pairs[0][2]
