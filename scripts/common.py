"""All generated state stays under the package's relative runs/ directory."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OPENPI = ROOT / "openpi"


def setup():
    for name in ("reports", "logs", "checkpoints", "evaluation", "data", "weights", "cache"):
        (RUNS / name).mkdir(parents=True, exist_ok=True)
    # Intentionally override external cache locations for a portable handoff.
    paths = {
        "HF_HOME": RUNS / "cache/huggingface",
        "HF_LEROBOT_HOME": RUNS / "data/lerobot",
        "OPENPI_DATA_HOME": RUNS / "weights",
        "UV_CACHE_DIR": RUNS / "cache/uv",
        "WANDB_DIR": RUNS / "logs/wandb",
        "LIBERO_CONFIG_PATH": RUNS / "cache/libero_config",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path[:0] = [str(OPENPI / "src"), str(OPENPI / "packages/openpi-client/src")]
    return ROOT


def simulator_setup():
    setup()
    import json
    libero = RUNS / "cache/LIBERO"
    if not (libero / "libero/libero/envs").is_dir():
        raise FileNotFoundError("Run bash scripts/setup_simulator.sh first")
    sys.path.insert(0, str(libero))
    base = libero / "libero/libero"
    paths = {"benchmark_root": base, "bddl_files": base / "bddl_files",
             "init_states": base / "init_files", "assets": base / "assets",
             "datasets": RUNS / "data/raw"}
    # JSON string quoting is valid YAML and handles spaces in project paths.
    text = "\n".join(f"{key}: {json.dumps(str(value))}" for key, value in paths.items())
    (Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").write_text(text + "\n")
    os.environ.setdefault("MUJOCO_GL", "egl")


def load_config(name, repo_id):
    setup()
    import dataclasses
    import json
    from openpi.training import config as cfg
    from openpi.training import optimizer
    spec = json.loads((ROOT / f"configs/{name}.json").read_text())
    base = cfg.get_config("pi05_libero")
    model = dataclasses.replace(base.model, spatial_conditioning=spec["spatial_conditioning"])
    result = dataclasses.replace(
        base, name=f"pi05_3d_{name}", model=model,
        data=dataclasses.replace(base.data, repo_id=repo_id),
        batch_size=spec["batch_size"], num_train_steps=spec["num_train_steps"],
        num_workers=spec["num_workers"], seed=spec["seed"],
        optimizer=optimizer.AdamW(weight_decay=1e-10, clip_gradient_norm=1.0),
        assets_base_dir=str(RUNS / "assets"), checkpoint_base_dir=str(RUNS / "checkpoints"),
        project_name="pi05_3d_injection", wandb_enabled=False, overwrite=False,
    )
    cfg._CONFIGS_DICT[result.name] = result
    return result
