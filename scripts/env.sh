#!/usr/bin/env bash
# Source this file; all paths derive from the checked-out repository.
PI05_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_ROOT
export HF_HOME="$PI05_ROOT/runs/cache/huggingface"
export HF_LEROBOT_HOME="$PI05_ROOT/runs/data/lerobot"
export OPENPI_DATA_HOME="$PI05_ROOT/runs/weights"
export UV_CACHE_DIR="$PI05_ROOT/runs/cache/uv"
export UV_PROJECT_ENVIRONMENT="$PI05_ROOT/runs/env/openpi"
export UV_LINK_MODE=copy
export WANDB_DIR="$PI05_ROOT/runs/logs/wandb"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$PI05_ROOT/runs"/{logs,reports,evaluation,data,weights,cache,env}
cd -- "$PI05_ROOT"
