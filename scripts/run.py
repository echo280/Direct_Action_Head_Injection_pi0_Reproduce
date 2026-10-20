"""Reuse official JAX training, norm-stat and policy code with our matched configs."""
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import os
from common import ROOT, RUNS, OPENPI, load_config, setup


def main():
    setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["norm", "train", "serve"])
    parser.add_argument("variant", choices=["baseline", "spatial"])
    parser.add_argument("--repo-id", default="local/pi05_3d_69")
    parser.add_argument("--exp-name", default="paper_seed42")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--geometry-cache", type=Path)
    args = parser.parse_args()
    config = load_config(args.variant, args.repo_id)
    changes = dict(exp_name=args.exp_name, fsdp_devices=args.fsdp_devices, resume=args.resume)
    if args.batch_size is not None:
        changes["batch_size"] = args.batch_size
    if args.steps is not None:
        changes["num_train_steps"] = args.steps
    config = dataclasses.replace(config, **changes)
    from openpi.training import config as registry
    registry._CONFIGS_DICT[config.name] = config
    geometry = args.geometry_cache or RUNS / "data/geometry" / args.repo_id / "cache.npz"
    if args.variant == "spatial" and args.mode != "serve":
        if not geometry.is_file():
            raise FileNotFoundError(f"Geometry is required: {geometry}. Run preparation and conversion first.")
        os.environ["PI05_GEOMETRY_CACHE"] = str(geometry.resolve())
    report = vars(args).copy()
    report.update(config_name=config.name, action_horizon=config.model.action_horizon,
                  seed=config.seed, global_batch=config.batch_size, steps=config.num_train_steps,
                  upstream=json.loads((ROOT / "UPSTREAM.json").read_text()))
    path = RUNS / "reports" / f"{args.variant}_{args.exp_name}_{args.mode}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.mode == "serve":
        if not args.checkpoint:
            raise ValueError("--checkpoint must point to a trained step directory")
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        policy = create_trained_policy(config, args.checkpoint)
        WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata).serve_forever()
        return
    script = OPENPI / "scripts" / ("train.py" if args.mode == "train" else "compute_norm_stats.py")
    spec = importlib.util.spec_from_file_location("handoff_official_" + args.mode, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.mode == "train":
        if config.batch_size % args.fsdp_devices:
            raise ValueError("Batch size must be divisible by fsdp_devices")
        import logging
        logging.basicConfig(level=logging.INFO, force=True)
        module.main(config)
    else:
        module.main(config.name)


if __name__ == "__main__":
    main()
