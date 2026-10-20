"""Paired ID / official LIBERO-PRO Task and Position rollouts."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import traceback
import numpy as np
from common import ROOT, RUNS, simulator_setup
from evaluation_protocol import effective_suite, load_protocol, sha256, task_semantics, verify_ood_assets


def rollout_episode(env, initial_state, args, protocol, client, geometry_api):
    from openpi.spatial.geometry import StageSwitch
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(protocol["settle_steps"]):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    instruction, source, placement = task_semantics(env)
    switch = StageSwitch(geometry_api.object_height(env, source))
    client.reset()
    plan = collections.deque()
    for step in range(protocol["suites"][args.suite]["max_steps"]):
        previous = switch.placing
        stage = switch.update(geometry_api.object_height(env, source))
        if switch.placing != previous:
            plan.clear()
        if not plan:
            front, wrist = geometry_api.model_images(obs)
            item = {"observation/image": front, "observation/wrist_image": wrist,
                    "observation/state": geometry_api.model_state(obs), "prompt": instruction}
            if args.variant == "spatial":
                item["spatial_displacement"] = geometry_api.geometry(env, obs, placement if stage else source)["displacement"]
            action_chunk = client.infer(item)["actions"]
            if not (0 < args.replan_steps <= len(action_chunk)):
                raise ValueError("Invalid replan length")
            plan.extend(action_chunk[:args.replan_steps])
        obs, _, _, _ = env.step(plan.popleft().tolist())
        if env.check_success():
            return True, step + 1
    return False, protocol["suites"][args.suite]["max_steps"]


def main():
    protocol = load_protocol()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "spatial"], required=True)
    parser.add_argument("--suite", choices=["libero_object", "libero_spatial"], required=True)
    parser.add_argument("--condition", choices=["id", "task", "position"], default="id")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=protocol["trials"])
    parser.add_argument("--replan-steps", type=int, default=protocol["replan_steps"])
    parser.add_argument("--seed", type=int, default=protocol["seed"])
    parser.add_argument("--max-tasks", type=int, help="Partial simulator acceptance run only")
    parser.add_argument("--run-name", default="paper_seed42", help="Result subdirectory; use a new name for repeated runs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("run-name must be a simple directory name")
    if args.trials < 1 or args.replan_steps < 1 or (args.max_tasks is not None and args.max_tasks < 1):
        raise ValueError("Trial/task/replan counts must be positive")
    backend = "libero" if args.condition == "id" else "libero_pro"
    os.environ["PI05_LIBERO_BACKEND"] = backend
    simulator_setup()
    if backend == "libero_pro":
        verify_ood_assets()
    from libero.libero import benchmark
    import libero.libero as libero_backend
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import sim_geometry
    expected_root = RUNS / "cache" / ("LIBERO" if backend == "libero" else "LIBERO-PRO")
    if not Path(libero_backend.__file__).resolve().is_relative_to(expected_root.resolve()):
        raise RuntimeError("Wrong LIBERO backend imported; run each condition in a separate Python process")
    np.random.seed(args.seed)
    registered_suite = effective_suite(args.suite, args.condition)
    suite = benchmark.get_benchmark_dict()[registered_suite]()
    if suite.n_tasks != protocol["suites"][args.suite]["tasks"]:
        raise ValueError(f"Expected 10 tasks, got {suite.n_tasks} in {registered_suite}")
    task_count = min(args.max_tasks or suite.n_tasks, suite.n_tasks)
    full = task_count == suite.n_tasks and args.trials == protocol["trials"]
    directory = RUNS / "evaluation" / args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{args.variant}_{args.suite}_{args.condition}.jsonl"
    if target.exists() or target.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"Refusing to mix repeated evaluations: {target}; use another --run-name")
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    server = client.get_server_metadata()
    if server.get("reproduction_variant") != args.variant:
        raise ValueError("Policy server variant mismatch/missing metadata; restart scripts/run.py serve with the matching variant")
    protocol_sha = sha256(ROOT / "configs/evaluation.json")
    metadata = {"arguments": vars(args), "protocol": protocol, "protocol_sha256": protocol_sha,
                "registered_suite": registered_suite, "backend": backend, "server": server,
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "scope": "full_condition" if full else "partial_acceptance",
                "ood_manifest_sha256": sha256(RUNS / "data/libero_pro/manifest.json") if backend == "libero_pro" else None}
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = []
    for task_id in range(task_count):
        task = suite.get_task(task_id)
        bddl = Path(suite.get_task_bddl_file_path(task_id))
        from libero.libero import get_libero_path
        init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        identity = {"bddl_sha256": sha256(bddl), "init_states_sha256": sha256(init_file)}
        initial_states = suite.get_task_init_states(task_id)
        env = sim_geometry.make_env(bddl)
        try:
            if args.trials > len(initial_states):
                raise ValueError("Not enough benchmark initial states")
            for trial in range(args.trials):
                np.random.seed(args.seed + trial)
                env.seed(args.seed + trial)
                success = False
                steps = 0
                error = None
                instruction, source, placement = task_semantics(env)
                try:
                    success, steps = rollout_episode(env, initial_states[trial], args, protocol, client, sim_geometry)
                except Exception:
                    error = traceback.format_exc()
                row = {"variant": args.variant, "suite": args.suite, "condition": args.condition, "task_id": task_id,
                       "task": task.name, "trial": trial, "success": success, "error": error,
                       "registered_suite": registered_suite, "instruction": instruction, "source_object": source,
                       "placement_target": placement, "initial_state_index": trial, "steps": steps,
                       "replan_steps": args.replan_steps, "seed": args.seed + trial, **identity}
                result.append(row)
                with target.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"{args.variant} {registered_suite} task={task_id} trial={trial} success={success} error={error is not None}", flush=True)
        finally:
            env.close()
    summary = {"variant": args.variant, "suite": args.suite, "condition": args.condition, "episodes": len(result),
               "successes": sum(row["success"] for row in result), "errors": sum(row["error"] is not None for row in result),
               "success_rate": float(np.mean([row["success"] for row in result])), "scope": metadata["scope"],
               "protocol_sha256": protocol_sha, "seed": args.seed, "replan_steps": args.replan_steps,
               "max_steps": protocol["suites"][args.suite]["max_steps"]}
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise RuntimeError("Rollout errors recorded: inspect JSONL before interpreting success rate")


if __name__ == "__main__":
    main()
