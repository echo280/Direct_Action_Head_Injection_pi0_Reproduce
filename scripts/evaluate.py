"""ID LIBERO rollout only. OOD evaluation intentionally requires the official PRO adapter."""
import argparse
import collections
import json
import traceback
import numpy as np
from common import ROOT, RUNS, simulator_setup


def main():
    simulator_setup()
    from libero.libero import benchmark
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi.spatial.geometry import StageSwitch
    from sim_geometry import make_env, goal_pair, object_height, geometry, model_images, model_state
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "spatial"], required=True)
    parser.add_argument("--suite", choices=["libero_object", "libero_spatial"], required=True)
    parser.add_argument("--condition", choices=["id", "task", "position"], default="id")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.condition != "id":
        raise NotImplementedError("LIBERO-PRO adapter is not implemented/validated. See docs/evaluation_protocol.md; do not label ID as OOD.")
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    target = RUNS / "evaluation" / f"{args.variant}_{args.suite}_id.jsonl"
    if target.exists():
        raise FileExistsError(f"Refusing to mix repeated evaluations: {target}")
    result = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        env = make_env(RUNS / "cache/LIBERO/libero/libero/bddl_files" / task.problem_folder / task.bddl_file)
        initial_states = suite.get_task_init_states(task_id)
        try:
            if args.trials > len(initial_states):
                raise ValueError("Not enough benchmark initial states")
            for trial in range(args.trials):
                env.seed(args.seed + trial)
                env.reset()
                obs = env.set_init_state(initial_states[trial])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.] * 6 + [-1.])
                source, placement = goal_pair(env)
                switch = StageSwitch(object_height(env, source))
                plan = collections.deque()
                success = False
                error = None
                max_steps = 280 if args.suite == "libero_object" else 220
                try:
                    for step in range(max_steps):
                        # Stage changes discard stale grasp actions before the next replan.
                        previous = switch.placing
                        stage = switch.update(object_height(env, source))
                        if switch.placing != previous:
                            plan.clear()
                        if not plan:
                            front, wrist = model_images(obs)
                            item = {"observation/image": front, "observation/wrist_image": wrist,
                                    "observation/state": model_state(obs), "prompt": task.language}
                            if args.variant == "spatial":
                                item["spatial_displacement"] = geometry(env, obs, placement if stage else source)["displacement"]
                            action_chunk = client.infer(item)["actions"]
                            if not (0 < args.replan_steps <= len(action_chunk)):
                                raise ValueError("Invalid replan length")
                            plan.extend(action_chunk[:args.replan_steps])
                        obs, _, _, _ = env.step(plan.popleft().tolist())
                        success = bool(env.check_success())
                        if success:
                            break
                except Exception:
                    error = traceback.format_exc()
                row = {"variant": args.variant, "suite": args.suite, "condition": "id", "task_id": task_id,
                       "task": task.name, "trial": trial, "success": success, "error": error,
                       "replan_steps": args.replan_steps, "seed": args.seed + trial}
                result.append(row)
                with target.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
        finally:
            env.close()
    summary = {"variant": args.variant, "suite": args.suite, "condition": "id", "episodes": len(result),
               "successes": sum(row["success"] for row in result), "errors": sum(row["error"] is not None for row in result),
               "success_rate": float(np.mean([row["success"] for row in result]))}
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise RuntimeError("Rollout errors recorded: inspect JSONL before interpreting success rate")


if __name__ == "__main__":
    main()
