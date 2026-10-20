"""Re-render state-bearing HDF5 demonstrations into matched RGB/action/geometry episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from common import ROOT, RUNS, simulator_setup


def main():
    simulator_setup()
    import h5py
    import robosuite
    from sim_geometry import make_env, goal_pair, object_height, geometry, model_images, model_state
    from openpi.spatial.geometry import StageSwitch
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[20, 69], default=69)
    parser.add_argument("--max-demos-per-task", type=int)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    tasks = json.loads((ROOT / f"configs/tasks_{args.scale}.json").read_text())
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    destination = RUNS / f"data/replayed_{args.scale}"
    destination.mkdir(exist_ok=True, parents=True)
    bddl_root = RUNS / "cache/LIBERO/libero/libero/bddl_files"
    completed = []
    for task in tasks:
        matches = list((RUNS / "data/raw").rglob(task["name"] + "_demo.hdf5"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Need one state-bearing HDF5 for {task['name']}; found {matches}")
        with h5py.File(matches[0], "r") as raw:
            env = make_env(bddl_root / task["bddl"])
            try:
                demos = sorted(raw["data"], key=lambda x: int(x.split("_")[-1]))
                if args.max_demos_per_task:
                    demos = demos[:args.max_demos_per_task]
                for demo_name in demos:
                    output = destination / f"{task['name']}__{demo_name}.npz"
                    if output.exists():
                        raise FileExistsError(f"Replay outputs already exist: {output}. Use a fresh handoff directory for a new dataset.")
                    demo = raw["data"][demo_name]
                    actions = np.asarray(demo["actions"], dtype=np.float32)
                    states = np.asarray(demo["states"])
                    if len(states) not in (len(actions), len(actions) + 1):
                        raise ValueError("Unexpected state/action alignment")
                    env.reset()
                    xml = demo.attrs.get("model_file")
                    if xml is None:
                        raise ValueError("Missing model_file: cannot faithfully replay geometry")
                    if isinstance(xml, bytes):
                        xml = xml.decode()
                    tree = ET.fromstring(xml)
                    # Relocate only known robosuite/LIBERO asset path components.
                    for node in tree.iter():
                        filename = node.get("file")
                        if not filename:
                            continue
                        parts = filename.replace("\\", "/").split("/")
                        if "robosuite" in parts:
                            index = max(i for i, p in enumerate(parts) if p == "robosuite")
                            node.set("file", str(Path(robosuite.__file__).parent.joinpath(*parts[index + 1:])))
                        elif "assets" in parts:
                            index = parts.index("assets")
                            node.set("file", str(RUNS.joinpath("cache/LIBERO/libero/libero/assets", *parts[index + 1:])))
                    env.reset_from_xml_string(ET.tostring(tree, encoding="unicode"))
                    obs = env.set_init_state(states[0])
                    target, placement = goal_pair(env)
                    switch = StageSwitch(object_height(env, target))
                    arrays = {key: [] for key in ("image", "wrist_image", "state", "displacement", "stage",
                                                  "target_base", "gripper_base", "target_uv", "intrinsics", "camera_to_base")}
                    for frame in range(len(actions)):
                        obs = env.set_init_state(states[frame])
                        stage = switch.update(object_height(env, target))
                        spatial = geometry(env, obs, placement if stage else target)
                        front, wrist = model_images(obs)
                        for key, value in dict(image=front, wrist_image=wrist, state=model_state(obs),
                                               stage=np.int32(stage), **spatial).items():
                            arrays[key].append(value)
                    info = {"task": task, "demo": demo_name, "source": matches[0].name,
                            "source_bytes": matches[0].stat().st_size, "frames": len(actions),
                            "alignment": "render saved states[i] with actions[i]; no no-op filtering",
                            "stage": "latched object height > initial + 0.01m", "target": target, "placement": placement}
                    np.savez_compressed(output, **{key: np.asarray(values) for key, values in arrays.items()},
                                        actions=actions, prompt=np.array(env.language_instruction),
                                        metadata_json=np.array(json.dumps(info)))
                    completed.append(info)
                    print(f"Saved {output.name}: {len(actions)} frames", flush=True)
                    (RUNS / "reports/replay_progress.json").write_text(json.dumps(completed, indent=2))
            finally:
                env.close()
    (destination / "manifest.json").write_text(json.dumps({"scale": args.scale,
        "task_count": len(tasks), "episodes": completed,
        "status": "rendered; RGB/geometry/state alignment requires simulator QA"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
