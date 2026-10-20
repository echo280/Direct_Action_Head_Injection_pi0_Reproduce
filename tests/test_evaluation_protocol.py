"""Protocol safeguards and task-perturbed inference without a GPU or simulator."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "openpi/src")]
from evaluate import rollout_episode
from evaluation_protocol import effective_suite, expected_pro_files, sha256, task_semantics, verify_asset_manifest
import common
from summarize_evaluation import collect


class EvaluationProtocolTests(unittest.TestCase):
    def test_backends_use_separate_source_config_and_asset_paths(self):
        old_path = sys.path.copy()
        try:
            with tempfile.TemporaryDirectory() as folder, mock.patch.object(common, "RUNS", Path(folder)), mock.patch.dict(common.os.environ):
                root = Path(folder)
                for backend, name in (("libero", "LIBERO"), ("libero_pro", "LIBERO-PRO")):
                    (root / "cache" / name / "libero/libero/envs").mkdir(parents=True)
                    common.os.environ["PI05_LIBERO_BACKEND"] = backend
                    common.simulator_setup()
                    config = Path(common.os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
                    text = config.read_text()
                    self.assertIn(str(root / "cache" / name / "libero/libero/assets").replace("\\", "\\\\"), text)
                    tasks = root / "data/libero_pro" if backend == "libero_pro" else root / "cache/LIBERO/libero/libero"
                    self.assertIn(str(tasks / "bddl_files").replace("\\", "\\\\"), text)
        finally:
            sys.path[:] = old_path

    def test_task_and_position_use_distinct_official_suites(self):
        self.assertEqual(effective_suite("libero_object", "id"), "libero_object")
        self.assertEqual(effective_suite("libero_object", "task"), "libero_object_task")
        self.assertEqual(effective_suite("libero_spatial", "position"), "libero_spatial_swap")
        with self.assertRaises(ValueError):
            effective_suite("libero_goal", "task")

    def test_manifest_rejects_missing_changed_or_wrong_revision_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "task.bddl").write_text("new instruction and goal")
            (root / "task.pruned_init").write_bytes(b"fixed states")
            files = ["task.bddl", "task.pruned_init"]
            source = {"dataset_revision": "fixed-revision"}
            manifest = {"source": source, "files": {name: sha256(root / name) for name in files}}
            verify_asset_manifest(root, files, manifest, source)
            with self.assertRaises(ValueError):
                verify_asset_manifest(root, files, manifest, {"dataset_revision": "different"})
            with self.assertRaises(ValueError):
                verify_asset_manifest(root, files + ["missing.bddl"], manifest, source)
            (root / "task.bddl").write_text("old instruction and goal")
            with self.assertRaises(ValueError):
                verify_asset_manifest(root, files, manifest, source)

    def test_registry_requires_all_four_suites_and_ten_tasks_each(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            target = checkout / "libero/libero/benchmark/libero_suite_task_map.py"
            target.parent.mkdir(parents=True)
            mapping = {suite + suffix: [f"task_{i}" for i in range(10)]
                       for suite in ("libero_object", "libero_spatial") for suffix in ("_task", "_swap")}
            target.write_text("libero_task_map = " + repr(mapping))
            files = expected_pro_files(checkout)
            self.assertEqual(len(files), 80)
            self.assertIn("init_files/libero_spatial_swap/task_9.pruned_init", files)
            mapping["libero_object_task"].pop()
            target.write_text("libero_task_map = " + repr(mapping))
            with self.assertRaises(ValueError):
                expected_pro_files(checkout)

    def make_env(self):
        class Env:
            # Official PRO can keep an alphabet-soup file name while changing the goal to cream cheese.
            language_instruction = "Pick the cream cheese and place it in the basket"
            env = SimpleNamespace(parsed_problem={"goal_state": [["in", "cream_cheese_1", "basket_1_contain_region"]]})
            steps = 0
            def reset(self):
                self.steps = 0
            def set_init_state(self, state):
                return {"state": state}
            def step(self, action):
                self.steps += 1
                return {}, 0, False, {}
            def check_success(self):
                return self.steps == 2
        return Env()

    def test_instruction_and_goal_come_from_bddl(self):
        instruction, source, placement = task_semantics(self.make_env())
        self.assertIn("cream cheese", instruction)
        self.assertEqual(source, "cream_cheese_1")
        self.assertEqual(placement, "basket_1_contain_region")
        env = self.make_env()
        env.env = SimpleNamespace(parsed_problem={"goal_state": []})
        with self.assertRaises(ValueError):
            task_semantics(env)

    def test_rollout_uses_new_instruction_target_and_phase_switch(self):
        self.check_rollout("spatial")

    def test_baseline_receives_same_new_instruction_without_spatial_signal(self):
        self.check_rollout("baseline")

    def check_rollout(self, variant):
        env = self.make_env()
        requests, targets = [], []
        heights = iter([0., 0., 0.02])
        def height(_env, source):
            self.assertEqual(source, "cream_cheese_1")
            return next(heights)
        def geometry(_env, _obs, target):
            targets.append(target)
            return {"displacement": np.ones(3)}
        def infer(item):
            requests.append(item)
            return {"actions": np.zeros((2, 7))}
        api = SimpleNamespace(object_height=height, geometry=geometry,
                              model_images=lambda obs: (0, 0), model_state=lambda obs: np.zeros(8))
        client = SimpleNamespace(reset=lambda: None, infer=infer)
        protocol = {"settle_steps": 0, "suites": {"libero_object": {"max_steps": 3}}}
        args = SimpleNamespace(suite="libero_object", variant=variant, replan_steps=2)
        success, steps = rollout_episode(env, np.zeros(4), args, protocol, client, api)
        self.assertTrue(success)
        self.assertEqual(steps, 2)
        # The second request proves the phase switch discarded the old remaining action.
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(item["prompt"] == env.language_instruction for item in requests))
        self.assertTrue(all(("spatial_displacement" in item) == (variant == "spatial") for item in requests))
        self.assertEqual(targets, ["cream_cheese_1", "basket_1_contain_region"] if variant == "spatial" else [])

    def test_result_table_rejects_partial_error_or_unpaired_inputs(self):
        protocol = {"conditions": {"id": "", "task": "_task", "position": "_swap"},
                    "suites": {"libero_object": {"tasks": 1}, "libero_spatial": {"tasks": 1}}, "trials": 2}
        with tempfile.TemporaryDirectory() as folder, mock.patch("summarize_evaluation.load_protocol", return_value=protocol):
            directory = Path(folder)
            for variant in ("baseline", "spatial"):
                for condition in protocol["conditions"]:
                    for suite in protocol["suites"]:
                        stem = f"{variant}_{suite}_{condition}"
                        rows = [{"task_id": 0, "task": "same", "trial": i, "initial_state_index": i,
                                 "bddl_sha256": "same-bddl", "init_states_sha256": "same-state",
                                 "instruction": "new task", "source_object": "new object", "placement_target": "basket",
                                 "seed": 7 + i, "success": bool(i), "error": None} for i in range(2)]
                        summary = {"scope": "full_condition", "episodes": 2, "errors": 0, "successes": 1,
                                   "protocol_sha256": "fixed", "seed": 7, "replan_steps": 5}
                        (directory / f"{stem}.summary.json").write_text(json.dumps(summary))
                        (directory / f"{stem}.metadata.json").write_text(json.dumps({"code_commit": "same",
                            "server": {"reproduction_checkpoint": variant + "/29999"}}))
                        (directory / f"{stem}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            table = collect(directory)
            self.assertEqual(len(table), 6)
            self.assertTrue(all(row["Avg"] == 50. for row in table))
            target = directory / "spatial_libero_object_task.jsonl"
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]["instruction"] = "old task"
            target.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                collect(directory)
            rows[0]["instruction"] = "new task"
            target.write_text("\n".join(json.dumps(row) for row in rows))
            target = directory / "spatial_libero_object_task.summary.json"
            summary = json.loads(target.read_text())
            for key, value in (("scope", "partial_acceptance"), ("errors", 1)):
                changed = dict(summary, **{key: value})
                target.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    collect(directory)


if __name__ == "__main__":
    unittest.main()
