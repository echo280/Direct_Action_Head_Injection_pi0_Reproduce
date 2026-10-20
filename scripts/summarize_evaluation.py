"""Build paired ID/Task/Position tables only from complete, error-free evaluations."""
import argparse
import csv
import json
from pathlib import Path
import re

from common import RUNS
from evaluation_protocol import load_protocol


def collect(directory):
    protocol = load_protocol()
    summaries, records, signatures = {}, {}, set()
    checkpoints = {variant: set() for variant in ("baseline", "spatial")}
    for variant in ("baseline", "spatial"):
        for condition in protocol["conditions"]:
            for suite in protocol["suites"]:
                stem = f"{variant}_{suite}_{condition}"
                summary = json.loads((directory / f"{stem}.summary.json").read_text())
                metadata = json.loads((directory / f"{stem}.metadata.json").read_text())
                checkpoints[variant].add(metadata["server"]["reproduction_checkpoint"])
                rows = [json.loads(line) for line in (directory / f"{stem}.jsonl").read_text().splitlines()]
                expected = protocol["suites"][suite]["tasks"] * protocol["trials"]
                if (summary["scope"] != "full_condition" or summary["episodes"] != expected
                        or len(rows) != expected or summary["errors"] or any(row["error"] for row in rows)):
                    raise ValueError(f"Incomplete/errored evaluation: {stem}")
                if sum(bool(row["success"]) for row in rows) != summary["successes"]:
                    raise ValueError(f"Summary does not match episode records: {stem}")
                expected_ids = {(task, trial) for task in range(protocol["suites"][suite]["tasks"])
                                for trial in range(protocol["trials"])}
                if {(row["task_id"], row["trial"]) for row in rows} != expected_ids:
                    raise ValueError(f"Missing or duplicate task/trial identities: {stem}")
                signatures.add((summary["protocol_sha256"], summary["seed"], summary["replan_steps"], metadata["code_commit"]))
                summaries[variant, condition, suite] = summary
                records[variant, condition, suite] = rows
    if len(signatures) != 1:
        raise ValueError("Paired evaluations have different protocols, code commits, seeds or replan lengths")
    if any(len(paths) != 1 for paths in checkpoints.values()):
        raise ValueError("A model variant used different checkpoints across evaluation conditions")
    fields = ("task_id", "task", "trial", "initial_state_index", "bddl_sha256", "init_states_sha256",
              "instruction", "source_object", "placement_target", "seed")
    table = []
    for condition in protocol["conditions"]:
        for suite in protocol["suites"]:
            pair = [records[variant, condition, suite] for variant in ("baseline", "spatial")]
            identities = [[tuple(row[key] for key in fields) for row in rows] for rows in pair]
            if identities[0] != identities[1]:
                raise ValueError(f"Baseline/spatial did not use the same task instructions and states: {condition}/{suite}")
        for variant in ("baseline", "spatial"):
            rates = {suite: 100 * summaries[variant, condition, suite]["successes"] /
                     summaries[variant, condition, suite]["episodes"] for suite in protocol["suites"]}
            table.append({"variant": variant, "condition": condition, "Object": rates["libero_object"],
                          "Spatial": rates["libero_spatial"], "Avg": sum(rates.values()) / len(rates)})
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="paper_seed42")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("Invalid run-name")
    directory = RUNS / "evaluation" / args.run_name
    table = collect(directory)
    (directory / "comparison.json").write_text(json.dumps({"unit": "percent", "results": table}, indent=2) + "\n")
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "condition", "Object", "Spatial", "Avg"])
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
