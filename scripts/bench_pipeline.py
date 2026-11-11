#!/usr/bin/env python3
"""Resumable costmap-benchmark pipeline: overfit tests -> learning-rate selection (validation) -> main runs.

    source data/costmap_bench/env.sh && $PY bench_pipeline.py --root data/costmap_bench [--seeds 0 1 2]

Every step is a bench_train.py run that is skipped when its metrics.json exists. STATUS.txt gets one line per step.
Main runs are ordered seed-major so a complete seed-0 table exists early.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
MODELS = ["pos", "pos_int", "rgb", "depth", "rgbd", "sngnn_sensor", "sngnn_priv"]
LABELS = ["native", "membership"]
LR_GRID = {"rgb": [1e-3, 3e-4], "depth": [1e-3, 3e-4], "rgbd": [1e-3, 3e-4], "pos": [1e-3, 3e-4], "pos_int": [1e-3, 3e-4],
           "sngnn_sensor": [5e-5, 2e-4], "sngnn_priv": [5e-5, 2e-4]}
BUDGET = 48000
BATCH = {"sngnn_sensor": 16, "sngnn_priv": 16}


def status(root: str, msg: str) -> None:
    with open(os.path.join(root, "STATUS.txt"), "a") as f:
        f.write(f"{msg} {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")


def run(root: str, out: str, args: list[str]) -> bool:
    if os.path.exists(os.path.join(out, "metrics.json")):
        return True
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "bench_train.py"), "--out", out, *args]
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
    status(root, f"BEGIN {os.path.relpath(out, root)}")
    with open(os.path.join(out, "stdout.log"), "a") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    status(root, f"EXIT {rc} {os.path.relpath(out, root)}")
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    status(root, "PIPELINE_BEGIN")
    # 1. overfit tests (32 training samples, evaluated on the same samples)
    for m in MODELS:
        steps_samples = 9600 if m not in BATCH else 6400
        run(root, os.path.join(root, "runs", "overfit", f"{m}_membership"),
            ["--model", m, "--label", "membership", "--seed", "0", "--overfit", "32", "--samples", str(steps_samples), "--val-every", "1600",
             *(["--batch", str(BATCH[m])] if m in BATCH else [])])
    # 2. learning-rate selection on validation (seed 0, native label, one third of the budget)
    chosen = {}
    for m in MODELS:
        scores = {}
        for lr in LR_GRID[m]:
            out = os.path.join(root, "runs", "tune", f"{m}_native_lr{lr:g}")
            ok = run(root, out, ["--model", m, "--label", "native", "--seed", "0", "--lr", str(lr), "--tune", "--samples", str(BUDGET // 3),
                                 "--val-every", str(BUDGET // 12), *(["--batch", str(BATCH[m])] if m in BATCH else [])])
            if ok:
                scores[lr] = json.load(open(os.path.join(out, "metrics.json")))["best_val_score"]
        chosen[m] = min(scores, key=scores.get) if scores else LR_GRID[m][0]
        status(root, f"LR_SELECTED {m} {chosen[m]:g} scores {json.dumps({f'{k:g}': round(v, 3) for k, v in scores.items()})}")
    json.dump(chosen, open(os.path.join(root, "selected_lr.json"), "w"), indent=1)
    # 3. main runs, seed-major
    for seed in args.seeds:
        for m in MODELS:
            for label in LABELS:
                run(root, os.path.join(root, "runs", "main", f"{m}_{label}_s{seed}"),
                    ["--model", m, "--label", label, "--seed", str(seed), "--lr", str(chosen[m]), "--samples", str(BUDGET), "--val-every", str(BUDGET // 8),
                     *(["--batch", str(BATCH[m])] if m in BATCH else [])])
        status(root, f"SEED_DONE {seed}")
    status(root, "PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
