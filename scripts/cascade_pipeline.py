#!/usr/bin/env python3
"""Resumable cascade + uncertainty pipeline (detached): stage-1 ensembles, fixes for the first benchmark, stage-2 runs.

    source data/costmap_bench/env.sh && $PY cascade_pipeline.py --root data/costmap_bench

 1. Test D re-alignment of the 42 first-benchmark runs (from saved predictions)
 2. an extra seed (3) for rgbd / depth native, whose seed 2 collapsed (seed 2 stays in the main table)
 3. stage 1: interaction predictor ensembles, tracks and tracks + RGB-D crops (5 members, 3 folds)
 4. stage-2 overfit tests, then main runs: pos_u, pos_int_u, cascade_tracks, cascade_vis x {native, membership} x seeds 0-2
 5. seed ensembles (epistemic + aleatoric) of the probabilistic models
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PROB = ["pos_u", "pos_int_u", "cascade_tracks", "cascade_vis"]
BUDGET = 48000


def status(root: str, msg: str) -> None:
    with open(os.path.join(root, "STATUS_cascade.txt"), "a") as f:
        f.write(f"{msg} {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")


def sh(root: str, name: str, cmd: list[str], log: str, done_file: str | None = None) -> bool:
    if done_file and os.path.exists(done_file):
        return True
    status(root, f"BEGIN {name}")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    status(root, f"EXIT {rc} {name}")
    return rc == 0


def train(root: str, out: str, args: list[str]) -> bool:
    return sh(root, os.path.relpath(out, root), [sys.executable, os.path.join(HERE, "bench_train.py"), "--out", out, *args],
              os.path.join(out, "stdout.log"), os.path.join(out, "metrics.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    py = sys.executable
    status(root, "PIPELINE_BEGIN")
    sh(root, "realign_test_d", [py, os.path.join(HERE, "bench_post.py"), "realign", "--root", root], os.path.join(root, "logs", "realign.log"))
    lr = json.load(open(os.path.join(root, "selected_lr.json")))
    for m in ("rgbd", "depth"):
        # seed 2 of these two models collapsed in the first benchmark; it stays in the main table (no cherry-picking) and an
        # extra seed (3) with the identical configuration measures how often the collapse happens
        train(root, os.path.join(root, "runs", "extra_seed", f"{m}_native_s3"),
              ["--model", m, "--label", "native", "--seed", "3", "--lr", str(lr[m]), "--samples", str(BUDGET), "--val-every", str(BUDGET // 8)])
    for var in ("tracks", "vis"):
        out = os.path.join(root, "stage1", var)
        sh(root, f"stage1_{var}", [py, os.path.join(HERE, "cascade_stage1.py"), "--variant", var, "--out", out, "--members", "5", "--folds", "3",
                                   "--steps", "3000", "--workers", "8"], os.path.join(out, "stdout.log"), os.path.join(out, "metrics.json"))
    for m in PROB:
        train(root, os.path.join(root, "runs", "overfit_u", f"{m}_membership"),
              ["--model", m, "--label", "membership", "--seed", "0", "--overfit", "32", "--samples", "6400", "--val-every", "1600"])
    for seed in (0, 1, 2):
        for m in PROB:
            for label in ("native", "membership"):
                train(root, os.path.join(root, "runs", "main", f"{m}_{label}_s{seed}"),
                      ["--model", m, "--label", label, "--seed", str(seed), "--lr", "0.001", "--samples", str(BUDGET), "--val-every", str(BUDGET // 8)])
        status(root, f"SEED_DONE {seed}")
    sh(root, "ensemble", [py, os.path.join(HERE, "bench_post.py"), "ensemble", "--root", root], os.path.join(root, "logs", "ensemble.log"))
    status(root, "PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
