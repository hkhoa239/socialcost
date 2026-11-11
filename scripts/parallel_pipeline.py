#!/usr/bin/env python3
"""Parallel scheduler for the cascade + TypeMix experiments (replaces the sequential cascade / typemix pipelines).

    $PY parallel_pipeline.py --root data/costmap_bench --jobs 3

Runs up to --jobs bench_train / stage-1 jobs at once, respecting dependencies (cascade_* and typemix_{tracks,vis*} need
their stage-1 predictions). Every job is skipped when its output exists, so the scheduler can be restarted at any time.
Then seed ensembles and reports. Status lines go to STATUS_cascade.txt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from cascade_pipeline import BUDGET, sh, status  # noqa: E402

MODELS = ["pos_u", "pos_int_u", "typemix_gt", "cascade_tracks", "typemix_tracks", "cascade_vis", "typemix_vis", "typemix_vis_nosem"]
NEEDS = {"cascade_tracks": "tracks", "typemix_tracks": "tracks", "cascade_vis": "vis", "typemix_vis": "vis", "typemix_vis_nosem": "vis"}


def jobs(root: str, py: str, workers: int) -> list[dict]:
    J = []
    for m, lr in (("rgbd", 0.0003), ("depth", 0.0003)):
        out = os.path.join(root, "runs", "extra_seed", f"{m}_native_s3")
        J.append({"name": os.path.relpath(out, root), "done": os.path.join(out, "metrics.json"), "needs": None, "log": os.path.join(out, "stdout.log"),
                  "cmd": [py, os.path.join(HERE, "bench_train.py"), "--out", out, "--model", m, "--label", "native", "--seed", "3", "--lr", str(lr),
                          "--samples", str(BUDGET), "--val-every", str(BUDGET // 8), "--workers", str(workers)]})
    for var in ("tracks", "vis"):
        out = os.path.join(root, "stage1", var)
        J.append({"name": f"stage1_{var}", "done": os.path.join(out, "metrics.json"), "needs": None, "log": os.path.join(out, "stdout.log"),
                  "cmd": [py, os.path.join(HERE, "cascade_stage1.py"), "--variant", var, "--out", out, "--members", "5", "--folds", "3", "--steps", "3000",
                          "--workers", str(workers)]})
    for m in MODELS:
        out = os.path.join(root, "runs", "overfit_u", f"{m}_membership")
        J.append({"name": os.path.relpath(out, root), "done": os.path.join(out, "metrics.json"), "needs": NEEDS.get(m), "log": os.path.join(out, "stdout.log"),
                  "cmd": [py, os.path.join(HERE, "bench_train.py"), "--out", out, "--model", m, "--label", "membership", "--seed", "0", "--overfit", "32",
                          "--samples", "6400", "--val-every", "1600", "--workers", str(workers)]})
    for seed in (0, 1, 2):
        for m in MODELS:
            for label in ("native", "membership"):
                out = os.path.join(root, "runs", "main", f"{m}_{label}_s{seed}")
                J.append({"name": os.path.relpath(out, root), "done": os.path.join(out, "metrics.json"), "needs": NEEDS.get(m),
                          "log": os.path.join(out, "stdout.log"),
                          "cmd": [py, os.path.join(HERE, "bench_train.py"), "--out", out, "--model", m, "--label", label, "--seed", str(seed), "--lr", "0.001",
                                  "--samples", str(BUDGET), "--val-every", str(BUDGET // 8), "--workers", str(workers)]})
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6, help="data-loader workers per job")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    py = sys.executable
    status(root, f"PARALLEL_BEGIN jobs={args.jobs}")
    todo = [j for j in jobs(root, py, args.workers) if not os.path.exists(j["done"])]
    running: dict = {}
    failed = []
    while todo or running:
        for name, (p, j) in list(running.items()):
            if p.poll() is not None:
                status(root, f"EXIT {p.returncode} {name}")
                if p.returncode != 0:
                    failed.append(name)
                del running[name]
        stage1_done = {v: os.path.exists(os.path.join(root, "stage1", v, "metrics.json")) for v in ("tracks", "vis")}
        stage1_failed = {v: f"stage1_{v}" in failed for v in ("tracks", "vis")}
        for j in list(todo):
            if len(running) >= args.jobs:
                break
            if j["needs"] and stage1_failed[j["needs"]]:
                todo.remove(j)
                status(root, f"SKIP {j['name']} (stage1_{j['needs']} failed)")
                continue
            if j["needs"] and not stage1_done[j["needs"]]:
                continue
            os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
            log = open(j["log"], "a")
            log.write(" ".join(j["cmd"]) + "\n")
            log.flush()
            running[j["name"]] = (subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT), j)
            todo.remove(j)
            status(root, f"BEGIN {j['name']}")
        time.sleep(10)
    sh(root, "ensemble_all", [py, os.path.join(HERE, "bench_post.py"), "ensemble", "--root", root], os.path.join(root, "logs", "ensemble_all.log"))
    rep = "/home/robotics/arena_ws/data/socialcost/reports"
    sh(root, "report_benchmark", [py, os.path.join(HERE, "bench_report.py"), "--root", root, "--out", rep], os.path.join(root, "logs", "report.log"))
    sh(root, "report_cascade", [py, os.path.join(HERE, "cascade_report.py"), "--root", root, "--out", rep], os.path.join(root, "logs", "report.log"))
    status(root, f"PARALLEL_DONE failed={len(failed)} {' '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
