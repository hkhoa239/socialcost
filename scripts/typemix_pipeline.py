#!/usr/bin/env python3
"""TypeMix phase (runs after cascade_pipeline.py): overfit tests, 24 main runs, seed ensembles, reports.

    $PY typemix_pipeline.py --root data/costmap_bench
Waits for PIPELINE_DONE in STATUS_cascade.txt (stage-1 predictions are needed), then trains
typemix_{tracks,vis,gt,vis_nosem} x {native, membership} x seeds 0-2 with the shared 48k-sample budget.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from cascade_pipeline import BUDGET, sh, status, train  # noqa: E402

MODELS = ["typemix_gt", "typemix_tracks", "typemix_vis", "typemix_vis_nosem"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    st = os.path.join(root, "STATUS_cascade.txt")
    while not (os.path.exists(st) and "PIPELINE_DONE" in open(st).read()):
        time.sleep(60)
    status(root, "TYPEMIX_BEGIN")
    py = sys.executable
    for m in MODELS:
        train(root, os.path.join(root, "runs", "overfit_u", f"{m}_membership"),
              ["--model", m, "--label", "membership", "--seed", "0", "--overfit", "32", "--samples", "6400", "--val-every", "1600"])
    for seed in (0, 1, 2):
        for m in MODELS:
            for label in ("native", "membership"):
                train(root, os.path.join(root, "runs", "main", f"{m}_{label}_s{seed}"),
                      ["--model", m, "--label", label, "--seed", str(seed), "--lr", "0.001", "--samples", str(BUDGET), "--val-every", str(BUDGET // 8)])
        status(root, f"TYPEMIX_SEED_DONE {seed}")
    os.remove(os.path.join(root, "logs", "ensemble.log")) if os.path.exists(os.path.join(root, "logs", "ensemble.log")) else None
    sh(root, "ensemble_all", [py, os.path.join(HERE, "bench_post.py"), "ensemble", "--root", root], os.path.join(root, "logs", "ensemble_all.log"))
    rep = "/home/robotics/arena_ws/data/socialcost/reports"
    sh(root, "report_benchmark", [py, os.path.join(HERE, "bench_report.py"), "--root", root, "--out", rep], os.path.join(root, "logs", "report.log"))
    sh(root, "report_cascade", [py, os.path.join(HERE, "cascade_report.py"), "--root", root, "--out", rep], os.path.join(root, "logs", "report.log"))
    status(root, "TYPEMIX_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
