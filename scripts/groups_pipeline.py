#!/usr/bin/env python3
"""Scheduler for stage 1 v2 (explicit groups, long history, wide crops) and its stage-2 TypeMix runs.

    $PY groups_pipeline.py --root data/costmap_bench --jobs 3

Stage 1: tracks2 and vis2 (8 s history, ensembles + out-of-fold export; vis2 also exports vis2raw = no grouping) and
the history ablation tracks2_h2s (2 s history, metrics only). Stage 2: TypeMix on tracks2 / vis2 / vis2raw, 2 labels x
3 seeds, same budget and protocol as the v1 runs. Then seed ensembles and the report. Restartable: finished jobs
(metrics.json present) are skipped. Status lines: STATUS_groups.txt.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
BUDGET = 48000
STAGE2 = {"typemix_tracks2": "tracks2", "typemix_vis2": "vis2", "typemix_vis2raw": "vis2raw"}
STAGE1_STEPS = 4000


STATUS = "STATUS_groups.txt"
REG3 = ["--dropout", "0.2", "--wd", "0.01", "--augment"]
STAGE2_V3 = {"typemix_tracks3raw": "tracks3raw", "typemix_vis3raw": "vis3raw"}
# counterfactual supervision: small backbone (2 s, d 64, 2 layers, wide crops), with / without the counterfactual losses
CFB = ["--hist", "10", "--d", "64", "--layers", "2", "--dropout", "0.1", "--wd", "0.001"]
STAGE2_CF = {"typemix_cfnoneraw": "cfnoneraw", "typemix_cffullraw": "cffullraw"}
STAGE2_COST = {"typemix_cfcostraw": "cfcostraw"}


def status(root: str, msg: str) -> None:
    with open(os.path.join(root, STATUS), "a") as f:
        f.write(f"{msg} {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")


def jobs(root: str, py: str, workers: int, v3: bool = False, cf: bool = False, cost: bool = False) -> list[dict]:
    J = []
    s1 = os.path.join(HERE, "cascade_stage1_v2.py")
    if cost:  # cost-aware stage 1 through the frozen TypeMix renderer (socialcost/bench/cost_aware.py)
        s1_jobs = (("cfcost", "vis2", CFB + ["--cf", "full", "--cost-weight", "0.1"]),
                   ("cfcost03", "vis2", CFB + ["--cf", "full", "--cost-weight", "0.3", "--no-oof"]),
                   ("costonly", "vis2", CFB + ["--cf", "none", "--cost-weight", "0.1", "--no-oof"]))
        stage2, steps = STAGE2_COST, 3000
    elif cf:
        s1_jobs = (("cffull", "vis2", CFB + ["--cf", "full"]), ("cfnone", "vis2", CFB + ["--cf", "none"]),
                   ("cfnoinv", "vis2", CFB + ["--cf", "noinv", "--no-oof"]), ("cfnorank", "vis2", CFB + ["--cf", "norank", "--no-oof"]),
                   ("cfnohidden", "vis2", CFB + ["--cf", "nohidden", "--no-oof"]), ("cftracks", "tracks2", CFB + ["--cf", "full", "--no-oof"]),
                   ("cftracksnone", "tracks2", CFB + ["--cf", "none", "--no-oof"]))
        stage2, steps = STAGE2_CF, 3000
    elif v3:  # regularised stage 1 (dropout, weight decay, augmentation, 3000 steps); stage 2 on the no-grouping export only
        s1_jobs = (("tracks3", "tracks2", REG3), ("vis3", "vis2", REG3), ("vis3_h2s", "vis2", REG3 + ["--hist", "10", "--no-oof"]))
        stage2, steps = STAGE2_V3, 3000
    else:
        s1_jobs = (("vis2", "vis2", []), ("tracks2", "tracks2", []), ("tracks2_h2s", "tracks2", ["--hist", "10", "--no-oof"]))
        stage2, steps = STAGE2, STAGE1_STEPS
    for name, var, extra in s1_jobs:
        out = os.path.join(root, "stage1", name)
        J.append({"name": f"stage1_{name}", "done": os.path.join(out, "metrics.json"), "needs": None, "log": os.path.join(out, "stdout.log"),
                  "cmd": [py, s1, "--variant", var, "--out", out, "--members", "5", "--folds", "3", "--steps", str(steps),
                          "--workers", str(workers)] + extra})
    for seed in (0, 1, 2):
        for m, need in stage2.items():
            for label in ("native", "membership"):
                out = os.path.join(root, "runs", "main", f"{m}_{label}_s{seed}")
                J.append({"name": os.path.relpath(out, root), "done": os.path.join(out, "metrics.json"), "needs": need, "log": os.path.join(out, "stdout.log"),
                          "cmd": [py, os.path.join(HERE, "bench_train.py"), "--out", out, "--model", m, "--label", label, "--seed", str(seed), "--lr", "0.001",
                                  "--samples", str(BUDGET), "--val-every", str(BUDGET // 8), "--workers", str(workers)]})
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--v3", action="store_true", help="regularised stage 1 v3 (status file STATUS_groups_v3.txt)")
    ap.add_argument("--cf", action="store_true", help="counterfactual supervision (status file STATUS_cf.txt, report counterfactual_stage1_results.md)")
    ap.add_argument("--cost", action="store_true", help="cost-aware stage 1 (status file STATUS_cost.txt)")
    args = ap.parse_args()
    global STATUS
    if args.v3:
        STATUS = "STATUS_groups_v3.txt"
    if args.cf:
        STATUS = "STATUS_cf.txt"
    if args.cost:
        STATUS = "STATUS_cost.txt"
    root = os.path.abspath(args.root)
    py = sys.executable
    status(root, f"GROUPS_BEGIN jobs={args.jobs}")
    todo = [j for j in jobs(root, py, args.workers, args.v3, args.cf, args.cost) if not os.path.exists(j["done"])]
    running: dict = {}
    failed: list[str] = []
    s1_job = {"tracks2": "stage1_tracks2", "vis2": "stage1_vis2", "vis2raw": "stage1_vis2", "tracks3raw": "stage1_tracks3", "vis3raw": "stage1_vis3",
              "cfnoneraw": "stage1_cfnone", "cffullraw": "stage1_cffull", "cfcostraw": "stage1_cfcost"}
    while todo or running:
        for name, (p, j) in list(running.items()):
            if p.poll() is not None:
                status(root, f"EXIT {p.returncode} {name}")
                if p.returncode != 0:
                    failed.append(name)
                del running[name]
        for j in list(todo):
            if len(running) >= args.jobs:
                break
            need = j["needs"]
            if need and s1_job[need] in failed:
                todo.remove(j)
                failed.append(j["name"])
                status(root, f"SKIP {j['name']} ({s1_job[need]} failed)")
                continue
            if need and not os.path.exists(os.path.join(root, "stage1", need, "metrics.json")):
                continue
            os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
            log = open(j["log"], "a")
            log.write(" ".join(j["cmd"]) + "\n")
            log.flush()
            running[j["name"]] = (subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT), j)
            todo.remove(j)
            status(root, f"BEGIN {j['name']}")
        time.sleep(10)
    for name, cmd in (("ensemble", [py, os.path.join(HERE, "bench_post.py"), "ensemble", "--root", root]),
                      ("report", [py, os.path.join(HERE, "cf_report.py" if (args.cf or args.cost) else "groups_report.py"), "--root", root, "--out",
                                  "/home/robotics/arena_ws/data/socialcost/reports"])):
        status(root, f"BEGIN {name}")
        with open(os.path.join(root, "logs", f"groups_{name}.log"), "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        status(root, f"EXIT {rc} {name}")
    status(root, f"GROUPS_DONE failed={len(failed)} {' '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
