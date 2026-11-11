#!/usr/bin/env python3
"""Costmap error around each person, by the person's time since last seen (Test A+B+C, saved predictions, seed mean).

Per (sample, person): mean |pred - target| over the cells within 1.5 m of the person where target or prediction has cost
("person error"), and, for people with an active raised hand, over that person's halt wedge (gesture component cells
within 3 m: "halt error") plus the fraction of wedge cells predicted >= 50 ("halt recall"). Buckets by
dv1/visibility.jsonl time_since_last_seen: visible (0), 0-1, 1-2, 2-5, > 5 s, never seen.

    $PY costmap_staleness_eval.py --label membership_v2 --models typemix_cffullraw typemix_cffullrawmemdecay_geo ... --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import ROOT, components_path, label_path, split_ids  # noqa: E402
from socialcost.dataset import read_jsonl  # noqa: E402

BENCH = "/home/robotics/arena_ws/data/costmap_bench"
BUCKETS = (("visible", 0, 0), ("0-1 s", 0, 1), ("1-2 s", 1, 2), ("2-5 s", 2, 5), ("> 5 s", 5, 1e9))
RES, N, O = 0.05, 200, -5.0


def bucket(t):
    if t is None:
        return "never seen"
    if t == 0:
        return "visible"
    for name, lo, hi in BUCKETS[1:]:
        if lo < t <= hi:
            return name
    return "> 5 s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="membership_v2")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = {}
    for m in a.models:
        runs = sorted(glob.glob(f"{BENCH}/runs/v2/{m}_{a.label}_s[0-9]/predictions.npz"))
        preds[m] = [np.load(r) for r in runs]
        print(m, len(runs), "seeds", flush=True)
    jj, ii = np.mgrid[0:N, 0:N]
    cx, cy = O + (ii + 0.5) * RES, O + (jj + 0.5) * RES
    acc = {m: defaultdict(lambda: defaultdict(float)) for m in a.models}
    eps = sum((split_ids(s) for s in ("test_a", "test_b", "test_c")), [])
    keys0 = set(preds[a.models[0]][0].files)
    for eid in eps:
        d = os.path.join(ROOT, "episodes", eid)
        if not os.path.exists(os.path.join(d, "dv1", "visibility.jsonl")):
            continue
        vis = {r["sample"]: {p["name"]: p for p in r["pedestrians"]} for r in read_jsonl(os.path.join(d, "dv1", "visibility.jsonl"))}
        robots = {r["sample"]: r for r in read_jsonl(os.path.join(d, "robot_states.jsonl"))}
        peds = {p["sample"]: p for p in read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))}
        for k in sorted(robots):
            key = f"{eid}|{k}"
            if key not in keys0 or k not in peds or k not in vis:
                continue
            tgt = np.load(label_path(eid, a.label, k)).astype(np.float32)
            gest = np.load(components_path(eid, a.label, k))["gesture"] > 0
            P = {m: np.mean([z[key].astype(np.float32) for z in preds[m] if key in z.files], 0) for m in a.models}
            pose = robots[k]["pose_map_base"]
            c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
            for p in peds[k]["pedestrians"]:
                name = p["name"].split("/")[-1]
                v = vis[k].get(name)
                if v is None:
                    continue
                dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
                lx, ly = c * dx + s * dy, -s * dx + c * dy
                if abs(lx) > 4.5 or abs(ly) > 4.5:
                    continue
                near = (cx - lx) ** 2 + (cy - ly) ** 2 <= 1.5 ** 2
                b = bucket(v.get("time_since_last_seen"))
                halt = v.get("halt_active") and gest.any()
                wedge = gest & ((cx - lx) ** 2 + (cy - ly) ** 2 <= 3.0 ** 2) if halt else None
                for m in a.models:
                    pr = P[m]
                    reg = near & ((tgt > 0) | (pr > 0))
                    if reg.any():
                        A = acc[m][b]
                        A["err"] += float(np.abs(pr[reg] - tgt[reg]).mean())
                        A["n"] += 1
                    if wedge is not None and wedge.any():
                        A = acc[m][b]
                        A["halt_err"] += float(np.abs(pr[wedge] - tgt[wedge]).mean())
                        A["halt_rec"] += float((pr[wedge] >= 50).mean())
                        A["halt_n"] += 1
    out = {m: {b: {"person_err": v["err"] / v["n"] if v["n"] else None, "persons": int(v["n"]),
                   "halt_err": v["halt_err"] / v["halt_n"] if v["halt_n"] else None, "halt_recall50": v["halt_rec"] / v["halt_n"] if v["halt_n"] else None,
                   "halt_persons": int(v["halt_n"])} for b, v in acc[m].items()} for m in a.models}
    json.dump(out, open(a.out, "w"), indent=1)
    order = [b[0] for b in BUCKETS] + ["never seen"]
    for key in ("person_err", "halt_err", "halt_recall50"):
        print(f"\n{key}: " + " | ".join(order))
        for m in a.models:
            print(f"  {m:32s} " + " | ".join("–" if out[m].get(b, {}).get(key) is None else f"{out[m][b][key]:.2f}" for b in order))
    print("\npersons per bucket:", {b: out[a.models[0]].get(b, {}).get("persons") for b in order}, "halt:", {b: out[a.models[0]].get(b, {}).get("halt_persons") for b in order})
    return 0


if __name__ == "__main__":
    sys.exit(main())
