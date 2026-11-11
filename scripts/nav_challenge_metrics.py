#!/usr/bin/env python3
"""Metrics and report for run_nav_challenge.py (navigation_challenge_v1, one run per configuration x arm).

Per run (simulator time):
  success              goal reached (outcome 2) within the time limit (default 180 s; a probe timeout is a failure)
  time_to_goal_s, path_length_m, detour (path / Route A length), stop_time_s
  min_clearance_m      min robot-pedestrian centre distance;  contact: min clearance < 0.35 m
  personal_zone_time_s time within 1.0 m of any pedestrian
  social_region_time_s time with the robot footprint inside an interaction term of the ground-truth extended cost
                       (conversation band, queue, raised-hand wedge), from socialcost.cost_meter
  social_exposure      integral over the run of the max ground-truth cost under the footprint / 252 (cost x seconds)
  region_exposure      the same integral for the interaction terms only
Aggregates per arm (all configurations and per kind): success rate with a Wilson 95 % interval, means with a t 95 %
interval over configurations, and paired per-configuration differences against bodies-only and against ours (typemix).
Infrastructure failures (status failed: ...) are listed and excluded. Repeats of a configuration x arm are averaged
per configuration for the paired comparisons. "valid" configurations: the ground-truth arm (oracle_ext) reaches the
goal in at least one repeat (a configuration nobody can solve with perfect social cost says nothing about estimators);
aggregates are reported for all and for valid configurations.

    python nav_challenge_metrics.py <challenge output dir> --challenge <navigation_challenge_v1.json> --out <json> --report <md>
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

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from nav_metrics import load_run, run_metrics  # noqa: E402
from nav_metrics_v2 import tci, wilson  # noqa: E402

CONTACT_M = 0.35
ORDER = ["bodies", "rgb", "rgbd", "pos_u", "typemix", "typemix_ucb", "typemix_gt", "oracle_ext",
         "posu_nat", "ours_nat", "gt_nat", "posu_ext", "ours_ext", "oursmem_ext", "gt_ext"]
METER = {"v1": ("cost_max", "inter_max"), "v2n": ("v2n_cost", "v2n_inter"), "v2e": ("v2e_cost", "v2e_inter")}
METER_LABEL = "v1"  # set from --meter-label
NAMES = {"bodies": "no social cost (bodies)", "pos_u": "position-only U-Net", "typemix": "**ours: CF stage 1 + TypeMix**",
         "typemix_ucb": "ours, μ + 1σ", "typemix_gt": "TypeMix + true types (privileged)", "oracle_ext": "ground-truth label (privileged)",
         "rgb": "RGB image U-Net (in view only)", "rgbd": "RGB-D image U-Net (in view only)",
         "posu_nat": "position U-Net (native v2)", "ours_nat": "**ours (native v2)**", "gt_nat": "ground truth native v2 (privileged)",
         "posu_ext": "position U-Net (extended v2)", "ours_ext": "**ours (extended v2)**", "oursmem_ext": "**ours + memory (extended v2)**",
         "gt_ext": "ground truth extended v2 (privileged)"}


def meter_metrics(d: str, ep: dict, t_end_ns: int) -> dict:
    p = os.path.join(d, "meter.jsonl")
    if not os.path.exists(p):
        return {}
    rows = [json.loads(x) for x in open(p) if x.strip()]
    rows = [r for r in rows if ep["start_ns"] <= r["t_ns"] <= t_end_ns]
    if len(rows) < 2:
        return {}
    t = np.array([r["t_ns"] for r in rows]) / 1e9
    dt = np.diff(t, append=t[-1])
    dt = np.clip(dt, 0, 0.5)
    fc, fi = METER[METER_LABEL]
    rows = [r for r in rows if fc in r]
    if len(rows) < 2:
        return {}
    t = np.array([r["t_ns"] for r in rows]) / 1e9
    dt = np.clip(np.diff(t, append=t[-1]), 0, 0.5)
    c = np.array([r[fc] for r in rows]) / 252.0
    ii = np.array([r[fi] for r in rows]) / 252.0
    return {"social_region_time_s": float(dt[ii > 0].sum()), "social_exposure": float((c * dt).sum()), "region_exposure": float((ii * dt).sum()),
            "region_peak": float(ii.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--challenge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--meter-label", default="v1", choices=sorted(METER), help="ground truth for exposure: v1 extended/2.0, v2 native, v2 extended")
    ap.add_argument("--oracle", default="oracle_ext", help="ground-truth arm name (defines valid configurations)")
    ap.add_argument("--methods", nargs="*", default=None, help="only these arms")
    args = ap.parse_args()
    global METER_LABEL
    METER_LABEL = args.meter_label
    doc = json.load(open(args.challenge))
    conf = {c["config_id"]: c for c in doc["configs"]}
    runs, failed = [], []
    for d in sorted(glob.glob(os.path.join(args.root, "*__*__*"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        meta = json.load(open(rj))
        if args.methods and meta.get("method") not in args.methods:
            continue
        if meta["status"].startswith("failed"):
            failed.append(meta)
            continue
        c = conf[meta["config_id"]]
        r = load_run(d)
        if r is None:
            m = {"success": 0.0, "outcome": "timeout"}
        else:
            ep = r["ep"]
            m = run_metrics(r, None, set())
            m["outcome"] = ep.get("outcome")
            t_end = ep["robot"][-1][0] if ep["robot"] else ep["start_ns"]
            if m["success"] and m["time_to_goal_s"] is not None and m["time_to_goal_s"] > args.time_limit:
                m["success"] = False
                m["outcome"] = "over_time_limit"
            m["success"] = float(bool(m["success"]))
            m["contact"] = m["min_clearance_m"] is not None and m["min_clearance_m"] < CONTACT_M
            m["detour"] = m["path_length_m"] / c["alternative_routes"]["route_a_m"] if m["success"] else None
            m.update(meter_metrics(d, ep, t_end))
        runs.append({**meta, **m})
    by = defaultdict(list)
    for r in runs:
        by[r["method"]].append(r)
    keys = ("time_to_goal_s", "path_length_m", "detour", "min_clearance_m", "personal_zone_time_s", "social_region_time_s", "social_exposure",
            "region_exposure", "stop_time_s")

    def agg(rs):
        out = {"n": len(rs), "success": wilson(sum(bool(r["success"]) for r in rs), len(rs)),
               "contact": wilson(sum(bool(r.get("contact")) for r in rs), len(rs))}
        for k in keys:
            out[k] = tci([r.get(k) for r in rs if r.get(k) is not None])
        return out

    valid = sorted({r["config_id"] for r in runs if r["method"] == args.oracle and r["success"]})
    res = {"valid_configs": valid,
           "per_method": {m: agg(rs) for m, rs in by.items()},
           "per_method_valid": {m: agg([r for r in rs if r["config_id"] in valid]) for m, rs in by.items()},
           "per_kind": {m: {k: agg([r for r in rs if r["kind"] == k]) for k in sorted({r["kind"] for r in rs})} for m, rs in by.items()}}
    # paired differences per configuration (repeats averaged), valid configurations
    idx = defaultdict(list)
    for r in runs:
        idx[(r["config_id"], r["method"])].append(r)

    def cm(c, m, k):
        v = [r.get(k) for r in idx.get((c, m), []) if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    paired = {}
    base_list = [b for b in ("bodies", "typemix") if b in by]
    for base in base_list:
        for m in by:
            if m == base:
                continue
            for k in ("social_region_time_s", "region_exposure", "social_exposure", "path_length_m", "personal_zone_time_s", "success"):
                d = [cm(c, m, k) - cm(c, base, k) for c in valid if cm(c, m, k) is not None and cm(c, base, k) is not None]
                paired.setdefault(f"{m}-{base}", {})[k] = tci(d)
            s = [(cm(c, m, "success"), cm(c, base, "success")) for c in valid if cm(c, m, "success") is not None and cm(c, base, "success") is not None]
            paired[f"{m}-{base}"]["success_wins_losses"] = [sum(a > b for a, b in s), sum(b > a for a, b in s), len(s)]
    res["paired"] = paired
    res["runs"] = runs
    res["infrastructure_failures"] = failed
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    if args.report:
        write_report(res, args.report, args.time_limit)
    print(json.dumps({m: {"n": v["n"], "success": v["success"]["rate"], "region_s": v["social_region_time_s"]["mean"]} for m, v in res["per_method"].items()}))
    return 0


def f(x, p=2):
    return "–" if x is None else f"{x:.{p}f}"


def ci_s(t, p=2):
    if not t or t.get("mean") is None:
        return "–"
    s = f(t["mean"], p)
    if t.get("ci95_t"):
        s += f" [{f(t['ci95_t'][0], p)}, {f(t['ci95_t'][1], p)}]"
    return s


def write_report(res: dict, path: str, limit: float) -> None:
    pm = res["per_method"]
    meths = [m for m in ORDER if m in pm] + [m for m in pm if m not in ORDER]
    L = ["# Closed-loop navigation: navigation_challenge_v1", "",
         "36 fixed configurations (conversation 8, raised hand 6, queue 8, group walking 6, occluded / mixed 8) in which the",
         "shortest path crosses a social region and a 2-60 % longer route avoids it. Same Nav2 stack (DWB, social layer = mask",
         "overlay of the estimate) in every arm; only the published social cost differs. Pedestrians replay identically",
         "across arms (same scenario and run seed). Social region / exposure use the ground-truth extended cost under the",
         f"robot footprint (socialcost.cost_meter). Success = goal within {limit:.0f} s of simulator time.", "",
         "| Arm | n | success | contact (<0.35 m) | time in social region (s) | region exposure | personal-zone time (s) | path length (m) | time to goal (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in meths:
        a = pm[m]
        L.append(f"| {NAMES.get(m, m)} | {a['n']} | {f(a['success']['rate'])} | {f(a['contact']['rate'])} | {ci_s(a['social_region_time_s'], 1)} | "
                 f"{ci_s(a['region_exposure'])} | {ci_s(a['personal_zone_time_s'], 1)} | {ci_s(a['path_length_m'], 1)} | {ci_s(a['time_to_goal_s'], 1)} |")
    L += ["", f"### Valid configurations only ({len(res['valid_configs'])}: ground-truth arm reaches the goal)", "",
          "| Arm | n | success | contact | time in social region (s) | region exposure | personal-zone time (s) | path length (m) | time to goal (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in meths:
        a = res["per_method_valid"][m]
        L.append(f"| {NAMES.get(m, m)} | {a['n']} | {f(a['success']['rate'])} | {f(a['contact']['rate'])} | {ci_s(a['social_region_time_s'], 1)} | "
                 f"{ci_s(a['region_exposure'])} | {ci_s(a['personal_zone_time_s'], 1)} | {ci_s(a['path_length_m'], 1)} | {ci_s(a['time_to_goal_s'], 1)} |")
    L += ["", "Means with t 95 % intervals over runs; success / contact are rates.", "",
          "## Time in the social region per kind (s)", "", "| Arm | " + " | ".join(sorted(next(iter(res["per_kind"].values())).keys())) + " |",
          "|---|" + "---|" * len(next(iter(res["per_kind"].values())))]
    kinds = sorted(next(iter(res["per_kind"].values())).keys())
    for m in meths:
        L.append(f"| {NAMES.get(m, m)} | " + " | ".join(ci_s(res["per_kind"][m].get(k, {}).get("social_region_time_s"), 1) for k in kinds) + " |")
    L += ["", "## Success per kind", "", "| Arm | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for m in meths:
        L.append(f"| {NAMES.get(m, m)} | " + " | ".join(f(res["per_kind"][m].get(k, {}).get("success", {}).get("rate")) for k in kinds) + " |")
    L += ["", "## Paired differences per valid configuration (arm - reference, repeats averaged)", "",
          "| Comparison | Δ success | Δ region time (s) | Δ region exposure | Δ path length (m) | success wins / losses / pairs |", "|---|---|---|---|---|---|"]
    for k, v in res["paired"].items():
        L.append(f"| {k} | {ci_s(v['success'])} | {ci_s(v['social_region_time_s'], 1)} | {ci_s(v['region_exposure'])} | {ci_s(v['path_length_m'], 1)} | "
                 f"{v['success_wins_losses'][0]} / {v['success_wins_losses'][1]} / {v['success_wins_losses'][2]} |")
    if res["infrastructure_failures"]:
        L += ["", f"Infrastructure failures excluded: {len(res['infrastructure_failures'])} runs: " +
              ", ".join(f"{r['tag']} ({r['status'][:60]})" for r in res["infrastructure_failures"])]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
