#!/usr/bin/env python3
"""Navigation metrics v2 (RGB-D diagnosis, Phase F). Same per-run definitions as nav_metrics.py, plus:

  - a probe timeout (episode never ended within the wall budget) is a navigation FAILURE (success 0),
    not a dropped run; warm-up failures are infrastructure failures and are listed separately
  - uncertainty: success rate with a Wilson 95 % interval; other metrics mean +- t(n-1) s / sqrt(n)
    (reported only for n >= 3; never a min/max range); paired differences per seed with the same t interval
  - pedestrian deflection against the robot-free reference of the same scenario and seed for every
    pedestrian (walkers: mean lateral offset and along-track delay; standing people: max displacement
    from their reference position at the same time); `ped_ref` says whether a reference existed
  - estimator runtime overhead: compute ms (mean, p95) and publish interval from estimator.jsonl

    python nav_metrics_v2.py <nav output dir> --out results.json [--baselines bodies rgbd]
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

from nav_metrics import ped_tracks, run_metrics  # noqa: E402

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
KEYS = ("scenario", "method", "controller", "seed", "tag")


def tci(v: list[float]) -> dict:
    v = [float(x) for x in v if x is not None]
    n = len(v)
    out = {"mean": float(np.mean(v)) if n else None, "n": n, "sd": float(np.std(v, ddof=1)) if n > 1 else None, "ci95_t": None}
    if n >= 3:
        h = T95.get(n, 1.96) * out["sd"] / math.sqrt(n)
        out["ci95_t"] = [out["mean"] - h, out["mean"] + h]
    return out


def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"rate": None, "k": 0, "n": 0, "ci95_wilson": None}
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": p, "k": k, "n": n, "ci95_wilson": [max(0.0, c - h), min(1.0, c + h)]}


def load(d: str) -> dict | None:
    rj = os.path.join(d, "run.json")
    if not os.path.exists(rj):
        return None
    run = json.load(open(rj))
    p = os.path.join(d, "episode.jsonl")
    lines = [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []
    run["ep"] = lines[0] if lines else None
    run["dir"] = d
    return run


def deflection_vs_ref(ep: dict, ref: dict | None) -> tuple[float | None, float | None, bool]:
    peds = ped_tracks(ep)
    if not peds:
        return None, None, False
    if ref is None or ref.get("ep") is None:
        return None, None, False
    rp = ped_tracks(ref["ep"])
    defl, delay = [], []
    for name, tr in peds.items():
        if name not in rp:
            continue
        rt = rp[name]
        tt = tr[:, 0]
        rx, ry = np.interp(tt, rt[:, 0], rt[:, 1]), np.interp(tt, rt[:, 0], rt[:, 2])
        vx, vy = np.gradient(rx, tt), np.gradient(ry, tt)
        v = np.hypot(vx, vy)
        ok = v > 0.1
        if ok.sum() > 10:
            ux, uy = vx[ok] / v[ok], vy[ok] / v[ok]
            ex, ey = tr[ok, 1] - rx[ok], tr[ok, 2] - ry[ok]
            defl.append(float(np.abs(-ex * uy + ey * ux).mean()))
            delay.append(float((-(ex * ux + ey * uy) / v[ok]).mean()))
        else:
            defl.append(float(np.hypot(tr[:, 1] - rx, tr[:, 2] - ry).max()))
    return (float(np.mean(defl)) if defl else None), (float(np.mean(delay)) if delay else None), bool(defl)


def runtime(d: str) -> dict:
    p = os.path.join(d, "estimator.jsonl")
    if not os.path.exists(p):
        return {}
    E = [json.loads(x) for x in open(p) if x.strip()]
    if len(E) < 2:
        return {}
    c = np.array([e["compute_ms"] for e in E])
    t = np.array([e["t_ns"] for e in E], float)
    iv = np.diff(t) / 1e6
    iv = iv[iv > 0]
    return {"compute_ms_mean": float(c.mean()), "compute_ms_p95": float(np.percentile(c, 95)),
            "publish_interval_ms_p50": float(np.median(iv)) if len(iv) else None, "publish_interval_ms_p95": float(np.percentile(iv, 95)) if len(iv) else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baselines", nargs="+", default=["bodies", "rgbd"])
    args = ap.parse_args()
    runs = [r for r in (load(d) for d in sorted(glob.glob(os.path.join(args.root, "*__*")))) if r]
    refs = {}
    for r in runs:
        if r["scenario"].endswith("_ref") and r.get("ep"):
            refs.setdefault((r["scenario"][:-4], r["seed"]), r)
    per, infra = [], []
    for r in runs:
        if r["scenario"].endswith("_ref"):
            continue
        base = {k: r[k] for k in KEYS} | {"status": r["status"]}
        if r["status"].startswith("failed"):
            infra.append(base)
            continue
        if r["ep"] is None:  # timeout: the robot never finished the episode
            per.append(base | {"success": False, "timeout": True} | runtime(r["dir"]))
            continue
        halt = {n for _, ps in r["ep"]["peds"][:1] for n, *_ in ps} if "halt" in r["scenario"] else set()
        m = run_metrics(r, None, halt)
        dfl, dly, has = deflection_vs_ref(r["ep"], refs.get((r["scenario"], r["seed"])))
        m["ped_deflection_m"], m["ped_delay_s"], m["ped_ref"] = dfl, dly, has
        per.append(base | {"timeout": False} | m | runtime(r["dir"]))
    groups = defaultdict(list)
    for p in per:
        groups[(p["scenario"], p["controller"], p["method"])].append(p)
    metrics = ("time_to_goal_s", "path_length_m", "stop_time_s", "min_clearance_m", "personal_zone_time_s", "halt_wedge_time_s",
               "ped_deflection_m", "ped_delay_s", "compute_ms_mean", "compute_ms_p95", "publish_interval_ms_p95")
    summary = {}
    for (s, c, m), L in groups.items():
        e = {"success": wilson(sum(1 for p in L if p["success"]), len(L)), "timeouts": sum(1 for p in L if p.get("timeout")),
             "pass_inside": wilson(sum(1 for p in L if p.get("pass_inside")), sum(1 for p in L if p.get("pass_inside") is not None)),
             "seeds": sorted(p["seed"] for p in L)}
        for k in metrics:
            e[k] = tci([p.get(k) for p in L if p.get(k) is not None])
        summary[f"{s}|{c}|{m}"] = e
    paired = {}
    for b in args.baselines:
        idx = {(p["scenario"], p["controller"], p["seed"]): p for p in per if p["method"] == b}
        acc = defaultdict(lambda: defaultdict(list))
        for p in per:
            q = idx.get((p["scenario"], p["controller"], p["seed"]))
            if q is None or p["method"] == b:
                continue
            for k in ("time_to_goal_s", "stop_time_s", "min_clearance_m", "personal_zone_time_s", "halt_wedge_time_s", "ped_deflection_m", "ped_delay_s"):
                if p.get(k) is not None and q.get(k) is not None:
                    acc[f"{p['scenario']}|{p['controller']}|{p['method']}"][k].append(p[k] - q[k])
            acc[f"{p['scenario']}|{p['controller']}|{p['method']}"]["success"].append(float(p["success"]) - float(q["success"]))
        paired[b] = {k: {kk: tci(v) for kk, v in d.items()} for k, d in acc.items()}
    json.dump({"per_run": per, "infra_failures": infra, "summary": summary, "paired": paired}, open(args.out, "w"), indent=1)
    for k in sorted(summary):
        e = summary[k]
        print(k, "succ", f"{e['success']['k']}/{e['success']['n']}", "T", None if e["time_to_goal_s"]["mean"] is None else round(e["time_to_goal_s"]["mean"], 1),
              "stop", None if e["stop_time_s"]["mean"] is None else round(e["stop_time_s"]["mean"], 1))
    print("infra failures:", len(infra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
