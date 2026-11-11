#!/usr/bin/env python3
"""Navigation metrics from run_nav_eval.py runs, per run and aggregated per scenario x controller x method.

Per run (simulator time, robot base and pedestrian centres in map):
  success, time_to_goal_s, path_length_m, stop_time_s (speed < 0.02 m/s after first motion)
  min_clearance_m          min robot-pedestrian centre distance
  personal_zone_time_s     time within personal_space_radius (1.0 m) of any pedestrian
  pass_inside              robot path crosses the segment between two members of one interaction
  halt_wedge_time_s        time inside the 2.5 m / 30 deg wedge in front of a person with a raised hand
  ped_deflection_m         walkers: mean lateral offset from their robot-free reference path (same seed),
                           standing people: max displacement from their start position
  ped_delay_s              walkers: mean along-track lag behind the reference, divided by reference speed
Means with 95 % bootstrap confidence intervals over seeds, and per-seed paired differences against
the bodies-only arm.

    python nav_metrics.py <nav output dir> --out results.json
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

PS_RADIUS = 1.0
HALT_LEN, HALT_HALF = 2.5, math.radians(30)


def load_run(d: str) -> dict | None:
    p = os.path.join(d, "episode.jsonl")
    if not os.path.exists(p):
        return None
    lines = [json.loads(x) for x in open(p) if x.strip()]
    if not lines:
        return None
    return {**json.load(open(os.path.join(d, "run.json"))), "ep": lines[0]}


def traj(ep: dict):
    r = np.array(ep["robot"], dtype=float)
    t = (r[:, 0] - ep["start_ns"]) / 1e9
    return t, r[:, 1:3], r[:, 3]


def ped_tracks(ep: dict) -> dict[str, np.ndarray]:
    out = defaultdict(list)
    for st, ps in ep["peds"]:
        for name, x, y, iid in ps:
            out[name].append(((st - ep["start_ns"]) / 1e9, x, y, iid))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def seg_cross(p1, p2, q1, q2) -> bool:
    o = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])  # noqa: E731
    return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0


def run_metrics(run: dict, ref: dict | None, halt_people: set[str]) -> dict:
    ep = run["ep"]
    t, xy, yaw = traj(ep)
    peds = ped_tracks(ep)
    dt = np.diff(t, prepend=t[0])
    sp = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)]
    moved = np.nonzero(sp > 0.02)[0]
    m = {"success": ep["outcome"] == 2, "time_to_goal_s": float(t[-1]) if ep["outcome"] == 2 else None,
         "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
         "stop_time_s": float(dt[moved[0]:][sp[moved[0]:] < 0.02].sum()) if moved.size else None}
    clear, zone, wedge, inside = [], 0.0, 0.0, False
    for name, tr in peds.items():
        px = np.interp(t, tr[:, 0], tr[:, 1])
        py = np.interp(t, tr[:, 0], tr[:, 2])
        d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
        clear.append(d.min())
    if peds:
        allp = np.stack([np.hypot(xy[:, 0] - np.interp(t, tr[:, 0], tr[:, 1]), xy[:, 1] - np.interp(t, tr[:, 0], tr[:, 2])) for tr in peds.values()])
        zone = float(dt[(allp < PS_RADIUS).any(axis=0)].sum())
    m["min_clearance_m"] = float(min(clear)) if clear else None
    m["personal_zone_time_s"] = zone
    # interaction segments: members sharing an id at the same stamp
    for st, ps in ep["peds"]:
        groups = defaultdict(list)
        for name, x, y, iid in ps:
            if iid >= 0:
                groups[iid].append((x, y))
        tt = (st - ep["start_ns"]) / 1e9
        k = np.searchsorted(t, tt)
        if 0 < k < len(t):
            for mem in groups.values():
                for a in range(len(mem)):
                    for b in range(a + 1, len(mem)):
                        if seg_cross(tuple(xy[k - 1]), tuple(xy[k]), mem[a], mem[b]):
                            inside = True
    m["pass_inside"] = inside
    for name in halt_people & set(peds):
        tr = peds[name]
        # the halt faces the robot's approach: from the person toward -x of map (the route comes from the west)
        px, py = np.interp(t, tr[:, 0], tr[:, 1]), np.interp(t, tr[:, 0], tr[:, 2])
        along = px - xy[:, 0]
        lat = np.abs(xy[:, 1] - py)
        inw = (along >= 0) & (along <= HALT_LEN) & (lat <= 0.5 + along * math.tan(HALT_HALF))
        wedge += float(dt[inw].sum())
    m["halt_wedge_time_s"] = wedge if halt_people else None
    # pedestrian disturbance
    defl, delay = [], []
    rpeds = ped_tracks(ref["ep"]) if ref else {}
    for name, tr in peds.items():
        if name in rpeds:
            rt = rpeds[name]
            tt = tr[:, 0]
            rx, ry = np.interp(tt, rt[:, 0], rt[:, 1]), np.interp(tt, rt[:, 0], rt[:, 2])
            vx, vy = np.gradient(rx, tt), np.gradient(ry, tt)
            v = np.hypot(vx, vy)
            ok = v > 0.1
            if ok.any():
                ux, uy = vx[ok] / v[ok], vy[ok] / v[ok]
                ex, ey = tr[ok, 1] - rx[ok], tr[ok, 2] - ry[ok]
                defl.append(float(np.abs(-ex * uy + ey * ux).mean()))
                delay.append(float((-(ex * ux + ey * uy) / v[ok]).mean()))
        else:
            defl.append(float(np.hypot(tr[:, 1] - tr[0, 1], tr[:, 2] - tr[0, 2]).max()))
    m["ped_deflection_m"] = float(np.mean(defl)) if defl else None
    m["ped_delay_s"] = float(np.mean(delay)) if delay else None
    return m


def ci(v: list[float]) -> list[float] | None:
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(v, len(v))) for _ in range(2000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = [r for r in (load_run(d) for d in sorted(glob.glob(os.path.join(args.root, "*__*")))) if r]
    refs = {(r["scenario"].replace("_ref", ""), r["controller"], r["seed"]): r for r in runs if r["scenario"].endswith("_ref")}
    per = []
    for r in runs:
        if r["scenario"].endswith("_ref"):
            continue
        halt = {n for _, ps in r["ep"]["peds"][:1] for n, *_ in ps} if "halt" in r["scenario"] else set()
        m = run_metrics(r, refs.get((r["scenario"], r["controller"], r["seed"])), halt)
        per.append({k: r[k] for k in ("scenario", "method", "controller", "seed", "tag")} | m)
    agg = defaultdict(lambda: defaultdict(list))
    for p in per:
        for k, v in p.items():
            if k not in ("scenario", "method", "controller", "seed", "tag") and v is not None:
                agg[(p["scenario"], p["controller"], p["method"])][k].append(float(v))
    summary = {f"{s}|{c}|{m}": {k: {"mean": float(np.mean(v)), "ci95": ci(v), "n": len(v)} for k, v in d.items()} for (s, c, m), d in agg.items()}
    # paired differences against bodies-only, matched by scenario, controller and seed
    base = {(p["scenario"], p["controller"], p["seed"]): p for p in per if p["method"] == "bodies"}
    paired = defaultdict(lambda: defaultdict(list))
    for p in per:
        b = base.get((p["scenario"], p["controller"], p["seed"]))
        if b is None or p["method"] == "bodies":
            continue
        for k in ("min_clearance_m", "personal_zone_time_s", "ped_deflection_m", "ped_delay_s", "time_to_goal_s", "halt_wedge_time_s"):
            if p.get(k) is not None and b.get(k) is not None:
                paired[f"{p['scenario']}|{p['controller']}|{p['method']}"][k].append(p[k] - b[k])
    paired_s = {k: {kk: {"mean": float(np.mean(v)), "ci95": ci(v), "n": len(v)} for kk, v in d.items()} for k, d in paired.items()}
    json.dump({"per_run": per, "summary": summary, "paired_vs_bodies": paired_s}, open(args.out, "w"), indent=2)
    for k, d in sorted(summary.items()):
        print(k, {kk: round(vv["mean"], 2) for kk, vv in d.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
