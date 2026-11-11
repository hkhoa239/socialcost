#!/usr/bin/env python3
"""Compare repro_probe.py logs of repeated launches with one seed.

Episodes are aligned on their simulation-time start (EpisodeRecord.start_time). The same episode
index in different launches has the same seed, so it should replay the same trajectories.
Robot and pedestrian positions are resampled on a 0.1 s grid; for each episode index the first
launch is the reference.

    python analyze_repro.py <dir> --mode free [--mode lockstep] --out <report.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def load(dir_: str, mode: str) -> dict[int, list[dict]]:
    runs = {}
    for path in sorted(glob.glob(os.path.join(dir_, f"{mode}_*.jsonl"))):
        launch = int(os.path.basename(path).split("_")[-1].split(".")[0])
        runs[launch] = [json.loads(line) for line in open(path) if line.strip()]
    return runs


def resample(ep: dict, horizon: float, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    t = np.arange(0.0, horizon + 1e-9, dt)
    r = np.array(ep["robot"], dtype=float)
    rt = (r[:, 0] - ep["start_ns"]) / 1e9
    robot = np.stack([np.interp(t, rt, r[:, 1], left=np.nan, right=np.nan), np.interp(t, rt, r[:, 2], left=np.nan, right=np.nan)], axis=1)
    peds = {}
    names = {p[0] for _, ps in ep["peds"] for p in ps}
    for name in names:
        pts = [(st, p[1], p[2]) for st, ps in ep["peds"] for p in ps if p[0] == name]
        a = np.array(pts, dtype=float)
        pt = (a[:, 0] - ep["start_ns"]) / 1e9
        peds[name] = np.stack([np.interp(t, pt, a[:, 1], left=np.nan, right=np.nan), np.interp(t, pt, a[:, 2], left=np.nan, right=np.nan)], axis=1)
    return t, robot, peds


def duration(ep: dict) -> float:
    return (ep["robot"][-1][0] - ep["start_ns"]) / 1e9 if ep["robot"] else 0.0


def analyze(runs: dict[int, list[dict]]) -> dict:
    by_index: dict[int, list[tuple[int, dict]]] = {}
    for launch, eps in runs.items():
        # episode id differs per launch only if a launch missed an episode; group by the episode id
        for ep in eps:
            by_index.setdefault(ep["episode_id"], []).append((launch, ep))
    out = {"launches": len(runs), "per_episode": {}}
    for eid, items in sorted(by_index.items()):
        items = [(la, ep) for la, ep in items if ep["robot"] and ep["peds"]]
        if not items:
            continue
        timing = {k: [ep[k] for _, ep in items if ep.get(k) is not None] for k in ("goal_rel_s", "first_cmd_rel_s", "first_motion_rel_s")}
        seeds = sorted({ep["seed"] for _, ep in items})
        horizon = min(duration(ep) for _, ep in items)
        ref_t, ref_robot, ref_peds = resample(items[0][1], horizon)
        robot_err, ped_err, inter = [], [], {}
        for la, ep in items:
            for name, t0 in ep["interactions"].items():
                inter.setdefault(name, []).append((t0 - ep["start_ns"]) / 1e9)
            if la == items[0][0]:
                continue
            _, robot, peds = resample(ep, horizon)
            d = np.linalg.norm(robot - ref_robot, axis=1)
            robot_err.append(d[np.isfinite(d)])
            for name, tr in peds.items():
                if name in ref_peds:
                    e = np.linalg.norm(tr - ref_peds[name], axis=1)
                    ped_err.append(e[np.isfinite(e)])
        re = np.concatenate(robot_err) if robot_err else np.array([])
        pe = np.concatenate(ped_err) if ped_err else np.array([])
        out["per_episode"][str(eid)] = {
            "launches_compared": len(items), "episode_seeds": seeds, "first_after_attach": [ep["first_after_attach"] for _, ep in items],
            "outcomes": [ep["outcome"] for _, ep in items], "durations_s": [round(duration(ep), 2) for _, ep in items],
            "timing_s": {k: {"values": [round(x, 3) for x in v], "range": round(max(v) - min(v), 3) if v else None} for k, v in timing.items()},
            "compared_horizon_s": round(horizon, 2),
            "robot_rmse_m": float(np.sqrt(np.mean(re ** 2))) if re.size else None, "robot_max_dev_m": float(re.max()) if re.size else None,
            "ped_rmse_m": float(np.sqrt(np.mean(pe ** 2))) if pe.size else None, "ped_max_dev_m": float(pe.max()) if pe.size else None,
            "interaction_activation_s": {k: {"values": [round(x, 3) for x in v], "range": round(max(v) - min(v), 3)} for k, v in inter.items()},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--mode", action="append", default=["free"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {m: analyze(load(args.dir, m)) for m in dict.fromkeys(args.mode)}
    json.dump(report, open(args.out, "w"), indent=2)
    for m, r in report.items():
        print(f"== {m}: {r['launches']} launches")
        for eid, e in r["per_episode"].items():
            print(f"  ep {eid}: n={e['launches_compared']} seeds={len(e['episode_seeds'])} motion={e['timing_s']['first_motion_rel_s']['values']} "
                  f"robot rmse={e['robot_rmse_m']} max={e['robot_max_dev_m']} ped rmse={e['ped_rmse_m']} max={e['ped_max_dev_m']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
