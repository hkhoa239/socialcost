#!/usr/bin/env python3
"""Recorded episodes of one scenario and seed from separate launches, compared slot by slot.

Samples sit on the same simulator-time grid (episode start + warmup + k / sample_hz), so the same
slot is the same moment of the episode in every launch. The first episode is the reference.

    python compare_repeats.py <dataset_root> --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.verify import load_episodes  # noqa: E402


def pose_at(e, t_rel_ns: float, origin_ns: float) -> tuple[float, float] | None:
    """Robot position at origin + t_rel, linear between recorded samples."""
    ts = np.array([r["stamp_ns"] for r in e.robots], dtype=float) - origin_ns
    if t_rel_ns < ts[0] or t_rel_ns > ts[-1]:
        return None
    x = np.interp(t_rel_ns, ts, [r["pose_map_base"]["x"] for r in e.robots])
    y = np.interp(t_rel_ns, ts, [r["pose_map_base"]["y"] for r in e.robots])
    return float(x), float(y)


def onset_ns(e) -> float:
    return e.meta["episode_start_ns"] + e.meta["stats"]["first_motion_rel_s"] * 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for e in load_episodes(args.root):
        groups[(e.scenario, e.meta.get("run_seed"))].append(e)
    report = {}
    for (scen, seed), eps in groups.items():
        ref = eps[0]
        ref_slot = {r["slot"]: i for i, r in enumerate(ref.robots)}
        rob, rob_yaw, ped, label = [], [], [], []
        per_launch = []
        for e in eps:
            slots = {r["slot"]: i for i, r in enumerate(e.robots)}
            rr, pp = [], []
            for s in sorted(set(slots) & set(ref_slot)):
                a, b = ref.robots[ref_slot[s]]["pose_map_base"], e.robots[slots[s]]["pose_map_base"]
                rr.append(math.dist((a["x"], a["y"]), (b["x"], b["y"])))
                rob_yaw.append(abs(math.degrees(math.atan2(math.sin(a["yaw"] - b["yaw"]), math.cos(a["yaw"] - b["yaw"])))))
                A = {p["name"]: p for p in ref.states(ref_slot[s])}
                B = {p["name"]: p for p in e.states(slots[s])}
                pp += [math.dist((A[n]["x"], A[n]["y"]), (B[n]["x"], B[n]["y"])) for n in A.keys() & B.keys()]
                label.append(float(np.abs(ref.label(ref_slot[s]).astype(int) - e.label(slots[s]).astype(int)).mean()))
            rob += rr
            ped += pp
            st = e.meta["stats"]
            per_launch.append({"episode": e.name, "episode_id": e.meta["episode_id"], "episode_seed": e.meta.get("episode_seed"),
                               "first_motion_rel_s": st.get("first_motion_rel_s"), "start_ok": st.get("start_ok"),
                               "robot_dev_max_m": max(rr) if rr else None, "ped_dev_max_m": max(pp) if pp else None})
        # robot aligned on its own motion onset instead of the episode start
        onset_err = []
        for e in eps[1:]:
            for r in ref.robots:
                t_rel = r["stamp_ns"] - onset_ns(ref)
                q = pose_at(e, t_rel, onset_ns(e))
                if q is not None:
                    onset_err.append(math.dist((r["pose_map_base"]["x"], r["pose_map_base"]["y"]), q))
        onset_err = np.array(onset_err)
        fm = [p["first_motion_rel_s"] for p in per_launch if p["first_motion_rel_s"] is not None]
        rob, ped = np.array(rob), np.array(ped)
        report[f"{scen}:{seed}"] = {
            "launches": len(eps), "episode_seeds_identical": len({p["episode_seed"] for p in per_launch}) == 1,
            "first_motion_rel_s": {"values": fm, "range": (max(fm) - min(fm)) if fm else None},
            "robot_rmse_m": float(np.sqrt((rob ** 2).mean())) if rob.size else None, "robot_max_dev_m": float(rob.max()) if rob.size else None,
            "robot_heading_max_dev_deg": max(rob_yaw) if rob_yaw else None,
            "robot_onset_aligned_rmse_m": float(np.sqrt((onset_err ** 2).mean())) if onset_err.size else None,
            "robot_onset_aligned_max_dev_m": float(onset_err.max()) if onset_err.size else None,
            "ped_rmse_m": float(np.sqrt((ped ** 2).mean())) if ped.size else None, "ped_max_dev_m": float(ped.max()) if ped.size else None,
            "label_mae_mean": float(np.mean(label)) if label else None, "per_launch": per_launch,
        }
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_launch"} for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
