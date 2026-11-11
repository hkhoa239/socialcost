#!/usr/bin/env python3
"""navigation_challenge_v1 robot-free pedestrian reference runs.

For every configuration, records scenario scv1_nav_<id>_ref (same pedestrians, robot parked >= 10 m away) over the
configuration's evaluation horizon, with task.scenario.linger_after_completion so the episode outlives the robot's
short parking route. Output: references/<id>/ (recorder format) and references/index.json with the checks:
full horizon recorded, pedestrian roster as configured, robot never within 10 m of a pedestrian.

    python dv1_nav_refs.py --root /opt/arena_ws/data/socialcost/navigation_challenge_v1 [--only N01,N02]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import dv1_collect as dc  # noqa: E402
from dv1_make_scenarios import build  # noqa: E402


def check(ref_dir: str, cfg: dict) -> dict:
    robots = [json.loads(line)["pose_map_base"] for line in open(os.path.join(ref_dir, "robot_states.jsonl"))]
    peds = [json.loads(line)["pedestrians"] for line in open(os.path.join(ref_dir, "pedestrian_states.jsonl"))]
    meta = json.load(open(os.path.join(ref_dir, "episode_metadata.json")))
    dmin = min((math.dist((r["x"], r["y"]), (p["x"], p["y"])) for r, ps in zip(robots, peds) for p in ps), default=None)
    names = sorted(p["name"].split("/")[-1] for p in peds[0]) if peds else []
    ok = (meta["samples"] >= 0.95 * meta["expected_samples"] and names == sorted(h["name"] for h in cfg["humans"]) and (dmin is None or dmin >= 10.0))
    return {"config_id": cfg["config_id"], "samples": meta["samples"], "expected": meta["expected_samples"], "horizon_s": cfg["evaluation_horizon_s"],
            "robot_min_dist_to_ped_m": round(dmin, 2) if dmin is not None else None, "roster_ok": names == sorted(h["name"] for h in cfg["humans"]),
            "end_reason": meta["stats"]["end_reason"], "ok": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    doc = json.load(open(os.path.join(args.root, "navigation_challenge_v1.json")))
    only = {x for x in args.only.split(",") if x}
    cfgs = [c for c in doc["configs"] if not only or c["config_id"] in only]
    cfgs.sort(key=lambda c: (c["world"], c["config_id"]))
    build()
    sensors = __import__("yaml").safe_load(open(dc.cd.resolve("", dc.CFG["sensors"], "sensors")))
    runner = dc.Runner(sensors, os.path.join(args.root, "references", "logs"))
    runner.extra_args = ["task.scenario.linger_after_completion:=true"]
    os.makedirs(os.path.join(args.root, "references", "logs"), exist_ok=True)
    index_path = os.path.join(args.root, "references", "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    for c in cfgs:
        cid = c["config_id"]
        if index.get(cid, {}).get("ok"):
            continue
        ref_dir = os.path.join(args.root, "references", cid)
        ep = {"episode_id": f"ref_{cid}", "scenario": f"scv1_nav_{cid}_ref", "world": c["world"], "run_seed": c["run_seed"], "pair_id": None,
              "condition": "na", "family": f"nav_reference_{c['kind']}", "split": "navigation_challenge"}
        dc.CFG["duration_s"] = float(c["evaluation_horizon_s"])
        res = None
        for attempt in range(3):
            if os.path.isdir(ref_dir):
                shutil.rmtree(ref_dir)
            try:
                runner.record(ep, ref_dir)
                res = check(ref_dir, c)
            except Exception as e:  # noqa: BLE001
                res = {"config_id": cid, "ok": False, "error": f"{type(e).__name__}: {e}"}
            res["attempts"] = attempt + 1
            dc.log(f"reference {cid}: {res}")
            if res.get("ok"):
                break
        index[cid] = res
        json.dump(index, open(index_path, "w"), indent=1)
    runner.stop()
    print(f"{sum(v.get('ok', False) for v in index.values())}/{len(doc['configs'])} references ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
