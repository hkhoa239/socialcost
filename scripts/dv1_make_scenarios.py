#!/usr/bin/env python3
"""Write the arena-scenario/2 directories (worlds/<world>/scenarios/scv1_<episode>) for manifest episodes.

    python dv1_make_scenarios.py --root /opt/arena_ws/data/socialcost/dataset_v1 [--stage subset|pilot|all] [--build]

Existing directories are rewritten from the manifest, except OFF episodes whose spawn poses were already
derived from their ON twin (manifests/derived_spawns.json), which are re-applied. --build runs the
colcon build of arena_simulation_setup that new scenario directories need.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.dv1.scenario import write_episode  # noqa: E402

BUILD = ("cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1 && colcon build --symlink-install "
         "--cmake-args -DPython3_EXECUTABLE=/opt/venv/bin/python --base-paths /opt/arena_ws/src --build-base /opt/arena_ws/build "
         "--install-base /opt/arena_ws/install --packages-select arena_simulation_setup")


def select(episodes: list[dict], stage: str) -> list[dict]:
    if stage == "subset":
        return [e for e in episodes if e["subset"]]
    if stage == "pilot":
        return [e for e in episodes if e["pilot"]]
    return episodes


def apply_derived(ep: dict, derived: dict) -> dict:
    d = derived.get(ep["episode_id"])
    if not d:
        return ep
    ep = json.loads(json.dumps(ep))
    for h in ep["humans"]:
        if h["name"] in d["spawn"]:
            h["spawn"] = d["spawn"][h["name"]]
    return ep


def build() -> None:
    r = subprocess.run(["bash", "-lc", BUILD], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("arena_simulation_setup build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", default="all", choices=["subset", "pilot", "all"])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.root, "manifests", "manifest_v1.json")))
    dpath = os.path.join(args.root, "manifests", "derived_spawns.json")
    derived = json.load(open(dpath)) if os.path.exists(dpath) else {}
    eps = select(man["episodes"], args.stage)
    for ep in eps:
        write_episode(apply_derived(ep, derived))
    print(f"wrote {len(eps)} scenario directories")
    if args.build:
        build()
        print("arena_simulation_setup built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
