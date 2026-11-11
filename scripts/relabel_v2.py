#!/usr/bin/env python3
"""Regenerate dataset_v1 labels with a socialcost_v2 spec into a separate tree (dataset_v1 stays untouched).

<out>/episodes/<id>/{social_cost, social_cost_membership, social_cost_extended}/<sample>.npy   (native / membership / extended/2.0)
<out>/episodes/<id>/components/<sample>.npz   personal_space, conversation, queue, gesture, mobility (benchmark metrics)
<out>/manifest.json   spec, equations, parameters, source checks, v1 spot-check results; splits are dataset_v1's (not copied)

Source states: recorded robot_states / pedestrian_states / interactions of each episode (no simulation). Every 40th sample
is also painted with the v1 spec and must equal the stored v1 labels (guards against drift in the shared code path).

    cd /opt/arena_ws/src/deps/socialcost && PYTHONPATH=. /opt/venv/bin/python scripts/relabel_v2.py --spec v2 --out <dir> [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.dataset import interaction_frame_at, read_jsonl  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402
from socialcost.labels_v2 import SPECS, labels_and_components, manifest  # noqa: E402

DV1 = os.environ.get("SC_DV1_ROOT", "/opt/arena_ws/data/socialcost/dataset_v1")
CHANNELS = {"native": "social_cost", "membership": "social_cost_membership", "extended": "social_cost_extended"}
COMP = ("personal_space", "conversation", "queue", "gesture", "mobility")


def check_sources(eid: str) -> dict:
    d = os.path.join(DV1, "episodes", eid)
    miss = [f for f in ("robot_states.jsonl", "pedestrian_states.jsonl", "interactions.jsonl", "episode_metadata.json") if not os.path.exists(os.path.join(d, f))]
    if miss:
        return {"episode": eid, "missing": miss}
    robots = {r["sample"] for r in read_jsonl(os.path.join(d, "robot_states.jsonl"))}
    peds = {p["sample"] for p in read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))}
    v1 = {int(f[:-4]) for f in os.listdir(os.path.join(d, "social_cost")) if f.endswith(".npy")} if os.path.isdir(os.path.join(d, "social_cost")) else set()
    return {"episode": eid, "missing": [], "samples": len(robots), "no_ped_state": sorted(robots - peds)[:5], "n_no_ped_state": len(robots - peds),
            "v1_label_samples": len(v1), "v1_not_regenerable": len(v1 - (robots & peds))}


def relabel(job) -> dict:
    eid, spec_name, out = job
    d = os.path.join(DV1, "episodes", eid)
    o = os.path.join(out, "episodes", eid)
    if os.path.exists(os.path.join(o, "DONE")):
        return json.load(open(os.path.join(o, "DONE")))
    meta = json.load(open(os.path.join(d, "episode_metadata.json")))
    base = LayerParams.from_mapping(meta["label"]["layer_params"])
    grid = GridSpec(**meta["label"]["grid"])
    robots = {r["sample"]: r for r in read_jsonl(os.path.join(d, "robot_states.jsonl"))}
    peds = {p["sample"]: p for p in read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))}
    frames = read_jsonl(os.path.join(d, "interactions.jsonl"))
    for c in list(CHANNELS.values()) + ["components"]:
        os.makedirs(os.path.join(o, c), exist_ok=True)
    spec = SPECS[spec_name]
    n, v1_checked, v1_bad, area = 0, 0, 0, {k: 0.0 for k in CHANNELS}
    for k in sorted(robots):
        if k not in peds:
            continue
        r = robots[k]
        pose = (r["pose_map_base"]["x"], r["pose_map_base"]["y"], r["pose_map_base"]["yaw"])
        fr = interaction_frame_at(frames, r["stamp_ns"])
        lab = labels_and_components(spec, base, grid, peds[k]["pedestrians"], fr, pose)
        name = f"{k:06d}"
        for ch, dn in CHANNELS.items():
            np.save(os.path.join(o, dn, name + ".npy"), lab[ch])
            area[ch] += float((lab[ch] > 0).sum())
        np.savez_compressed(os.path.join(o, "components", name + ".npz"), **{c: lab[c] for c in COMP})
        if k % 40 == 0:
            ref = labels_and_components(SPECS["v1"], base, grid, peds[k]["pedestrians"], fr, pose)
            v1_checked += 1
            v1_bad += int(any(not np.array_equal(ref[ch], np.load(os.path.join(d, dn, name + ".npy"))) for ch, dn in CHANNELS.items()))
        n += 1
    res = {"episode": eid, "samples": n, "v1_spotchecks": v1_checked, "v1_mismatch": v1_bad,
           "mean_cost_cells": {ch: a / max(n, 1) for ch, a in area.items()}}
    json.dump(res, open(os.path.join(o, "DONE"), "w"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v2", choices=sorted(SPECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eps = sorted(e for e in os.listdir(os.path.join(DV1, "episodes")) if os.path.isdir(os.path.join(DV1, "episodes", e)))
    if args.limit:
        eps = eps[:args.limit]
    t0 = time.time()
    with Pool(args.workers) as pool:
        src = pool.map(check_sources, eps)
    bad = [s for s in src if s["missing"] or s.get("v1_not_regenerable")]
    print(f"{len(eps)} episodes; missing sources: {len([s for s in src if s['missing']])}; with v1 samples not regenerable: "
          f"{len([s for s in src if s.get('v1_not_regenerable')])}", flush=True)
    ok = [s["episode"] for s in src if not s["missing"]]
    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(relabel, [(e, args.spec, args.out) for e in ok])):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(ok)} episodes, {time.time() - t0:.0f} s", flush=True)
    base_meta = json.load(open(os.path.join(DV1, "episodes", ok[0], "episode_metadata.json")))
    man = manifest(SPECS[args.spec], LayerParams.from_mapping(base_meta["label"]["layer_params"]), GridSpec(**base_meta["label"]["grid"]))
    man.update({"source_dataset": DV1, "splits": os.path.join(DV1, "splits") + " (identical splits and counterfactual pairs; not copied)",
                "episodes": len(results), "samples": sum(r["samples"] for r in results),
                "source_check": {"episodes_missing_sources": [s for s in src if s["missing"]], "episodes_with_unregenerable_v1_samples": bad},
                "v1_spotcheck": {"samples": sum(r["v1_spotchecks"] for r in results), "mismatches": sum(r["v1_mismatch"] for r in results)},
                "channels": CHANNELS, "components": list(COMP), "wall_s": round(time.time() - t0)})
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("episodes", "samples", "v1_spotcheck", "wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
