#!/usr/bin/env python3
"""Label the visibility category each partial-observability episode actually achieved.

The designed category (manifest `visibility_category`) is kept. From the target's per-sample states in
dv1/visibility.jsonl this adds `achieved_visibility_category` and the evidence to
episode_metadata.json["dv1"] and writes metadata/visibility_achieved.json (per episode + per-category shortfall).

    python dv1_achieved_visibility.py --root /opt/arena_ws/data/socialcost/dataset_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from socialcost.dataset import read_jsonl  # noqa: E402

SEEN = ("visible", "partially_occluded")


def evidence(states: list[str | None]) -> dict:
    seen = [s in SEEN for s in states]
    first = next((i for i, v in enumerate(seen) if v), None)
    exit_i = next((i for i in range(first + 1, len(states)) if states[i] == "outside_fov"), None) if first is not None else None
    re_i = next((i for i in range(exit_i + 1, len(states)) if seen[i]), None) if exit_i is not None else None
    occl = sum(s in ("partially_occluded", "fully_occluded") for s in states)
    full = sum(s == "fully_occluded" for s in states)
    return {"first_seen": first, "exit_sample": exit_i, "reentry_sample": re_i, "occluded_samples": occl, "fully_occluded_samples": full,
            "states": dict(Counter(states))}


def achieved(ev: dict) -> str:
    """Strongest partial-observability event observed, in the order re-entry > exit > occlusion."""
    if ev["first_seen"] is None:
        return "never_visible"
    if ev["reentry_sample"] is not None:
        return "fov_reentry"
    if ev["exit_sample"] is not None:
        return "fov_exit"
    if ev["occluded_samples"] >= 3:
        return "occlusion"
    return "visible_throughout"


def satisfies(designed: str, ev: dict) -> bool:
    if designed == "fov_exit":
        return ev["exit_sample"] is not None
    if designed == "fov_reentry":
        return ev["reentry_sample"] is not None
    if designed == "static_occlusion":
        return ev["occluded_samples"] >= 3
    return ev["occluded_samples"] >= 2  # dynamic_occlusion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.root, "manifests", "manifest_v1.json")))["episodes"]
    out, per_cat = {}, defaultdict(Counter)
    for e in man:
        if e["family"] != "partial_observability":
            continue
        d = os.path.join(args.root, "episodes", e["episode_id"])
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        tgt = next(h["name"] for h in e["humans"] if h["role"] == "target")
        rows = read_jsonl(os.path.join(d, "dv1", "visibility.jsonl"))
        states = [next((p["state"] for p in r["pedestrians"] if p["name"] == tgt), None) for r in rows]
        ev = evidence(states)
        ok = satisfies(e["visibility_category"], ev)
        rec = {"designed": e["visibility_category"], "achieved": e["visibility_category"] if ok else achieved(ev), "design_met": ok,
               "split": e["split"], "world": e["world_key"], "target": tgt, **ev}
        out[e["episode_id"]] = rec
        per_cat[e["visibility_category"]]["met" if ok else "not_met"] += 1
        mp = os.path.join(d, "episode_metadata.json")
        meta = json.load(open(mp))
        meta["dv1"]["achieved_visibility_category"] = rec["achieved"]
        meta["dv1"]["visibility_design_met"] = ok
        meta["dv1"]["visibility_evidence"] = {k: rec[k] for k in ("first_seen", "exit_sample", "reentry_sample", "occluded_samples", "fully_occluded_samples")}
        json.dump(meta, open(mp, "w"), indent=2)
    summary = {c: dict(v) for c, v in per_cat.items()}
    achieved_counts = dict(Counter(r["achieved"] for r in out.values()))
    json.dump({"definition": "designed = manifest visibility_category; achieved = designed when its event is observed for the target "
                             "(exit: seen then outside FOV; re-entry: seen, outside, seen again; static occlusion: >= 3 occluded samples; "
                             "dynamic occlusion: >= 2), otherwise the strongest event observed (fov_reentry > fov_exit > occlusion > "
                             "visible_throughout, or never_visible)",
               "per_designed_category": summary, "achieved_counts": achieved_counts, "episodes": out},
              open(os.path.join(args.root, "metadata", "visibility_achieved.json"), "w"), indent=1)
    print(json.dumps({"per_designed_category": summary, "achieved_counts": achieved_counts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
