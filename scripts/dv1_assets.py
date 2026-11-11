#!/usr/bin/env python3
"""dataset_v1 pedestrian asset audit: inventory of every human body in the asset bucket, pool selection
and the held-out body set.

    python dv1_assets.py --fetched /tmp/dv1_ann --out /opt/arena_ws/data/socialcost/dataset_v1/metadata

--fetched is a directory filled by `arena_assets net default fetch -o <dir> <Domain>/Human/<name> ...`.
Writes asset_manifest.json (all bodies), body_pools.json (pool, known / held-out) and copies thumbnails.
Held-out selection is stratified so no tag value (gender, age, ethnicity, role, domain) is exclusive to
the held-out set; behaviour and labels never depend on these tags.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

import yaml

ARENA = os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena")
LOCAL = os.path.join(ARENA, "_assets", "default")
POOL_SIZE = 44  # walking identities used by dataset_v1 (wheelchair variants share an identity)
HELDOUT_FRACTION = 0.25
SEED = 20261101


def tags_of(ann: dict) -> dict:
    out = {}
    for t in ann.get("tags", []):
        parts = t.split("::")
        if parts[0] == "human" and len(parts) == 3:
            out[parts[1]] = parts[2]
        elif parts[0] == "domain":
            out["domain"] = parts[1]
    return out


def entry(root: str, domain: str, name: str) -> dict:
    d = os.path.join(root, domain, "Human", name)
    ann_path = os.path.join(d, "annotation.yaml")
    ann = yaml.safe_load(open(ann_path)) if os.path.exists(ann_path) else {}
    bb = ann.get("bounding_box")
    clips = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(d, "clips"))) if os.path.isdir(os.path.join(d, "clips")) else []
    t = tags_of(ann)
    wheel = "wheelchair" in name or t.get("mobility") == "wheelchair"
    note = ann.get("note", "")
    rig = "cmu_mb" if "cmu_mb" in note else ("cmu" if "CMU" in note or "CMU" in ann.get("desc", "") else "unknown")
    ref = name if domain == "Common" else f"{domain}/{name}"
    return {
        "body_id": ref, "domain": domain, "name": name, "identity": ref.replace("_wheelchair", ""),
        "annotation": bool(ann), "bounding_box": bb,
        "size_m": {"length": round(bb[0][1] - bb[0][0], 3), "width": round(bb[1][1] - bb[1][0], 3), "height": round(bb[2][1] - bb[2][0], 3)} if bb else None,
        "tags": t, "mobility_aid": "wheelchair" if wheel else None,
        "clips": clips, "rig": rig,
        # GaitGenerator drives walk / run from gait profiles for every CMU-rigged body; gestures (head look,
        # arm point / halt, body clips such as talk_with_arm_gesture) are joint overlays on the same rig
        "walking_compatible": not wheel and "walk" in clips,
        "gesture_compatible": rig in ("cmu", "cmu_mb"),
        "wheelchair_locomotion": wheel,
        "thumbnail": f"thumbnails/{domain}_{name}.png" if os.path.exists(os.path.join(d, f"{name}_thumb.png")) else None,
        "locally_cached_before_dv1": os.path.isdir(os.path.join(LOCAL, domain, "Human", name)),
        "attribution": os.path.exists(os.path.join(d, "ATTRIBUTION.md")),
    }


def select_pool(bodies: list[dict]) -> tuple[list[str], list[str], dict]:
    rng = random.Random(SEED)
    walk = [b for b in bodies if b["walking_compatible"] and b["annotation"] and b["gesture_compatible"]]
    wheel_ids = {b["identity"] for b in bodies if b["mobility_aid"]}
    by_id = {b["identity"]: b for b in walk}
    # every wheelchair identity's walking twin is in the pool (counterfactual OFF condition)
    pool = [i for i in sorted(wheel_ids) if i in by_id]
    strata = defaultdict(list)
    for b in walk:
        if b["identity"] in pool:
            continue
        strata[(b["domain"], str(b["tags"].get("gender")), str(b["tags"].get("age")))].append(b["identity"])
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    while len(pool) < POOL_SIZE and any(strata.values()):
        rng.shuffle(keys)
        for k in keys:
            if strata[k] and len(pool) < POOL_SIZE:
                pool.append(strata[k].pop())
    pool.sort()
    # held-out: one wheelchair identity, the rest stratified; reject draws that make a tag value exclusive
    n_hold = round(HELDOUT_FRACTION * len(pool))
    wheel_sorted = sorted(i for i in wheel_ids if i in pool)
    def tag(i, key):
        return by_id[i]["domain"] if key == "domain" else str(by_id[i]["tags"].get(key))

    keys5 = ("gender", "age", "ethnicity", "role", "domain")
    pool_frac = {k: Counter(tag(i, k) for i in pool) for k in keys5}
    best, best_score, valid = None, 1e9, 0
    for attempt in range(20000):
        cand = [wheel_sorted[-1]] + rng.sample([i for i in pool if i not in wheel_ids], n_hold - 1)
        known = [i for i in pool if i not in cand]
        if any(not {tag(i, k) for i in cand} <= {tag(i, k) for i in known} for k in keys5):
            continue
        valid += 1
        # squared deviation of held-out tag proportions from the pool's
        score = sum((Counter(tag(i, k) for i in cand)[v] / len(cand) - n / len(pool)) ** 2 for k in keys5 for v, n in pool_frac[k].items())
        if score < best_score:
            best, best_score = cand, score
    if best is None:
        raise RuntimeError("no stratified held-out set found")
    cand = best
    known = [i for i in pool if i not in cand]
    stats = {"candidates_valid": valid, "balance_score": round(best_score, 4)}
    return known, sorted(cand), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "thumbnails"), exist_ok=True)
    bodies, missing = [], []
    for domain in sorted(os.listdir(args.fetched)):
        hd = os.path.join(args.fetched, domain, "Human")
        if not os.path.isdir(hd):
            continue
        for name in sorted(os.listdir(hd)):
            e = entry(args.fetched, domain, name)
            bodies.append(e)
            if not e["annotation"]:
                missing.append(e["body_id"])
            src = os.path.join(hd, name, f"{name}_thumb.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, "thumbnails", f"{domain}_{name}.png"))
    known, held, stats = select_pool(bodies)
    wheel = {b["identity"]: b["body_id"] for b in bodies if b["mobility_aid"]}
    for b in bodies:
        b["pool"] = "known" if b["identity"] in known else ("heldout" if b["identity"] in held else "unused")
    by_id = {b["body_id"]: b for b in bodies}

    def tagcount(ids, key):
        return dict(Counter((by_id[i]["domain"] if key == "domain" else by_id[i]["tags"].get(key)) for i in ids))

    pools = {
        "known_walking": known, "heldout_walking": held,
        "wheelchair_known": {i: wheel[i] for i in known if i in wheel}, "wheelchair_heldout": {i: wheel[i] for i in held if i in wheel},
        "heldout_fraction": round(len(held) / (len(known) + len(held)), 3), "selection": {"seed": SEED, "pool_size": POOL_SIZE, **stats},
        "tag_balance": {k: {"known": tagcount(known, k), "heldout": tagcount(held, k)} for k in ("domain", "gender", "age", "ethnicity", "role")},
        "rule": "held-out identities (and their wheelchair variants) never appear in train, val, test_a or test_b episodes",
    }
    json.dump({"bodies": bodies, "total": len(bodies), "without_annotation": missing,
               "counts": {"walking": sum(b["walking_compatible"] for b in bodies), "wheelchair": sum(bool(b["mobility_aid"]) for b in bodies),
                          "by_domain": dict(Counter(b["domain"] for b in bodies))}},
              open(os.path.join(args.out, "asset_manifest.json"), "w"), indent=2)
    json.dump(pools, open(os.path.join(args.out, "body_pools.json"), "w"), indent=2)
    print(f"{len(bodies)} bodies, {len(missing)} without annotation {missing}; pool known {len(known)}, held-out {len(held)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
