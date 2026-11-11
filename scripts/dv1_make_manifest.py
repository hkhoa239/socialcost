#!/usr/bin/env python3
"""Stage A: generate and freeze the complete 600-episode dataset_v1 manifest and split files.

    python dv1_make_manifest.py --root /opt/arena_ws/data/socialcost/dataset_v1 [--force]

Writes manifests/manifest_v1.json (every episode: world, site, seed, bodies, counts, trajectories,
condition, visibility category, pair, split, label versions, stage), splits/<split>.txt,
splits/test_d_counterfactual.json, manifests/manifest_v1.sha256. Refuses to overwrite a frozen
manifest unless --force.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.dv1 import spec  # noqa: E402
from socialcost.dv1.manifest import generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    mdir = os.path.join(args.root, "manifests")
    out = os.path.join(mdir, "manifest_v1.json")
    if os.path.exists(os.path.join(mdir, "FROZEN")) and not args.force:
        print("manifest is frozen (manifests/FROZEN); use --force to regenerate", file=sys.stderr)
        return 1
    pools = json.load(open(os.path.join(args.root, "metadata", "body_pools.json")))
    episodes, info = generate(pools)
    doc = {"dataset": spec.DATASET, "spec_version": spec.SPEC_VERSION, "master_seed": spec.MASTER_SEED,
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "recording": {"sample_hz": spec.SAMPLE_HZ, "duration_s": spec.DURATION_S, "warmup_s": spec.WARMUP_S},
           "worlds": spec.WORLDS, "sites": {k: {"world": v[0], "kind": v[1], "route": v[2]} for k, v in spec.SITES.items()},
           "label_versions": spec.LABELS, "ped_count_target": {f"{a}-{b}": f for (a, b), f in spec.PED_COUNT_BINS.items()},
           "ped_count_allocation": {fam: {f"{a}-{b}": n for (a, b), n in d.items()} for fam, d in spec.PED_COUNT_ALLOCATION.items()},
           "body_pools": {k: pools[k] for k in ("known_walking", "heldout_walking", "wheelchair_known", "wheelchair_heldout")},
           "generation": info, "episodes": episodes}
    os.makedirs(mdir, exist_ok=True)
    dsp = os.path.join(mdir, "derived_spawns.json")
    if os.path.exists(dsp):  # derived OFF spawns belong to the previous manifest
        os.replace(dsp, dsp + ".stale_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    raw = json.dumps(doc, indent=1, sort_keys=False)
    open(out, "w").write(raw)
    open(os.path.join(mdir, "manifest_v1.sha256"), "w").write(hashlib.sha256(raw.encode()).hexdigest() + "  manifest_v1.json\n")
    sdir = os.path.join(args.root, "splits")
    os.makedirs(sdir, exist_ok=True)
    for s in spec.SPLITS:
        with open(os.path.join(sdir, f"{s}.txt"), "w") as f:
            f.writelines(f"{e['episode_id']}\n" for e in episodes if e["split"] == s)
    test_d = [{"pair_id": e["pair_id"], "family": e["family"], "split": e["split"], "observability": e["observability"],
               "on": e["episode_id"], "off": next(x["episode_id"] for x in episodes if x["pair_id"] == e["pair_id"] and x["condition"] == "off")}
              for e in episodes if e["pair_id"] and e["condition"] == "on" and e["split"].startswith("test")]
    json.dump({"definition": "Test D: indexed subset of the counterfactual pairs in test_a, test_b and test_c. Not counted again; "
                             "both conditions of a pair are in the same primary split. observability 'hidden_state' pairs (queues) are "
                             "controls, not evidence of visual understanding.", "pairs": test_d},
              open(os.path.join(sdir, "test_d_counterfactual.json"), "w"), indent=1)
    c = Counter((e["family"], e["split"]) for e in episodes)
    print(f"{len(episodes)} episodes, {info['units']} units; per split {dict(Counter(e['split'] for e in episodes))}")
    print("per world", dict(Counter(e["world_key"] for e in episodes)), "pilot", sum(e["pilot"] for e in episodes), "subset", sum(e["subset"] for e in episodes))
    print("test_d pairs", len(test_d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
