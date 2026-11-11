#!/usr/bin/env python3
"""Experiment split for the estimators, from episode metadata, with leakage checks.

  train   map_empty body sets bA bB (pairs, group-walk bA, controls)
  val     map_empty bC
  test_A  in-distribution: map_empty bD (training bodies, new placement)
  test_B  unseen world: reception
  test_C  unseen appearance: map_empty bH (Office / Hospital bodies)
  test_D  paired semantic challenge: the ON/OFF pairs of test_A, test_B, test_C

Episodes with valid=False are excluded. Both conditions of a pair always land in one split.

    python make_experiment_split.py <dataset_root> [...] --out split.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def assign(meta: dict) -> str | None:
    if meta.get("world") == "reception":
        return "test_B"
    bs = meta.get("body_set")
    return {"bA": "train", "bB": "train", "bC": "val", "bD": "test_A", "bH": "test_C"}.get(bs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split = defaultdict(list)
    pair_split, scen_split, excluded = {}, {}, []
    for root in args.roots:
        for d in sorted(glob.glob(os.path.join(root, "episodes", "episode_*"))):
            m = json.load(open(os.path.join(d, "episode_metadata.json")))
            key = f"{os.path.basename(root.rstrip('/'))}/{os.path.basename(d)}"
            if not m.get("valid", True):
                excluded.append(key)
                continue
            s = assign(m)
            if s is None:
                excluded.append(key)
                continue
            split[s].append(key)
            pid, sc = m.get("pair_id"), (m.get("scenario"), m.get("run_seed"))
            for table, k in ((pair_split, pid), (scen_split, sc)):
                if k and table.setdefault(k, s) != s:
                    raise SystemExit(f"leakage: {k} in {table[k]} and {s}")
            if s.startswith("test") and m.get("condition") in ("on", "off"):
                split["test_D"].append(key)
    out = dict(split)
    out["_excluded"] = excluded
    out["_counts"] = {k: len(v) for k, v in split.items()}
    json.dump(out, open(args.out, "w"), indent=2)
    print(out["_counts"], "excluded", len(excluded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
