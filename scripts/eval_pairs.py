#!/usr/bin/env python3
"""Paired semantic challenge: does a model's prediction change between ON and OFF like the label does?

For every held-out pair and every slot present in both conditions (same simulator time, matched
geometry), with D = C_on - C_off (labels) and P = P_on - P_off (predictions), in cost units:

  region R = cells where D != 0 (where the social condition changes the target), optionally
  intersected with the camera FOV
  d_pair      mean |P - D| over R               (paired response fidelity, lower is better)
  true_diff   mean |D| over R
  pred_diff   mean |P| over R
  response    sum(P * sign(D)) / sum(|D|) over R  (1 = full change in the right direction, 0 = no change)
  spill       mean |P| outside R                 (difference predicted where the label has none)

    python eval_pairs.py --pred <experiment>/predictions.npz --split split.json --label native --roots <dataset roots> --out pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.learn.data import LABEL_DIRS, EpisodeIndex, fov_mask  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--set", default="test_D")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split = json.load(open(args.split))
    preds = np.load(args.pred)
    by_root = {os.path.basename(r.rstrip("/")): r for r in args.roots}
    eps = {}
    for key in split[args.set]:
        root, ep = key.split("/")
        eps[key] = EpisodeIndex(os.path.join(by_root[root], "episodes", ep))
    pairs = defaultdict(dict)
    for key, e in eps.items():
        pairs[e.meta["pair_id"]][e.meta["condition"]] = (key, e)
    e0 = next(iter(eps.values()))
    fov = fov_mask(e0.grid, e0.K, e0.T_bc)
    pred_split = {k.split("|")[0] for k in preds.files}
    results, agg = [], defaultdict(lambda: defaultdict(list))
    for pid, c in sorted(pairs.items()):
        if "on" not in c or "off" not in c:
            continue
        (kon, eon), (koff, eoff) = c["on"], c["off"]
        son = {r["slot"]: i for i, r in enumerate(eon.robots)}
        soff = {r["slot"]: i for i, r in enumerate(eoff.robots)}
        rows = defaultdict(list)
        for s in sorted(set(son) & set(soff)):
            i, j = son[s], soff[s]
            pk = [f"{ps}|{kon}|{i}" for ps in pred_split if f"{ps}|{kon}|{i}" in preds.files]
            qk = [f"{ps}|{koff}|{j}" for ps in pred_split if f"{ps}|{koff}|{j}" in preds.files]
            if not pk or not qk:
                continue
            D = np.load(os.path.join(eon.dir, LABEL_DIRS[args.label], f"{i:06d}.npy")).astype(float) - np.load(os.path.join(eoff.dir, LABEL_DIRS[args.label], f"{j:06d}.npy")).astype(float)
            P = preds[pk[0]].astype(float) - preds[qk[0]].astype(float)
            for rname, mask in (("all", np.ones_like(fov)), ("fov", fov)):
                R = (D != 0) & mask
                if not R.any():
                    continue
                rows[f"{rname}_d_pair"].append(float(np.abs(P - D)[R].mean()))
                rows[f"{rname}_true_diff"].append(float(np.abs(D)[R].mean()))
                rows[f"{rname}_pred_diff"].append(float(np.abs(P)[R].mean()))
                rows[f"{rname}_response"].append(float((P[R] * np.sign(D[R])).sum() / np.abs(D[R]).sum()))
                rows[f"{rname}_spill"].append(float(np.abs(P)[(~R) & mask].mean()))
        r = {"pair_id": pid, "family": eon.meta.get("family"), "observability": eon.meta.get("observability"), "slots": len(rows.get("all_d_pair", [])),
             **{k: float(np.mean(v)) for k, v in rows.items()}}
        results.append(r)
        for k, v in rows.items():
            agg[r["family"]][k] += v
            agg["ALL"][k] += v
    summary = {fam: {**{k: float(np.mean(v)) for k, v in d.items()}, "slots": len(d.get("all_d_pair", []))} for fam, d in agg.items()}
    json.dump({"per_pair": results, "by_family": summary}, open(args.out, "w"), indent=2)
    for fam, s in summary.items():
        print(fam, {k: round(v, 3) for k, v in s.items() if k.startswith("fov") or k == "slots"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
