#!/usr/bin/env python3
"""Non-learned baselines, scored like a trained model (same metrics, same predictions.npz format).

  rule             personal-space term only, from the people's positions (Arena's own function,
                   no interaction or attribute information) - the position-only rule baseline
  oracle_native    the exact arena-native label from privileged simulator state
  oracle_extended  the exact extended label (zero error against itself; the upper reference)

    python eval_baselines.py --split split.json --roots <dataset roots> --label extended --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.learn.data import LABEL_DIRS, EpisodeIndex, fov_mask  # noqa: E402
from socialcost.learn.metrics import Accumulator  # noqa: E402
from socialcost.labels import Ped, paint  # noqa: E402


def predict(e: EpisodeIndex, k: int, method: str) -> np.ndarray:
    if method == "rule":
        pose = e.robots[k]["pose_map_base"]
        return paint(e.params, [Ped(p["x"], p["y"], p["animation_state"]) for p in e.peds[k]["pedestrians"]], e.grid, (pose["x"], pose["y"], pose["yaw"]))
    return np.load(os.path.join(e.dir, LABEL_DIRS["native" if method == "oracle_native" else "extended"], f"{k:06d}.npy"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split = json.load(open(args.split))
    by_root = {os.path.basename(r.rstrip("/")): r for r in args.roots}
    for method in ("rule", "oracle_native", "oracle_extended"):
        out = os.path.join(args.out, f"{method}_{args.label}")
        os.makedirs(out, exist_ok=True)
        metrics, preds = {}, {}
        for name, keys in split.items():
            if name.startswith("_") or name == "train":
                continue
            acc = Accumulator()
            for key in keys:
                root, ep = key.split("/")
                e = EpisodeIndex(os.path.join(by_root[root], "episodes", ep))
                fov = torch.from_numpy(fov_mask(e.grid, e.K, e.T_bc))[None]
                for k in range(len(e.robots)):
                    p = predict(e, k, method)
                    t = np.load(os.path.join(e.dir, LABEL_DIRS[args.label], f"{k:06d}.npy"))
                    r = predict(e, k, "rule")
                    tt = lambda a: torch.from_numpy(a.astype(np.float32) / 252.0)[None]  # noqa: E731
                    fam = e.meta.get("family") or e.meta.get("scenario")
                    for g in ("overall", f"family:{fam}"):
                        acc.add(g, tt(p), tt(t), tt(r), fov)
                    preds[f"{name}|{key}|{k}"] = p.astype(np.uint8)
            metrics[name] = acc.result()
        json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
        json.dump({"name": f"{method}_{args.label}", "method": method, "label": args.label, "params": 0,
                   "note": "non-learned; oracle methods read privileged simulator state"}, open(os.path.join(out, "run.json"), "w"), indent=2)
        np.savez_compressed(os.path.join(out, "predictions.npz"), **preds)
        print(method, {k: {kk: round(vv, 2) for kk, vv in v["overall"].items() if kk in ("mae_nonzero_fov", "mae_social_extra_fov", "iou_nonzero_fov") and vv is not None} for k, v in metrics.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
