#!/usr/bin/env python3
"""Fit the publication threshold of a probabilistic costmap model on the validation split (never on test).

The labels have no cost between 0 and their floor (12 = cost_bias x 252): every labelled cell is >= 12. A sigmoid mean
leaves a diffuse low-cost tail around every prediction; in closed loop that tail is a wide shallow field the local
controller has to cross. The threshold zeroes cells below tau and is chosen on validation by the IoU of the non-zero
region (pred >= tau vs label > 0); MAE and false-positive area are reported alongside.

    python fit_cost_threshold.py --model typemix_cffullraw --label extended --out <spec.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from bench_train import build_model, variant  # noqa: E402
from socialcost.bench.data import Dv1Samples, collate, split_ids  # noqa: E402

BENCH = "/home/robotics/arena_ws/data/costmap_bench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="typemix_cffullraw")
    ap.add_argument("--label", default="extended")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    models = []
    for s in args.seeds.split(","):
        m = build_model(args.model)
        m.load_state_dict(torch.load(os.path.join(BENCH, "runs", "main", f"{args.model}_{args.label}_s{s}", "best.pt"), map_location="cpu")["model"])
        models.append(m.to(dev).eval())
    ds = Dv1Samples(split_ids("val"), args.label, variant(args.model), stride=args.stride)
    dl = DataLoader(ds, batch_size=16, num_workers=6, collate_fn=collate)
    taus = np.arange(0, 41, 2)
    inter = np.zeros(len(taus))
    union = np.zeros(len(taus))
    abs_err = np.zeros(len(taus))
    fp = np.zeros(len(taus))
    n_cells = 0
    with torch.no_grad():
        for b in dl:
            x = {k: v.to(dev) for k, v in b["x"].items()}
            mu = torch.stack([torch.sigmoid(m(x)[:, 0].float()) for m in models]).mean(0).cpu().numpy() * 252
            t = b["target"].numpy() * 252
            lab = t > 0
            for i, tau in enumerate(taus):
                p = np.where(mu >= tau, mu, 0)
                pm = p > 0
                inter[i] += (pm & lab).sum()
                union[i] += (pm | lab).sum()
                abs_err[i] += np.abs(p - t).sum()
                fp[i] += (pm & ~lab).sum()
            n_cells += t.size
    res = {"tau": taus.tolist(), "iou_nonzero": (inter / union).tolist(), "mae": (abs_err / n_cells).tolist(),
           "fp_cells_per_sample": (fp / (n_cells / (200 * 200))).tolist()}
    best = int(np.argmax(res["iou_nonzero"]))
    spec = {"model": args.model, "label": args.label, "seeds": args.seeds, "split": "val", "threshold": int(taus[best]),
            "criterion": "max IoU of the non-zero region on validation", "curve": res}
    json.dump(spec, open(args.out, "w"), indent=1)
    for i in range(len(taus)):
        print(f"tau {taus[i]:3d}  IoU {res['iou_nonzero'][i]:.3f}  MAE {res['mae'][i]:.2f}  FP cells {res['fp_cells_per_sample'][i]:.0f}")
    print("best", spec["threshold"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
