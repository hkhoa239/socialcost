#!/usr/bin/env python3
"""Costmap benchmark split by camera visibility, from saved predictions (no retraining).

Regions of the 200 x 200 robot-frame label grid (same FOV mask as metrics.mae_fov: horizontal camera FOV, 0.3-8 m):
  in_view        inside the camera FOV
  in_view_near   inside the FOV and <= 4 m from the camera
  out_of_view    everything else
Per region, over Test A+B+C (and per split), label units 0..252:
  mae            every cell of the region
  mae_cost       cells where the target has cost (> 0)
  hc_iou         IoU of {pred >= 100} and {target >= 100}
  recall_12      fraction of target cells >= 12 predicted >= 12
  fp_m2          area per sample predicted >= 12 where the target is 0
  mass_ratio     sum pred / sum target over cells where the target has cost (1 = right amount, < 1 under-predicts)

    $PY fov_split_eval.py --root data/costmap_bench --out <json> --report <md>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import LABELS, ROOT, Episode, split_ids  # noqa: E402
from socialcost.learn.data import fov_mask  # noqa: E402

MODELS = {"rgb": "RGB (image)", "rgbd": "RGB-D (image)", "depth": "Depth (image)", "pos": "Position", "pos_u": "Position (prob.)",
          "pos_int": "Position+Interaction", "sngnn_sensor": "SNGNN2D-v2 (sensor graph)", "typemix_vis": "TypeMix, crops (vis)",
          "typemix_cffullraw": "TypeMix + CF (ours, crops)", "typemix_gt": "TypeMix, true types"}
SPLITS = ("test_a", "test_b", "test_c")
HIGH, HAS, CELL = 100, 12, 0.0025
KEYS = ("abs", "n", "c_abs", "c_n", "hi", "hu", "r_hit", "r_n", "fp", "p_mass", "t_mass")


def regions(e0: Episode) -> dict:
    fov = fov_mask(e0.grid, e0.K, e0.T_bc)
    g = e0.grid
    i, j = np.meshgrid(np.arange(g.width), np.arange(g.height))
    x = g.origin_x + (i + 0.5) * g.resolution - e0.T_bc[0, 3]
    y = g.origin_y + (j + 0.5) * g.resolution - e0.T_bc[1, 3]
    near = np.hypot(x, y) <= 4.0
    return {"in_view": fov, "in_view_near": fov & near, "out_of_view": ~fov}


def accumulate(P: np.ndarray, T: np.ndarray, R: dict, acc: dict, n_samples: int) -> None:
    P, T = P.astype(np.float32), T.astype(np.float32)
    E = np.abs(P - T)
    for name, m in R.items():
        a = acc[name]
        Pm, Tm, Em = P[:, m], T[:, m], E[:, m]
        c = Tm > 0
        ph, th = Pm >= HIGH, Tm >= HIGH
        tr = Tm >= HAS
        a["abs"] += Em.sum(); a["n"] += Em.size
        a["c_abs"] += Em[c].sum(); a["c_n"] += c.sum()
        a["hi"] += (ph & th).sum(); a["hu"] += (ph | th).sum()
        a["r_hit"] += (tr & (Pm >= HAS)).sum(); a["r_n"] += tr.sum()
        a["fp"] += ((Pm >= HAS) & (Tm == 0)).sum()
        a["p_mass"] += Pm[c].sum(); a["t_mass"] += Tm[c].sum()
        a["samples"] += n_samples


def finish(a: dict) -> dict:
    d = lambda x, y: float(x / y) if y else None  # noqa: E731
    return {"mae": d(a["abs"], a["n"]), "mae_cost": d(a["c_abs"], a["c_n"]), "hc_iou": d(a["hi"], a["hu"]), "recall_12": d(a["r_hit"], a["r_n"]),
            "fp_m2": d(a["fp"] * CELL, a["samples"]), "mass_ratio": d(a["p_mass"], a["t_mass"]),
            "target_cost_cells_per_sample": d(a["c_n"], a["samples"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/robotics/arena_ws/data/costmap_bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--labels", nargs="+", default=["membership", "native"])
    args = ap.parse_args()
    split_of = {e: sp for sp in SPLITS for e in split_ids(sp)}
    R = regions(Episode(split_ids("test_a")[0]))
    print({k: round(float(v.mean()), 3) for k, v in R.items()}, flush=True)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for label in args.labels:
        tcache: dict = {}
        for model in MODELS:
            for d in sorted(glob.glob(os.path.join(args.root, "runs", "main", f"{model}_{label}_s[0-9]"))):
                run = os.path.basename(d)
                f = os.path.join(d, "predictions.npz")
                if run in res or not os.path.exists(f):
                    continue
                z = np.load(f)
                keys = [k for k in z.files if k.split("|")[0] in split_of]
                acc = {sp: {r: defaultdict(float) for r in R} for sp in SPLITS + ("all",)}
                by = defaultdict(list)
                for k in keys:
                    by[split_of[k.split("|")[0]]].append(k)
                for sp, ks in by.items():
                    for i in range(0, len(ks), 256):
                        ch = ks[i:i + 256]
                        for k in ch:
                            if k not in tcache:
                                e, s = k.split("|")
                                tcache[k] = np.load(os.path.join(ROOT, "episodes", e, LABELS[label], f"{int(s):06d}.npy")).astype(np.uint8)
                        P = np.stack([z[k] for k in ch]); T = np.stack([tcache[k] for k in ch])
                        accumulate(P, T, R, acc[sp], len(ch))
                        accumulate(P, T, R, acc["all"], len(ch))
                res[run] = {"model": model, "label": label, **{sp: {r: finish(acc[sp][r]) for r in R} for sp in acc}}
                print(run, {r: round(res[run]["all"][r]["mae_cost"], 1) for r in R}, flush=True)
                json.dump(res, open(args.out, "w"), indent=1)
    write_report(res, R, args.report)
    return 0


def write_report(res: dict, R: dict, path: str) -> None:
    f = lambda v, p=1: "–" if v is None else f"{v:.{p}f}"  # noqa: E731
    L = ["# Costmap benchmark split by camera visibility", "",
         "Saved test predictions (Test A+B+C, seeds 0-2: mean ± std), re-scored per region of the robot-frame grid.",
         f"In view = camera horizontal FOV, 0.3-8 m ({100 * R['in_view'].mean():.0f} % of the grid); near = in view and <= 4 m "
         f"({100 * R['in_view_near'].mean():.0f} %). Label units 0..252.",
         "mae_cost: error on cells with true cost (lower better). hc_iou: overlap of high-cost (>= 100) cells (higher better). "
         "recall: share of true cost cells >= 12 predicted >= 12. mass: predicted / true cost on cost cells (1 = right amount). "
         "FP m²: area per frame predicted >= 12 where there is no cost.", ""]
    for label in ("membership", "native"):
        runs = defaultdict(list)
        for r in res.values():
            if r["label"] == label:
                runs[r["model"]].append(r)
        if not runs:
            continue
        for reg in ("in_view", "in_view_near", "out_of_view"):
            L += [f"## {label} label, {reg.replace('_', ' ')}", "",
                  "| Model | mae_cost | hc_iou | recall_12 | mass | FP m² | mae (all cells) |", "|---|---|---|---|---|---|---|"]
            order = sorted(runs, key=lambda m: np.mean([x["all"][reg]["mae_cost"] for x in runs[m]]))
            for m in order:
                def ms(k, p=1, scale=1.0):
                    v = [x["all"][reg][k] * scale for x in runs[m] if x["all"][reg][k] is not None]
                    return f"{np.mean(v):.{p}f} ± {np.std(v):.{p}f}" if v else "–"
                L.append(f"| {MODELS[m]} ({len(runs[m])}) | {ms('mae_cost')} | {ms('hc_iou', 3)} | {ms('recall_12', 2)} | {ms('mass_ratio', 2)} | "
                         f"{ms('fp_m2', 2)} | {ms('mae', 2)} |")
            L.append("")
        L += [f"## {label} label, in view, per split (mae_cost)", "", "| Model | Test A | Test B | Test C |", "|---|---|---|---|"]
        for m in runs:
            L.append(f"| {MODELS[m]} | " + " | ".join(f(np.mean([x[sp]['in_view']['mae_cost'] for x in runs[m]])) for sp in SPLITS) + " |")
        L.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
