#!/usr/bin/env python3
"""Phase A: stage-by-stage consistency of the estimator's intermediate grids (estimator_node --dump-dir).

Stages per dumped tick
  robot_raw  (label grid, base frame, 0.05 m, 200 x 200)  network output / oracle paint
  robot_post (same grid)                                   after --post (if any)
  fov-masked (same grid)                                   what the model is allowed to publish
  cost       (costmap frame, costmap lattice, 0.10 m)      resampled at the input stamp's pose
  occ        (OccupancyGrid int8)                           round(cost * 100 / 254), max 99
Checks
  - fov masking: published robot grid == where(fov, post, 0)
  - resample: an independent re-implementation (cv2.warpAffine, nearest, inverse map) reproduces `cost`
    exactly; and the information lost by nearest sampling 0.05 -> 0.10 m vs a 2x2 max-pool
  - encoding: |occ * 254 // 100 - cost| <= 2 (3 at clipping), clipped cells
  - value ranges and frames/origins/resolution per stage
  - shadow oracle (learned arms): agreement in the robot grid inside the FOV

    python stage_consistency.py <arm dir> [<arm dir> ...] --out stage_report.json --figdir figs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np


def resample_indep(robot: np.ndarray, pose_f: np.ndarray, origin: np.ndarray, res: float, n: int) -> np.ndarray:
    """Target cell (i, j) centre -> robot frame -> label cell, written as one affine map for cv2.remap."""
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    wx = origin[0] + (i + 0.5) * res - pose_f[0]
    wy = origin[1] + (j + 0.5) * res - pose_f[1]
    c, s = math.cos(pose_f[2]), math.sin(pose_f[2])
    lx, ly = c * wx + s * wy, -s * wx + c * wy
    u = ((lx + 5.0) / 0.05).astype(np.float32)  # continuous label-grid coordinates, cell k spans [k, k+1)
    v = ((ly + 5.0) / 0.05).astype(np.float32)
    # floor semantics: remap with INTER_NEAREST rounds to nearest pixel centre; shift by -0.5 to get floor
    out = cv2.remap(robot, u - 0.5, v - 0.5, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def resample_maxpool(robot: np.ndarray, pose_f, origin, res, n) -> np.ndarray:
    """Same lattice, but each target cell takes the max of the label cells whose centres fall in it (no aliasing)."""
    out = np.zeros((n, n), np.uint8)
    jj, ii = np.nonzero(robot)
    if not len(jj):
        return out
    lx = -5.0 + (ii + 0.5) * 0.05
    ly = -5.0 + (jj + 0.5) * 0.05
    c, s = math.cos(pose_f[2]), math.sin(pose_f[2])
    wx = pose_f[0] + c * lx - s * ly
    wy = pose_f[1] + s * lx + c * ly
    ti = np.floor((wx - origin[0]) / res).astype(int)
    tj = np.floor((wy - origin[1]) / res).astype(int)
    ok = (ti >= 0) & (tj >= 0) & (ti < n) & (tj < n)
    np.maximum.at(out, (tj[ok], ti[ok]), robot[jj[ok], ii[ok]])
    return out


def arm_report(d: str, figdir: str | None) -> dict:
    files = sorted(glob.glob(os.path.join(d, "dumps", "tick_*.npz")))
    R = {"ticks": len(files), "fov_mask_ok": 0, "resample_exact": 0, "resample_mismatch_cells_max": 0, "encode_err_max": 0,
         "clipped_max": 0, "nearest_vs_maxpool": [], "frames": set(), "res": set(), "origin_on_lattice": 0, "shadow": []}
    for k, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        cost, occ, fov = z["cost"], z["occ"], z["fov"]
        R["frames"].add(str(z["frame"]))
        R["res"].add(float(z["res"]))
        res, origin, pose_f = float(z["res"]), z["origin"], z["pose_f"]
        R["origin_on_lattice"] += int(np.allclose(origin / res, np.round(origin / res), atol=1e-6))
        enc = (occ.astype(np.int32) * 254) // 100
        R["encode_err_max"] = max(R["encode_err_max"], int(np.abs(enc - cost.astype(np.int32)).max()))
        R["clipped_max"] = max(R["clipped_max"], int((occ >= 99).sum()))
        if "robot_raw" in z.files and fov.ndim == 2:
            post = z["robot_post"] if "robot_post" in z.files else z["robot_raw"]
            pub = np.where(fov, post, 0).astype(np.uint8)
            R["fov_mask_ok"] += 1
            n = cost.shape[0]
            ind = resample_indep(pub, pose_f, origin, res, n)
            mism = int((ind != cost).sum())
            R["resample_exact"] += int(mism == 0)
            R["resample_mismatch_cells_max"] = max(R["resample_mismatch_cells_max"], mism)
            mp = resample_maxpool(pub, pose_f, origin, res, n)
            hi = mp >= 100
            R["nearest_vs_maxpool"].append({"maxpool_hi_cells": int(hi.sum()), "nearest_hi_cells": int((cost >= 100).sum()),
                                            "mean_abs_diff_on_maxpool_nonzero": float(np.abs(mp.astype(int) - cost.astype(int))[mp > 0].mean()) if (mp > 0).any() else 0.0,
                                            "peak_loss": int(mp.max()) - int(cost.max())})
            if "shadow_robot" in z.files:
                sh = z["shadow_robot"].astype(np.int16)
                a, b = post.astype(np.int16)[fov], sh[fov]
                R["shadow"].append({"mae_fov": float(np.abs(a - b).mean()), "pred_nz": int((a > 0).sum()), "lab_nz": int((b > 0).sum()),
                                    "pred_ge50": int((a >= 50).sum()), "lab_ge50": int((b >= 50).sum()), "both_ge50": int(((a >= 50) & (b >= 50)).sum())})
            if figdir and k in (len(files) // 4, len(files) // 2, 3 * len(files) // 4):
                figure(z, pub, mp, os.path.join(figdir, f"{os.path.basename(d)}_tick{k:03d}.png"))
    out = {k: (sorted(v) if isinstance(v, set) else v) for k, v in R.items() if k not in ("nearest_vs_maxpool", "shadow")}
    nv = R["nearest_vs_maxpool"]
    if nv:
        out["nearest_vs_maxpool"] = {"hi_cells_maxpool_mean": float(np.mean([x["maxpool_hi_cells"] for x in nv])),
                                     "hi_cells_nearest_mean": float(np.mean([x["nearest_hi_cells"] for x in nv])),
                                     "mean_abs_diff": float(np.mean([x["mean_abs_diff_on_maxpool_nonzero"] for x in nv])),
                                     "peak_loss_mean": float(np.mean([x["peak_loss"] for x in nv]))}
    if R["shadow"]:
        S = R["shadow"]
        out["shadow_robot_grid"] = {"ticks": len(S), "mae_fov": float(np.mean([s["mae_fov"] for s in S])),
                                    "pred_nonzero_over_label_nonzero": float(np.sum([s["pred_nz"] for s in S]) / max(np.sum([s["lab_nz"] for s in S]), 1)),
                                    "precision@50": float(np.sum([s["both_ge50"] for s in S]) / max(np.sum([s["pred_ge50"] for s in S]), 1)),
                                    "recall@50": float(np.sum([s["both_ge50"] for s in S]) / max(np.sum([s["lab_ge50"] for s in S]), 1))}
    return out


def figure(z, pub, mp, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("robot_raw", z["robot_raw"])]
    if "robot_post" in z.files:
        panels.append(("robot_post", z["robot_post"]))
    panels += [("fov-masked (published, robot grid)", pub)]
    if "shadow_robot" in z.files:
        panels.append(("shadow oracle_ext (robot grid)", np.where(z["fov"], z["shadow_robot"], 0)))
    panels += [("resampled cost (costmap lattice)", z["cost"]), ("maxpool resample (reference)", mp), ("occ*254/100 (what Nav2 reads)", (z["occ"].astype(int) * 254) // 100)]
    fig, ax = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.1))
    for a, (t, im) in zip(ax, panels):
        if im.shape[0] == 200:  # robot grid: x forward up
            a.imshow(np.rot90(im, 1), vmin=0, vmax=252, cmap="magma")
        else:
            a.imshow(im, vmin=0, vmax=252, cmap="magma", origin="lower")
        a.set_title(t, fontsize=7)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{os.path.basename(os.path.dirname(os.path.dirname(path)))} stamp {int(z['stamp_ns']) / 1e9:.1f}s", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()
    rep = {os.path.basename(a.rstrip("/")): arm_report(a, args.figdir) for a in args.arms}
    json.dump(rep, open(args.out, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
