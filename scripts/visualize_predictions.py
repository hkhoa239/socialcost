#!/usr/bin/env python3
"""Ground truth vs. predicted social costmaps, with the RGB-D input and a bird's-eye view of the scene.

Per sample, two rows (all costmaps in the robot frame, x forward up, y left left, 10 m x 10 m):
  row 1  RGB (pedestrians projected) | depth (m) | bird's-eye scene (robot, FOV, pedestrians,
         headings, interaction links) | ground-truth label
  row 2  one predicted costmap per model, each titled with its MAE inside the camera FOV;
         the dashed wedge is the camera FOV (the region the image models can observe)
  row 3  prediction - ground truth for each model (blue: too low, red: too high); learned models are
         shown inside the camera FOV only (grey outside): they are trained there only (--raw shows all)

Outputs <output>/<episode>_<family>_<condition>/sample_NNNNNN.png and <...>.mp4, plus index.md.

    python visualize_predictions.py --experiments <dir> --split split.json --roots <dataset roots> \
        --models pilot_rgbd_extended pilot_pos_extended rule_extended --label extended --output <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from socialcost.learn.data import LABEL_DIRS, EpisodeIndex, fov_mask  # noqa: E402
from socialcost.qa import pose_matrix  # noqa: E402

_spec = importlib.util.spec_from_file_location("viz", os.path.join(HERE, "visualize_dataset.py"))
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)

RAW = False
PRETTY = {"pilot_rgbd_extended": "RGB-D", "pilot_depth_extended": "depth only", "pilot_rgb_extended": "RGB only",
          "pilot_pos_extended": "position (all peds)", "pilot_pos_fov_extended": "position (FOV peds)", "rule_extended": "personal-space rule",
          "oracle_native_extended": "Arena-native oracle", "pilot_rgbd_native": "RGB-D", "pilot_pos_native": "position (all peds)",
          "rule_native": "personal-space rule"}


def heat(ax, arr, ext, title, cmap="magma", vmin=0, vmax=252):
    im = ax.imshow(arr.T, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.plot(0, 0, marker=(3, 0, 0), color="lime", ms=7)
    return im


def fov_wedge(ax, hfov: float, reach: float, color="w") -> None:
    for sgn in (-1, 1):
        ax.plot([0, reach * math.sin(sgn * hfov / 2)], [0, reach * math.cos(hfov / 2)], "--", color=color, lw=0.6)


def draw(fig, e: EpisodeIndex, k: int, label: str, preds: dict[str, np.ndarray], fov: np.ndarray) -> None:
    fig.clf()
    n = max(len(preds), 4)
    axes = fig.subplots(3, n, gridspec_kw={"wspace": 0.3, "hspace": 0.35})
    for ax in axes.ravel():
        ax.set_visible(False)
    rb = e.robots[k]
    pose = rb["pose_map_base"]
    peds = e.peds[k]["pedestrians"]
    gt = np.load(os.path.join(e.dir, LABEL_DIRS[label], f"{k:06d}.npy"))
    ext = viz.label_extent(e.grid)
    W, H = e.cam["width"], e.cam["height"]
    hfov = 2 * math.atan(W / (2 * e.K[0, 0]))
    fig.suptitle(f"{e.key}  {e.meta.get('scenario')}  family {e.meta.get('family')}  condition {e.meta.get('condition') or '-'}  "
                 f"sample {k}  sim t = {rb['stamp_ns'] / 1e9:.2f} s   label: {label}", fontsize=10)

    # row 1: inputs, scene, ground truth
    ax = axes[0, 0]
    ax.set_visible(True)
    ax.imshow(cv2.imread(os.path.join(e.dir, "rgb", f"{k:06d}.png"))[:, :, ::-1])
    T = np.linalg.inv(pose_matrix(rb["pose_map_camera"]))
    for p in peds:
        pc = T @ np.array([p["x"], p["y"], 1.0, 1.0])
        if pc[2] > 0.3:
            u, v = (e.K @ pc[:3])[:2] / pc[2]
            if 0 <= u < W:
                ax.plot(u, v, "o", ms=6, mfc="none", mec=viz.color(p["interaction_id"]), mew=1.5)
    ax.set_title("RGB input", fontsize=8)
    ax.axis("off")
    ax = axes[0, 1]
    ax.set_visible(True)
    d = np.load(os.path.join(e.dir, "depth", f"{k:06d}.npy"))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.65")
    im = ax.imshow(np.ma.masked_invalid(d), cmap=cmap, vmin=0, vmax=12)
    cb = plt.colorbar(im, ax=ax, fraction=0.035)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("m", fontsize=7)
    ax.set_title("depth input (grey: no return)", fontsize=8)
    ax.axis("off")

    ax = axes[0, 2]
    ax.set_visible(True)
    ax.set_facecolor("0.93")
    local = {}
    for p in peds:
        lx, ly = viz.map_to_base(pose, p["x"], p["y"])
        local[p["name"]] = (lx, ly)
        col = viz.color(p["interaction_id"]) if p["interaction_id"] >= 0 else "0.25"
        hy = p["yaw"] - pose["yaw"]
        ax.add_patch(plt.Circle((ly, lx), 0.25, fc=col, ec="k", lw=0.5, alpha=0.8))
        ax.arrow(ly, lx, 0.6 * math.sin(hy), 0.6 * math.cos(hy), head_width=0.18, color="k", lw=0.8)
        tags = [g.get("slot") for g in p.get("gestures") or [] if isinstance(g, dict) and str(g.get("slot", "")).startswith("halt")]
        if tags:
            ax.text(ly, lx + 0.45, "stop hand", fontsize=6, ha="center", color="crimson")
        if "wheelchair" in os.path.basename(str(p.get("model_uri", ""))).lower():
            ax.text(ly, lx - 0.6, "wheelchair", fontsize=6, ha="center", color="navy")
    groups: dict[int, list] = {}
    for p in peds:
        if p["interaction_id"] >= 0:
            groups.setdefault(p["interaction_id"], []).append(p)
    for iid, mem in groups.items():
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                (ax_, ay_), (bx_, by_) = local[mem[a]["name"]], local[mem[b]["name"]]
                ax.plot([ay_, by_], [ax_, bx_], "-", color=viz.color(iid), lw=1.5)
        cx = np.mean([local[m["name"]][0] for m in mem])
        cy = np.mean([local[m["name"]][1] for m in mem])
        ax.text(cy, cx + 0.7, viz.INTERACTION_NAMES.get(mem[0]["interaction_type"], ""), fontsize=6, ha="center", color=viz.color(iid))
    ax.plot(0, 0, marker=(3, 0, 0), color="green", ms=10)
    fov_wedge(ax, hfov, ext[3], color="green")
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_title("bird's-eye scene, robot frame\n(colour = interaction, arrow = heading)", fontsize=8)
    ax.set_xlabel("y left [m]", fontsize=7)
    ax.set_ylabel("x forward [m]", fontsize=7)
    ax.tick_params(labelsize=6)

    ax = axes[0, 3]
    ax.set_visible(True)
    im = heat(ax, gt, ext, f"ground truth ({label})\nmax {gt.max()}")
    fov_wedge(ax, hfov, ext[3])
    plt.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=6)

    # rows 2-3: predictions and errors. Learned models are trained on the camera FOV only: outside it
    # their output is untrained, so it is masked (as the closed-loop estimator does) unless --raw.
    for c, (name, p) in enumerate(preds.items()):
        learned = name.startswith("pilot_")
        err = p.astype(float) - gt.astype(float)
        mae_fov = float(np.abs(err)[fov].mean())
        if learned and not RAW:
            p = np.where(fov, p, 0)
            err = np.where(fov, err, np.nan)
        ax = axes[1, c]
        ax.set_visible(True)
        heat(ax, p, ext, f"{PRETTY.get(name, name)}\nMAE in FOV {mae_fov:.1f}" + ("  (FOV only)" if learned and not RAW else ""))
        fov_wedge(ax, hfov, ext[3])
        ax = axes[2, c]
        ax.set_visible(True)
        cm = plt.get_cmap("coolwarm").copy()
        cm.set_bad("0.75")
        im = heat(ax, np.ma.masked_invalid(err), ext, f"{PRETTY.get(name, name)} - truth", cmap=cm, vmin=-150, vmax=150)
        fov_wedge(ax, hfov, ext[3], color="k")
        if c == len(preds) - 1:
            plt.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--label", default="extended")
    ap.add_argument("--sets", nargs="+", default=["test_A", "test_B", "test_C"])
    ap.add_argument("--episode", action="append", default=[], help="restrict to these episode keys (root/episode_NNNNNN)")
    ap.add_argument("--snapshots", type=int, default=5)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw", action="store_true", help="show learned predictions outside the camera FOV too (untrained there)")
    args = ap.parse_args()
    global RAW
    RAW = args.raw
    split = json.load(open(args.split))
    by_root = {os.path.basename(r.rstrip("/")): r for r in args.roots}
    # load every prediction now: the training chain may rewrite these files later
    preds = {}
    for m in args.models:
        z = np.load(os.path.join(args.experiments, m, "predictions.npz"))
        preds[m] = {f: z[f] for f in z.files}
    os.makedirs(args.output, exist_ok=True)
    index = ["# Prediction visualizations", "", f"label: {args.label}; models: {', '.join(PRETTY.get(m, m) for m in args.models)}", "",
             "| episode | split | family | condition | snapshots | video |", "|---|---|---|---|---|---|"]
    fig = plt.figure(figsize=(4.2 * max(len(args.models), 4), 12))
    for s in args.sets:
        for key in split.get(s, []):
            if args.episode and key not in args.episode:
                continue
            root, ep = key.split("/")
            e = EpisodeIndex(os.path.join(by_root[root], "episodes", ep))
            n = len(e.robots)
            have = [m for m in args.models if f"{s}|{key}|0" in preds[m]]
            if not have or n == 0:
                continue
            fov = fov_mask(e.grid, e.K, e.T_bc)
            tag = f"{ep}_{e.meta.get('family')}_{e.meta.get('condition') or 'na'}_{s}"
            out = os.path.join(args.output, tag)
            os.makedirs(out, exist_ok=True)
            picks = sorted({int(round(x)) for x in np.linspace(0, n - 1, args.snapshots)})
            writer = None
            for k in range(n):
                if args.no_video and k not in picks:
                    continue
                pk = {m: preds[m][f"{s}|{key}|{k}"] for m in have if f"{s}|{key}|{k}" in preds[m]}
                draw(fig, e, k, args.label, pk, fov)
                fig.canvas.draw()
                img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                if k in picks:
                    cv2.imwrite(os.path.join(out, f"sample_{k:06d}.png"), img[:, :, ::-1])
                if not args.no_video:
                    if writer is None:
                        writer = cv2.VideoWriter(out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (img.shape[1], img.shape[0]))
                    writer.write(np.ascontiguousarray(img[:, :, ::-1]))
            if writer is not None:
                writer.release()
            index.append(f"| {key} | {s} | {e.meta.get('family')} | {e.meta.get('condition') or '-'} | [{tag}/]({tag}/) | "
                         f"{'-' if args.no_video else f'[{tag}.mp4]({tag}.mp4)'} |")
            print(tag, flush=True)
    open(os.path.join(args.output, "index.md"), "w").write("\n".join(index) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
