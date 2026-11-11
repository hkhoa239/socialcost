#!/usr/bin/env python3
"""Seven-panel semantic view of recorded samples, from stored data only:

  1 RGB   2 pedestrians (positions, headings, published vs. completed membership)
  3 personal-space cost only   4 native interaction cost only   5 arena-native label (stored)
  6 extended label (stored)    7 extended - native

Panels 3-7 share the label grid (base_link frame). Panels 3-4 are painted with the layer's own
functions (personal-space clip 0 for 4); 5 = max(3, 4) exactly.

    python visualize_semantics.py --dataset <root> [--episode N] --output <dir> [--snapshots 6] [--no-video]
"""

from __future__ import annotations

import argparse
import importlib.util
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

from socialcost import semantics as S  # noqa: E402
from socialcost import verify as V  # noqa: E402
from socialcost.dataset import peds_for_layer  # noqa: E402

_spec = importlib.util.spec_from_file_location("viz", os.path.join(HERE, "visualize_dataset.py"))
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)


def heat(ax, arr, ext, vmax, title, cmap="magma", vmin=0):
    im = ax.imshow(arr.T, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.plot(0, 0, marker=(3, 0, 0), color="lime", ms=7)
    return im


def draw(fig, ep: V.Ep, k: int) -> None:
    fig.clf()
    axes = fig.subplots(1, 7, gridspec_kw={"width_ratios": [1.3, 1, 1, 1, 1, 1, 1], "wspace": 0.25})
    rb = ep.robots[k]
    pose = ep.pose(k)
    published = ep.states(k)
    completed = ep.states(k, membership=True)
    vmax = ep.params.max_cost_value
    ext = viz.label_extent(ep.grid)
    comp = S.components(ep.params, peds_for_layer(published), ep.grid, pose)
    native, extended = ep.label(k, S.NATIVE), ep.label(k, S.EXTENDED)
    fig.suptitle(f"{ep.scenario}  seed {ep.meta.get('run_seed')}  {ep.name}  sample {k}  sim t = {rb['stamp_ns'] / 1e9:.2f} s  "
                 f"(labels: social_cost = arena-native/1.0, social_cost_membership = extended-membership/1.0)", fontsize=10)

    ax = axes[0]
    ax.imshow(cv2.imread(os.path.join(ep.dir, "rgb", f"{k:06d}.png"))[:, :, ::-1])
    ax.set_title("1 RGB", fontsize=8)
    ax.axis("off")

    ax = axes[1]
    ax.set_facecolor("0.12")
    pub = {p["name"]: p["interaction_id"] for p in published}
    for p in completed:
        lx, ly = viz.map_to_base({"x": pose[0], "y": pose[1], "yaw": pose[2]}, p["x"], p["y"])
        hy = p["yaw"] - pose[2]
        col = viz.color(p["interaction_id"])
        queued_only = pub.get(p["name"], -1) < 0 <= p["interaction_id"]
        ax.plot(ly, lx, "s" if queued_only else "o", color=col, ms=5, mfc="none" if queued_only else col)
        ax.arrow(ly, lx, 0.45 * math.sin(hy), 0.45 * math.cos(hy), head_width=0.15, color=col, lw=1)
        if p["interaction_id"] >= 0:
            ax.text(ly, lx - 0.5, f"{viz.INTERACTION_NAMES.get(p['interaction_type'], p['interaction_type'])}#{p['interaction_id']}{' (queued)' if queued_only else ''}",
                    color=col, fontsize=5, ha="center")
    ax.plot(0, 0, marker=(3, 0, 0), color="lime", ms=7)
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_title("2 pedestrians, base_link\n(filled: published id, hollow: queued)", fontsize=8)
    ax.tick_params(labelsize=6)

    heat(axes[2], comp["personal"], ext, vmax, f"3 personal space only\nnonzero {int((comp['personal'] > 0).sum())}")
    heat(axes[3], comp["interaction"], ext, vmax, f"4 native interaction only\nnonzero {int((comp['interaction'] > 0).sum())}")
    heat(axes[4], native, ext, vmax, f"5 arena-native (stored)\nmax {native.max()}")
    heat(axes[5], extended, ext, vmax, f"6 extended (stored)\nmax {extended.max()}")
    d = extended.astype(int) - native.astype(int)
    im = heat(axes[6], d, ext, max(1, int(np.abs(d).max())), f"7 extended - native\n{int((d != 0).sum())} cells, min {d.min()}", cmap="coolwarm", vmin=-max(1, int(np.abs(d).max())))
    plt.colorbar(im, ax=axes[6], fraction=0.045).ax.tick_params(labelsize=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--episode", action="append", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--snapshots", type=int, default=6)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    fig = plt.figure(figsize=(30, 4.6))
    for ep in V.load_episodes(*args.dataset):
        if args.episode and not any(ep.name.endswith(e) or ep.name.endswith(f"episode_{int(e):06d}" if e.isdigit() else e) for e in args.episode):
            continue
        n = len(ep.robots)
        tag = f"{ep.name.replace('/', '_')}_{ep.scenario}"
        os.makedirs(os.path.join(args.output, tag), exist_ok=True)
        picks = sorted({int(round(x)) for x in np.linspace(0, n - 1, args.snapshots)})
        writer = None
        for k in range(n):
            if args.no_video and k not in picks:
                continue
            draw(fig, ep, k)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if k in picks:
                cv2.imwrite(os.path.join(args.output, tag, f"sample_{k:06d}.png"), img[:, :, ::-1])
            if not args.no_video:
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(args.output, tag + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), ep.meta.get("sample_hz", 5.0), (img.shape[1], img.shape[0]))
                writer.write(np.ascontiguousarray(img[:, :, ::-1]))
        if writer is not None:
            writer.release()
        print(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
