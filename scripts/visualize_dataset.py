#!/usr/bin/env python3
"""Offline visualization of recorded SOCIALCOST episodes, from the stored data only.

Per sample, four panels:
  1. RGB, pedestrians projected through the recorded intrinsics and T_map_camera
  2. depth in metres (NaN = invalid, grey)
  3. ground-truth social cost in the label grid (robot frame, x forward up, y left left)
  4. top-down overlay in the same robot frame: the label, robot pose and camera FOV, pedestrians
     transformed map -> base_link with the recorded pose at the image stamp, headings,
     interaction ids, personal-space circles and interaction-region (capsule) boundaries

Per episode: PNG snapshots, an MP4 of every sample, a temporal plot of total social cost and
active interaction count.

    python visualize_dataset.py --dataset <root> [--episode episode_000001] --output <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.dataset import interaction_frame_at, read_jsonl, with_membership  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402
from socialcost.qa import episodes, pose_matrix  # noqa: E402

INTERACTION_NAMES = {0: "TALK_TO", 1: "GROUP_CONV", 3: "SIT_ON", 4: "LIE_ON", 5: "USE", 6: "QUEUE_USE", 7: "WAVE_AT",
                     8: "BLOCK", 9: "SERVICE", 10: "HUG", 11: "SHAKE_HAND", 12: "GROUP_WALK"}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]


def color(iid: int) -> str:
    return COLORS[iid % len(COLORS)] if iid >= 0 else "white"


def map_to_base(pose: dict, x: float, y: float) -> tuple[float, float]:
    """Point in map -> base_link (planar), using the recorded base pose at the image stamp."""
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    dx, dy = x - pose["x"], y - pose["y"]
    return c * dx + s * dy, -s * dx + c * dy


class Episode:
    def __init__(self, ep_dir: str, params: LayerParams, grid: GridSpec, channel: str) -> None:
        self.dir = ep_dir
        self.meta = json.load(open(os.path.join(ep_dir, "episode_metadata.json")))
        self.cam = json.load(open(os.path.join(ep_dir, "camera_info.json")))
        self.robots = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))
        self.peds = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))
        self.frames = read_jsonl(os.path.join(ep_dir, "interactions.jsonl"))
        self.params, self.grid, self.channel = params, grid, channel
        self.K = np.array(self.cam["K"]).reshape(3, 3)
        self.t0 = self.robots[0]["stamp_ns"] if self.robots else 0

    def label(self, k: int) -> np.ndarray:
        return np.load(os.path.join(self.dir, self.channel, f"{k:06d}.npy"))

    def states(self, k: int) -> list[dict]:
        """Pedestrians as the label saw them: membership-completed for that channel."""
        ps = self.peds[k]["pedestrians"]
        if self.channel == "social_cost_membership":
            ps = with_membership(ps, interaction_frame_at(self.frames, self.robots[k]["stamp_ns"]))
        return ps


def label_extent(grid: GridSpec) -> tuple[float, float, float, float]:
    # imshow(label.T): rows run along x, columns along y; column 0 is y = origin_y
    return (grid.origin_y, grid.origin_y + grid.height * grid.resolution, grid.origin_x, grid.origin_x + grid.width * grid.resolution)


def draw(fig, ep: Episode, k: int) -> None:
    fig.clf()
    axes = fig.subplots(1, 4, gridspec_kw={"width_ratios": [1.25, 1.25, 1, 1], "wspace": 0.35})
    rb, peds = ep.robots[k], ep.states(k)
    pose = rb["pose_map_base"]
    W, H = ep.cam["width"], ep.cam["height"]
    hfov = 2 * math.atan(W / (2 * ep.K[0, 0]))
    t_rel = (rb["stamp_ns"] - ep.t0) / 1e9
    fig.suptitle(f"{ep.meta.get('scenario')}  seed {ep.meta.get('run_seed')}  {os.path.basename(ep.dir)}  sample {k}  "
                 f"sim t = {rb['stamp_ns'] / 1e9:.2f} s (+{t_rel:.1f} s)  label: {ep.channel}", fontsize=10)

    # 1. RGB
    ax = axes[0]
    rgb = cv2.imread(os.path.join(ep.dir, "rgb", f"{k:06d}.png"))[:, :, ::-1]
    ax.imshow(rgb)
    T_cm = np.linalg.inv(pose_matrix(rb["pose_map_camera"]))
    proj = {}
    for p in peds:
        pc = T_cm @ np.array([p["x"], p["y"], 1.0, 1.0])
        if pc[2] > 0.3:
            u, v = (ep.K @ pc[:3])[:2] / pc[2]
            if -40 < u < W + 40:
                proj[p["name"]] = (u, v, p)
                ax.plot(u, v, "o", ms=6, mfc="none", mec=color(p["interaction_id"]), mew=1.5)
    for a in proj.values():
        for b in proj.values():
            if a[2]["name"] < b[2]["name"] and a[2]["interaction_id"] >= 0 and a[2]["interaction_id"] == b[2]["interaction_id"]:
                ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=color(a[2]["interaction_id"]), lw=1.2)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("RGB (rings: pedestrians at 1 m height)", fontsize=8)
    ax.axis("off")

    # 2. depth
    ax = axes[1]
    d = np.load(os.path.join(ep.dir, "depth", f"{k:06d}.npy"))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.65")
    im = ax.imshow(np.ma.masked_invalid(d), cmap=cmap, vmin=0, vmax=12)
    cb = plt.colorbar(im, ax=ax, fraction=0.035)
    cb.set_label("depth [m]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(f"depth (planar, m), invalid {np.isnan(d).mean():.0%} (grey)", fontsize=8)
    ax.axis("off")

    # 3. ground-truth costmap
    lab = ep.label(k)
    ext = label_extent(ep.grid)
    ax = axes[2]
    im = ax.imshow(lab.T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=ep.params.max_cost_value, interpolation="nearest")
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"ground truth, base_link frame\nmax {lab.max()}, nonzero {int((lab > 0).sum())} cells", fontsize=8)
    ax.set_xlabel("y left [m]", fontsize=7)
    ax.set_ylabel("x forward [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(labelsize=6)

    # 4. top-down overlay, same frame and grid as panel 3
    ax = axes[3]
    ax.imshow(lab.T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=ep.params.max_cost_value, alpha=0.85, interpolation="nearest")
    r_ps, r_g = ep.params.personal_space_radius, ep.params.group_extra_radius
    local = {}
    for p in peds:
        lx, ly = map_to_base(pose, p["x"], p["y"])
        local[p["name"]] = (lx, ly)
        col = color(p["interaction_id"])
        hy = p["yaw"] - pose["yaw"]
        ax.add_patch(plt.Circle((ly, lx), r_ps, fc="none", ec="cyan", lw=0.6, ls=":"))
        ax.plot(ly, lx, "o", color=col, ms=4)
        ax.arrow(ly, lx, 0.5 * math.sin(hy), 0.5 * math.cos(hy), head_width=0.15, color=col, lw=1)
        if p["interaction_id"] >= 0:
            ax.text(ly, lx - 0.45, f"#{p['interaction_id']}", color="w", fontsize=6, ha="center")
    groups: dict[int, list[dict]] = {}
    for p in peds:
        if p["interaction_id"] >= 0:
            groups.setdefault(p["interaction_id"], []).append(p)
    yy, xx = np.meshgrid(np.linspace(ext[0], ext[1], 300), np.linspace(ext[2], ext[3], 300))
    for iid, mem in groups.items():
        kind = mem[0]["interaction_type"]
        w = ep.params.interaction_weights[kind] if kind < len(ep.params.interaction_weights) else 0.0
        name = INTERACTION_NAMES.get(kind, str(kind))
        cy = np.mean([local[m["name"]][1] for m in mem])
        cx = np.mean([local[m["name"]][0] for m in mem])
        ax.text(cy, cx + 0.6, f"{name} #{iid} w={w:g}", color=color(iid), fontsize=6, ha="center", fontweight="bold")
        if w <= 0 or len(mem) < 2:
            continue
        inside = np.zeros_like(xx, dtype=bool)
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                (ax_, ay_), (bx_, by_) = local[mem[a]["name"]], local[mem[b]["name"]]
                ab = np.array([bx_ - ax_, by_ - ay_])
                t = np.clip(((xx - ax_) * ab[0] + (yy - ay_) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
                inside |= (xx - (ax_ + t * ab[0])) ** 2 + (yy - (ay_ + t * ab[1])) ** 2 <= r_g ** 2
                ax.plot([ay_, by_], [ax_, bx_], "-", color=color(iid), lw=1.2)
        if inside.any():
            ax.contour(yy, xx, inside.astype(float), levels=[0.5], colors=[color(iid)], linewidths=1.0, linestyles="--")
    ax.plot(0, 0, marker=(3, 0, 0), color="lime", ms=9)
    reach = ext[3]
    for sgn in (-1, 1):
        ax.plot([0, reach * math.sin(sgn * hfov / 2)], [0, reach * math.cos(hfov / 2)], "--", color="lime", lw=0.7)
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_title("top-down overlay, base_link frame\n(dotted: personal space r, dashed: interaction region)", fontsize=8)
    ax.set_xlabel("y left [m]", fontsize=7)
    ax.tick_params(labelsize=6)


def temporal_plot(ep: Episode, out: str) -> None:
    t = np.array([(r["stamp_ns"] - ep.t0) / 1e9 for r in ep.robots])
    total = {ch: [] for ch in ("social_cost", "social_cost_membership")}
    for k in range(len(ep.robots)):
        for ch in total:
            total[ch].append(int(np.load(os.path.join(ep.dir, ch, f"{k:06d}.npy")).astype(np.int64).sum()))
    published = [len({p["interaction_id"] for p in ps["pedestrians"] if p["interaction_id"] >= 0}) for ps in ep.peds]
    active = []
    for r in ep.robots:
        f = interaction_frame_at(ep.frames, r["stamp_ns"])
        active.append(sum(1 for i in f["interactions"] if i["outcome"] == 1) if f else 0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    a1.plot(t, total["social_cost"], label="social_cost (Arena layer)")
    a1.plot(t, total["social_cost_membership"], "--", label="social_cost_membership")
    a1.set_ylabel("sum of cell costs")
    a1.legend(fontsize=7)
    a2.step(t, published, where="post", label="interactions on Pedestrian.interaction_id")
    a2.step(t, active, "--", where="post", label="ACTIVE interactions in InteractionFrame")
    a2.set_ylabel("count")
    a2.set_xlabel(f"seconds since first sample (sim t0 = {ep.t0 / 1e9:.2f} s)")
    a2.legend(fontsize=7)
    fig.suptitle(f"{ep.meta.get('scenario')} seed {ep.meta.get('run_seed')} {os.path.basename(ep.dir)}")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", action="append", default=[], help="episode_NNNNNN or NNNNNN, repeatable; default all")
    ap.add_argument("--output", required=True)
    ap.add_argument("--channel", default="social_cost", choices=["social_cost", "social_cost_membership"])
    ap.add_argument("--snapshots", type=int, default=6)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dataset, "metadata.json")))
    params = LayerParams.from_mapping(meta["label_params"])
    grid = GridSpec(**meta["label_grid"])
    wanted = {e if e.startswith("episode_") else f"episode_{int(e):06d}" for e in args.episode}
    os.makedirs(args.output, exist_ok=True)
    fig = plt.figure(figsize=(19, 4.6))
    for ep_dir in episodes(args.dataset):
        name = os.path.basename(ep_dir)
        if wanted and name not in wanted:
            continue
        ep = Episode(ep_dir, params, grid, args.channel)
        n = len(ep.robots)
        if n == 0:
            continue
        tag = f"{name}_{ep.meta.get('scenario')}_seed{ep.meta.get('run_seed')}_{args.channel}"
        snap_dir = os.path.join(args.output, tag)
        os.makedirs(snap_dir, exist_ok=True)
        picks = sorted({int(round(x)) for x in np.linspace(0, n - 1, args.snapshots)})
        writer = None
        for k in range(n):
            if args.no_video and k not in picks:
                continue
            draw(fig, ep, k)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if k in picks:
                cv2.imwrite(os.path.join(snap_dir, f"sample_{k:06d}.png"), img[:, :, ::-1])
            if not args.no_video:
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(args.output, tag + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             ep.meta.get("sample_hz", 5.0), (img.shape[1], img.shape[0]))
                writer.write(np.ascontiguousarray(img[:, :, ::-1]))
        if writer is not None:
            writer.release()
        temporal_plot(ep, os.path.join(args.output, tag + "_temporal.png"))
        print(f"{tag}: {len(picks)} snapshots, {'no video' if args.no_video else 'video'}, temporal plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
