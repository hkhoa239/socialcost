#!/usr/bin/env python3
"""dataset_v1 visual report: per-episode sample sheets, temporal videos, distribution plots.

    python dv1_visualize.py --root /opt/arena_ws/data/socialcost/dataset_v1 --out <dir> [--stage subset|pilot|all] [--videos N]

Sheet panels: RGB with visibility overlay, depth, top-down (pedestrians, headings, robot and trajectories,
camera FOV), arena-native, extended/2.1, component decomposition, ON-OFF difference for pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from socialcost.dataset import read_jsonl  # noqa: E402
from socialcost.dv1 import spec  # noqa: E402

# reference palette (dataviz skill, light surface)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
STATE_COL = {"visible": (27, 175, 122), "partially_occluded": (0, 161, 237), "fully_occluded": (72, 73, 227), "outside_fov": (160, 160, 160)}
COMP = ["personal_space", "conversation", "queue", "gesture", "appearance_or_geometry_extension", "body_occupancy"]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})


def lab_img(a: np.ndarray) -> np.ndarray:
    """Robot-frame label to an image with forward up and left on the left."""
    return np.flipud(np.fliplr(a.T)) if a.ndim == 2 else a


def sheet(root: str, ep: dict, k: int, out: str, twin: dict | None = None) -> None:
    d = os.path.join(root, "episodes", ep["episode_id"])
    rgb = cv2.cvtColor(cv2.imread(os.path.join(d, "rgb", f"{k:06d}.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(os.path.join(d, "depth", f"{k:06d}.npy"))
    comp = np.load(os.path.join(d, "components", f"{k:06d}.npz"))
    vis = read_jsonl(os.path.join(d, "dv1", "visibility.jsonl"))[k]
    robots = read_jsonl(os.path.join(d, "robot_states.jsonl"))
    peds = read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))
    over = rgb.copy()
    for p in vis["pedestrians"]:
        col = STATE_COL[p["state"]]
        cv2.putText(over, f"{p['name'][:10]} {p['state'][:4]} {p['visibility_fraction']:.2f}", (4, 12 + 11 * vis["pedestrians"].index(p)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col[::-1], 1)
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 6)
    ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(over); ax.set_title("RGB + visibility state"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(depth, cmap="Blues_r", vmin=0.3, vmax=12); ax.set_title("depth (m, 0.3-12)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 3]); topdown(ax, ep, robots, peds, k)
    ax = fig.add_subplot(gs[0, 4]); ax.imshow(lab_img(comp["combined_native"]), cmap="Blues", vmin=0, vmax=252); ax.set_title("arena-native/1.1"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 5]); ax.imshow(lab_img(comp["combined_extended"]), cmap="Blues", vmin=0, vmax=252); ax.set_title("extended/2.1"); ax.axis("off")
    for i, c in enumerate(COMP[:5]):
        ax = fig.add_subplot(gs[1, i]); ax.imshow(lab_img(comp[c]), cmap="Blues", vmin=0, vmax=252 if c != "body_occupancy" else 1)
        ax.set_title(c.replace("appearance_or_geometry_extension", "mobility hyp.") + f" (max {int(comp[c].max())})", fontsize=8); ax.axis("off")
    ax = fig.add_subplot(gs[1, 5])
    if twin is not None:
        dt = os.path.join(root, "episodes", twin["episode_id"])
        ct = np.load(os.path.join(dt, "components", f"{k:06d}.npz"))
        a, b = (comp, ct) if ep["condition"] == "on" else (ct, comp)
        diff = a["combined_extended"].astype(int) - b["combined_extended"].astype(int)
        ax.imshow(lab_img(diff), cmap="RdBu_r", vmin=-252, vmax=252); ax.set_title(f"extended ON - OFF ({int((diff != 0).sum())} cells)", fontsize=8)
    else:
        ax.imshow(lab_img(comp["body_occupancy"]), cmap="Greys", vmin=0, vmax=1); ax.set_title("body occupancy", fontsize=8)
    ax.axis("off")
    fig.suptitle(f"{ep['episode_id']}  {ep['family']} / {ep['subtype']} / {ep['condition']}   world {ep['world_key']} ({ep['world']})   split {ep['split']}   "
                 f"sample {k}   peds {ep['n_peds']}", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def topdown(ax, ep, robots, peds, k) -> None:
    rx = [r["pose_map_base"]["x"] for r in robots]
    ry = [r["pose_map_base"]["y"] for r in robots]
    ax.plot(rx, ry, color=INK2, lw=2, label="robot")
    r = robots[k]["pose_map_base"]
    for side in (-1, 1):
        a = r["yaw"] + side * math.radians(spec.CAMERA_HFOV_DEG / 2)
        ax.plot([r["x"], r["x"] + 6 * math.cos(a)], [r["y"], r["y"] + 6 * math.sin(a)], color=GRID, lw=1)
    ax.plot(r["x"], r["y"], "s", color=INK, ms=7)
    names = [p["name"] for p in peds[0]["pedestrians"]]
    for n in names:
        xs = [next((p["x"] for p in s["pedestrians"] if p["name"] == n), np.nan) for s in peds]
        ys = [next((p["y"] for p in s["pedestrians"] if p["name"] == n), np.nan) for s in peds]
        ax.plot(xs, ys, color=S1, lw=1, alpha=0.6)
    for p in peds[k]["pedestrians"]:
        col = S2 if p["interaction_id"] >= 0 else S1
        ax.plot(p["x"], p["y"], "o", color=col, ms=8, mec=SURF, mew=2)
        ax.arrow(p["x"], p["y"], 0.5 * math.cos(p["yaw"]), 0.5 * math.sin(p["yaw"]), color=col, width=0.03, head_width=0.15)
    ax.set_aspect("equal")
    ax.set_title("top-down (map): robot path, peds, FOV", fontsize=8)
    ax.tick_params(labelsize=7)


def video(root: str, ep: dict, out: str) -> None:
    d = os.path.join(root, "episodes", ep["episode_id"])
    n = len(os.listdir(os.path.join(d, "rgb")))
    vis = read_jsonl(os.path.join(d, "dv1", "visibility.jsonl"))
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), spec.SAMPLE_HZ, (320 + 240 + 240, 240))
    for k in range(n):
        rgb = cv2.imread(os.path.join(d, "rgb", f"{k:06d}.png"))
        comp = np.load(os.path.join(d, "components", f"{k:06d}.npz"))
        ext = cv2.resize(lab_img(comp["combined_extended"]).astype(np.uint8), (240, 240), interpolation=cv2.INTER_NEAREST)
        ext = cv2.applyColorMap(255 - ext, cv2.COLORMAP_BONE)
        nat = cv2.resize(lab_img(comp["combined_native"]).astype(np.uint8), (240, 240), interpolation=cv2.INTER_NEAREST)
        nat = cv2.applyColorMap(255 - nat, cv2.COLORMAP_BONE)
        for i, p in enumerate(vis[k]["pedestrians"]):
            cv2.putText(rgb, f"{p['name'][:10]} {p['state'][:4]}", (4, 12 + 11 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.33, STATE_COL[p["state"]], 1)
        cv2.putText(rgb, f"{ep['episode_id']} {ep['condition']} t={k / spec.SAMPLE_HZ:.1f}s", (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(nat, "native", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(ext, "extended/2.1", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        w.write(np.hstack([rgb, nat, ext]))
    w.release()


def bar(ax, counts: dict, title: str, color=S1, order=None, rot=0) -> None:
    keys = order or sorted(counts, key=lambda k: str(k))
    vals = [counts.get(k, 0) for k in keys]
    b = ax.bar([str(k) for k in keys], vals, color=color, width=0.6, edgecolor=SURF, linewidth=2)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(v), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=rot, labelsize=7)
    ax.grid(axis="x", visible=False)


def distributions(root: str, eps: list[dict], out_dir: str, recorded_only: bool) -> dict:
    """Planned (manifest) and recorded distributions. Recorded measures use episodes with DONE."""
    done = [e for e in eps if os.path.exists(os.path.join(root, "episodes", e["episode_id"], "DONE"))]
    use = done if recorded_only else eps
    stats = {}
    fig, axs = plt.subplots(3, 4, figsize=(17, 11))
    bins = ["0-1", "2", "3-4", "5-6", "7-10"]

    def binof(n):
        return "0-1" if n <= 1 else "2" if n == 2 else "3-4" if n <= 4 else "5-6" if n <= 6 else "7-10"
    planned = Counter(binof(e["n_peds"]) for e in eps)
    rec = Counter(binof(e["n_peds"]) for e in done)
    ax = axs[0, 0]
    x = np.arange(len(bins))
    tgt = [100 * spec.PED_COUNT_BINS[b] for b in [(0, 1), (2, 2), (3, 4), (5, 6), (7, 10)]]
    ax.bar(x - 0.2, tgt, 0.38, color=S1, label="target %", edgecolor=SURF, linewidth=2)
    ax.bar(x + 0.2, [100 * planned[b] / max(1, len(eps)) for b in bins], 0.38, color=S2, label="manifest %", edgecolor=SURF, linewidth=2)
    ax.set_xticks(x, bins); ax.set_title("pedestrian count (% of episodes)"); ax.legend(frameon=False, fontsize=7); ax.grid(axis="x", visible=False)
    stats["ped_count"] = {"target_pct": dict(zip(bins, tgt)), "manifest": dict(planned), "recorded": dict(rec)}
    fam = Counter(e["family"] for e in use)
    bar(axs[0, 1], fam, "episodes per family", rot=30)
    bar(axs[0, 2], Counter(e["world_key"] + " " + e["world"] for e in use), "world")
    bar(axs[0, 3], Counter(e["visibility_category"] for e in use), "visibility category", rot=30)
    bodies = Counter(b for e in use for b in e["bodies"])
    stats["body_asset_use"] = dict(bodies)
    axs[1, 0].hist(list(bodies.values()), bins=15, color=S1, edgecolor=SURF, linewidth=2); axs[1, 0].set_title(f"body asset use ({len(bodies)} bodies): episodes per body")
    bar(axs[1, 1], Counter(t for e in use for t in e["trajectory_classes"]), "trajectory class (person-level, any)", rot=45)
    bar(axs[1, 2], Counter(str(e["interaction_condition"]) for e in use), "interaction condition", rot=30)
    # recorded measures
    vis_counts, occl, area, mag, speed, dist = [], Counter(), [], [], [], []
    for e in done:
        d = os.path.join(root, "episodes", e["episode_id"])
        for row in read_jsonl(os.path.join(d, "dv1", "visibility.jsonl"))[::5]:
            vis_counts.append(sum(p["state"] in ("visible", "partially_occluded") for p in row["pedestrians"]))
            for p in row["pedestrians"]:
                occl[p["state"]] += 1
                if p["state"] != "outside_fov":
                    dist.append(p["distance_m"])
        for k in range(0, 125, 10):
            f = os.path.join(d, "components", f"{k:06d}.npz")
            if os.path.exists(f):
                z = np.load(f)["combined_extended"]
                area.append(float((z > 0).sum() * 0.0025))
                if z.any():
                    mag.append(int(z.max()))
        for s in read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))[::5]:
            speed += [math.hypot(p["vx"], p["vy"]) for p in s["pedestrians"]]
    if done:
        bar(axs[1, 3], Counter(vis_counts), f"camera-visible pedestrians per sample (recorded, n={len(done)} eps)")
        bar(axs[2, 0], occl, "visibility state (recorded samples x peds)", order=["visible", "partially_occluded", "fully_occluded", "outside_fov"], rot=20)
        axs[2, 1].hist(dist, bins=24, color=S1, edgecolor=SURF, linewidth=1); axs[2, 1].set_title("camera distance of in-FOV peds (m)")
        axs[2, 2].hist(area, bins=24, color=S1, edgecolor=SURF, linewidth=1); axs[2, 2].set_title("social cost area per sample (m^2, extended)")
        axs[2, 3].hist(speed, bins=24, color=S1, edgecolor=SURF, linewidth=1); axs[2, 3].set_title("pedestrian speed (m/s)")
        stats["recorded"] = {"episodes": len(done), "visible_per_sample": dict(Counter(vis_counts)), "visibility_states": dict(occl),
                             "cost_area_m2_mean": float(np.mean(area)) if area else 0, "cost_max_hist": dict(Counter(mag)),
                             "speed_mean": float(np.mean(speed)) if speed else 0, "camera_distance_mean": float(np.mean(dist)) if dist else 0}
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distributions.png"), dpi=80)
    plt.close(fig)
    if mag:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(mag, bins=30, color=S1, edgecolor=SURF, linewidth=1); ax.set_title("max cost per labelled sample (extended/2.1)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "cost_magnitude.png"), dpi=80); plt.close(fig)
    json.dump(stats, open(os.path.join(out_dir, "distributions.json"), "w"), indent=1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--per-family", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    man = json.load(open(os.path.join(args.root, "manifests", "manifest_v1.json")))
    eps = man["episodes"]
    sel = {"subset": [e for e in eps if e["subset"]], "pilot": [e for e in eps if e["pilot"]]}.get(args.stage, eps)
    done = [e for e in sel if os.path.exists(os.path.join(args.root, "episodes", e["episode_id"], "DONE"))]
    by_pair = {(e["pair_id"], e["condition"]): e for e in eps if e["pair_id"]}
    rejected_path = os.path.join(os.path.dirname(args.out.rstrip("/")), "dataset_v1_validation", args.stage, "rejected.json")
    rejected = set(json.load(open(rejected_path))) if os.path.exists(rejected_path) else set()
    # representative sheets: per family and split group, plus failure cases
    picked = []
    for fam in spec.FAMILIES:
        for group in (("train",), ("val",), ("test_a", "test_b", "test_c")):
            c = [e for e in done if e["family"] == fam and e["split"] in group and e["condition"] != "off"]
            picked += c[: args.per_family]
    picked += [e for e in done if e["episode_id"] in rejected]
    for e in picked:
        twin = by_pair.get((e["pair_id"], "off" if e["condition"] == "on" else "on")) if e["pair_id"] else None
        if twin and not os.path.exists(os.path.join(args.root, "episodes", twin["episode_id"], "DONE")):
            twin = None
        tag = "FAIL_" if e["episode_id"] in rejected else ""
        for k in (30, 90):
            sheet(args.root, e, k, os.path.join(args.out, "sheets", f"{tag}{e['episode_id']}_{e['family']}_{e['condition']}_s{k:03d}.png"), twin)
            if twin:
                sheet(args.root, twin, k, os.path.join(args.out, "sheets", f"{tag}{twin['episode_id']}_{twin['family']}_{twin['condition']}_s{k:03d}.png"), e)
    # videos for temporal scenarios
    temporal = [e for e in done if e["family"] in ("partial_observability", "group_walking") or e["subtype"] in ("lifecycle", "approach_and_stop")
                or "pulse" in str(e["subtype"]) or "advancing" in str(e["subtype"])]
    for e in temporal[: args.videos]:
        video(args.root, e, os.path.join(args.out, "videos", f"{e['episode_id']}_{e['family']}_{e['condition']}.mp4"))
    distributions(args.root, eps if args.stage == "all" else sel, args.out, recorded_only=False)
    print(f"{len(picked)} sheet episodes, {min(len(temporal), args.videos)} videos, {len(done)} recorded episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
