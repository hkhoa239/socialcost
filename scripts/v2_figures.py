#!/usr/bin/env python3
"""Figures for the socialcost_v2 results report -> <out>/fig_*.png (host costmap_bench venv)."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import ROOT, label_path  # noqa: E402
from socialcost.dataset import read_jsonl  # noqa: E402

B = "/home/robotics/arena_ws/data/costmap_bench"
NAMES = {"typemix_gt": "TypeMix, true types*", "typemix_cffullrawmemhold": "Ours + memory (hold)", "typemix_cffullrawmemdecay": "Ours + memory (decay)",
         "typemix_cffullrawmemdecay_geo": "Ours + memory (decay+geo)", "typemix_cffullraw": "Ours (no memory)", "typemix_vis": "TypeMix, crop stage 1",
         "typemix_tracks": "TypeMix, track stage 1", "cascade_vis": "Cascade U-Net (crops)", "cascade_tracks": "Cascade U-Net (tracks)", "pos": "Position U-Net",
         "pos_u": "Position U-Net (prob.)", "sngnn_priv": "SNGNN2D-v2, privileged*", "sngnn_sensor": "SNGNN2D-v2, sensor", "rgbd": "RGB-D U-Net",
         "depth": "Depth U-Net", "rgb": "RGB U-Net"}
OURS = {"typemix_cffullraw", "typemix_cffullrawmemhold", "typemix_cffullrawmemdecay", "typemix_cffullrawmemdecay_geo"}
BUCK = ["visible", "0-1 s", "1-2 s", "2-5 s", "> 5 s"]


def fig_bench(S, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    for a, (lab, key, title, rng) in zip(ax, (("membership_v2", "mae_nonzero", "Error on cost cells (membership v2)", None),
                                               ("extended_v2", "mae_nonzero", "Error on cost cells (extended v2, navigation label)", None),
                                               ("membership_v2", "inter_iou", "Interaction-region IoU (membership v2)", None))):
        rows = sorted(S[lab].items(), key=lambda kv: kv[1][key][0], reverse=(key != "inter_iou"))
        rows = [r for r in rows if r[0] not in ("rgb", "rgbd", "depth")] if key == "mae_nonzero" else rows
        y = np.arange(len(rows))
        col = ["#2e7d4f" if m in OURS else ("#9a9a9a" if m.endswith("gt") or m == "sngnn_priv" else "#4a6fa5") for m, _ in rows]
        a.barh(y, [v[key][0] for _, v in rows], xerr=[v[key][1] for _, v in rows], color=col, alpha=.9, capsize=2)
        a.set_yticks(y, [f"{NAMES[m]} ({v['n']})" for m, v in rows], fontsize=8)
        a.set_title(title, fontsize=10)
        a.grid(axis="x", alpha=.3)
        for i, (_, v) in enumerate(rows):
            a.text(v[key][0], i, f" {v[key][0]:.{3 if key == 'inter_iou' else 1}f}", va="center", fontsize=7)
    fig.text(0.01, 0.01, "Test A+B+C, mean ± std over seeds (n). Green: ours. Grey: privileged. Image U-Nets omitted from error panels (whole-grid error ~60: "
                         "they see only the camera view).", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_memory(E, ST, out):
    modes = [("none", "no memory", "-o", "#4a6fa5"), ("hold", "hard hold", ":s", "#b3403a"), ("decay", "hold + decay", "--^", "#9a6b12"),
             ("decay_geo", "hold + decay + geometry", "-D", "#2e7d4f")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, key, title in zip(ax[:3], ("inter_f1", "halt_f1", "wheel_f1"), ("Stage 1: interaction F1", "Stage 1: raised-hand F1", "Stage 1: wheelchair F1")):
        for mode, lab, st, c in modes:
            t = E[mode]["test"]
            a.plot(BUCK, [t.get(b, {}).get(key) or 0 for b in BUCK], st, label=lab, color=c)
        a.set_title(title, fontsize=10); a.set_xlabel("time since last seen"); a.grid(alpha=.3); a.tick_params(labelsize=8)
    ax[0].legend(fontsize=7)
    a = ax[3]
    for m, lab, st, c in (("typemix_cffullraw", "ours (no memory)", "-o", "#4a6fa5"), ("typemix_cffullrawmemdecay_geo", "ours + memory", "-D", "#2e7d4f"),
                          ("pos_u", "position U-Net", ":", "#888"), ("typemix_gt", "true types*", "--", "#aaa")):
        a.plot(BUCK, [ST[m].get(b, {}).get("person_err") for b in BUCK], st, label=lab, color=c)
    a.set_title("Costmap error within 1.5 m of the person\n(extended v2)", fontsize=10); a.set_xlabel("time since last seen"); a.grid(alpha=.3)
    a.legend(fontsize=7); a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_timeline(eid, j, out):
    c = np.load(f"{B}/cascade_cache/{eid}.npz")
    raw = np.load(f"{B}/stage1/cffullraw/preds/{eid}.npz")
    mem = np.load(f"{B}/stage1/cffullrawmemdecay_geo/preds/{eid}.npz")
    T = len(c["vis"])
    t = np.arange(T) * 0.2
    vis = c["vis"][:, j]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ax[0].fill_between(t, 0, vis, color="#4a6fa5", alpha=.4, step="mid", label="visibility fraction")
    ax[0].axhline(0.1, color="#4a6fa5", lw=.8, ls=":")
    ax[0].fill_between(t, 0, c["halt"][:, j] * 1.0, color="#b3403a", alpha=.15, step="mid", label="raised hand (ground truth)")
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8, loc="upper right"); ax[0].set_ylabel("person")
    ax[1].plot(t, raw["node_p"][:, j, 0].astype(float), color="#4a6fa5", label="p(raised hand), no memory")
    ax[1].plot(t, mem["node_p"][:, j, 0].astype(float), color="#2e7d4f", lw=2, label="p(raised hand), memory")
    ax[1].plot(t, mem["node_u"][:, j, 0].astype(float), color="#9a6b12", ls="--", label="staleness uncertainty u")
    unseen = vis < 0.1
    for k in range(T):
        if unseen[k]:
            ax[1].axvspan(t[k] - 0.1, t[k] + 0.1, color="#000", alpha=.05, lw=0)
    ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("belief"); ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(alpha=.3)
    ax[0].set_title(f"Episode {eid}: a person with a raised hand leaves the camera view (grey = not visible)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def grid_people(eid, k):
    robots = {r["sample"]: r for r in read_jsonl(os.path.join(ROOT, "episodes", eid, "robot_states.jsonl"))}
    peds = {p["sample"]: p for p in read_jsonl(os.path.join(ROOT, "episodes", eid, "pedestrian_states.jsonl"))}
    pose = robots[k]["pose_map_base"]
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    pts = []
    for p in peds[k]["pedestrians"]:
        dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
        pts.append((c * dx + s * dy, -s * dx + c * dy))
    return pts


def pred(model, label, key):
    for f in sorted(glob.glob(f"{B}/runs/v2/{model}_{label}_s[0-9]/predictions.npz")):
        z = np.load(f)
        if key in z.files:
            return z[key].astype(float)
    return None


def draw(a, img, pts, title, fov=True):
    a.imshow(img, origin="lower", extent=(-5, 5, -5, 5), cmap="magma_r", vmin=0, vmax=252)
    if pts:
        a.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, c="#2e7d4f", edgecolors="w", linewidths=.5)
    a.scatter([0], [0], marker=">", s=40, c="#2f5d8a")
    if fov:
        h = math.radians(36)
        a.plot([0, 8 * math.cos(h)], [0, 8 * math.sin(h)], color="#2f5d8a", lw=.6)
        a.plot([0, 8 * math.cos(h)], [0, -8 * math.sin(h)], color="#2f5d8a", lw=.6)
    a.set_xlim(-5, 5); a.set_ylim(-5, 5); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=8)


def fig_examples(samples, out, label="extended_v2"):
    cols = [("gt", "ground truth (v2)"), ("typemix_cffullraw", "ours"), ("typemix_cffullrawmemdecay_geo", "ours + memory"), ("typemix_gt", "true types*"),
            ("pos_u", "position U-Net"), ("sngnn_sensor", "SNGNN2D-v2"), ("rgbd", "RGB-D U-Net")]
    fig, ax = plt.subplots(len(samples), len(cols), figsize=(2.1 * len(cols), 2.25 * len(samples)))
    for r, (eid, k, desc) in enumerate(samples):
        key = f"{eid}|{k}"
        pts = grid_people(eid, k)
        for cidx, (m, t) in enumerate(cols):
            img = np.load(label_path(eid, label, k)).astype(float) if m == "gt" else pred(m, label, key)
            if img is None:
                ax[r, cidx].axis("off"); continue
            draw(ax[r, cidx], img, pts, t if r == 0 else "")
        ax[r, 0].set_ylabel(desc, fontsize=8)
        ax[r, 0].yaxis.label.set_visible(True)
    fig.text(0.005, 0.005, "Robot-centred 10 m grid, robot at centre facing right; lines = camera field of view; green dots = people. *privileged.", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_before_after(eid, j, out, label="extended_v2"):
    c = np.load(f"{B}/cascade_cache/{eid}.npz")
    vis = c["vis"][:, j]
    seen = np.where(vis >= 0.1)[0]
    last = int(seen[seen < len(vis) - 20][-1]) if len(seen) else 10
    ks = [last - last % 2, last + 10 - (last + 10) % 2, last + 30 - (last + 30) % 2]
    ks = [min(k, len(vis) - 2) for k in ks]
    rows = [("gt", "ground truth"), ("typemix_cffullraw", "ours (no memory)"), ("typemix_cffullrawmemdecay_geo", "ours + memory")]
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.2, 7.2))
    for ci, k in enumerate(ks):
        pts = grid_people(eid, k)
        tt = (k - last) * 0.2
        for ri, (m, t) in enumerate(rows):
            img = np.load(label_path(eid, label, k)).astype(float) if m == "gt" else pred(m, label, f"{eid}|{k}")
            if img is None:
                ax[ri, ci].axis("off"); continue
            draw(ax[ri, ci], img, pts, (f"last seen {'+' if tt >= 0 else ''}{tt:.1f} s" if ri == 0 else ""))
            if ci == 0:
                ax[ri, ci].set_ylabel(t, fontsize=9)
    fig.suptitle(f"Episode {eid}: costmap before and after the person leaves the camera view", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(f"{B}/v2_results_summary.json"))
    fig_bench(S, os.path.join(a.out, "fig_bench.png"))
    E = json.load(open(f"{B}/memory/eval_cffullraw.json"))
    ST = json.load(open(f"{B}/v2_eval/staleness_extended_v2.json"))
    fig_memory(E, ST, os.path.join(a.out, "fig_memory.png"))
    fig_timeline("000155", 0, os.path.join(a.out, "fig_timeline_000155.png"))
    fig_before_after("000155", 0, os.path.join(a.out, "fig_before_after_000155.png"))
    # examples: one per family from Test A/B/C (even samples = saved predictions)
    import random
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "manifest_v1.json")))["episodes"]}
    from socialcost.bench.data import split_ids
    rng = random.Random(3)
    picks = []
    for fam in ("conversation", "queue", "stop_gesture", "wheelchair", "group_walking"):
        ids = [e for s in ("test_a", "test_b", "test_c") for e in split_ids(s) if man[e]["family"] == fam]
        eid = rng.choice(ids)
        n = len(read_jsonl(os.path.join(ROOT, "episodes", eid, "robot_states.jsonl")))
        best, bk = -1, 0
        for k in range(0, n, 10):
            t = np.load(label_path(eid, "extended_v2", k))
            if (t > 0).sum() > best:
                best, bk = (t > 0).sum(), k
        picks.append((eid, bk, f"{fam.replace('_', ' ')} ({eid})"))
    fig_examples(picks, os.path.join(a.out, "fig_examples.png"))
    print("figures in", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
