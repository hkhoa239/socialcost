#!/usr/bin/env python3
"""Geometry validation of the socialcost_v2 labels against v1 (synthetic scenes + dataset_v1 recorded states).

Writes <out>/labels_v2_geometry.json, <out>/fig_*.png and prints a markdown summary.

    cd /opt/arena_ws/src/deps/socialcost && PYTHONPATH=. /opt/venv/bin/python scripts/labels_v2_validate.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.dataset import interaction_frame_at, read_jsonl  # noqa: E402
from socialcost.labels import GridSpec, LayerParams, Ped  # noqa: E402
from socialcost.labels_v2 import SPECS, labels_and_components, paint_v2  # noqa: E402

BASE = LayerParams.from_overlay()
RES = 0.02
G = GridSpec(resolution=RES, width=700, height=500, origin_x=-7.0, origin_y=-5.0, frame="map")
CELL = RES * RES
VARIANTS = ["v1", "v2_fixedsigma", "v2", "v2_allpairs"]
THR = (1, 50, 100)
DV1 = os.environ.get("SC_DV1_ROOT", "/opt/arena_ws/data/socialcost/dataset_v1")


def stats(c: np.ndarray) -> dict:
    return {"peak": int(c.max()), **{f"area_ge{t}_m2": round(float((c >= t).sum() * CELL), 2) for t in THR}}


def profile(c: np.ndarray, y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    j = int((y - G.origin_y) / RES)
    x = G.origin_x + (np.arange(G.width) + 0.5) * RES
    return x, c[j]


def talk_pair(d: float) -> list[Ped]:
    return [Ped(-d / 2, 0, 0, 1, 0), Ped(d / 2, 0, 0, 1, 0)]


def group(n: int, side: float) -> list[Ped]:
    r = side / (2 * math.sin(math.pi / n))
    return [Ped(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), 0, 1, 1) for k in range(n)]


def queue(n: int, gap: float = 0.8, bend_at: int | None = None) -> list[Ped]:
    pts, x, y, dx, dy = [], -3.0, -2.0, gap, 0.0
    for k in range(n):
        pts.append((x, y))
        if bend_at is not None and k + 1 == bend_at:
            dx, dy = 0.0, gap
        x, y = x + dx, y + dy
    return [Ped(px, py, 0, 3, 6) for px, py in pts]


def reach_beyond(c: np.ndarray, thr: int, ped_x: float) -> float:
    x, v = profile(c)
    m = (v >= thr) & (x > ped_x)
    return round(float(x[m].max() - ped_x), 2) if m.any() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=60, help="dataset_v1 episodes for the area statistics")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R: dict = {"grid_res_m": RES, "params": {v: SPECS[v].to_dict() for v in VARIANTS}}
    paint = lambda v, peds: paint_v2(SPECS[v], BASE, peds, G, (0.0, 0.0, 0.0))  # noqa: E731

    # personal space (unchanged)
    one = [Ped(0, 0, 0)]
    R["personal_space"] = {v: {**stats(paint(v, one)), "radius_ge1_m": reach_beyond(paint(v, one), 1, 0.0)} for v in ("v1", "v2")}

    # conversation pairs vs spacing
    R["conversation_pair"] = {}
    for d in (0.8, 1.0, 1.2, 1.5, 2.0):
        R["conversation_pair"][str(d)] = {}
        for v in VARIANTS[:3]:
            c = paint(v, talk_pair(d))
            x, prof = profile(c)
            jx = int((0 - G.origin_x) / RES)
            col = c[:, jx]
            R["conversation_pair"][str(d)][v] = {**stats(c), **{f"beyond_person_ge{t}_m": reach_beyond(c, t, d / 2) for t in THR},
                                                  **{f"halfwidth_ge{t}_m": round(float(((col >= t).sum() * RES) / 2), 2) for t in THR},
                                                  "edge_step": int(prof[(prof > 0)].min()) if (prof > 0).any() else 0}
    # groups of 3-4: is the interior covered?
    R["groups"] = {}
    for n, side in ((3, 1.68), (4, 1.63), (4, 2.3)):
        c = paint("v2", group(n, side))
        cv1 = paint("v1", group(n, side))
        ctr = c[int(-G.origin_y / RES), int(-G.origin_x / RES)]
        R["groups"][f"{n}x{side}"] = {"v2_centre_cost": int(ctr), "v1_centre_cost": int(cv1[int(-G.origin_y / RES), int(-G.origin_x / RES)]),
                                      "v2": stats(c), "v1": stats(cv1)}
    # queues
    R["queue"] = {}
    for n in range(3, 9):
        for shape, bend in (("straight", None), ("bent", max(2, n // 2))):
            peds = queue(n, bend_at=bend)
            R["queue"][f"{n}_{shape}"] = {v: stats(paint(v, peds)) for v in ("v1", "v2_allpairs", "v2")}
    # narrow corridor: a pair talking across a 2.4 m corridor (walls at y = +-1.2): free width left at each threshold
    R["corridor"] = {}
    for d in (1.0, 1.4):
        peds = [Ped(0, -d / 2, 0, 1, 0), Ped(0, d / 2, 0, 1, 0)]
        for v in VARIANTS[:3]:
            c = paint(v, peds)
            yy = G.origin_y + (np.arange(G.height) + 0.5) * RES
            inside = np.abs(yy) <= 1.2
            # blocked length along the corridor (x) where every in-corridor cell is >= thr
            blocked = {}
            for t in THR:
                full = (c[inside] >= t).all(axis=0)
                blocked[f"fully_blocked_length_ge{t}_m"] = round(float(full.sum() * RES), 2)
            R["corridor"][f"pair{d}_{v}"] = blocked

    # dataset_v1 area statistics (membership label), recorded states, robot-centred 10 m grid
    eps = sorted(os.listdir(os.path.join(DV1, "episodes")))
    random.Random(0).shuffle(eps)
    areas = {v: [] for v in ("v1", "v2", "v2_allpairs")}
    inter = {v: [] for v in ("v1", "v2", "v2_allpairs")}
    n_eps = 0
    for eid in eps:
        d = os.path.join(DV1, "episodes", eid)
        if not os.path.exists(os.path.join(d, "pedestrian_states.jsonl")):
            continue
        meta = json.load(open(os.path.join(d, "episode_metadata.json")))
        base = LayerParams.from_mapping(meta["label"]["layer_params"])
        grid = GridSpec(**meta["label"]["grid"])
        robots = {r["sample"]: r for r in read_jsonl(os.path.join(d, "robot_states.jsonl"))}
        peds = {p["sample"]: p for p in read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))}
        frames = read_jsonl(os.path.join(d, "interactions.jsonl"))
        for k in sorted(robots)[::20]:
            r = robots[k]
            pose = (r["pose_map_base"]["x"], r["pose_map_base"]["y"], r["pose_map_base"]["yaw"])
            for v in areas:
                out = labels_and_components(SPECS[v], base, grid, peds[k]["pedestrians"], interaction_frame_at(frames, r["stamp_ns"]), pose)
                areas[v].append(float((out["membership"] > 0).sum()) * grid.resolution ** 2)
                inter[v].append(float((np.maximum(out["conversation"], out["queue"]) > 0).sum()) * grid.resolution ** 2)
        n_eps += 1
        if n_eps >= args.episodes:
            break
    R["dataset_membership_area"] = {v: {"mean_social_area_m2": round(float(np.mean(a)), 2), "mean_interaction_area_m2": round(float(np.mean(inter[v])), 2),
                                        "samples": len(a)} for v, a in areas.items()}
    R["dataset_membership_area"]["episodes"] = n_eps
    json.dump(R, open(os.path.join(args.out, "labels_v2_geometry.json"), "w"), indent=1)

    # ------------------------------------------------------------------ figures
    ext = [G.origin_x, G.origin_x + G.width * RES, G.origin_y, G.origin_y + G.height * RES]
    scenes = [("person alone", one), ("talk pair 1.2 m", talk_pair(1.2)), ("group of 3", group(3, 1.68)), ("queue of 7, bent", queue(7, bend_at=4))]
    fig, ax = plt.subplots(len(scenes), 3, figsize=(13, 3.1 * len(scenes)))
    for r_, (title, peds) in enumerate(scenes):
        for c_, v in enumerate(("v1", "v2_fixedsigma", "v2")):
            a = ax[r_, c_]
            a.imshow(paint(v, peds), origin="lower", extent=ext, cmap="magma_r", vmin=0, vmax=252)
            a.scatter([p.x for p in peds], [p.y for p in peds], s=14, c="tab:green")
            for t, ls in ((50, "--"), (100, "-")):
                a.contour(np.linspace(ext[0], ext[1], G.width), np.linspace(ext[2], ext[3], G.height), paint(v, peds), levels=[t], colors="k", linewidths=0.6, linestyles=ls)
            a.set_title(f"{title} - {v}", fontsize=9)
            a.set_xlim(-5, 5); a.set_ylim(-4, 4); a.set_aspect("equal"); a.tick_params(labelsize=7)
    fig.suptitle("Social cost labels v1 vs v2 (dashed: cost 50, solid: cost 100)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_footprints.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for v, st in (("v1", "-"), ("v2_fixedsigma", ":"), ("v2", "--")):
        x, p = profile(paint(v, talk_pair(1.2)))
        ax[0].plot(x, p, st, label=v)
        jx = int((0 - G.origin_x) / RES)
        yy = G.origin_y + (np.arange(G.height) + 0.5) * RES
        ax[1].plot(yy, paint(v, talk_pair(1.2))[:, jx], st, label=v)
    ax[0].set_title("talk pair 1.2 m: cost along the pair axis"); ax[1].set_title("cost across the pair (through the midpoint)")
    for a in ax:
        a.set_xlabel("m"); a.set_ylabel("cost"); a.legend(fontsize=8); a.set_xlim(-4, 4); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_profiles.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ns = list(range(3, 9))
    for shape, a in (("straight", ax[0]), ("bent", ax[1])):
        for v, st in (("v1", "-o"), ("v2_allpairs", ":s"), ("v2", "--^")):
            a.plot(ns, [R["queue"][f"{n}_{shape}"][v]["area_ge1_m2"] for n in ns], st, label=v)
        a.set_title(f"queue ({shape}): area with cost > 0"); a.set_xlabel("people in queue"); a.set_ylabel("m²"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_queue_area.png"), dpi=110); plt.close(fig)
    print(json.dumps({k: R[k] for k in ("personal_space", "groups", "dataset_membership_area")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
