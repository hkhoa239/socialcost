#!/usr/bin/env python3
"""Side-by-side bird's-eye video of several navigation arms on one configuration (synchronised by time since episode start).

Panel per arm: occupancy map, that run's robot path / pose / camera view, that run's pedestrians (+ last 3 s of track), start
and goal, and the social cost the arm used: the costmap it published to Nav2 (estimator dump/, learned arms) or, for the
ground-truth arm, the true socialcost_v2 label painted from the run's pedestrian positions and the configuration's reference
attributes (interaction membership, gestures; nav_viz_export.align). Panel titles carry the run's metrics.

    python nav_compare_video.py --runs <challenge dir> --config N09 --arms gt_ext ours_ext oursmem_ext posu_ext --metrics <json> --out N09.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from nav_video import (BG, GOAL, PED, ROBOT, START, TXT, WALL, View, blend_cost, colorbar, load_dumps, occupancy,  # noqa: E402
                       pred_to_view, put, robot_at)
from nav_viz_export import CHALLENGE, align, jsonl, load_reference, ref_at  # noqa: E402
from socialcost.labels import LayerParams  # noqa: E402
from socialcost.labels_v2 import SPECS, labels_and_components  # noqa: E402

TITLES = {"gt_ext": "ground truth (extended v2)", "gt_nat": "ground truth (native v2)", "ours_ext": "ours", "oursmem_ext": "ours + memory",
          "posu_ext": "position U-Net", "ours_nat": "ours (native)", "posu_nat": "position U-Net (native)", "bodies": "no social cost"}
FPS = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--channel", default="extended", choices=["extended", "native"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--panel", type=int, default=440)
    ap.add_argument("--layout", default="auto", choices=["auto", "grid", "row"])
    a = ap.parse_args()
    doc = json.load(open(os.path.join(CHALLENGE, "navigation_challenge_v1.json")))
    c = next(x for x in doc["configs"] if x["config_id"] == a.config)
    M = {r["tag"]: r for r in json.load(open(a.metrics))["runs"]}
    ref = load_reference(a.config)
    params = LayerParams.from_overlay()
    arms = []
    for arm in a.arms:
        d = os.path.join(a.runs, f"{a.config}__{arm}__dwb")
        ep = jsonl(os.path.join(d, "episode.jsonl"))[-1]
        robot = np.array(ep["robot"], float)
        peds = [(t, {n: (x, y) for n, x, y, *_ in ps}) for t, ps in ep["peds"]]
        dumps = load_dumps(d) if os.path.isdir(os.path.join(d, "dump")) else []
        arms.append({"arm": arm, "ep": ep, "robot": robot, "peds": peds, "ped_t": np.array([t for t, _ in peds]), "dumps": dumps, "di": 0,
                     "shift": align(ref, ep) if ref is not None else 0.0, "m": M.get(f"{a.config}__{arm}__dwb", {}), "trail": []})
    ep0 = arms[0]["ep"]
    off = np.round((np.array(ep0["start_pose"][:2]) - np.array(c["robot"]["start"][:2])) * 4) / 4
    start, goal = np.array(c["robot"]["start"][:2]) + off, np.array(c["robot"]["goal"][:2]) + off
    pts = [start, goal]
    for A in arms:
        pts += [r[1:3] for r in A["robot"][::5]] + [np.array(p) for _, ps in A["peds"][::10] for p in ps.values()]
    pts = np.array(pts)
    lo, hi = pts.min(0) - 1.5, pts.max(0) + 1.5
    if (hi - lo).max() > 24:
        mid = (start + goal) / 2
        lo, hi = mid - 12, mid + 12
    view = View(lo, hi, a.panel / (hi - lo).max())
    WX, WY = view.world()
    grid = view.grid()
    occ = occupancy(c, view, off)
    base = np.full((view.H, view.W, 3), BG, np.uint8)
    base[occ] = WALL
    half = math.radians(36.0)
    durs = [(A["robot"][-1, 0] - A["ep"]["start_ns"]) / 1e9 for A in arms]
    T = max(durs)
    n = len(arms)
    tall = view.H > 1.5 * view.W
    if a.layout == "row" or (a.layout == "auto" and tall):
        view = View(lo, hi, 620 / (hi - lo)[1] if tall else a.panel / (hi - lo).max())
        WX, WY = view.world()
        grid = view.grid()
        occ = occupancy(c, view, off)
        base = np.full((view.H, view.W, 3), BG, np.uint8)
        base[occ] = WALL
        cols = n
    else:
        cols = 2 if n == 4 else n
    rows_ = int(math.ceil(n / cols))
    gap, head, ptitle, foot = 12, 40, 34, 52
    Wtot = cols * view.W + (cols + 1) * gap
    Htot = head + rows_ * (view.H + ptitle) + foot
    bar = colorbar(Wtot)
    raw = a.out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wtot, Htot))
    times = list(np.arange(0, T, a.speed / FPS)) + [T] * (2 * FPS)
    for tr in times:
        frame = np.full((Htot, Wtot, 3), 255, np.uint8)
        put(frame, f"{a.config}  {c['kind']} ({c['world']})   t = {tr:5.1f} s   x{a.speed:g}", (gap, 26), 0.6)
        for i, A in enumerate(arms):
            s0 = A["ep"]["start_ns"]
            t = s0 + min(tr, durs[i]) * 1e9
            rp = robot_at(A["robot"], t)
            k = int(np.clip(np.searchsorted(A["ped_t"], t), 0, len(A["peds"]) - 1))
            live = A["peds"][k][1]
            if A["dumps"]:
                while A["di"] + 1 < len(A["dumps"]) and A["dumps"][A["di"] + 1]["t"] <= t:
                    A["di"] += 1
                d = A["dumps"][A["di"]]
                cost = pred_to_view(d, robot_at(A["robot"], d["t"])[1:], view, WX, WY) if d["t"] <= t + 5e8 else np.zeros_like(WX)
            elif ref is not None:
                st, fr = ref_at(ref, (t - s0) / 1e9 + A["shift"])
                states = [dict(p, **({"x": live[p["name"]][0], "y": live[p["name"]][1]} if p["name"] in live else {})) for p in st["pedestrians"]]
                cost = labels_and_components(SPECS["v2"], params, grid, states, fr, (0.0, 0.0, 0.0))[a.channel][::-1].astype(np.float32)
            else:
                cost = np.zeros_like(WX)
            if not A["trail"] or rp[1:3].tolist() != A["trail"][-1]:
                A["trail"].append(rp[1:3].tolist())
            img = base.copy()
            blend_cost(img, cost)
            img[occ] = WALL
            x, y, th = rp[1:]
            fov = [view.px(x, y)] + [view.px(x + 8 * math.cos(th + q), y + 8 * math.sin(th + q)) for q in np.linspace(-half, half, 10)]
            cv2.polylines(img, [np.array(fov, np.int32)], True, (190, 160, 120), 1, cv2.LINE_AA)
            cv2.drawMarker(img, view.px(*start), START, cv2.MARKER_SQUARE, 10, 2)
            cv2.drawMarker(img, view.px(*goal), GOAL, cv2.MARKER_STAR, 16, 2)
            k0 = int(np.searchsorted(A["ped_t"], t - 3e9))
            for nm in live:
                tr_ = [view.px(*A["peds"][q][1][nm]) for q in range(k0, k + 1, 2) if nm in A["peds"][q][1]]
                if len(tr_) > 1:
                    cv2.polylines(img, [np.array(tr_, np.int32)], False, (150, 200, 150), 2, cv2.LINE_AA)
            for nm, (px_, py_) in live.items():
                cv2.circle(img, view.px(px_, py_), max(3, int(0.25 * view.s)), PED, -1, cv2.LINE_AA)
            if len(A["trail"]) > 1:
                cv2.polylines(img, [np.array([view.px(*p) for p in A["trail"]], np.int32)], False, ROBOT, 2, cv2.LINE_AA)
            cv2.circle(img, view.px(x, y), max(4, int(0.2 * view.s)), ROBOT, -1, cv2.LINE_AA)
            cv2.line(img, view.px(x, y), view.px(x + 0.45 * math.cos(th), y + 0.45 * math.sin(th)), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(img, (8, view.H - 10), (8 + int(2 * view.s), view.H - 10), TXT, 2)
            put(img, "2 m", (10, view.H - 15), 0.38)
            r_, c_ = divmod(i, cols)
            x0 = gap + c_ * (view.W + gap)
            y0 = head + r_ * (view.H + ptitle) + ptitle
            frame[y0:y0 + view.H, x0:x0 + view.W] = img
            m = A["m"]
            done = tr >= durs[i]
            title = TITLES.get(A["arm"], A["arm"])
            if view.W < 300 and title.startswith("ground truth"):
                title = "ground truth"
            put(frame, title, (x0, y0 - 18), 0.45 if view.W < 300 else 0.5, TXT, 1)
            narrow = view.W < 300
            stats = (f"int {m.get('region_exposure', 0):.2f} soc {m.get('social_exposure', 0):.2f}" if narrow else
                     f"interaction exp. {m.get('region_exposure', 0):.2f}   social exp. {m.get('social_exposure', 0):.2f}")
            if done:
                stats += f" {'GOAL' if m.get('success') else 'FAIL'} {durs[i]:.0f}s" if narrow else f"   {'GOAL' if m.get('success') else 'FAIL'} {durs[i]:.1f} s"
            put(frame, stats, (x0, y0 - 4), 0.4, GOAL if done and not m.get("success") else (90, 90, 90))
        frame[Htot - foot + 2:Htot - foot + 36] = bar
        put(frame, "blue: robot + path   green: pedestrians + last 3 s   outline: camera view   red star: goal   colour: social cost each arm used",
            (gap, Htot - 6), 0.4, (90, 90, 90))
        vw.write(frame)
    vw.release()
    print(json.dumps({"out": raw, "frames": len(times), "size": [Wtot, Htot]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
