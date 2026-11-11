#!/usr/bin/env python3
"""Top-down trajectory video of closed-loop runs (robot + pedestrians), several arms side by side.

Panels share the simulator clock (time since episode start). Shown per panel: robot footprint with its
heading and the camera FOV wedge (what an image-based estimator can see), robot trail, pedestrians with
their 1 m personal-space ring and trail, a line between members of the same interaction, start and goal.
A finished panel freezes on its last frame and shows the outcome.

    python animate_trajectories.py --root <nav_v2 dir> --scenario socialcost_nav_conversation --controller dwb --seed 1 \
        --arms bodies pos oracle_ext oracle_ext_replay rgbd rgbd_t13 --out video.mp4 [--speed 4] [--fps 10]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402

FOV_HALF_DEG = 35.8   # camera horizontal half-FOV (oakd, from the dataset intrinsics)
FOV_RANGE = 8.0       # range the estimator publishes within
GOAL_DELTA = (14.0, 0.0)  # all socialcost_nav scenarios: goto = start + 14 m along x
COL = {"bodies": "#8c8c8c", "pos": "#4c72b0", "oracle_ext": "#2ca02c", "oracle_ext_replay": "#5fbf5f", "rgbd": "#d62728", "rgbd_t13": "#ff7f0e"}


def load(run_dir: str) -> dict | None:
    p = os.path.join(run_dir, "episode.jsonl")
    if not os.path.exists(p):
        return None
    e = json.loads(open(p).readline())
    r = np.array(e["robot"], float)
    t0 = e["start_ns"]
    peds = defaultdict(list)
    for st, ps in e["peds"]:
        for name, x, y, iid in ps:
            peds[name].append(((st - t0) / 1e9, x, y, iid))
    return {"t": (r[:, 0] - t0) / 1e9, "xy": r[:, 1:3], "yaw": r[:, 3], "peds": {k: np.array(v, float) for k, v in peds.items()},
            "start": e["start_pose"], "outcome": e["outcome"], "info": e["outcome_info"]}


def at(tr: np.ndarray, t: float):
    k = np.searchsorted(tr[:, 0], t)
    return tr[min(k, len(tr) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--controller", default="dwb")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=["bodies", "pos", "oracle_ext", "oracle_ext_replay", "rgbd", "rgbd_t13"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=4.0, help="simulated seconds per video second")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--tmax", type=float, default=180.0, help="cap on simulated time shown")
    args = ap.parse_args()
    runs = []
    for a in args.arms:
        d = load(os.path.join(args.root, f"{args.scenario}__{a}__{args.controller}__s{args.seed}"))
        if d:
            runs.append((a, d))
    if not runs:
        raise SystemExit("no runs found")
    T = min(max(d["t"][-1] for _, d in runs), args.tmax)
    # common view: everything any panel visits
    xs = np.concatenate([d["xy"][:, 0] for _, d in runs] + [p[:, 1] for _, d in runs for p in d["peds"].values()])
    ys = np.concatenate([d["xy"][:, 1] for _, d in runs] + [p[:, 2] for _, d in runs for p in d["peds"].values()])
    x0, x1 = xs.min() - 1.5, xs.max() + 1.5
    ym = 0.5 * (ys.min() + ys.max())
    half_h = max(3.5, 0.5 * (ys.max() - ys.min()) + 1.5)
    n = len(runs)
    cols = 2 if n > 3 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8.0 * cols, 2.9 * rows + 0.6), squeeze=False)
    axes = [a for row in axes for a in row]
    for a in axes[n:]:
        a.axis("off")
    title = fig.suptitle("", fontsize=11)
    dyn = []
    for ax, (arm, d) in zip(axes, runs):
        ax.set_xlim(x0, x1)
        ax.set_ylim(ym - half_h, ym + half_h)
        ax.set_aspect("equal")
        ax.set_facecolor("#fafafa")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6)
        sx, sy = d["start"][0], d["start"][1]
        ax.plot(sx, sy, "s", color="k", ms=5)
        ax.plot(sx + GOAL_DELTA[0], sy + GOAL_DELTA[1], "*", color="gold", mec="k", ms=12)
        ax.set_title(arm, fontsize=10, color=COL.get(arm, "k"), fontweight="bold", loc="left")
        st = {"arm": arm, "d": d, "ax": ax}
        st["wedge"] = ax.add_patch(Wedge((sx, sy), FOV_RANGE, -FOV_HALF_DEG, FOV_HALF_DEG, color=COL.get(arm, "k"), alpha=0.08))
        st["trail"], = ax.plot([], [], "-", color=COL.get(arm, "k"), lw=1.6)
        st["body"] = ax.add_patch(Circle((sx, sy), 0.22, color=COL.get(arm, "k"), zorder=5))
        st["head"], = ax.plot([], [], "-", color="k", lw=1.5, zorder=6)
        st["peds"] = {}
        for name in d["peds"]:
            ring = ax.add_patch(Circle((0, 0), 1.0, fill=False, ls="--", lw=0.8, color="#7f3fbf", alpha=0.7))
            dot = ax.add_patch(Circle((0, 0), 0.2, color="#7f3fbf", zorder=4))
            tr, = ax.plot([], [], ":", color="#7f3fbf", lw=0.8)
            st["peds"][name] = (ring, dot, tr)
        st["links"], = ax.plot([], [], "-", color="#7f3fbf", lw=2, alpha=0.5)
        st["txt"] = ax.text(0.99, 0.04, "", transform=ax.transAxes, ha="right", fontsize=8, family="monospace",
                            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        dyn.append(st)
    frames = int(T / args.speed * args.fps) + 1

    def draw(f):
        t = min(f * args.speed / args.fps, T)
        title.set_text(f"{args.scenario.replace('socialcost_nav_', '')} / {args.controller} / seed {args.seed}    t = {t:5.1f} s  "
                       f"(×{args.speed:g})   ◼ start  ★ goal   wedge = camera FOV (8 m)   dashed ring = 1 m personal space")
        out = []
        for st in dyn:
            d = st["d"]
            tt = min(t, d["t"][-1])
            k = min(np.searchsorted(d["t"], tt), len(d["t"]) - 1)
            x, y = d["xy"][k]
            yaw = d["yaw"][k]
            st["body"].center = (x, y)
            st["head"].set_data([x, x + 0.45 * math.cos(yaw)], [y, y + 0.45 * math.sin(yaw)])
            st["wedge"].set_center((x, y))
            st["wedge"].set_theta1(math.degrees(yaw) - FOV_HALF_DEG)
            st["wedge"].set_theta2(math.degrees(yaw) + FOV_HALF_DEG)
            st["trail"].set_data(d["xy"][: k + 1, 0], d["xy"][: k + 1, 1])
            groups = defaultdict(list)
            dmin = 99.0
            for name, (ring, dot, tr) in st["peds"].items():
                p = d["peds"][name]
                q = at(p, tt)
                ring.center = dot.center = (q[1], q[2])
                kk = np.searchsorted(p[:, 0], tt)
                tr.set_data(p[: kk + 1, 1], p[: kk + 1, 2])
                if q[3] >= 0:
                    groups[int(q[3])].append((q[1], q[2]))
                dmin = min(dmin, math.hypot(q[1] - x, q[2] - y))
            lx, ly = [], []
            for mem in groups.values():
                for i in range(len(mem)):
                    for j in range(i + 1, len(mem)):
                        lx += [mem[i][0], mem[j][0], np.nan]
                        ly += [mem[i][1], mem[j][1], np.nan]
            st["links"].set_data(lx, ly)
            v = 0.0
            if k > 0:
                v = float(np.linalg.norm(d["xy"][k] - d["xy"][max(0, k - 10)]) / max(d["t"][k] - d["t"][max(0, k - 10)], 1e-6))
            done = t >= d["t"][-1]
            status = (("GOAL " if d["outcome"] == 2 else "END ") + f"{d['t'][-1]:.1f}s") if done else f"v {v:4.2f} m/s"
            st["txt"].set_text(f"{status}\nclosest ped {dmin:4.2f} m")
            st["txt"].set_color("red" if dmin < 1.0 else "black")
            out += [st["body"], st["head"], st["wedge"], st["trail"], st["links"], st["txt"]]
        return out

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
    draw(frames - 1)
    fig.savefig(os.path.splitext(args.out)[0] + "_final.png", dpi=100)
    print(args.out, frames, "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
