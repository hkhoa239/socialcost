#!/usr/bin/env python3
"""Bird's-eye video of one closed-loop navigation run: ground-truth vs predicted social costmap, side by side.

Left panel   ground-truth extended social cost (extended/2.1, as oracle_ext / the navigation metrics): the run's own
             pedestrian positions + per-person attributes (heading, interaction membership, gestures) of the
             configuration's reference recording at the aligned time (same construction as nav_viz_export.py)
Right panel  the costmap the arm actually published to Nav2 (estimator --dump-dir grids, odom frame, after the label
             floor), moved into the map frame with the robot pose at its stamp
Both panels  occupancy map, start / goal, robot (footprint, heading, path so far, camera field of view),
             pedestrians (body, last 3 s of track). Same colour scale (cost 0..252).

Needs a run directory with dump/tick_*.npz (estimator started with --dump-dir {run_dir}/dump).
Writes an mp4v file; transcode with ffmpeg for H.264 (see make_nav_videos.sh).

    python nav_video.py --run <run_dir> --config N02 --out N02_ours.mp4 [--speed 2] [--title "..."]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from nav_viz_export import CHALLENGE, WORLDS, align, jsonl, load_reference, ref_at, thin  # noqa: E402
from socialcost.dataset import EXTENDED_CONFIG  # noqa: E402
from socialcost.extensions import ExtParams, paint_extended  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402

FPS = 10
MAXC = 252.0
FOV_RANGE = 8.0
BG = (250, 250, 250)
WALL = (70, 70, 70)
ROBOT = (200, 110, 20)       # BGR: blue
PED = (40, 140, 40)          # green
GOAL = (40, 40, 220)         # red
START = (120, 120, 120)
TXT = (30, 30, 30)
OUTCOME = {2: "goal reached", -1: "timeout", -6: "Nav2 aborted"}


def cost_lut() -> np.ndarray:
    """256 x 3 BGR colours (inferno, reversed lightness: low cost pale yellow, high cost dark purple)."""
    ramp = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0]
    return ramp[::-1].copy()


LUT = cost_lut()


def blend_cost(img: np.ndarray, cost: np.ndarray) -> None:
    """Alpha-blend a cost image (H x W, 0..252, row 0 = top) onto img in place."""
    m = cost > 0
    if not m.any():
        return
    v = np.clip(cost[m] / MAXC * 255, 0, 255).astype(np.uint8)
    col = LUT[np.clip(64 + v.astype(int) * 3 // 4, 0, 255)].astype(np.float32)
    a = (0.35 + 0.55 * cost[m] / MAXC)[:, None]
    img[m] = (img[m] * (1 - a) + col * a).astype(np.uint8)


class View:
    """Map-frame crop -> panel pixels (row 0 = top)."""

    def __init__(self, lo, hi, scale):
        self.lo, self.hi, self.s = np.asarray(lo, float), np.asarray(hi, float), float(scale)
        self.W = int(round((self.hi[0] - self.lo[0]) * self.s))
        self.H = int(round((self.hi[1] - self.lo[1]) * self.s))

    def px(self, x, y):
        return int(round((x - self.lo[0]) * self.s)), int(round((self.hi[1] - y) * self.s))

    def grid(self) -> GridSpec:
        return GridSpec(resolution=1.0 / self.s, width=self.W, height=self.H, origin_x=float(self.lo[0]), origin_y=float(self.lo[1]), frame="map")

    def world(self):
        jj, ii = np.mgrid[0:self.H, 0:self.W]
        return self.lo[0] + (ii + 0.5) / self.s, self.hi[1] - (jj + 0.5) / self.s


def occupancy(c, view: View, off) -> np.ndarray:
    my = yaml.safe_load(open(os.path.join(WORLDS, c["world"], "map", "0.yaml")))
    img = np.array(Image.open(os.path.join(WORLDS, c["world"], "map", my["image"])).convert("L"))
    mres, (mox, moy) = my["resolution"], my["origin"][:2]
    wx, wy = view.world()
    wx, wy = wx - off[0], wy - off[1]  # the occupancy map is in the scenario frame
    pc = np.floor((wx - mox) / mres).astype(int)
    pr = img.shape[0] - 1 - np.floor((wy - moy) / mres).astype(int)
    ok = (pc >= 0) & (pc < img.shape[1]) & (pr >= 0) & (pr < img.shape[0])
    occ = np.zeros(wx.shape, bool)
    occ[ok] = img[pr[ok], pc[ok]] < 100
    return occ


def load_dumps(run: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(run, "dump", "tick_*.npz"))):
        z = np.load(f)
        out.append({"t": int(z["stamp_ns"]), "cost": z["cost"].astype(np.float32), "origin": z["origin"].astype(float), "res": float(z["res"]),
                    "pose_f": z["pose_f"].astype(float)})
    out.sort(key=lambda d: d["t"])
    return out


def robot_at(robot, t_ns):
    k = int(np.clip(np.searchsorted(robot[:, 0], t_ns), 0, len(robot) - 1))
    return robot[k]


def pred_to_view(d: dict, robot_map, view: View, WX, WY) -> np.ndarray:
    """Published odom-frame grid -> panel, via map pose (robot_map) and odom pose (pose_f) at the grid's stamp."""
    xo, yo, tho = d["pose_f"]
    xm, ym, thm = robot_map
    dth = tho - thm  # map -> odom rotation
    c, s = math.cos(dth), math.sin(dth)
    dx, dy = WX - xm, WY - ym
    ox, oy = xo + c * dx - s * dy, yo + s * dx + c * dy
    i = np.floor((ox - d["origin"][0]) / d["res"]).astype(int)
    j = np.floor((oy - d["origin"][1]) / d["res"]).astype(int)
    h, w = d["cost"].shape
    ok = (i >= 0) & (i < w) & (j >= 0) & (j < h)
    out = np.zeros(WX.shape, np.float32)
    out[ok] = d["cost"][j[ok], i[ok]]
    return out


def put(img, text, org, scale=0.5, color=TXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def colorbar(width: int) -> np.ndarray:
    bar = np.full((34, width, 3), 255, np.uint8)
    x0, x1 = 60, width - 60
    for x in range(x0, x1):
        c = (x - x0) / (x1 - x0) * MAXC
        tmp = np.full((1, 1, 3), 255, np.uint8)
        blend_cost(tmp, np.array([[max(c, 0.5)]]))
        bar[4:16, x] = tmp[0, 0]
    for v in (0, 63, 126, 189, 252):
        x = int(x0 + v / MAXC * (x1 - x0))
        cv2.line(bar, (x, 16), (x, 20), TXT, 1)
        put(bar, str(v), (x - 10, 31), 0.4)
    put(bar, "social cost", (4, 14), 0.4)
    return bar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed (sim seconds per video second)")
    ap.add_argument("--title", default="ours: counterfactual stage 1 + TypeMix")
    ap.add_argument("--panel", type=int, default=640, help="longest panel side in pixels")
    args = ap.parse_args()
    doc = json.load(open(os.path.join(CHALLENGE, "navigation_challenge_v1.json")))
    c = next(x for x in doc["configs"] if x["config_id"] == args.config)
    ep = jsonl(os.path.join(args.run, "episode.jsonl"))[-1]
    meta = json.load(open(os.path.join(args.run, "run.json")))
    s0 = ep["start_ns"]
    robot = np.array(ep["robot"], float)
    peds = ep["peds"]
    dumps = load_dumps(args.run)
    if not dumps:
        print("no dump/ grids in", args.run)
        return 2
    ref = load_reference(args.config)
    if ref is None:
        print("no reference recording for", args.config, "(ground truth needs it)")
        return 2
    shift = align(ref, ep)
    off = np.round((np.array(ep["start_pose"][:2]) - np.array(c["robot"]["start"][:2])) * 4) / 4
    start = np.array(c["robot"]["start"][:2]) + off
    goal = np.array(c["robot"]["goal"][:2]) + off
    pts = [start, goal] + [r[1:3] for r in robot[::5]] + [np.array(p[1:3]) for _, ps in peds[::10] for p in ps]
    pts = np.array(pts)
    lo, hi = pts.min(0) - 2.0, pts.max(0) + 2.0
    if (hi - lo).max() > 30:
        mid = (start + goal) / 2
        lo, hi = mid - 15, mid + 15
    view = View(lo, hi, args.panel / (hi - lo).max())
    WX, WY = view.world()
    grid = view.grid()
    occ = occupancy(c, view, off)
    params = LayerParams.from_overlay()
    ext = dataclasses.replace(ExtParams.load(EXTENDED_CONFIG), mobility=False)
    camf = os.path.join(CHALLENGE, "references", args.config, "camera_info.json")
    half = math.radians(36.0)
    if os.path.exists(camf):
        ci = json.load(open(camf))
        K = np.array(ci.get("k") or ci.get("K")).reshape(3, 3)
        half = math.atan(K[0, 2] / K[0, 0])
    t_end = robot[-1, 0]
    outcome = OUTCOME.get(ep.get("outcome"), str(ep.get("outcome_info")))
    ped_rows = [(t, {n: (x, y) for n, x, y, *_ in ps}) for t, ps in peds]
    ped_t = np.array([t for t, _ in ped_rows])
    gap, head, foot = 16, 78, 70
    Wtot = 2 * view.W + 3 * gap
    Htot = head + view.H + foot
    bar = colorbar(Wtot)
    raw = args.out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wtot, Htot))
    base = np.full((view.H, view.W, 3), BG, np.uint8)
    base[occ] = WALL
    dt = args.speed / FPS
    times = list(np.arange(s0, t_end, dt * 1e9)) + [t_end] * (2 * FPS)  # hold the last frame 2 s
    di = 0
    trail = []
    for f, t in enumerate(times):
        tr = (t - s0) / 1e9
        rp = robot_at(robot, t)
        k = int(np.clip(np.searchsorted(ped_t, t), 0, len(ped_rows) - 1))
        live = ped_rows[k][1]
        # ground truth at this time
        s, fr = ref_at(ref, tr + shift)
        states = []
        for p in s["pedestrians"]:
            q = dict(p)
            if p["name"] in live:
                q["x"], q["y"] = live[p["name"]]
            states.append(q)
        gt, _ = paint_extended(params, ext, states, fr, grid, (0.0, 0.0, 0.0))
        gt = gt[::-1].astype(np.float32)
        # latest published grid at this time
        while di + 1 < len(dumps) and dumps[di + 1]["t"] <= t:
            di += 1
        d = dumps[di]
        pred = pred_to_view(d, robot_at(robot, d["t"])[1:], view, WX, WY) if d["t"] <= t + 5e8 else np.zeros_like(WX, np.float32)
        if f == 0 or rp[1:3].tolist() != (trail[-1] if trail else None):
            trail.append(rp[1:3].tolist())
        panels = []
        for kind, cost in (("gt", gt), ("pred", pred)):
            img = base.copy()
            blend_cost(img, cost)
            img[occ] = WALL
            # camera field of view
            x, y, th = rp[1:]
            fov = [view.px(x, y)] + [view.px(x + FOV_RANGE * math.cos(th + a), y + FOV_RANGE * math.sin(th + a)) for a in np.linspace(-half, half, 12)]
            cv2.polylines(img, [np.array(fov, np.int32)], True, (190, 160, 120), 1, cv2.LINE_AA)
            # start, goal
            cv2.drawMarker(img, view.px(*start), START, cv2.MARKER_SQUARE, 12, 2)
            gx, gy = view.px(*goal)
            cv2.drawMarker(img, (gx, gy), GOAL, cv2.MARKER_STAR, 18, 2)
            # pedestrian tracks and bodies
            k0 = int(np.searchsorted(ped_t, t - 3e9))
            for n in live:
                pts_ = [view.px(*ped_rows[i][1][n]) for i in range(k0, k + 1, 2) if n in ped_rows[i][1]]
                if len(pts_) > 1:
                    cv2.polylines(img, [np.array(pts_, np.int32)], False, (150, 200, 150), 2, cv2.LINE_AA)
            for n, (px_, py_) in live.items():
                cv2.circle(img, view.px(px_, py_), max(3, int(0.25 * view.s)), PED, -1, cv2.LINE_AA)
                cv2.circle(img, view.px(px_, py_), max(3, int(0.25 * view.s)), (20, 70, 20), 1, cv2.LINE_AA)
            # robot path, footprint, heading
            if len(trail) > 1:
                cv2.polylines(img, [np.array([view.px(*p) for p in trail], np.int32)], False, ROBOT, 2, cv2.LINE_AA)
            r = max(4, int(0.2 * view.s))
            cv2.circle(img, view.px(x, y), r, ROBOT, -1, cv2.LINE_AA)
            cv2.line(img, view.px(x, y), view.px(x + 0.45 * math.cos(th), y + 0.45 * math.sin(th)), (255, 255, 255), 2, cv2.LINE_AA)
            # scale bar 2 m
            cv2.line(img, (10, view.H - 12), (10 + int(2 * view.s), view.H - 12), TXT, 2)
            put(img, "2 m", (12, view.H - 18), 0.4)
            panels.append(img)
        frame = np.full((Htot, Wtot, 3), 255, np.uint8)
        frame[head:head + view.H, gap:gap + view.W] = panels[0]
        frame[head:head + view.H, 2 * gap + view.W:2 * gap + 2 * view.W] = panels[1]
        put(frame, f"{args.config}  {c['kind']} ({c['world']})   {args.title}", (gap, 22), 0.6, TXT, 1)
        status = f"t = {min(tr, (t_end - s0) / 1e9):5.1f} s   x{args.speed:g} speed"
        if t >= t_end:
            status += f"   |   {outcome} at {(t_end - s0) / 1e9:.1f} s"
        put(frame, status, (gap, 46), 0.5, GOAL if (t >= t_end and ep.get("outcome") != 2) else TXT)
        put(frame, "ground-truth social cost (extended label)", (gap, head - 8), 0.45)
        put(frame, "predicted social cost (published to Nav2)", (2 * gap + view.W, head - 8), 0.45)
        frame[head + view.H + 4:head + view.H + 38] = bar
        put(frame, "robot (blue) + path   pedestrians (green) + last 3 s   camera view (outline)   goal (red star)",
            (gap, Htot - 12), 0.42, (90, 90, 90))
        vw.write(frame)
    vw.release()
    print(json.dumps({"out": raw, "frames": len(times), "size": [Wtot, Htot], "shift_s": shift, "outcome": outcome, "status": meta["status"],
                      "dumps": len(dumps)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
