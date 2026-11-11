"""Per-episode cache for the interaction-prediction cascade (stage 1).

For every sample t and person p of an episode (roster fixed per episode):
  track[t, p]   x, y, yaw, vx, vy in the map frame (simulator tracks - the "perfect tracker" input)
  robot[t]      x, y, yaw of the robot
  vis[t, p]     camera visibility fraction from dv1/visibility.jsonl (projection + depth consistency; computable on the
                robot from tracks and the depth image, no simulator interaction state)
  crop[t, p]    RGB-D crop around the projected body, 4 x 64 x 32 uint8 (RGB, depth 0-8 m), zero when not in the image
Labels (stage-1 supervision, never stage-1 inputs):
  inter[t, p]   membership-completed interaction id (-1 none) and type, from the InteractionFrame at the image stamp
  halt[t, p]    raised stop hand (gesture stream at the stamp);  wheel[p]  body asset is a wheelchair
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from ..dataset import interaction_frame_at, read_jsonl, with_membership
from ..qa import pose_matrix
from .data import ROOT

CACHE = os.environ.get("SC_CASCADE_CACHE", "/home/robotics/arena_ws/data/costmap_bench/cascade_cache")
CROP_H, CROP_W = 64, 32


def person_box(K, Tcm, x, y, height=1.75, r=0.3, W=320, H=240):
    pts = []
    for dx, dy, z in ((-r, -r, 0.0), (r, r, 0.0), (-r, r, height), (r, -r, height), (r, r, height), (-r, -r, height)):
        pc = Tcm @ np.array([x + dx, y + dy, z, 1.0])
        if pc[2] > 0.3:
            pts.append((K @ pc[:3])[:2] / pc[2])
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    u0, v0 = np.floor(pts.min(0)).astype(int)
    u1, v1 = np.ceil(pts.max(0)).astype(int)
    if u1 <= 0 or v1 <= 0 or u0 >= W or v0 >= H:
        return None
    return max(0, u0), max(0, v0), min(W, u1), min(H, v1)


def build(eid: str, root: str = ROOT, crops: bool = True) -> str:
    out = os.path.join(CACHE, f"{eid}.npz")
    if os.path.exists(out):
        return out
    d = os.path.join(root, "episodes", eid)
    peds = read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))
    robots = read_jsonl(os.path.join(d, "robot_states.jsonl"))
    frames = read_jsonl(os.path.join(d, "interactions.jsonl"))
    vis_rows = read_jsonl(os.path.join(d, "dv1", "visibility.jsonl"))
    cam = json.load(open(os.path.join(d, "camera_info.json")))
    K = np.array(cam["K"]).reshape(3, 3)
    T = len(robots)
    names = [p["name"].split("/")[-1] for p in peds[0]["pedestrians"]] if peds and peds[0]["pedestrians"] else []
    P = len(names)
    track = np.zeros((T, P, 5), np.float32)
    robot = np.zeros((T, 3), np.float32)
    vis = np.zeros((T, P), np.float32)
    inter = -np.ones((T, P), np.int32)
    itype = -np.ones((T, P), np.int32)
    halt = np.zeros((T, P), np.uint8)
    wheel = np.zeros(P, np.uint8)
    crop = np.zeros((T, P, 4, CROP_H, CROP_W), np.uint8) if crops else np.zeros((0,), np.uint8)
    for t in range(T):
        r = robots[t]["pose_map_base"]
        robot[t] = (r["x"], r["y"], r["yaw"])
        st = {p["name"].split("/")[-1]: p for p in peds[t]["pedestrians"]}
        full = {p["name"].split("/")[-1]: p for p in with_membership(peds[t]["pedestrians"], interaction_frame_at(frames, robots[t]["stamp_ns"]))}
        vr = {p["name"]: p for p in vis_rows[t]["pedestrians"]} if t < len(vis_rows) else {}
        if crops and P:
            rgb = cv2.imread(os.path.join(d, "rgb", f"{t:06d}.png"))[:, :, ::-1]
            dep = np.load(os.path.join(d, "depth", f"{t:06d}.npy"))
            dep8 = (np.clip(np.nan_to_num(dep, nan=0.0), 0, 8) / 8 * 255).astype(np.uint8)
            Tcm = np.linalg.inv(pose_matrix(robots[t]["pose_map_camera"]))
        for j, n in enumerate(names):
            p = st.get(n)
            if p is None:
                continue
            track[t, j] = (p["x"], p["y"], p["yaw"], p["vx"], p["vy"])
            f = full[n]
            inter[t, j], itype[t, j] = f["interaction_id"], f["interaction_type"] if f["interaction_id"] >= 0 else -1
            halt[t, j] = any(isinstance(g, dict) and str(g.get("slot", "")).startswith("halt") for g in p.get("gestures") or [])
            if t == 0:
                wheel[j] = "wheelchair" in os.path.basename(str(p.get("model_uri", ""))).lower()
            v = vr.get(n)
            vis[t, j] = v["visibility_fraction"] if v else 0.0
            if crops and vis[t, j] > 0.05:
                b = person_box(K, Tcm, p["x"], p["y"])
                if b and b[2] - b[0] >= 2 and b[3] - b[1] >= 4:
                    u0, v0, u1, v1 = b
                    crop[t, j, :3] = cv2.resize(rgb[v0:v1, u0:u1], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                    crop[t, j, 3] = cv2.resize(dep8[v0:v1, u0:u1], (CROP_W, CROP_H), interpolation=cv2.INTER_NEAREST)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, names=np.array(names), track=track, robot=robot, vis=vis, inter=inter, itype=itype, halt=halt, wheel=wheel, crop=crop)
    return out


# stage-1 v2 crops: wider box (outstretched / raised arm: +-0.55 m lateral, 2.3 m tall) at 128 x 64, close to the native
# resolution of people within ~5 m (fx = 222 px on the 320 x 240 camera)
CROP2_H, CROP2_W = 128, 64
CROP2_DIR = os.path.join(CACHE, "crops2")


def build_crops2(eid: str, root: str = ROOT) -> str:
    out = os.path.join(CROP2_DIR, f"{eid}.npz")
    if os.path.exists(out):
        return out
    d = os.path.join(root, "episodes", eid)
    z = np.load(os.path.join(CACHE, f"{eid}.npz"))
    names, track, vis = list(z["names"]), z["track"], z["vis"]
    robots = read_jsonl(os.path.join(d, "robot_states.jsonl"))
    cam = json.load(open(os.path.join(d, "camera_info.json")))
    K = np.array(cam["K"]).reshape(3, 3)
    T, P = track.shape[:2]
    crop = np.zeros((T, P, 4, CROP2_H, CROP2_W), np.uint8)
    for t in range(T):
        if not P or not (vis[t] > 0.05).any():
            continue
        rgb = cv2.imread(os.path.join(d, "rgb", f"{t:06d}.png"))[:, :, ::-1]
        dep = np.load(os.path.join(d, "depth", f"{t:06d}.npy"))
        dep8 = (np.clip(np.nan_to_num(dep, nan=0.0), 0, 8) / 8 * 255).astype(np.uint8)
        Tcm = np.linalg.inv(pose_matrix(robots[t]["pose_map_camera"]))
        for j in range(P):
            if vis[t, j] <= 0.05:
                continue
            b = person_box(K, Tcm, float(track[t, j, 0]), float(track[t, j, 1]), height=2.3, r=0.55)
            if b and b[2] - b[0] >= 2 and b[3] - b[1] >= 4:
                u0, v0, u1, v1 = b
                crop[t, j, :3] = cv2.resize(rgb[v0:v1, u0:u1], (CROP2_W, CROP2_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                crop[t, j, 3] = cv2.resize(dep8[v0:v1, u0:u1], (CROP2_W, CROP2_H), interpolation=cv2.INTER_NEAREST)
    os.makedirs(CROP2_DIR, exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", crop=crop)
    os.replace(out + ".tmp.npz", out)
    return out


if __name__ == "__main__":
    import sys
    from multiprocessing import Pool

    ids = sorted(os.listdir(os.path.join(ROOT, "episodes")))
    fn = build_crops2 if "--crops2" in sys.argv else build
    nproc = int(next((a for a in sys.argv[1:] if a.isdigit()), 12))
    with Pool(nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, ids)):
            if i % 50 == 0:
                print(i, flush=True)
    print("done", len(ids))
