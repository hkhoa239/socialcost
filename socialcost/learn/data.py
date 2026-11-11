"""Training data: samples, input construction per model family, evaluation masks.

Inputs a model may see are built here and nowhere else:
  rgbd / depth / rgb models: the RGB image, the depth image, camera intrinsics and the static camera
                             mount (sensor calibration), nothing from the simulator's pedestrian state
  position models:           rasterised pedestrian positions and headings (clean simulator positions;
                             a diagnostic, not a tracker) - optionally only those inside the camera FOV
Labels, evaluation masks and pair metadata are returned separately under "target"/"meta" and are
never passed to a model's forward().
"""

from __future__ import annotations

import glob
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..dataset import read_jsonl, with_membership, interaction_frame_at
from ..labels import GridSpec, Ped, paint, LayerParams
from ..qa import pose_matrix

LABEL_DIRS = {"native": "social_cost", "membership": "social_cost_membership", "extended": "social_cost_extended"}
BEV_RES = 0.1  # m, internal grid of the image models (100 x 100 over the 10 m label)
BEV_N = 100
Z_RANGE = (0.05, 2.2)  # m above the floor kept when lifting depth points


def episode_dirs(roots: list[str]) -> list[str]:
    return [d for r in roots for d in sorted(glob.glob(os.path.join(r, "episodes", "episode_*")))]


class EpisodeIndex:
    """Per-episode static data loaded once: metadata, poses, privileged states, calibration."""

    def __init__(self, ep_dir: str) -> None:
        self.dir = ep_dir
        self.meta = json.load(open(os.path.join(ep_dir, "episode_metadata.json")))
        self.cam = json.load(open(os.path.join(ep_dir, "camera_info.json")))
        self.robots = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))
        self.peds = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))
        self.frames = read_jsonl(os.path.join(ep_dir, "interactions.jsonl"))
        self.K = np.array(self.cam["K"], dtype=np.float64).reshape(3, 3)
        self.T_bc = pose_matrix(self.cam["T_base_camera"])
        self.params = LayerParams.from_mapping(self.meta["label"]["layer_params"])
        self.grid = GridSpec(**self.meta["label"]["grid"])
        self.key = f"{os.path.basename(os.path.dirname(os.path.dirname(ep_dir)))}/{os.path.basename(ep_dir)}"


def fov_mask(grid: GridSpec, K: np.ndarray, T_bc: np.ndarray, max_range: float = 8.0) -> np.ndarray:
    """Label cells whose centre is inside the camera's horizontal FOV, in front of it, within range."""
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    x = grid.origin_x + (i + 0.5) * grid.resolution
    y = grid.origin_y + (j + 0.5) * grid.resolution
    cx, cy = T_bc[0, 3], T_bc[1, 3]
    fwd = T_bc[:3, :3] @ np.array([0, 0, 1.0])
    yaw = math.atan2(fwd[1], fwd[0])
    half = math.atan(K[0, 2] / K[0, 0])
    ang = np.arctan2(y - cy, x - cx) - yaw
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    r = np.hypot(x - cx, y - cy)
    return (np.abs(ang) <= half) & (r <= max_range) & (r > 0.3)


def raster_peds(states: list[dict], pose: dict, grid: GridSpec, sigma: float = 0.15, fov: tuple | None = None) -> np.ndarray:
    """3 x H x W: centre heatmap, heatmap * sin(heading), heatmap * cos(heading), in the label grid."""
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    x = grid.origin_x + (i + 0.5) * grid.resolution
    y = grid.origin_y + (j + 0.5) * grid.resolution
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    out = np.zeros((3, grid.height, grid.width), np.float32)
    for p in states:
        dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        if fov is not None:
            K, T_bc = fov
            fwd = T_bc[:3, :3] @ np.array([0, 0, 1.0])
            ang = math.atan2(ly - T_bc[1, 3], lx - T_bc[0, 3]) - math.atan2(fwd[1], fwd[0])
            if abs(math.atan2(math.sin(ang), math.cos(ang))) > math.atan(K[0, 2] / K[0, 0]) or lx - T_bc[0, 3] < 0.3:
                continue
        h = np.exp(-((x - lx) ** 2 + (y - ly) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        rel = p["yaw"] - pose["yaw"]
        out[0] = np.maximum(out[0], h)
        out[1] += h * math.sin(rel)
        out[2] += h * math.cos(rel)
    return out


def lift_indices(depth: np.ndarray, K: np.ndarray, T_bc: np.ndarray, stride: int = 2, mode: str = "depth", grid: GridSpec | None = None) -> np.ndarray:
    """For the feature map at 1/stride resolution: BEV cell index (row-major in BEV_N x BEV_N) or -1.

    mode "depth": back-project each pixel by its measured depth. mode "ground": intersect the pixel
    ray with the floor (z = 0) - the RGB-only model's geometry, no depth used.
    """
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    u = u + 0.5 * stride
    v = v + 0.5 * stride
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)  # optical frame, z = 1
    R, t = T_bc[:3, :3], T_bc[:3, 3]
    rb = rays @ R.T  # ray directions in base frame
    if mode == "depth":
        z = depth[::stride, ::stride]
        pts = rb * np.nan_to_num(z, nan=0.0)[..., None] + t
        ok = np.isfinite(z) & (z > 0.2)
    else:
        lam = -t[2] / np.where(np.abs(rb[..., 2]) < 1e-6, -1e-6, rb[..., 2])
        pts = rb * lam[..., None] + t
        ok = (lam > 0) & (rb[..., 2] < -1e-3)
        pts[..., 2] = 0.1  # inside Z_RANGE
    ok &= (pts[..., 2] >= Z_RANGE[0]) & (pts[..., 2] <= Z_RANGE[1]) & (np.abs(pts[..., 0]) < 5.0) & (np.abs(pts[..., 1]) < 5.0)
    ix = np.floor((pts[..., 0] + 5.0) / BEV_RES).astype(np.int64)
    iy = np.floor((pts[..., 1] + 5.0) / BEV_RES).astype(np.int64)
    idx = np.where(ok, iy * BEV_N + ix, -1)
    return idx.astype(np.int64)


class SocialCostSamples(Dataset):
    def __init__(self, episodes: list[EpisodeIndex], label: str, inputs: str, cache_fov: bool = True) -> None:
        """inputs: rgbd | depth | rgb | pos | pos_fov"""
        self.eps = episodes
        self.label = LABEL_DIRS[label]
        self.inputs = inputs
        self.items = [(e, k) for e in episodes for k in range(len(e.robots))]
        e0 = episodes[0]
        self.fov = fov_mask(e0.grid, e0.K, e0.T_bc)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        e, k = self.items[i]
        name = f"{k:06d}"
        target = np.load(os.path.join(e.dir, self.label, name + ".npy")).astype(np.float32) / e.params.max_cost_value
        pose = e.robots[k]["pose_map_base"]
        states = e.peds[k]["pedestrians"]
        # evaluation-only maps (never model inputs): personal-space-only rule on the same people
        rule = paint(e.params, [Ped(p["x"], p["y"], p["animation_state"]) for p in states], e.grid, (pose["x"], pose["y"], pose["yaw"])).astype(np.float32) / e.params.max_cost_value
        x: dict = {}
        if self.inputs in ("rgbd", "depth", "rgb"):
            rgb = cv2.imread(os.path.join(e.dir, "rgb", name + ".png"))[:, :, ::-1].astype(np.float32) / 255.0
            depth = np.load(os.path.join(e.dir, "depth", name + ".npy"))
            valid = np.isfinite(depth).astype(np.float32)
            d = np.nan_to_num(depth, nan=0.0) / 10.0
            chans = []
            if self.inputs in ("rgbd", "rgb"):
                chans.append(rgb.transpose(2, 0, 1) - 0.5)
            if self.inputs in ("rgbd", "depth"):
                chans += [d[None], valid[None]]
            x["image"] = torch.from_numpy(np.concatenate(chans, 0).astype(np.float32))
            x["lift"] = torch.from_numpy(lift_indices(depth, e.K, e.T_bc, mode="ground" if self.inputs == "rgb" else "depth"))
        else:
            fov = (e.K, e.T_bc) if self.inputs == "pos_fov" else None
            x["raster"] = torch.from_numpy(raster_peds(states, pose, e.grid, fov=fov))
        meta = {"episode": e.key, "sample": k, "family": e.meta.get("family") or e.meta.get("scenario"), "pair_id": e.meta.get("pair_id") or "",
                "condition": e.meta.get("condition") or "", "slot": e.robots[k]["slot"]}
        return {"x": x, "target": torch.from_numpy(target), "rule": torch.from_numpy(rule), "fov": torch.from_numpy(self.fov), "meta": meta}
