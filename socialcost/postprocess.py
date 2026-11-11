"""Post-processing of a predicted social cost grid (robot-centred label grid, uint8 cost 0..252).

Used identically offline (calibration ablations, scripts/analyze_cost_distribution.py) and live
(estimator_node --post <spec.json>). Every step uses only the prediction itself and the robot's own
sensor data (depth image, intrinsics, camera mount); never pedestrian states or labels.

spec keys (all optional; an empty spec is the identity):
  visibility      bool   zero cells beyond the depth-observed range along their bearing (+ margin_m)
  margin_m        float  how far behind the farthest depth return along a bearing a cell still counts as seen (default 0.6)
  blur_sigma      float  Gaussian smoothing, in cells (0 = off)
  lut             list   253 ints: validation-fitted monotone map cost -> cost (isotonic)
  scale           float  multiply cost (after lut), clip to 252
  threshold       int    zero cells with cost < threshold
  min_peak        int    drop connected components (cost > 0) whose peak is below this
  min_area_cells  int    drop connected components smaller than this
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from .labels import GridSpec

MAX = 252


def load_spec(path: str | None) -> dict:
    return json.load(open(path)) if path else {}


def visibility_mask(depth: np.ndarray, K: np.ndarray, T_bc: np.ndarray, grid: GridSpec, margin_m: float = 0.6,
                    bin_deg: float = 1.0, max_range: float = 8.0, stride: int = 4) -> np.ndarray:
    """Cells whose range from the camera is within the farthest valid depth return on the same bearing (+ margin).

    A bearing with no valid return is unobserved. Only sensor data is used."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0.2)
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(z)], -1)
    R, t = T_bc[:3, :3], T_bc[:3, 3]
    pts = (rays * np.nan_to_num(z, nan=0.0)[..., None]) @ R.T + t
    dx, dy = pts[..., 0] - t[0], pts[..., 1] - t[1]
    rng = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    nb = int(round(360 / bin_deg))
    far = np.zeros(nb)
    b = (np.floor((ang[ok] + 180.0) / bin_deg).astype(int)) % nb
    np.maximum.at(far, b, np.minimum(rng[ok], max_range))
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    cx = grid.origin_x + (i + 0.5) * grid.resolution - t[0]
    cy = grid.origin_y + (j + 0.5) * grid.resolution - t[1]
    cb = (np.floor((np.degrees(np.arctan2(cy, cx)) + 180.0) / bin_deg).astype(int)) % nb
    lim = far[cb]
    return (lim > 0) & (np.hypot(cx, cy) <= lim + margin_m)


def apply(cost: np.ndarray, spec: dict, vis: np.ndarray | None = None) -> np.ndarray:
    """cost: uint8 (H, W) in 0..252. vis: visibility mask (needed when spec['visibility'])."""
    c = cost.astype(np.float32)
    if spec.get("visibility"):
        if vis is None:
            raise ValueError("visibility requested but no mask given")
        c = np.where(vis, c, 0.0)
    s = float(spec.get("blur_sigma", 0) or 0)
    if s > 0:
        c = cv2.GaussianBlur(c, (0, 0), s)
    lut = spec.get("lut")
    if lut is not None:
        c = np.asarray(lut, np.float32)[np.clip(np.round(c), 0, MAX).astype(np.int64)]
    if spec.get("scale") is not None:
        c = c * float(spec["scale"])
    c = np.clip(np.round(c), 0, MAX)
    thr = int(spec.get("threshold", 0) or 0)
    if thr > 0:
        c[c < thr] = 0
    mp, ma = int(spec.get("min_peak", 0) or 0), int(spec.get("min_area_cells", 0) or 0)
    if mp > 0 or ma > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((c > 0).astype(np.uint8), connectivity=8)
        if n > 1:
            peak = np.zeros(n)
            np.maximum.at(peak, lab.ravel(), c.ravel())
            drop = (peak < mp) | (stats[:, cv2.CC_STAT_AREA] < ma)
            drop[0] = False
            c[drop[lab]] = 0
    return c.astype(np.uint8)


def fit_isotonic_lut(pred: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None) -> list[int]:
    """Monotone non-decreasing map pred cost (0..252) -> mean target cost, pool-adjacent-violators on 253 bins."""
    p = pred.ravel().astype(np.int64)
    t = target.ravel().astype(np.float64)
    w = np.ones_like(t) if weight is None else weight.ravel().astype(np.float64)
    sw = np.bincount(p, w, MAX + 1)
    st = np.bincount(p, w * t, MAX + 1)
    blocks = []  # [sum_w, sum_t, start, end]
    for k in range(MAX + 1):
        if sw[k] <= 0:
            continue
        blocks.append([sw[k], st[k], k, k])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][3] = b[3]
    xs, ys = [], []
    for b in blocks:
        xs.append(0.5 * (b[2] + b[3]))
        ys.append(b[1] / b[0])
    lut = np.interp(np.arange(MAX + 1), xs, ys) if xs else np.arange(MAX + 1, dtype=float)
    lut = np.maximum.accumulate(lut)
    lut[0] = 0.0
    return [int(round(min(max(v, 0), MAX))) for v in lut]


def describe(spec: dict) -> str:
    if not spec:
        return "raw"
    parts = []
    for k in ("visibility", "blur_sigma", "lut", "scale", "threshold", "min_peak", "min_area_cells"):
        if k in spec and spec[k] not in (None, 0, False):
            parts.append("lut" if k == "lut" else ("vis" if k == "visibility" else f"{k}={spec[k]}"))
    return "+".join(parts) or "raw"

