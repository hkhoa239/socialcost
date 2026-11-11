"""World geometry for dataset_v1: occupancy, routes, clearance, path lengths.

All coordinates are in the world (scenario) frame of `worlds/<name>/world.yaml`, which is the frame
of `map/0.png` (origin and resolution from `map/0.yaml`). Recorded data is in the Arena `map` TF
frame, offset from this one by a constant per world (see `map_offset`).
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from PIL import Image

ARENA = os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena")
if not os.path.isdir(ARENA):
    ARENA = "/home/robotics/arena_ws/src/Arena"
WORLDS_DIR = os.path.join(ARENA, "arena_simulation_setup", "worlds")
ASSETS_DIR = os.path.join(ARENA, "_assets", "default")

# fallback footprints (length, width in m) by model-name keyword, used when no annotation is cached
_FALLBACK = [("sofa", 2.0, 0.9), ("bench", 1.8, 0.55), ("bed", 2.1, 1.0), ("desk", 1.6, 0.8), ("table", 1.2, 0.8),
             ("counter", 1.8, 1.4), ("shelf", 1.2, 0.5), ("rack", 1.2, 0.5), ("cabinet", 1.0, 0.5), ("cupboard", 1.0, 0.5),
             ("chair", 0.55, 0.55), ("plant", 0.5, 0.5), ("trash", 0.4, 0.4), ("extinguisher", 0.3, 0.3), ("vending", 1.0, 0.8),
             ("machine", 0.8, 0.7), ("cooler", 0.4, 0.4), ("dispenser", 0.4, 0.4), ("box", 0.5, 0.5), ("curtain", 0.3, 2.0), ("personenleitsystem", 1.5, 0.15)]
# objects mounted on walls or above head height do not block pedestrians
_IGNORE = ("tv", "mirror", "dryer", "painting", "keyboard", "monitor", "phone", "laptop", "mouse", "paper", "pencil",
           "donut", "coffee", "ringbinder", "smartphone", "tray", "markerboard")


def _pose(o: dict) -> tuple[float, float, float]:
    p = o.get("pose")
    if isinstance(p, dict):
        return float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("yaw", p.get("theta", 0)))
    return float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0


def object_footprint(model: str) -> tuple[float, float, float, float, str] | None:
    """(xmin, xmax, ymin, ymax) in the object frame and the source, None when it does not block."""
    low = model.lower()
    if any(k in low for k in _IGNORE):
        return None
    dom, _, name = model.rpartition("/")
    for d in ([dom] if dom else ["Common", "Office", "Hospital", "Residential"]):
        ann = os.path.join(ASSETS_DIR, d, "Object", name, "annotation.yaml")
        if os.path.exists(ann):
            bb = yaml.safe_load(open(ann)).get("bounding_box")
            if bb and bb[2][0] < 1.0:  # starts below 1 m: blocks walking
                return bb[0][0], bb[0][1], bb[1][0], bb[1][1], "annotation"
            if bb:
                return None
    for key, length, width in _FALLBACK:
        if key in low:
            return -length / 2, length / 2, -width / 2, width / 2, "fallback:" + key
    return -0.4, 0.4, -0.4, 0.4, "fallback:default"


@dataclass
class World:
    name: str
    origin: tuple[float, float]
    resolution: float
    occ: np.ndarray  # bool [H, W], row 0 = top (max y)
    objects: list[dict] = field(default_factory=list)
    _dist: np.ndarray | None = None

    @classmethod
    def load(cls, name: str) -> "World":
        wdir = os.path.join(WORLDS_DIR, name)
        meta = yaml.safe_load(open(os.path.join(wdir, "map", "0.yaml")))
        img = np.asarray(Image.open(os.path.join(wdir, "map", meta["image"])).convert("L"), dtype=np.float32) / 255.0
        occ = (1.0 - img) > meta.get("occupied_thresh", 0.65) if not meta.get("negate", 0) else img > meta.get("occupied_thresh", 0.65)
        w = cls(name=name, origin=(float(meta["origin"][0]), float(meta["origin"][1])), resolution=float(meta["resolution"]), occ=occ.copy())
        doc = yaml.safe_load(open(os.path.join(wdir, "world.yaml")))
        levels = doc["levels"]
        level = levels[0] if isinstance(levels, list) else next(iter(levels.values()))
        for o in level.get("objects", []) or []:
            model = o.get("model", "")
            fp = object_footprint(model)
            x, y, yaw = _pose(o)
            w.objects.append({"name": o.get("name"), "model": model, "pose": [x, y, yaw], "footprint": fp[:4] if fp else None,
                              "footprint_source": fp[4] if fp else "non-blocking"})
            if fp:
                w.stamp_box(x, y, yaw, fp[:4])
        return w

    # pixel <-> world
    def to_px(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self.origin[0]) / self.resolution)
        r = self.occ.shape[0] - 1 - int((y - self.origin[1]) / self.resolution)
        return r, c

    def to_world(self, r: int, c: int) -> tuple[float, float]:
        return self.origin[0] + (c + 0.5) * self.resolution, self.origin[1] + (self.occ.shape[0] - 1 - r + 0.5) * self.resolution

    def stamp_box(self, x: float, y: float, yaw: float, fp: tuple[float, float, float, float]) -> None:
        x0, x1, y0, y1 = fp
        c, s = math.cos(yaw), math.sin(yaw)
        step = self.resolution / 2
        for u in np.arange(x0, x1 + 1e-9, step):
            for v in np.arange(y0, y1 + 1e-9, step):
                r, cc = self.to_px(x + c * u - s * v, y + s * u + c * v)
                if 0 <= r < self.occ.shape[0] and 0 <= cc < self.occ.shape[1]:
                    self.occ[r, cc] = True
        self._dist = None

    def add_obstacles(self, items: list[dict]) -> "World":
        """Copy with scenario objects stamped in (items: {model, pose})."""
        w = World(self.name, self.origin, self.resolution, self.occ.copy(), list(self.objects))
        for it in items:
            fp = object_footprint(it["model"])
            if fp:
                w.stamp_box(it["pose"][0], it["pose"][1], it["pose"][2] if len(it["pose"]) > 2 else 0.0, fp[:4])
        return w

    @property
    def dist(self) -> np.ndarray:
        """Distance (m) of every cell to the nearest occupied cell or the map border."""
        if self._dist is None:
            from scipy.ndimage import distance_transform_edt
            free = ~self.occ
            free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
            self._dist = distance_transform_edt(free) * self.resolution
        return self._dist

    def clearance(self, x: float, y: float) -> float:
        r, c = self.to_px(x, y)
        if not (0 <= r < self.occ.shape[0] and 0 <= c < self.occ.shape[1]):
            return 0.0
        return float(self.dist[r, c])

    def ray_free(self, x: float, y: float, ang: float, max_d: float = 8.0) -> float:
        """Free distance from (x, y) along `ang` until an occupied cell."""
        d = 0.0
        while d < max_d:
            r, c = self.to_px(x + d * math.cos(ang), y + d * math.sin(ang))
            if not (0 <= r < self.occ.shape[0] and 0 <= c < self.occ.shape[1]) or self.occ[r, c]:
                return d
            d += self.resolution / 2
        return max_d

    def segment_clear(self, a: tuple[float, float], b: tuple[float, float], margin: float) -> bool:
        n = max(2, int(math.dist(a, b) / 0.1))
        return all(self.clearance(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) >= margin for t in np.linspace(0, 1, n))

    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        n = max(2, int(math.dist(a, b) / (self.resolution / 2)))
        for t in np.linspace(0, 1, n)[1:-1]:
            r, c = self.to_px(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if self.occ[r, c]:
                return False
        return True

    def path_length(self, a: tuple[float, float], b: tuple[float, float], margin: float = 0.3, cost: np.ndarray | None = None,
                    cost_weight: float = 0.0, box: float | None = None) -> tuple[float, list[tuple[float, float]]] | None:
        """8-connected A* on cells with clearance >= margin. Optional per-cell cost (same grid, 0..1) adds
        cost_weight * cost per metre (a social costmap); returns (geometric length, path)."""
        free = self.dist >= margin
        H, W = free.shape
        s, g = self.to_px(*a), self.to_px(*b)
        if not all(0 <= p[0] < H and 0 <= p[1] < W for p in (s, g)) or not (free[s] and free[g]):
            return None
        res = self.resolution
        if box is not None:  # search only inside the endpoints' bounding box grown by `box` metres
            m = int(box / res)
            r0, r1 = max(0, min(s[0], g[0]) - m), min(H - 1, max(s[0], g[0]) + m)
            c0, c1 = max(0, min(s[1], g[1]) - m), min(W - 1, max(s[1], g[1]) + m)
            inside = np.zeros_like(free)
            inside[r0:r1 + 1, c0:c1 + 1] = True
            free = free & inside
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        best = {s: 0.0}
        prev = {}
        pq = [(0.0, 0.0, s)]
        while pq:
            _, gcost, u = heapq.heappop(pq)
            if u == g:
                break
            if gcost > best.get(u, 1e18) + 1e-9:
                continue
            for dr, dc, w in moves:
                v = (u[0] + dr, u[1] + dc)
                if not (0 <= v[0] < H and 0 <= v[1] < W) or not free[v]:
                    continue
                step = w * res
                if cost is not None:
                    step *= 1.0 + cost_weight * float(cost[v])
                nc = gcost + step
                if nc < best.get(v, 1e18) - 1e-12:
                    best[v] = nc
                    prev[v] = u
                    h = math.hypot(v[0] - g[0], v[1] - g[1]) * res
                    heapq.heappush(pq, (nc + h, nc, v))
        if g not in prev and g != s:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        cells.reverse()
        pts = [self.to_world(*c) for c in cells]
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        return length, pts[::4] + [pts[-1]]


def map_offset(world: str) -> tuple[float, float]:
    """Arena `map` frame = world frame + offset. Measured on map_empty (+5.30) and reception (+5.25) as
    5.0 - map origin per axis; confirmed per world by the Stage A probe (scripts/dv1_world_probe_report.py)."""
    meta = yaml.safe_load(open(os.path.join(WORLDS_DIR, world, "map", "0.yaml")))
    return 5.0 - float(meta["origin"][0]), 5.0 - float(meta["origin"][1])


class Route:
    """Robot route polyline with arc-length parameterisation and a local (s, lateral) frame."""

    def __init__(self, pts: list[tuple[float, float]]):
        self.pts = [tuple(map(float, p)) for p in pts]
        self.seg = [math.dist(self.pts[i], self.pts[i + 1]) for i in range(len(self.pts) - 1)]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.length = float(self.cum[-1])

    def at(self, s: float) -> tuple[float, float, float]:
        """Point and heading at arc length s; beyond the ends the first / last segment is extended."""
        if s <= 0.0:
            i = 0
        elif s >= self.length:
            i = len(self.seg) - 1
        else:
            i = min(int(np.searchsorted(self.cum, s, side="right") - 1), len(self.seg) - 1)
        t = (s - self.cum[i]) / self.seg[i] if self.seg[i] > 0 else 0.0
        (x0, y0), (x1, y1) = self.pts[i], self.pts[i + 1]
        return float(x0 + (x1 - x0) * t), float(y0 + (y1 - y0) * t), float(math.atan2(y1 - y0, x1 - x0))

    def local_to_world(self, s: float, v: float) -> tuple[float, float]:
        x, y, h = self.at(s)
        return float(x - v * math.sin(h)), float(y + v * math.cos(h))

    def heading(self, s: float) -> float:
        return self.at(s)[2]

    def corners(self) -> list[float]:
        return [float(c) for c in self.cum[1:-1]]

    def lateral_free(self, world: World, s: float) -> tuple[float, float]:
        x, y, h = self.at(s)
        return world.ray_free(x, y, h + math.pi / 2), world.ray_free(x, y, h - math.pi / 2)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
