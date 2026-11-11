"""Offline social-cost labels from pedestrian states, through arena_social_cost_layer's cost functions.

The parameters are read from the same Nav2 overlay the live layer loads
(arena_robots/config/nav2/overlays/social_cost_layer.yaml), so a label and the
live layer never disagree on a number.
"""

from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import yaml

# INSCRIBED_INFLATED_OBSTACLE - 1, the layer clamps max_cost_value to this.
MAX_LAYER_COST = 252

DEFAULT_OVERLAY = os.path.join(
    os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena"),
    "arena_robots/arena_robots/config/nav2/overlays/social_cost_layer.yaml",
)


@dataclass(frozen=True)
class LayerParams:
    """One `social_cost_layer` parameter block, defaults as in social_cost_layer.hpp."""

    personal_space_radius: float = 1.0
    personal_space_sigma: float = 0.35
    group_extra_radius: float = 1.0
    group_sigma: float = 0.4
    falloff_shape: float = 2.0
    max_cost_value: int = 200
    cost_scale: float = 1.0
    cost_bias: float = 0.0
    stale_timeout: float = 1.0
    interaction_weights: tuple[float, ...] = ()

    @classmethod
    def from_overlay(cls, path: str = DEFAULT_OVERLAY, costmap: str = "local_costmap") -> "LayerParams":
        """The block of `costmap` (local_costmap or global_costmap) in a Nav2 overlay yaml."""
        with open(path) as f:
            doc = yaml.safe_load(f)
        block = doc[costmap][costmap]["ros__parameters"]["social_cost_layer"]
        return cls.from_mapping(block)

    @classmethod
    def from_mapping(cls, block: Mapping) -> "LayerParams":
        names = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in block.items() if k in names}
        if "interaction_weights" in kwargs:
            kwargs["interaction_weights"] = tuple(float(w) for w in kwargs["interaction_weights"])
        if "max_cost_value" in kwargs:
            kwargs["max_cost_value"] = int(min(max(int(kwargs["max_cost_value"]), 0), MAX_LAYER_COST))
        return cls(**kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["interaction_weights"] = list(self.interaction_weights)
        return d


@dataclass(frozen=True)
class GridSpec:
    """A label grid. Cell (i, j) is out[j, i], its centre at (origin_x + (i+.5) res, origin_y + (j+.5) res)
    in the grid frame. A robot-centred grid uses frame `base_footprint`, x forward, y left."""

    resolution: float = 0.05
    width: int = 200
    height: int = 200
    origin_x: float = -5.0
    origin_y: float = -5.0
    frame: str = "base_footprint"

    @classmethod
    def centered(cls, size_m: float, resolution: float, frame: str = "base_footprint") -> "GridSpec":
        n = int(round(size_m / resolution))
        return cls(resolution=resolution, width=n, height=n, origin_x=-n * resolution / 2, origin_y=-n * resolution / 2, frame=frame)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ped:
    """The fields of arena_people_msgs/Pedestrian the layer reads."""

    x: float
    y: float
    animation_state: int = 0
    interaction_id: int = -1
    interaction_type: int = 0
    extra: dict = field(default_factory=dict)


class _CParams(ctypes.Structure):
    _fields_ = [
        ("personal_space_radius", ctypes.c_double),
        ("personal_space_sigma", ctypes.c_double),
        ("group_extra_radius", ctypes.c_double),
        ("group_sigma", ctypes.c_double),
        ("falloff_shape", ctypes.c_double),
        ("max_cost_value", ctypes.c_int),
        ("cost_scale", ctypes.c_double),
        ("cost_bias", ctypes.c_double),
    ]


class _CPed(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("animation_state", ctypes.c_uint8),
        ("interaction_id", ctypes.c_int32),
        ("interaction_type", ctypes.c_uint8),
    ]


def _find_library() -> str:
    explicit = os.environ.get("SOCIALCOST_GRID_LIB")
    if explicit:
        return explicit
    try:
        from ament_index_python.packages import get_package_prefix

        return os.path.join(get_package_prefix("socialcost"), "lib", "libsocialcost_grid.so")
    except Exception:  # noqa: BLE001 - fall back to the default install path
        return "/opt/arena_ws/install/socialcost/lib/libsocialcost_grid.so"


_lib = None


def _library():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_find_library())
        lib.sc_paint.restype = ctypes.c_int
        lib.sc_paint.argtypes = [
            ctypes.POINTER(_CParams), ctypes.POINTER(_CPed), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        _lib = lib
    return _lib


def paint(params: LayerParams, peds: Iterable[Ped], grid: GridSpec, grid_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """The social layer alone on `grid`, placed at grid_pose = (x, y, yaw) in the pedestrians' frame.

    Returns a (height, width) uint8 array in [0, params.max_cost_value]; 0 is free.
    """
    peds = list(peds)
    lib = _library()
    cp = _CParams(
        params.personal_space_radius, params.personal_space_sigma, params.group_extra_radius,
        params.group_sigma, params.falloff_shape, int(params.max_cost_value), params.cost_scale, params.cost_bias,
    )
    arr = (_CPed * max(len(peds), 1))()
    for k, p in enumerate(peds):
        arr[k] = _CPed(float(p.x), float(p.y), int(p.animation_state) & 0xFF, int(p.interaction_id), int(p.interaction_type) & 0xFF)
    weights = (ctypes.c_double * max(len(params.interaction_weights), 1))(*params.interaction_weights)
    out = np.zeros((grid.height, grid.width), dtype=np.uint8)
    gx, gy, gyaw = (float(v) for v in grid_pose)
    rc = lib.sc_paint(
        ctypes.byref(cp), arr, len(peds), weights, len(params.interaction_weights),
        gx, gy, gyaw, grid.origin_x, grid.origin_y, grid.resolution, grid.width, grid.height,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if rc != 0:
        raise ValueError(f"sc_paint failed with {rc} for grid {grid}")
    return out


def peds_from_msg(msg) -> list[Ped]:
    """Ped list from an arena_people_msgs/Pedestrians message, roster order kept (it decides group typing)."""
    return [
        Ped(p.pose.position.x, p.pose.position.y, p.animation_state, p.interaction_id, p.interaction_type)
        for p in msg.pedestrians
    ]


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
