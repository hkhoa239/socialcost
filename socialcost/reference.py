"""An independent numpy rewrite of the arena_social_cost_layer formulas, for verification only.

Written from the layer's README and social_cost_layer.cpp, without the C++ shim, so a label
painted by labels.paint can be checked against a second implementation on any grid pose.
"""

from __future__ import annotations

import math

import numpy as np

from .labels import GridSpec, LayerParams, Ped

_ALERT = (4, 5)  # SURPRISED, CURIOUS
_DANGER = (3, 6)  # PANIC, THREATENING


def cell_centres(grid: GridSpec, grid_pose: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """World coordinates of every cell centre, arrays indexed [j, i]."""
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    lx = grid.origin_x + (i + 0.5) * grid.resolution
    ly = grid.origin_y + (j + 0.5) * grid.resolution
    gx, gy, gyaw = grid_pose
    c, s = math.cos(gyaw), math.sin(gyaw)
    return gx + c * lx - s * ly, gy + s * lx + c * ly


def reference_paint(params: LayerParams, peds: list[Ped], grid: GridSpec, grid_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    wx, wy = cell_centres(grid, grid_pose)
    scale = params.personal_space_sigma / 0.45

    def to_cell(v: np.ndarray) -> np.ndarray:
        return (np.clip(v * params.cost_scale + params.cost_bias, 0.0, 1.0) * params.max_cost_value).astype(np.uint8)

    out = np.zeros((grid.height, grid.width), np.uint8)
    for p in peds:
        if p.animation_state in _DANGER:
            sigma, amp = 1.2 * scale, 1.0
        elif p.animation_state in _ALERT:
            sigma, amp = 0.75 * scale, 0.75
        else:
            sigma, amp = params.personal_space_sigma, 0.5
        d2 = (wx - p.x) ** 2 + (wy - p.y) ** 2
        v = amp * np.exp(-np.power(d2 / (2 * sigma * sigma), params.falloff_shape))
        out = np.maximum(out, np.where(d2 <= params.personal_space_radius ** 2, to_cell(v), 0).astype(np.uint8))
    groups: dict[int, list[Ped]] = {}
    for p in peds:
        if p.interaction_id >= 0:
            groups.setdefault(p.interaction_id, []).append(p)
    for members in groups.values():
        kind = members[0].interaction_type
        w = params.interaction_weights[kind] if kind < len(params.interaction_weights) else 0.0
        if w <= 0:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pa, pb = members[a], members[b]
                abx, aby = pb.x - pa.x, pb.y - pa.y
                ab2 = abx * abx + aby * aby
                t = np.clip(((wx - pa.x) * abx + (wy - pa.y) * aby) / ab2, 0, 1) if ab2 > 1e-9 else 0.0
                d2 = (wx - (pa.x + t * abx)) ** 2 + (wy - (pa.y + t * aby)) ** 2
                v = w * np.exp(-np.power(d2 / (2 * params.group_sigma ** 2), params.falloff_shape))
                out = np.maximum(out, np.where(d2 <= params.group_extra_radius ** 2, to_cell(v), 0).astype(np.uint8))
    return out


def to_cell_cost(params: LayerParams, normalized: float) -> int:
    return int(min(max(normalized * params.cost_scale + params.cost_bias, 0.0), 1.0) * params.max_cost_value)
