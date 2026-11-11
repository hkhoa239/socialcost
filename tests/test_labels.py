"""Unit tests of the offline label path, no ROS graph needed.

    python -m pytest tests/test_labels.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from socialcost.dataset import with_membership
from socialcost.labels import GridSpec, LayerParams, Ped, paint
from socialcost.recorder import interpolate_peds

PARAMS = LayerParams.from_overlay(costmap="local_costmap")


def reference(params: LayerParams, peds: list[Ped], grid: GridSpec) -> np.ndarray:
    """The layer's formulas written out in numpy (README of arena_social_cost_layer), world-aligned grid."""
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    wx = grid.origin_x + (i + 0.5) * grid.resolution
    wy = grid.origin_y + (j + 0.5) * grid.resolution
    scale = params.personal_space_sigma / 0.45
    sigma = {3: 1.2 * scale, 6: 1.2 * scale, 4: 0.75 * scale, 5: 0.75 * scale}
    amp = {3: 1.0, 6: 1.0, 4: 0.75, 5: 0.75}

    def cell(v):
        return (np.clip(v * params.cost_scale + params.cost_bias, 0, 1) * params.max_cost_value).astype(np.uint8)

    out = np.zeros((grid.height, grid.width), np.uint8)
    for p in peds:
        d2 = (wx - p.x) ** 2 + (wy - p.y) ** 2
        s = sigma.get(p.animation_state, params.personal_space_sigma)
        a = amp.get(p.animation_state, 0.5)
        v = a * np.exp(-((d2 / (2 * s * s)) ** params.falloff_shape))
        out = np.maximum(out, np.where(d2 <= params.personal_space_radius ** 2, cell(v), 0))
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
                ab = np.array([pb.x - pa.x, pb.y - pa.y])
                t = np.clip(((wx - pa.x) * ab[0] + (wy - pa.y) * ab[1]) / max(ab @ ab, 1e-9), 0, 1) if ab @ ab > 1e-9 else 0.0
                d2 = (wx - (pa.x + t * ab[0])) ** 2 + (wy - (pa.y + t * ab[1])) ** 2
                v = w * np.exp(-((d2 / (2 * params.group_sigma ** 2)) ** params.falloff_shape))
                out = np.maximum(out, np.where(d2 <= params.group_extra_radius ** 2, cell(v), 0))
    return out


def random_peds(rng: np.random.Generator) -> list[Ped]:
    peds = [Ped(rng.uniform(-4, 4), rng.uniform(-4, 4), int(rng.integers(0, 7))) for _ in range(3)]
    for gid, kind in enumerate((1, 6, 12)):
        cx, cy = rng.uniform(-3, 3, 2)
        peds += [Ped(cx + rng.uniform(-1, 1), cy + rng.uniform(-1, 1), 0, gid, kind) for _ in range(3)]
    return peds


@pytest.mark.parametrize("block", ["local_costmap", "global_costmap"])
def test_shim_matches_numpy_reference(block: str) -> None:
    params = LayerParams.from_overlay(costmap=block)
    grid = GridSpec(resolution=0.05, width=160, height=140, origin_x=-4.0, origin_y=-3.5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        peds = random_peds(rng)
        diff = np.abs(paint(params, peds, grid).astype(int) - reference(params, peds, grid).astype(int))
        # float rounding at the truncation boundary may move a few cells by one
        assert diff.max() <= 1 and (diff > 0).mean() < 1e-3


def test_rotated_grid_is_the_same_field() -> None:
    """A robot-frame grid at yaw 0 equals the world grid shifted to the robot; at yaw 90 it is that grid rotated."""
    peds = random_peds(np.random.default_rng(1))
    g = GridSpec.centered(6.0, 0.05)
    robot = (0.7, -0.3)
    world = paint(PARAMS, peds, GridSpec(resolution=0.05, width=g.width, height=g.height, origin_x=robot[0] + g.origin_x, origin_y=robot[1] + g.origin_y))
    assert np.array_equal(paint(PARAMS, peds, g, (robot[0], robot[1], 0.0)), world)
    rot = paint(PARAMS, peds, g, (robot[0], robot[1], math.pi / 2))
    # robot x (forward) is world +y, so rot[j, i] = world[i, N-1-j], a counter-clockwise quarter turn
    assert (np.abs(rot.astype(int) - np.rot90(world, k=1).astype(int)) > 1).mean() < 1e-3


def test_group_walk_and_queue_weights() -> None:
    """GROUP_WALK (12) has weight 0 in the overlay: two walkers 1.6 m apart leave the gap between them free."""
    g = GridSpec(resolution=0.05, width=100, height=40, origin_x=-2.5, origin_y=-1.0)
    walkers = [Ped(-0.8, 0.0, 1, 7, 12), Ped(0.8, 0.0, 1, 7, 12)]
    queue = [Ped(-0.8, 0.0, 0, 7, 6), Ped(0.8, 0.0, 0, 7, 6)]
    mid = (20, 50)  # cell holding (0, 0)
    assert paint(PARAMS, walkers, g)[mid] < paint(PARAMS, queue, g)[mid]
    assert paint(PARAMS, queue, g)[mid] >= int((1.0 * PARAMS.cost_scale + PARAMS.cost_bias) * PARAMS.max_cost_value) - 1


def test_membership_adds_queued_members_only() -> None:
    states = [{"id": 1, "interaction_id": 4, "interaction_type": 6}, {"id": 2, "interaction_id": -1, "interaction_type": 0},
              {"id": 3, "interaction_id": -1, "interaction_type": 0}, {"id": 9, "interaction_id": 2, "interaction_type": 0}]
    frame = {"interactions": [{"interaction_id": 4, "interaction_type": 6, "outcome": 1, "member_ids": [1, 2, 3], "member_roles": [0, 1, 1]},
                              {"interaction_id": 5, "interaction_type": 1, "outcome": 0, "member_ids": [9], "member_roles": [0]}]}
    out = with_membership(states, frame)
    assert [(p["interaction_id"], p["interaction_type"]) for p in out] == [(4, 6), (4, 6), (4, 6), (2, 0)]
    assert states[1]["interaction_id"] == -1  # input untouched


def test_interpolation_between_rosters() -> None:
    a = (0, [{"name": "p", "x": 0.0, "y": 0.0, "z": 0.0, "vx": 1.0, "vy": 0.0, "yaw": 3.1, "animation_state": 0, "interaction_id": -1}])
    b = (100, [{"name": "p", "x": 1.0, "y": 2.0, "z": 0.0, "vx": 1.0, "vy": 0.0, "yaw": -3.1, "animation_state": 1, "interaction_id": 3}])
    out, sync = interpolate_peds(a, b, 75)
    p = out[0]
    assert (p["x"], p["y"]) == pytest.approx((0.75, 1.5))
    assert abs(abs(p["yaw"]) - math.pi) < 0.1  # wraps through pi, not through 0
    # discrete state is the one in force at the stamp: the interaction starts only at t = 100
    assert p["interaction_id"] == -1 and p["animation_state"] == 0 and sync["alpha"] == pytest.approx(0.75)
