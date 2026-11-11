"""socialcost_v2 label tests (run in the arena container: needs libsocialcost_grid.so).

    cd /opt/arena_ws/src/deps/socialcost && PYTHONPATH=. /opt/venv/bin/python -m pytest tests/test_labels_v2.py -q
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import random

import numpy as np
import pytest

from socialcost.dataset import interaction_frame_at, read_jsonl
from socialcost.labels import GridSpec, LayerParams, Ped, paint
from socialcost.labels_v2 import SPECS, expand_queue_groups, labels_and_components, mst_edges, paint_v2

DV1 = os.environ.get("SC_DV1_ROOT", "/opt/arena_ws/data/socialcost/dataset_v1")
BASE = LayerParams.from_overlay()
GRID = GridSpec()


# ------------------------------------------------------------------ independent reference of the written equations
def reference_paint(P: LayerParams, peds: list[Ped], grid: GridSpec, pose, pairs_of=None) -> np.ndarray:
    """NumPy transcription of the manifest equations (not the C code)."""
    ox, oy, res = grid.origin_x, grid.origin_y, grid.resolution
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    lx, ly = ox + (i + 0.5) * res, oy + (j + 0.5) * res
    gx, gy, gyaw = pose
    c, s = math.cos(gyaw), math.sin(gyaw)
    wx, wy = (gx + lx, gy + ly) if gyaw == 0.0 else (gx + c * lx - s * ly, gy + s * lx + c * ly)
    sc = P.personal_space_sigma / 0.45
    sig = {"base": P.personal_space_sigma, "alert": 0.75 * sc, "danger": 1.2 * sc}
    amp = {"base": 0.5, "alert": 0.75, "danger": 1.0}

    def cell(v):
        return np.floor(np.clip(v * P.cost_scale + P.cost_bias, 0, 1) * P.max_cost_value).astype(np.int32)

    best = np.zeros(wx.shape, np.int32)
    for p in peds:
        k = "danger" if p.animation_state in (3, 6) else "alert" if p.animation_state in (4, 5) else "base"
        d2 = (wx - p.x) ** 2 + (wy - p.y) ** 2
        v = amp[k] * np.exp(-np.power(d2 / (2 * sig[k] ** 2), P.falloff_shape))
        best = np.maximum(best, np.where(d2 <= P.personal_space_radius ** 2, cell(v), 0))
    groups: dict[int, list[Ped]] = {}
    for p in peds:
        if p.interaction_id >= 0:
            groups.setdefault(p.interaction_id, []).append(p)
    for g in groups.values():
        w = P.interaction_weights[g[0].interaction_type] if g[0].interaction_type < len(P.interaction_weights) else 0.0
        if w <= 0:
            continue
        for a, b in itertools.combinations(g, 2):
            abx, aby = b.x - a.x, b.y - a.y
            ab2 = abx * abx + aby * aby
            t = np.clip(((wx - a.x) * abx + (wy - a.y) * aby) / ab2, 0, 1) if ab2 > 1e-9 else 0.0
            d2 = (wx - (a.x + t * abx)) ** 2 + (wy - (a.y + t * aby)) ** 2
            v = w * np.exp(-np.power(d2 / (2 * P.group_sigma ** 2), P.falloff_shape))
            best = np.maximum(best, np.where(d2 <= P.group_extra_radius ** 2, cell(v), 0))
    return best.astype(np.uint8)


def random_scene(rng: random.Random, n: int, queue: bool = False) -> list[Ped]:
    peds = []
    for k in range(n):
        st = rng.choice([0, 0, 0, 1, 4, 3])
        peds.append(Ped(rng.uniform(-4, 4), rng.uniform(-4, 4), st))
    # two interactions: a talk pair/group and (optionally) a queue
    m = rng.randint(2, min(4, n))
    for k in range(m):
        peds[k] = dataclasses.replace(peds[k], interaction_id=7, interaction_type=rng.choice([0, 1]))
    if queue and n - m >= 3:
        for k in range(m, n):
            peds[k] = dataclasses.replace(peds[k], interaction_id=9, interaction_type=6)
    return peds


@pytest.mark.parametrize("spec", ["v1", "v2", "v2_fixedsigma"])
def test_c_painter_matches_written_equations(spec):
    rng = random.Random(0)
    P = SPECS[spec].layer_params(BASE)
    worst, cells = 0, 0
    for trial in range(25):
        peds = random_scene(rng, rng.randint(2, 7))
        pose = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.choice([0.0, rng.uniform(-3, 3)]))
        a = paint(P, peds, GRID, pose).astype(int)
        b = reference_paint(P, peds, GRID, pose).astype(int)
        worst = max(worst, int(np.abs(a - b).max()))
        cells += int((a != b).sum())
    # identical up to libm rounding at floor() boundaries
    assert worst <= 1 and cells <= 25 * 40000 * 1e-4, (worst, cells)


def test_edge_decomposition_is_exact():
    """Painting each pair as its own 2-person interaction == the layer's all-pairs group term."""
    rng = random.Random(1)
    P = SPECS["v2"].layer_params(BASE)
    for trial in range(20):
        n = rng.randint(3, 8)
        peds = [Ped(rng.uniform(-4, 4), rng.uniform(-4, 4), 0, 5, 6) for _ in range(n)]
        whole = paint(P, peds, GRID, (0.0, 0.0, 0.0))
        split = [Ped(p.x, p.y, 0) for p in peds]
        for nid, (a, b) in enumerate(itertools.combinations(range(n), 2), start=100):
            split += [Ped(peds[a].x, peds[a].y, 0, nid, 6), Ped(peds[b].x, peds[b].y, 0, nid, 6)]
        assert np.array_equal(whole, paint(P, split, GRID, (0.0, 0.0, 0.0)))


def test_mst_line_and_bend():
    line = np.array([[0, 0], [0.8, 0], [1.6, 0], [2.4, 0], [3.2, 0]], float)
    assert sorted(mst_edges(line)) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    shuffled = line[[3, 0, 4, 1, 2]]
    e = mst_edges(shuffled)
    assert len(e) == 4 and all(np.hypot(*(shuffled[a] - shuffled[b])) < 0.81 for a, b in e)
    bend = np.array([[0, 0], [0.8, 0], [1.6, 0], [1.6, 0.8], [1.6, 1.6]], float)
    assert len(mst_edges(bend)) == 4


def test_mst_queue_equals_all_pairs_for_collinear_queue():
    """For people on a straight line the corridor and all pairs paint the same cells (all chords lie on the line)."""
    P = SPECS["v2"].layer_params(BASE)
    peds = [Ped(-2.0 + 0.8 * k, 0.5, 0, 3, 6) for k in range(6)]
    a = paint_v2(SPECS["v2"], BASE, peds, GRID, (0.0, 0.0, 0.0))
    b = paint_v2(SPECS["v2_allpairs"], BASE, peds, GRID, (0.0, 0.0, 0.0))
    assert np.array_equal(a, b)
    assert np.array_equal(b, paint(P, peds, GRID, (0.0, 0.0, 0.0)))


def test_mst_queue_drops_chord_across_bend():
    peds = [Ped(x, y, 0, 3, 6) for x, y in [(-2, -2), (-1.2, -2), (-0.4, -2), (0.4, -2), (0.4, -1.2), (0.4, -0.4), (0.4, 0.4), (0.4, 1.2)]]
    a = paint_v2(SPECS["v2"], BASE, peds, GRID, (0.0, 0.0, 0.0))
    b = paint_v2(SPECS["v2_allpairs"], BASE, peds, GRID, (0.0, 0.0, 0.0))
    assert (b >= a).all() and (b > 0).sum() > (a > 0).sum()


def test_non_queue_groups_untouched_by_expansion():
    peds = [Ped(0, 0, 0, 1, 1), Ped(1, 0, 0, 1, 1), Ped(0.5, 1, 0, 1, 1), Ped(3, 3, 0)]
    assert expand_queue_groups(peds, SPECS["v2"]) == peds


def _episodes(n=6):
    eps = sorted(os.listdir(os.path.join(DV1, "episodes")))
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(DV1, "manifests", "manifest_v1.json")))["episodes"]}
    fams = {}
    for e in eps:
        f = man.get(e, {}).get("family")
        if f and f not in fams and os.path.exists(os.path.join(DV1, "episodes", e, "pedestrian_states.jsonl")):
            fams[f] = e
    return list(fams.values())[:n] + eps[:1]


@pytest.mark.skipif(not os.path.isdir(os.path.join(DV1, "episodes")), reason="dataset_v1 not mounted")
def test_v1_spec_reproduces_stored_v1_labels():
    """The v2 code path with v1 parameters writes exactly the stored dataset_v1 labels and components."""
    for eid in _episodes():
        d = os.path.join(DV1, "episodes", eid)
        meta = json.load(open(os.path.join(d, "episode_metadata.json")))
        base = LayerParams.from_mapping(meta["label"]["layer_params"])
        grid = GridSpec(**meta["label"]["grid"])
        robots = {r["sample"]: r for r in read_jsonl(os.path.join(d, "robot_states.jsonl"))}
        peds = {p["sample"]: p for p in read_jsonl(os.path.join(d, "pedestrian_states.jsonl"))}
        frames = read_jsonl(os.path.join(d, "interactions.jsonl"))
        for k in sorted(robots)[::25]:
            r = robots[k]
            pose = (r["pose_map_base"]["x"], r["pose_map_base"]["y"], r["pose_map_base"]["yaw"])
            out = labels_and_components(SPECS["v1"], base, grid, peds[k]["pedestrians"], interaction_frame_at(frames, r["stamp_ns"]), pose)
            name = f"{k:06d}"
            assert np.array_equal(out["native"], np.load(os.path.join(d, "social_cost", name + ".npy"))), (eid, k, "native")
            assert np.array_equal(out["membership"], np.load(os.path.join(d, "social_cost_membership", name + ".npy"))), (eid, k, "membership")
            assert np.array_equal(out["extended"], np.load(os.path.join(d, "social_cost_extended", name + ".npy"))), (eid, k, "extended")
            comp = np.load(os.path.join(d, "components", name + ".npz"))
            for key in ("personal_space", "conversation", "queue", "gesture"):
                assert np.array_equal(out[key], comp[key]), (eid, k, key)
