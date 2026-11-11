"""Synchronization and geometry checks on recorded datasets, as pytest cases.

    SOCIALCOST_DATASETS=/opt/arena_ws/data/socialcost/smoke_test:/opt/arena_ws/data/socialcost/verify_scenarios \
        python -m pytest tests/test_dataset_geometry.py -q
"""

from __future__ import annotations

import os

import pytest

from socialcost import verify as V

ROOTS = [r for r in os.environ.get("SOCIALCOST_DATASETS", "/opt/arena_ws/data/socialcost/smoke_test").split(":") if r]
EPISODES = V.load_episodes(*ROOTS)

if not EPISODES:
    pytest.skip(f"no recorded episodes under {ROOTS}", allow_module_level=True)


@pytest.fixture(scope="module", params=EPISODES, ids=[e.name for e in EPISODES])
def ep(request) -> V.Ep:
    return request.param


@pytest.fixture(scope="module")
def geometry(ep: V.Ep) -> dict:
    return V.geometry(ep)


def test_inventory_complete(ep: V.Ep) -> None:
    inv = V.inventory(ep)
    p = inv["problems"]
    assert inv["samples"] == inv["expected_samples"], "missing samples"
    assert not p["empty_files"] and not p["unreadable"] and not p["count_mismatch"]
    assert not p["nan_only_depth_samples"] and p["inf_depth_values"] == 0


def test_synchronization(ep: V.Ep) -> None:
    r = V.synchronization(ep)
    m = r["metrics"]
    assert m["rgb_minus_depth_ms"]["max_abs"] == 0.0
    assert m["robot_pose_minus_image_ms"]["max_abs"] == 0.0
    assert m["bracketing_peds_msgs_ms"]["brackets_image_stamp"] and m["bracketing_peds_msgs_ms"]["nearest_max"] <= 50.0
    assert m["monotonic"] and m["inside_episode_window"] and not m["missing_slots"]
    assert r["status"] == V.PASS, r["failing"]


@pytest.mark.parametrize("check", ["tf_base_camera", "grid_definition", "metric_units", "values_and_invalid"])
def test_geometry(geometry: dict, check: str) -> None:
    assert geometry[check]["status"] == V.PASS, (geometry[check]["metrics"], geometry[check]["failing"])


def test_label_orientation_from_depth(geometry: dict) -> None:
    r = geometry["label_orientation_from_depth"]
    if r["status"] == V.NOT_TESTED:
        pytest.skip("no visible pedestrian in this episode")
    assert r["status"] == V.PASS, r["failing"]
