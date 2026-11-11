"""Unit tests for the temporal interaction memory (socialcost/bench/memory.py). Pure numpy, runs anywhere.

    cd src/deps/socialcost && PYTHONPATH=. python -m pytest tests/test_memory.py -q
"""

import numpy as np
import pytest

from socialcost.bench.memory import BeliefCache, MemoryConfig, run_episode

TALK = np.array([0.05, 0.9, 0.02, 0.02, 0.01])
NONE = np.array([0.95, 0.02, 0.01, 0.01, 0.01])


def frame(p_pair, P=2, node=None):
    e = np.zeros((P, P, 5))
    e[..., 0] = 1
    e[0, 1] = e[1, 0] = p_pair
    n = np.zeros((P, 2)) if node is None else np.array(node, float)
    return e, n


def xy(sep=1.0):
    return np.array([[0.0, 0.0], [sep, 0.0]])


def run(cfg, seq):
    """seq: list of (t, vis_a, vis_b, stage1_pair_p, sep)."""
    c = BeliefCache(cfg)
    outs = []
    for t, va, vb, p, sep in seq:
        e, n = frame(p)
        outs.append(c.step(t, ["a", "b"], xy(sep), np.array([va, vb]), e, n))
    return outs


def test_none_mode_passes_stage1_through():
    o = run(MemoryConfig(mode="none"), [(0, 1, 1, TALK, 1), (1, 0, 0, NONE, 1)])
    assert np.allclose(o[1]["edge_p"][0, 1], NONE)
    assert o[1]["stale_edge"][0, 1] == pytest.approx(1.0)


def test_hold_keeps_last_observation_without_decay():
    o = run(MemoryConfig(mode="hold"), [(0, 1, 1, TALK, 1), (10, 0, 0, NONE, 1)])
    assert np.allclose(o[1]["edge_p"][0, 1], TALK)


def test_decay_holds_then_mixes_to_prior():
    cfg = MemoryConfig(mode="decay", hold_s=2, decay_s=3)
    o = run(cfg, [(0, 1, 1, TALK, 1), (1.5, 0, 0, NONE, 1), (3.5, 0, 0, NONE, 1), (6, 0, 0, NONE, 1)])
    prior = np.array(cfg.edge_prior)
    assert np.allclose(o[1]["edge_p"][0, 1], TALK)
    assert np.allclose(o[2]["edge_p"][0, 1], 0.5 * TALK + 0.5 * prior)
    assert np.allclose(o[3]["edge_p"][0, 1], prior)
    # explicit staleness uncertainty grows over the whole unobserved interval (0.3, 0.7, 1.0)
    U = [x["u_edge"][0, 1] for x in o[1:]]
    assert U[0] < U[1] < U[2] == pytest.approx(1.0) and U[0] == pytest.approx(0.3)


def test_hold_mode_has_no_uncertainty_growth():
    o = run(MemoryConfig(mode="hold"), [(0, 1, 1, TALK, 1), (4, 0, 0, NONE, 1)])
    assert o[1]["u_edge"][0, 1] == 0


def test_one_visible_person_is_not_pair_evidence():
    o = run(MemoryConfig(mode="decay"), [(0, 1, 1, TALK, 1), (1, 1, 0, NONE, 1)])
    assert np.allclose(o[1]["edge_p"][0, 1], TALK)  # held, not overwritten by the one-sided frame
    assert o[1]["stale_edge"][0, 1] == pytest.approx(1.0)


def test_new_joint_observation_overwrites_with_ema():
    o = run(MemoryConfig(mode="decay", ema=0.5), [(0, 1, 1, TALK, 1), (1, 1, 1, NONE, 1)])
    exp = 0.5 * NONE + 0.5 * TALK
    assert np.allclose(o[1]["edge_p"][0, 1], exp / exp.sum())


def test_geometry_resets_separated_talk_pair():
    cfg = MemoryConfig(mode="decay_geo", hold_s=5, decay_s=5, geo_sep_growth=1.5)
    o = run(cfg, [(0, 1, 1, TALK, 1.0), (1, 0, 0, NONE, 1.5), (2, 0, 0, NONE, 3.0)])
    assert np.allclose(o[1]["edge_p"][0, 1], TALK)
    assert np.allclose(o[2]["edge_p"][0, 1], cfg.edge_prior) and o[2]["geo_reset"][0, 1]


def test_never_observed_pair_uses_stage1():
    o = run(MemoryConfig(mode="decay"), [(0, 1, 0, TALK, 1), (1, 0, 0, TALK, 1)])
    assert np.allclose(o[1]["edge_p"][0, 1], TALK) and np.isinf(o[1]["stale_edge"][0, 1])


def test_track_loss_and_reappearance_start_fresh():
    c = BeliefCache(MemoryConfig(mode="hold"))
    e, n = frame(TALK)
    c.step(0, ["a", "b"], xy(), np.array([1, 1]), e, n)
    e1, n1 = np.ones((1, 1, 5)) / 5, np.zeros((1, 2))
    c.step(1, ["a"], np.zeros((1, 2)), np.array([0]), e1, n1)  # b's track lost
    e2, n2 = frame(NONE)
    o = c.step(2, ["a", "b"], xy(), np.array([0, 0]), e2, n2)  # b re-appears (new track)
    assert np.allclose(o["edge_p"][0, 1], NONE) and np.isinf(o["stale_edge"][0, 1])


def test_chronological_order_enforced_and_reset():
    c = BeliefCache(MemoryConfig())
    e, n = frame(TALK)
    c.step(5, ["a", "b"], xy(), np.array([1, 1]), e, n)
    with pytest.raises(ValueError):
        c.step(4, ["a", "b"], xy(), np.array([1, 1]), e, n)
    c.reset()
    o = c.step(0, ["a", "b"], xy(), np.array([0, 0]), *frame(NONE))
    assert np.allclose(o["edge_p"][0, 1], NONE)


def test_wheelchair_retained_longer_than_raised_hand():
    cfg = MemoryConfig(mode="decay", halt_hold_s=1, halt_decay_s=2, wheel_hold_s=30, wheel_decay_s=30)
    c = BeliefCache(cfg)
    e, _ = frame(NONE)
    c.step(0, ["a", "b"], xy(), np.array([1, 1]), e, np.array([[0.9, 0.9], [0, 0]]))
    o = c.step(4, ["a", "b"], xy(), np.array([0, 1]), e, np.array([[0.0, 0.0], [0, 0]]))
    assert o["node_p"][0, 0] == pytest.approx(cfg.node_prior[0])   # raised hand back to prior after 3 s
    assert o["node_p"][0, 1] == pytest.approx(0.9)                  # wheelchair still held


def test_run_episode_shapes_and_causality():
    T, P = 20, 3
    rng = np.random.default_rng(0)
    track = rng.normal(size=(T, P, 5))
    vis = (rng.random((T, P)) > 0.5).astype(float)
    ep = rng.dirichlet(np.ones(5), size=(T, P, P))
    npp = rng.random((T, P, 2))
    cfg = MemoryConfig(mode="decay")
    full = run_episode(cfg, ["a", "b", "c"], track, vis, ep, npp)
    part = run_episode(cfg, ["a", "b", "c"], track[:10], vis[:10], ep[:10], npp[:10])
    # outputs up to t depend only on frames <= t
    assert np.allclose(full["edge_p"][:10], part["edge_p"]) and np.allclose(full["node_p"][:10], part["node_p"])
