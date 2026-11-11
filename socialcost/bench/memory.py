"""Temporal interaction memory: persistent, uncertainty-aware beliefs over stage-1 (GroupNet) outputs.

Problem: GroupNet's input window (2 s of tracks + RGB-D crops) loses the visual evidence when a person leaves the camera's
view (the crop becomes the 'unseen' embedding), so its relation and gesture probabilities drift back toward the
track-only answer, usually `none`, and the interaction cost disappears while the person is still tracked.

BeliefCache is a strictly causal filter over the per-frame stage-1 outputs, keyed by stable track names:

  person p   observed  <=> sensor visibility fraction(p, t) >= vis_thr (no ground truth)
  pair (a,b) observed  <=> a and b both observed at t (one visible person is not evidence about the pair)

  observed      belief <- ema * stage1(t) + (1 - ema) * belief          (ema = 1: take the new calibrated output)
  unobserved    age = t - t_last_observation
     mode none        belief <- stage1(t)                               (current pipeline)
     mode hold        belief <- belief at the last observation, u = 0  (no uncertainty growth: ablation)
     mode decay       age <= hold: held belief; hold < age <= hold + decay: linear mix toward the decay target;
                      afterwards: the target. Target "prior" = class prior (train frequencies); "stage1" = the current
                      stage-1 output, which without vision is GroupNet's track-only answer (an informed prior). Explicit staleness uncertainty u = min(1, age / (hold + decay)) rises from
                      0 at visual loss to 1 (entropy alone cannot carry it: the prior is concentrated on `none`, so a
                      held `talk` mixed toward it first gains and then loses entropy). TypeMix adds u x E^2 of the
                      held primitive to its semantic variance (typemix.py), so uncertainty grows over the whole interval.
     mode decay_geo   decay, plus track-geometry consistency for held talk / queue / group-walk relations: the pair
                      is reset to the prior when its separation grew by more than geo_sep_growth m since the last
                      joint observation, or exceeds geo_max_sep m (talk) / (queue, group walk) absolute limits.
  nodes          raised hand: short hold/decay (transient gesture); wheelchair: long hold/decay (persistent attribute)

Never observed: the stage-1 output is used as is (there is nothing to remember). Tracks that disappear lose their
beliefs; a name that reappears starts fresh (re-identification is treated as a new track). reset() between episodes.
Edge classes: 0 none, 1 talk, 2 queue, 3 group walk, 4 other. Node columns: 0 raised hand, 1 wheelchair.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

MODES = ("none", "hold", "decay", "decay_geo")


@dataclasses.dataclass
class MemoryConfig:
    mode: str = "decay"
    vis_thr: float = 0.1
    ema: float = 1.0
    hold_s: float = 2.0
    decay_s: float = 3.0
    halt_hold_s: float = 1.0
    halt_decay_s: float = 2.0
    wheel_hold_s: float = 30.0
    wheel_decay_s: float = 30.0
    geo_sep_growth: float = 1.5
    geo_max_sep_talk: float = 3.0
    geo_max_sep_queue: float = 6.0
    geo_max_sep_walk: float = 4.0
    decay_target: str = "prior"  # "prior": class prior (train frequencies) | "stage1": the current track-based stage-1 output
    edge_prior: tuple = (0.80, 0.08, 0.05, 0.05, 0.02)
    node_prior: tuple = (0.03, 0.05)

    def tag(self) -> str:
        return f"{self.mode}_{self.decay_target}_h{self.hold_s:g}_d{self.decay_s:g}_e{self.ema:g}"


def _mix(held: np.ndarray, prior: np.ndarray, age: float, hold: float, decay: float) -> tuple[np.ndarray, float]:
    """Held belief -> prior after `hold` s over `decay` s; returns (belief, weight of the prior)."""
    if age <= hold:
        return held, 0.0
    w = 1.0 if decay <= 0 else min(1.0, (age - hold) / decay)
    return (1 - w) * held + w * prior, w


class BeliefCache:
    def __init__(self, cfg: MemoryConfig) -> None:
        assert cfg.mode in MODES, cfg.mode
        self.cfg = cfg
        self.ep = np.asarray(cfg.edge_prior, np.float64)
        self.np_ = np.asarray(cfg.node_prior, np.float64)
        self.reset()

    def reset(self) -> None:
        self.t_prev: float | None = None
        self.node: dict[str, dict] = {}   # name -> {"p": (2,), "t": last obs time}
        self.edge: dict[tuple, dict] = {}  # (a, b) sorted -> {"p": (5,), "t": last joint obs, "sep": separation then}

    # ------------------------------------------------------------------------------------------------ one frame
    def step(self, t: float, names: list[str], xy: np.ndarray, vis: np.ndarray, edge_p: np.ndarray, node_p: np.ndarray) -> dict:
        """t seconds (increasing within an episode); names (P,), xy (P, 2) tracked positions, vis (P,) visibility fraction,
        edge_p (P, P, 5) and node_p (P, 2) stage-1 outputs at t. Returns filtered edge_p / node_p and per-pair / per-person
        staleness (s since last observation, inf if never) and prior weight."""
        if self.t_prev is not None and t < self.t_prev:
            raise ValueError("BeliefCache.step called out of chronological order")
        self.t_prev = t
        c = self.cfg
        P = len(names)
        alive = set(names)
        for k in [k for k in self.node if k not in alive]:
            del self.node[k]
        for k in [k for k in self.edge if k[0] not in alive or k[1] not in alive]:
            del self.edge[k]
        obs = np.asarray(vis, np.float64) >= c.vis_thr
        out_e = np.array(edge_p, np.float64, copy=True)
        out_n = np.array(node_p, np.float64, copy=True)
        stale_n = np.full(P, np.inf)
        stale_e = np.full((P, P), np.inf)
        prior_w_e = np.zeros((P, P))
        u_e = np.zeros((P, P))
        u_n = np.zeros((P, 2))
        geo_reset = np.zeros((P, P), bool)
        # ---- persons
        for i, n in enumerate(names):
            m = self.node.get(n)
            if obs[i]:
                p = node_p[i].astype(np.float64)
                if m is not None and c.ema < 1.0:
                    p = c.ema * p + (1 - c.ema) * m["p"]
                self.node[n] = {"p": p, "t": t}
                out_n[i] = p
                stale_n[i] = 0.0
            elif m is not None:
                age = t - m["t"]
                stale_n[i] = age
                if c.mode == "none":
                    continue
                if c.mode == "hold":
                    out_n[i] = m["p"]
                else:
                    tgt = node_p[i].astype(np.float64) if c.decay_target == "stage1" else self.np_
                    out_n[i, 0], _ = _mix(m["p"][0], tgt[0], age, c.halt_hold_s, c.halt_decay_s)
                    out_n[i, 1], _ = _mix(m["p"][1], tgt[1], age, c.wheel_hold_s, c.wheel_decay_s)
                    u_n[i, 0] = min(1.0, age / max(1e-6, c.halt_hold_s + c.halt_decay_s))
                    u_n[i, 1] = min(1.0, age / max(1e-6, c.wheel_hold_s + c.wheel_decay_s))
        # ---- pairs
        for i in range(P):
            for j in range(i + 1, P):
                key = (names[i], names[j]) if names[i] < names[j] else (names[j], names[i])
                sep = float(math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
                m = self.edge.get(key)
                if obs[i] and obs[j]:
                    p = edge_p[i, j].astype(np.float64)
                    if m is not None and c.ema < 1.0:
                        p = c.ema * p + (1 - c.ema) * m["p"]
                    p = p / p.sum()
                    self.edge[key] = {"p": p, "t": t, "sep": sep}
                    out_e[i, j] = out_e[j, i] = p
                    stale_e[i, j] = stale_e[j, i] = 0.0
                    continue
                if m is None:
                    continue  # never jointly observed: stage-1 output as is
                age = t - m["t"]
                stale_e[i, j] = stale_e[j, i] = age
                if c.mode == "none":
                    continue
                if c.mode == "hold":
                    b, w = m["p"], 0.0
                else:
                    tgt = edge_p[i, j].astype(np.float64) if c.decay_target == "stage1" else self.ep
                    b, w = _mix(m["p"], tgt, age, c.hold_s, c.decay_s)
                    u_e[i, j] = u_e[j, i] = min(1.0, age / max(1e-6, c.hold_s + c.decay_s))
                    if c.mode == "decay_geo":
                        cls = int(np.argmax(m["p"]))
                        limit = {1: c.geo_max_sep_talk, 2: c.geo_max_sep_queue, 3: c.geo_max_sep_walk}.get(cls)
                        if limit is not None and (sep > limit or sep > m["sep"] + c.geo_sep_growth):
                            b, w = (edge_p[i, j].astype(np.float64) if c.decay_target == "stage1" else self.ep.copy()), 1.0
                            u_e[i, j] = u_e[j, i] = 1.0
                            geo_reset[i, j] = geo_reset[j, i] = True
                out_e[i, j] = out_e[j, i] = b
                prior_w_e[i, j] = prior_w_e[j, i] = w
        idx = np.arange(P)
        out_e[idx, idx] = (1, 0, 0, 0, 0)
        return {"edge_p": out_e, "node_p": out_n, "stale_edge": stale_e, "stale_node": stale_n, "prior_w_edge": prior_w_e,
                "geo_reset": geo_reset, "observed": obs, "u_edge": u_e, "u_node": u_n}


def run_episode(cfg: MemoryConfig, names: list[str], track: np.ndarray, vis: np.ndarray, edge_p: np.ndarray, node_p: np.ndarray,
                dt: float = 0.2) -> dict:
    """Filter a stored episode (T frames, chronological). track (T, P, >=2) x, y in the map frame; vis (T, P)."""
    cache = BeliefCache(cfg)
    T, P = vis.shape
    out = {"edge_p": np.zeros_like(edge_p, np.float32), "node_p": np.zeros_like(node_p, np.float32),
           "stale_edge": np.zeros((T, P, P), np.float32), "stale_node": np.zeros((T, P), np.float32), "geo_reset": np.zeros((T, P, P), bool),
           "edge_u": np.zeros((T, P, P), np.float32), "node_u": np.zeros((T, P, 2), np.float32)}
    for t in range(T):
        # dataset_v1 tracks every pedestrian for the whole episode (no track loss); live use passes only current tracks
        sub = np.array([i for i in range(P) if np.isfinite(track[t, i, 0])], int)
        r = cache.step(t * dt, [names[i] for i in sub], track[t][sub, :2], vis[t][sub], edge_p[t][np.ix_(sub, sub)], node_p[t][sub])
        out["edge_p"][t] = edge_p[t]
        out["node_p"][t] = node_p[t]
        out["edge_p"][t][np.ix_(sub, sub)] = r["edge_p"]
        out["node_p"][t][sub] = r["node_p"]
        se = np.full((P, P), np.inf, np.float32)
        se[np.ix_(sub, sub)] = r["stale_edge"]
        out["stale_edge"][t] = se
        sn = np.full(P, np.inf, np.float32)
        sn[sub] = r["stale_node"]
        out["stale_node"][t] = sn
        g = np.zeros((P, P), bool)
        g[np.ix_(sub, sub)] = r["geo_reset"]
        out["geo_reset"][t] = g
        out["edge_u"][t][np.ix_(sub, sub)] = r["u_edge"]
        out["node_u"][t][sub] = r["u_node"]
    return out
