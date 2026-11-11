"""Counterfactual supervision for stage 1 (who interacts, raised hands, mobility aids).

The dataset records every counterfactual unit twice: ON and OFF share seed, roster and (almost exactly) the tracks; only
the social cue or the hidden social state differs. Trained as independent samples, a model can fit both labels only by
memorising nuisance details of an episode (shortcut learning). Here the pair itself is the supervision, following
learning from counterfactual examples (Teney, Abbasnejad & van den Hengel, ECCV 2020; Kaushik, Hovy & Lipton, ICLR
2020), adapted to a sensor-dependent notion of observability:

  for every output entry (pair same-group logit, person halt / mobility-aid logit) of a matched sample pair (ON t, OFF t)
    labels differ, cue observable    -> ranking: s * (z_on - z_off) >= margin, s = sign(y_on - y_off)
                                        ("the prediction must move with the cue")
    labels differ, cue not observable -> both targets replaced by the pair mean (0.5) + invariance
                                        ("identical evidence, different truth: predict the posterior, not a guess")
    labels equal                     -> invariance ("nothing else may change")
  invariance = symmetric KL of the two predictive distributions; for Bernoulli logits a, b:
               KL(p||q) + KL(q||p) = (p - q)(a - b);  for the type softmax: sum_c (p_c - q_c)(log p_c - log q_c)

Observability is a property of the sensor, not of the family: a raised hand, a wheelchair or talking gestures are
observable for the crop model when the cue holder is visible (visibility > 0.3); never for the tracks-only model; the
queue (hidden_state) family is never observable.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import ROOT
from .group_net import sample_tensors2

VIS_OBS = 0.3


def train_pairs(ids: set[str], root: str = ROOT) -> list[dict]:
    """Counterfactual units with both conditions in `ids` (same roster required)."""
    man = json.load(open(os.path.join(root, "manifests", "manifest_v1.json")))["episodes"]
    by: dict = {}
    for e in man:
        if e.get("pair_id") and e["episode_id"] in ids:
            by.setdefault(e["pair_id"], {})[e["condition"]] = e
    return [{"on": d["on"]["episode_id"], "off": d["off"]["episode_id"], "family": d["on"]["family"],
             "observable": d["on"].get("observability", "") != "hidden_state"} for d in by.values() if set(d) == {"on", "off"}]


class CFPairs(Dataset):
    """Matched samples (ON t, OFF t) of counterfactual units, sharing the episode cache of the main dataset."""

    def __init__(self, pairs: list[dict], cache: dict, hist: int, with_crops: bool, stride: int = 2) -> None:
        self.cache, self.hist, self.with_crops = cache, hist, with_crops
        self.items, self.skipped = [], 0
        for p in pairs:
            a, b = cache.get(p["on"]), cache.get(p["off"])
            if a is None or b is None or not a.P or a.names != b.names:
                self.skipped += 1
                continue
            for t in range(0, min(a.T, b.T), stride):
                self.items.append((p["on"], p["off"], t, p["observable"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        on, off, t, obs = self.items[i]
        a = sample_tensors2(self.cache[on], t, self.hist, self.with_crops)
        b = sample_tensors2(self.cache[off], t, self.hist, self.with_crops)
        return a, b, obs


def collate_cf(batch):
    on = {k: torch.stack([x[0][k] for x in batch]) for k in batch[0][0]}
    off = {k: torch.stack([x[1][k] for x in batch]) for k in batch[0][1]}
    obs = torch.tensor([x[2] for x in batch], dtype=torch.bool)
    return on, off, obs


def concat(*bs: dict) -> dict:
    return {k: torch.cat([b[k] for b in bs]) for k in bs[0] if torch.is_tensor(bs[0][k])}


def cf_plan(on: dict, off: dict, obs_family: torch.Tensor, with_crops: bool) -> dict:
    """Entry masks (rank / soft / invariant) for pairs and nodes, and the soft-target edits. Shapes: pairs [B, P, P], nodes [B, P, 2]."""
    mask = on["mask"]
    B, P = mask.shape
    pm = mask.unsqueeze(1) & mask.unsqueeze(2) & ~torch.eye(P, dtype=torch.bool, device=mask.device).unsqueeze(0)
    vis = torch.maximum(on["vis_now"], off["vis_now"]) > VIS_OBS  # B, P
    fam = obs_family.view(B, 1) & with_crops
    # pairs: the cue is carried by the members; observable if either member is visible
    pair_obs = fam.unsqueeze(-1) & (vis.unsqueeze(1) | vis.unsqueeze(2))
    dpair = (on["same"] != off["same"]) & pm
    node_y_on = torch.stack([on["halt"], on["wheel"]], -1)
    node_y_off = torch.stack([off["halt"], off["wheel"]], -1)
    nm = mask.unsqueeze(-1).expand(-1, -1, 2)
    dnode = (node_y_on != node_y_off) & nm
    node_obs = (fam & vis).unsqueeze(-1).expand(-1, -1, 2)
    return {"pm": pm, "nm": nm,
            "pair_rank": dpair & pair_obs, "pair_soft": dpair & ~pair_obs, "pair_inv": (pm & ~dpair) | (dpair & ~pair_obs),
            "node_rank": dnode & node_obs, "node_soft": dnode & ~node_obs, "node_inv": (nm & ~dnode) | (dnode & ~node_obs),
            "pair_sign": (on["same"] - off["same"]).sign(), "node_sign": (node_y_on - node_y_off).sign()}


def apply_soft_targets(on: dict, off: dict, plan: dict) -> None:
    """Hidden-state entries: both conditions get the pair-mean target (0.5) in the supervised loss."""
    for b in (on, off):
        b["same"] = torch.where(plan["pair_soft"], torch.full_like(b["same"], 0.5), b["same"])
        b["halt"] = torch.where(plan["node_soft"][..., 0], torch.full_like(b["halt"], 0.5), b["halt"])
        b["wheel"] = torch.where(plan["node_soft"][..., 1], torch.full_like(b["wheel"], 0.5), b["wheel"])


def _sym_bern(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(a) - torch.sigmoid(b)) * (a - b)


def cf_loss(out_on, out_off, on: dict, off: dict, plan: dict, margin: float, w_rank: float, w_inv: float) -> tuple[torch.Tensor, dict]:
    s_on, t_on, n_on = (x.float() for x in out_on)
    s_off, t_off, n_off = (x.float() for x in out_off)
    zero = s_on.sum() * 0
    parts = {}

    def mean(x, m):
        return x[m].mean() if m.any() else zero

    rank = mean(torch.relu(margin - plan["pair_sign"] * (s_on - s_off)), plan["pair_rank"]) + \
        mean(torch.relu(margin - plan["node_sign"] * (n_on - n_off)), plan["node_rank"])
    inv = mean(_sym_bern(s_on, s_off), plan["pair_inv"]) + mean(_sym_bern(n_on, n_off), plan["node_inv"])
    both = (on["ptype"] >= 0) & (off["ptype"] >= 0)
    if both.any():
        lp, lq = torch.log_softmax(t_on[both], -1), torch.log_softmax(t_off[both], -1)
        inv = inv + ((lp.exp() - lq.exp()) * (lp - lq)).sum(-1).mean()
    parts.update({"cf_rank": float(rank), "cf_inv": float(inv), "n_rank": int(plan["pair_rank"].sum() + plan["node_rank"].sum()),
                  "n_soft": int(plan["pair_soft"].sum() + plan["node_soft"].sum())})
    return w_rank * rank + w_inv * inv, parts


def cf_test_d(sink: dict, cache: dict, pairs: list[dict], raw: bool = True) -> dict:
    """Counterfactual response of stage 1 on held-out pairs, per family:
    shift  = mean over label-differing entries of sign(y_on - y_off) * (p_on - p_off)   (1 = full, 0 = blind)
    spur   = mean over label-equal entries of |p_on - p_off|                            (0 = invariant to non-causal changes)
    p_hid  = mean prediction on label-differing entries (both conditions) - for hidden families the posterior is ~0.5
    entries: pair P(interacting) = 1 - edge_p[..., 0]; node halt / mobility-aid probability."""
    key = "raw_edge_p" if raw else "edge_p"
    acc: dict = {}
    for p in pairs:
        if p["on"] not in sink or p["off"] not in sink or cache[p["on"]].names != cache[p["off"]].names:
            continue
        a, b = cache[p["on"]], cache[p["off"]]
        r = acc.setdefault(p["family"], {"shift": [], "spur": [], "p_diff": []})
        P = a.P
        iu = np.triu_indices(P, 1)
        for t in sorted(set(sink[p["on"]]) & set(sink[p["off"]])):
            ra, rb = sink[p["on"]][t], sink[p["off"]][t]
            ya = ((a.inter[t][:, None] >= 0) & (a.inter[t][:, None] == a.inter[t][None, :]))[iu].astype(float)
            yb = ((b.inter[t][:, None] >= 0) & (b.inter[t][:, None] == b.inter[t][None, :]))[iu].astype(float)
            pa, pb = 1 - ra[key][..., 0][iu], 1 - rb[key][..., 0][iu]
            na = np.stack([a.halt[t], a.wheel], -1).reshape(-1).astype(float)
            nb = np.stack([b.halt[t], b.wheel], -1).reshape(-1).astype(float)
            qa, qb = ra["node_p"].reshape(-1), rb["node_p"].reshape(-1)
            for y1, y2, p1, p2 in ((ya, yb, pa, pb), (na, nb, qa, qb)):
                d = y1 != y2
                r["shift"] += list(np.sign(y1 - y2)[d] * (p1 - p2)[d])
                r["spur"] += list(np.abs(p1 - p2)[~d])
                r["p_diff"] += list(p1[d]) + list(p2[d])
    return {f: {"shift": float(np.mean(v["shift"])) if v["shift"] else None, "spurious": float(np.mean(v["spur"])) if v["spur"] else None,
                "p_on_differing": float(np.mean(v["p_diff"])) if v["p_diff"] else None, "n_differing": len(v["shift"])} for f, v in acc.items()}
