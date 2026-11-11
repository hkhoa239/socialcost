"""Stage 1 of the cascade: predict social interactions (who interacts with whom, which kind), raised stop hands and
mobility aids from tracks (and optionally RGB-D person crops).

Design (see reports/cascade_uncertainty_design.md):
  node encoder   GRU over a 2 s track history in the current robot frame (position, heading, velocity, visibility)
                 + optional crop CNN over 3 frames (t, t-0.4 s, t-0.8 s), masked mean over frames where the person is seen
  relational     NRI-style encoder (Kipf et al. 2018): edge = f([n_i, n_j, pair features]); 2 rounds of
                 node <- mean_j edge_ij, edge <- f([n_i, n_j, edge_ij]); DANTE-style pairwise affinity (Swofford et al. 2020)
  heads          edge: 5 classes {none, talk, queue, group_walk, other}, logits symmetrised over (i, j) and (j, i)
                 node: halt (binary), mobility aid (binary)
  uncertainty    deep ensemble of M members (Lakshminarayanan et al. 2017) + temperature scaling on validation
                 (Guo et al. 2017): predictive probability = mean_m softmax(z_m / T), total uncertainty = entropy,
                 epistemic = mutual information = H(mean p) - mean H(p_m)
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .cascade_data import CACHE, CROP_H, CROP_W

HIST = 10  # samples at 5 Hz = 2 s
CROP_T = (0, 2, 4)  # crop frames t, t-0.4 s, t-0.8 s
P_MAX = 10
CLASSES = ["none", "talk", "queue", "group_walk", "other"]
TYPE_TO_CLASS = {0: 1, 1: 1, 6: 2, 12: 3}  # other interaction types -> 4
NODE_F = 8
EDGE_F = 10


def type_class(t: int) -> int:
    return TYPE_TO_CLASS.get(int(t), 4)


class EpisodeCache:
    def __init__(self, eid: str, with_crops: bool) -> None:
        z = np.load(os.path.join(CACHE, f"{eid}.npz"))
        self.track, self.robot, self.vis = z["track"], z["robot"], z["vis"]
        self.inter, self.itype, self.halt, self.wheel = z["inter"], z["itype"], z["halt"], z["wheel"]
        self.crop = z["crop"] if with_crops else None
        self.T, self.P = self.track.shape[:2]


def sample_tensors(c: EpisodeCache, t: int, with_crops: bool) -> dict:
    """Inputs and labels of sample t, padded to P_MAX people, in the robot frame at t."""
    P = c.P
    rx, ry, ryaw = c.robot[t]
    cs, sn = math.cos(ryaw), math.sin(ryaw)
    node = np.zeros((HIST, P_MAX, NODE_F), np.float32)
    for h in range(HIST):
        tt = max(0, t - (HIST - 1 - h))
        tr = c.track[tt]
        dx, dy = tr[:, 0] - rx, tr[:, 1] - ry
        lx, ly = cs * dx + sn * dy, -sn * dx + cs * dy
        yaw = tr[:, 2] - ryaw
        vx, vy = cs * tr[:, 3] + sn * tr[:, 4], -sn * tr[:, 3] + cs * tr[:, 4]
        node[h, :P] = np.stack([lx / 5, ly / 5, np.sin(yaw), np.cos(yaw), vx, vy, np.hypot(vx, vy), c.vis[tt]], 1)
    cur = node[-1, :P]
    x, y, sy, cy = cur[:, 0] * 5, cur[:, 1] * 5, cur[:, 2], cur[:, 3]
    yaw = np.arctan2(sy, cy)
    old = node[0, :P]
    edge = np.zeros((P_MAX, P_MAX, EDGE_F), np.float32)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            d = math.hypot(dx, dy) + 1e-6
            ci, si = math.cos(yaw[i]), math.sin(yaw[i])
            lxi, lyi = ci * dx + si * dy, -si * dx + ci * dy
            dyaw = yaw[j] - yaw[i]
            face_i = (ci * dx + si * dy) / d
            cj, sj = math.cos(yaw[j]), math.sin(yaw[j])
            face_j = -(cj * dx + sj * dy) / d
            dv = math.hypot(cur[i, 4] - cur[j, 4], cur[i, 5] - cur[j, 5])
            d_old = math.hypot((old[j, 0] - old[i, 0]) * 5, (old[j, 1] - old[i, 1]) * 5)
            edge[i, j] = (d / 3, lxi / 3, lyi / 3, math.cos(dyaw), math.sin(dyaw), face_i, face_j, dv, d_old / 3, math.exp(-d))
    mask = np.zeros(P_MAX, bool)
    mask[:P] = True
    ecls = np.full((P_MAX, P_MAX), -100, np.int64)
    for i in range(P):
        for j in range(P):
            if i != j:
                same = c.inter[t, i] >= 0 and c.inter[t, i] == c.inter[t, j]
                ecls[i, j] = type_class(c.itype[t, i]) if same else 0
    halt = np.zeros(P_MAX, np.float32)
    halt[:P] = c.halt[t]
    wheel = np.zeros(P_MAX, np.float32)
    wheel[:P] = c.wheel
    out = {"node": torch.from_numpy(node), "edge": torch.from_numpy(edge), "mask": torch.from_numpy(mask), "ecls": torch.from_numpy(ecls),
           "halt": torch.from_numpy(halt), "wheel": torch.from_numpy(wheel)}
    if with_crops:
        cr = np.zeros((P_MAX, len(CROP_T), 4, CROP_H, CROP_W), np.uint8)
        cv = np.zeros((P_MAX, len(CROP_T)), np.float32)
        for k, back in enumerate(CROP_T):
            tt = max(0, t - back)
            cr[:P, k] = c.crop[tt]
            cv[:P, k] = (c.vis[tt] > 0.05) & (c.crop[tt].reshape(P, -1).max(1) > 0)
        out["crop"] = torch.from_numpy(cr)
        out["crop_vis"] = torch.from_numpy(cv)
    return out


class Stage1Samples(Dataset):
    def __init__(self, ids: list[str], stride: int, with_crops: bool) -> None:
        self.with_crops = with_crops
        self.cache = {e: EpisodeCache(e, with_crops) for e in ids}
        self.items = [(e, t) for e in ids for t in range(0, self.cache[e].T, stride) if self.cache[e].P > 0]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        e, t = self.items[i]
        d = sample_tensors(self.cache[e], t, self.with_crops)
        d["key"] = (e, t)
        return d


def collate1(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "key"}
    out["key"] = [b["key"] for b in batch]
    return out


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o), nn.ReLU(inplace=True))


class CropEncoder(nn.Module):
    def __init__(self, out: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 32, 3, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, out, 3, 2, 1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.unseen = nn.Parameter(torch.zeros(out))

    def forward(self, crop: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        B, P, Tn = crop.shape[:3]
        f = self.net(crop.flatten(0, 2).float() / 255.0 - 0.5).flatten(1).view(B, P, Tn, -1)
        w = vis.unsqueeze(-1)
        seen = (w.sum(2) > 0).float()
        mean = (f * w).sum(2) / w.sum(2).clamp(min=1)
        return seen * mean + (1 - seen) * self.unseen


class InteractionNet(nn.Module):
    def __init__(self, with_crops: bool, hid: int = 96, rounds: int = 2) -> None:
        super().__init__()
        self.with_crops = with_crops
        self.inp = nn.Linear(NODE_F, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.crop = CropEncoder(64) if with_crops else None
        self.node0 = mlp(64 + (64 if with_crops else 0), hid, hid)
        self.edge0 = mlp(2 * hid + EDGE_F, hid, hid)
        self.node_up = nn.ModuleList([mlp(2 * hid, hid, hid) for _ in range(rounds)])
        self.edge_up = nn.ModuleList([mlp(3 * hid, hid, hid) for _ in range(rounds)])
        self.edge_head = nn.Linear(hid, len(CLASSES))
        self.node_head = nn.Linear(2 * hid, 2)

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        node, edge, mask = b["node"], b["edge"], b["mask"]
        B, H, P, _ = node.shape
        h = F.relu(self.inp(node)).permute(0, 2, 1, 3).reshape(B * P, H, 64)
        _, hn = self.gru(h)
        n = hn[-1].view(B, P, 64)
        if self.crop is not None:
            n = torch.cat([n, self.crop(b["crop"], b["crop_vis"])], -1)
        n = self.node0(n)
        pair = mask.unsqueeze(1) & mask.unsqueeze(2) & ~torch.eye(P, dtype=torch.bool, device=node.device).unsqueeze(0)
        ni, nj = n.unsqueeze(2).expand(-1, -1, P, -1), n.unsqueeze(1).expand(-1, P, -1, -1)
        e = self.edge0(torch.cat([ni, nj, edge], -1))
        pw = pair.unsqueeze(-1).float()
        for nu, eu in zip(self.node_up, self.edge_up):
            agg = (e * pw).sum(2) / pw.sum(2).clamp(min=1)
            n = n + nu(torch.cat([n, agg], -1))
            ni, nj = n.unsqueeze(2).expand(-1, -1, P, -1), n.unsqueeze(1).expand(-1, P, -1, -1)
            e = e + eu(torch.cat([ni, nj, e], -1))
        logits = self.edge_head(e)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        agg = (e * pw).sum(2) / pw.sum(2).clamp(min=1)
        return logits, self.node_head(torch.cat([n, agg], -1))


def stage1_loss(edge_logits, node_logits, b, class_w: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
    le = F.cross_entropy(edge_logits.flatten(0, 2), b["ecls"].flatten(), weight=class_w, ignore_index=-100)
    m = b["mask"].float()
    ln = (F.binary_cross_entropy_with_logits(node_logits[..., 0], b["halt"], pos_weight=pos_w[0], reduction="none") * m).sum() / m.sum()
    lw = (F.binary_cross_entropy_with_logits(node_logits[..., 1], b["wheel"], pos_weight=pos_w[1], reduction="none") * m).sum() / m.sum()
    return le + ln + lw


def ensemble_stats(edge_logits: list[torch.Tensor], node_logits: list[torch.Tensor], t_edge: float = 1.0, t_node: float = 1.0) -> dict:
    """Ensemble mean probabilities, total entropy and mutual information (epistemic) for edges and nodes."""
    pe = torch.stack([F.softmax(z / t_edge, -1) for z in edge_logits])  # M,B,P,P,C
    pm = pe.mean(0)
    H = -(pm * pm.clamp(min=1e-8).log()).sum(-1)
    He = -(pe * pe.clamp(min=1e-8).log()).sum(-1).mean(0)
    pn = torch.stack([torch.sigmoid(z / t_node) for z in node_logits])  # M,B,P,2
    pnm = pn.mean(0)

    def bern_h(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log())

    return {"edge_p": pm, "edge_entropy": H, "edge_mi": (H - He).clamp(min=0), "node_p": pnm, "node_mi": (bern_h(pnm) - bern_h(pn).mean(0)).clamp(min=0)}
