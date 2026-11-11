"""Stage 1 v2 of the cascade: explicit social groups from long track histories (and optionally wide RGB-D crops).

Changes against v1 (interaction_net.py), each one an answer to a measured v1 failure:
  history     8 s (40 samples at 5 Hz) per person instead of 2 s, plus a pair-level temporal encoder over the 8 s of
              pairwise geometry (distance, relative bearing, mutual facing, velocity difference) - the trajectory-level
              pair relation of Jahangard et al. 2023 ("Real-time trajectory-based social group detection", their
              track-level GIoU averaged over the overlap window), in metric robot-frame space instead of image boxes
  relational  edge-aware graph transformer layers (Dwivedi & Bresson 2021, the layer used by Jahangard et al.)
  factorised  pair output = P(same group) x P(type | same group) instead of one 5-way softmax; the type is a group
              property in the labels (the layer types a group by its first member and paints every member pair)
  group loss  normalised-cut consistency of the ground-truth groups under the predicted affinity
              (Shi & Malik 2000; the spectral grouping loss of Jahangard et al. in a batched, eigen-free form)
  decoding    explicit groups: perturb-and-MAP (Papandreou & Yuille 2011) correlation clustering of the affinity logits
              (Bansal et al. 2004; greedy average linkage). Each sample is a consistent partition (transitive, so the
              A-B, B-C, but not A-C "V" cannot occur); the marginal over samples and ensemble members is exported in the
              v1 format (edge_p [P, P, 5]) so stage 2 is unchanged
  crops       128 x 64 crops of a wider box (+-0.55 m, 2.3 m: an outstretched / raised arm stays inside), the
              per-frame features and the temporal difference (t vs t-0.8 s: the arm moving up)
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .cascade_data import CACHE, CROP2_DIR
from .interaction_net import CLASSES, P_MAX, type_class

NODE_F2 = 9
PAIR_F = 10
PAIR_STRIDE = 2  # pair sequence every 0.4 s
CROP_T2 = (0, 2, 4)
N_TYPES = 4  # talk, queue, group_walk, other (CLASSES[1:])


class EpisodeCache2:
    def __init__(self, eid: str, with_crops: bool) -> None:
        z = np.load(os.path.join(CACHE, f"{eid}.npz"))
        self.names = list(z["names"])
        self.track, self.robot, self.vis = z["track"], z["robot"], z["vis"]
        self.inter, self.itype, self.halt, self.wheel = z["inter"], z["itype"], z["halt"], z["wheel"]
        self.T, self.P = self.track.shape[:2]
        self.present = np.abs(self.track).sum(-1) > 0
        self.crop = np.load(os.path.join(CROP2_DIR, f"{eid}.npz"))["crop"] if with_crops and self.P else None


def sample_tensors2(c: EpisodeCache2, t: int, hist: int, with_crops: bool, rng: np.random.Generator | None = None) -> dict:
    """Inputs and labels of sample t, padded to P_MAX, all geometry in the robot frame at t.
    rng (training only, v3): mirror the scene about the robot's x axis (crops flipped too), 5 cm position / velocity
    noise, and drop 10 % of the past samples (never the current one)."""
    P = c.P
    rx, ry, ryaw = c.robot[t]
    cs, sn = math.cos(ryaw), math.sin(ryaw)
    hs = t - (hist - 1) + np.arange(hist)
    ok_t = hs >= 0
    hc = np.clip(hs, 0, None)
    tr = c.track[hc]  # H, P, 5
    pres = c.present[hc] & ok_t[:, None]
    dx, dy = tr[..., 0] - rx, tr[..., 1] - ry
    lx, ly = cs * dx + sn * dy, -sn * dx + cs * dy
    yaw = tr[..., 2] - ryaw
    vx, vy = cs * tr[..., 3] + sn * tr[..., 4], -sn * tr[..., 3] + cs * tr[..., 4]
    flip = False
    if rng is not None:
        flip = bool(rng.random() < 0.5)
        if flip:
            ly, yaw, vy = -ly, -yaw, -vy
        lx = lx + rng.normal(0, 0.05, lx.shape)
        ly = ly + rng.normal(0, 0.05, ly.shape)
        vx = vx + rng.normal(0, 0.05, vx.shape)
        vy = vy + rng.normal(0, 0.05, vy.shape)
        drop = rng.random(pres.shape) < 0.1
        drop[-1] = False
        pres = pres & ~drop
    sp = np.hypot(vx, vy)
    node = np.zeros((hist, P_MAX, NODE_F2), np.float32)
    feats = np.stack([lx / 5, ly / 5, np.sin(yaw), np.cos(yaw), vx, vy, sp, c.vis[hc]], -1) * pres[..., None]
    node[:, :P, :8] = feats
    node[:, :P, 8] = pres
    # pair sequence (every PAIR_STRIDE samples, ending at t)
    ph = np.arange(hist - 1, -1, -PAIR_STRIDE)[::-1]
    S = len(ph)
    pair = np.zeros((S, P_MAX, P_MAX, PAIR_F), np.float32)
    if P > 1:
        X, Y, Yw, VX, VY, PR = lx[ph], ly[ph], yaw[ph], vx[ph], vy[ph], pres[ph]  # S, P
        ddx = X[:, None, :] - X[:, :, None]  # S, i, j : j - i
        ddy = Y[:, None, :] - Y[:, :, None]
        d = np.hypot(ddx, ddy) + 1e-6
        ci, si = np.cos(Yw)[:, :, None], np.sin(Yw)[:, :, None]
        cj, sj = np.cos(Yw)[:, None, :], np.sin(Yw)[:, None, :]
        lxi, lyi = ci * ddx + si * ddy, -si * ddx + ci * ddy
        dyaw = Yw[:, None, :] - Yw[:, :, None]
        face_i = lxi / d
        face_j = -(cj * ddx + sj * ddy) / d
        dv = np.hypot(VX[:, :, None] - VX[:, None, :], VY[:, :, None] - VY[:, None, :])
        both = (PR[:, :, None] & PR[:, None, :]).astype(np.float32)
        f = np.stack([d / 3, lxi / 3, lyi / 3, np.cos(dyaw), np.sin(dyaw), face_i, face_j, dv, np.exp(-d)], -1) * both[..., None]
        pair[:, :P, :P, :9] = f
        pair[:, :P, :P, 9] = both
        eye = np.eye(P_MAX, dtype=bool)
        pair[:, eye] = 0.0
    mask = np.zeros(P_MAX, bool)
    mask[:P] = True
    # labels
    gid = c.inter[t]
    same = np.zeros((P_MAX, P_MAX), np.float32)
    ptype = np.full((P_MAX, P_MAX), -100, np.int64)
    ecls = np.full((P_MAX, P_MAX), -100, np.int64)
    G = np.zeros((P_MAX, P_MAX), np.float32)
    if P:
        s_ = (gid[:, None] >= 0) & (gid[:, None] == gid[None, :]) & ~np.eye(P, dtype=bool)
        same[:P, :P] = s_
        tcls = np.array([type_class(x) for x in c.itype[t]])  # 1..4
        ptype[:P, :P] = np.where(s_, tcls[:, None] - 1, -100)
        ecls[:P, :P] = np.where(s_, tcls[:, None], 0)
        np.fill_diagonal(ecls, -100)
        for k, g in enumerate(sorted({int(x) for x in gid if x >= 0})):
            G[:P, k] = gid == g
    halt = np.zeros(P_MAX, np.float32)
    halt[:P] = c.halt[t]
    wheel = np.zeros(P_MAX, np.float32)
    wheel[:P] = c.wheel
    vis_now = np.zeros(P_MAX, np.float32)
    vis_now[:P] = c.vis[t]
    out = {"node": torch.from_numpy(node), "pair": torch.from_numpy(pair), "mask": torch.from_numpy(mask), "same": torch.from_numpy(same),
           "ptype": torch.from_numpy(ptype), "ecls": torch.from_numpy(ecls), "G": torch.from_numpy(G), "halt": torch.from_numpy(halt),
           "wheel": torch.from_numpy(wheel), "vis_now": torch.from_numpy(vis_now)}
    if with_crops:
        cr = np.zeros((P_MAX, len(CROP_T2), 4) + c.crop.shape[-2:] if c.crop is not None else (P_MAX, len(CROP_T2), 4, 128, 64), np.uint8)
        cv = np.zeros((P_MAX, len(CROP_T2)), np.float32)
        if c.crop is not None:
            for k, back in enumerate(CROP_T2):
                tt = max(0, t - back)
                cr[:P, k] = c.crop[tt]
                cv[:P, k] = (c.vis[tt] > 0.05) & (c.crop[tt].reshape(P, -1).max(1) > 0)
        if flip:
            cr = np.ascontiguousarray(cr[..., ::-1])
        out["crop"] = torch.from_numpy(cr)
        out["crop_vis"] = torch.from_numpy(cv)
    return out


class Stage1Samples2(Dataset):
    def __init__(self, ids: list[str], hist: int, with_crops: bool, augment: bool = False, cache: dict | None = None) -> None:
        self.hist, self.with_crops, self.augment = hist, with_crops, augment
        self.cache = cache if cache is not None else {e: EpisodeCache2(e, with_crops) for e in ids}
        self.items = [(e, t) for e in ids for t in range(self.cache[e].T) if self.cache[e].P > 0]
        self._n = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        e, t = self.items[i]
        rng = None
        if self.augment:
            self._n += 1
            rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, i, self._n])
        d = sample_tensors2(self.cache[e], t, self.hist, self.with_crops, rng)
        d["key"] = (e, t)
        return d


def collate2(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "key"}
    out["key"] = [b["key"] for b in batch]
    return out


class EdgeGTLayer(nn.Module):
    """Graph transformer layer with edge features (Dwivedi & Bresson 2021): attention score of i -> j is
    (Q_i * K_j * E_ij) / sqrt(d_k) per channel; the per-channel score updates the edge, its head sum gates the message."""

    def __init__(self, d: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.drop = nn.Dropout(dropout)
        self.q, self.k, self.v, self.ee = (nn.Linear(d, d) for _ in range(4))
        self.on, self.oe = nn.Linear(d, d), nn.Linear(d, d)
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.fn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, d))
        self.fe = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, d))

    def forward(self, n: torch.Tensor, e: torch.Tensor, pm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, D = n.shape
        q = self.q(n).view(B, P, 1, self.h, self.dk)
        k = self.k(n).view(B, 1, P, self.h, self.dk)
        v = self.v(n).view(B, P, self.h, self.dk)
        s = q * k * self.ee(e).view(B, P, P, self.h, self.dk) / math.sqrt(self.dk)  # B, i, j, h, dk
        logit = s.sum(-1).float().clamp(-5, 5).masked_fill(~pm.unsqueeze(-1), -1e4)
        a = torch.softmax(logit, 2) * pm.unsqueeze(-1).float()
        agg = torch.einsum("bijh,bjhd->bihd", a.to(v.dtype), v).reshape(B, P, D)
        n = self.ln[0](n + self.drop(self.on(agg)))
        n = self.ln[1](n + self.drop(self.fn(n)))
        e = self.ln[2](e + self.drop(self.oe(s.reshape(B, P, P, D))))
        e = self.ln[3](e + self.drop(self.fe(e)))
        return n, e


class CropEncoder2(nn.Module):
    def __init__(self, out: int) -> None:
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]  # noqa: E731
        self.net = nn.Sequential(*c(4, 32), *c(32, 64), *c(64, 96), *c(96, 128), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(256, out)
        self.unseen = nn.Parameter(torch.zeros(out))

    def forward(self, crop: torch.Tensor, vis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, P, Tn = crop.shape[:3]
        out = self.unseen.view(1, 1, -1).expand(B, P, -1).clone()
        if not mask.any():
            return out
        sel = crop[mask]  # N, Tn, 4, H, W (only real people)
        f = self.net(sel.flatten(0, 1).float() / 255.0 - 0.5).flatten(1).view(sel.shape[0], Tn, -1)
        w = vis[mask].unsqueeze(-1)
        mean = (f * w).sum(1) / w.sum(1).clamp(min=1)
        both = (w[:, 0] * w[:, -1])
        diff = (f[:, 0] - f[:, -1]) * both  # t vs t - 0.8 s: the arm going up
        seen = (w.sum(1) > 0).float()
        emb = self.proj(torch.cat([mean, diff], -1))
        out[mask] = (seen * emb + (1 - seen) * self.unseen).to(out.dtype)
        return out


class GroupNet(nn.Module):
    def __init__(self, with_crops: bool, d: int = 96, heads: int = 4, layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.with_crops = with_crops
        self.node_in = nn.Linear(NODE_F2, 64)
        self.node_gru = nn.GRU(64, d, batch_first=True)
        self.pair_in = nn.Linear(PAIR_F, 48)
        self.pair_gru = nn.GRU(48, d, batch_first=True)
        self.crop = CropEncoder2(d) if with_crops else None
        self.node_mix = nn.Sequential(nn.Linear(d * (2 if with_crops else 1), d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.LayerNorm(d))
        self.edge0 = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.LayerNorm(d))
        self.layers = nn.ModuleList([EdgeGTLayer(d, heads, dropout) for _ in range(layers)])
        head = lambda o: nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * d, d), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d, o))  # noqa: E731
        self.same_head, self.type_head = head(1), head(N_TYPES)
        self.node_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(inplace=True), nn.Linear(d, 2))

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node, pair, mask = b["node"], b["pair"], b["mask"]
        B, H, P, _ = node.shape
        h = F.relu(self.node_in(node)).permute(0, 2, 1, 3).reshape(B * P, H, -1)
        _, hn = self.node_gru(h)
        n = hn[-1].view(B, P, -1)
        if self.crop is not None:
            n = torch.cat([n, self.crop(b["crop"], b["crop_vis"], mask).to(n.dtype)], -1)
        n = self.drop(self.node_mix(n))
        S = pair.shape[1]
        ph = F.relu(self.pair_in(pair)).permute(0, 2, 3, 1, 4).reshape(B * P * P, S, -1)
        _, pe = self.pair_gru(ph)
        pe = pe[-1].view(B, P, P, -1)
        pm = mask.unsqueeze(1) & mask.unsqueeze(2) & ~torch.eye(P, dtype=torch.bool, device=node.device).unsqueeze(0)
        ni, nj = n.unsqueeze(2).expand(-1, -1, P, -1), n.unsqueeze(1).expand(-1, P, -1, -1)
        e = self.drop(self.edge0(torch.cat([ni, nj, pe], -1)))
        for layer in self.layers:
            n, e = layer(n, e, pm)
        ni, nj = n.unsqueeze(2).expand(-1, -1, P, -1), n.unsqueeze(1).expand(-1, P, -1, -1)
        z = torch.cat([e + e.transpose(1, 2), ni + nj, (ni - nj).abs()], -1)  # symmetric in (i, j)
        same = self.same_head(z).squeeze(-1)
        typ = self.type_head(z)
        pw = pm.unsqueeze(-1).to(e.dtype)
        agg = (e * pw).sum(2) / pw.sum(2).clamp(min=1)
        return same, typ, self.node_head(torch.cat([n, agg], -1))


def group_loss(same, typ, node, b, w: dict, cut_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
    """BCE on P(same group) + CE on the type of ground-truth group pairs + halt / mobility-aid BCE
    + normalised-cut consistency: for every ground-truth group g (>= 2 members), cut(g) / assoc(g) under W = sigmoid(same)."""
    same, typ, node = same.float(), typ.float(), node.float()
    mask = b["mask"]
    P = mask.shape[1]
    pm = mask.unsqueeze(1) & mask.unsqueeze(2) & ~torch.eye(P, dtype=torch.bool, device=mask.device).unsqueeze(0)
    l_same = F.binary_cross_entropy_with_logits(same[pm], b["same"][pm], pos_weight=w["same_pos"])
    tm = b["ptype"] >= 0
    l_type = F.cross_entropy(typ[tm], b["ptype"][tm], weight=w["type"]) if tm.any() else same.sum() * 0
    m = mask.float()
    l_halt = (F.binary_cross_entropy_with_logits(node[..., 0], b["halt"], pos_weight=w["node_pos"][0], reduction="none") * m).sum() / m.sum()
    l_wheel = (F.binary_cross_entropy_with_logits(node[..., 1], b["wheel"], pos_weight=w["node_pos"][1], reduction="none") * m).sum() / m.sum()
    W = torch.sigmoid(same) * pm.float()
    G = b["G"]
    assoc = torch.einsum("bpg,bpq->bg", G, W)
    within = torch.einsum("bpg,bpq,bqg->bg", G, W, G)
    gm = G.sum(1) >= 2
    l_cut = ((assoc - within) / (assoc + 1e-3))[gm].mean() if gm.any() else same.sum() * 0
    total = l_same + l_type + l_halt + l_wheel + cut_weight * l_cut
    return total, {"same": float(l_same), "type": float(l_type), "halt": float(l_halt), "wheel": float(l_wheel), "cut": float(l_cut)}


# ---------------------------------------------------------------- explicit groups (numpy, per sample)

def correlation_clustering(w: np.ndarray) -> np.ndarray:
    """Greedy average-linkage correlation clustering of a symmetric logit matrix w [P, P]: merge the two clusters with
    the highest mean cross logit while it is > 0 (i.e. while the merge raises the correlation-clustering objective)."""
    P = w.shape[0]
    lab = np.arange(P)
    if P < 2:
        return lab
    S = w.astype(np.float64).copy()
    np.fill_diagonal(S, 0.0)
    n = np.ones(P)
    alive = np.ones(P, bool)
    while True:
        M = S / np.outer(n, n)
        M[~alive, :] = -np.inf
        M[:, ~alive] = -np.inf
        np.fill_diagonal(M, -np.inf)
        k = int(np.argmax(M))
        a, c = divmod(k, P)
        if not M[a, c] > 0:
            break
        S[a, :] += S[c, :]
        S[:, a] += S[:, c]
        S[a, a] = 0.0
        n[a] += n[c]
        alive[c] = False
        lab[lab == c] = a
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def group_types(lab: np.ndarray, p_same: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Type distribution [P, 4] of every person's group: affinity-weighted mean of the pair type posteriors inside the
    group (zeros for singletons)."""
    P = len(lab)
    out = np.zeros((P, N_TYPES))
    for g in np.unique(lab):
        idx = np.flatnonzero(lab == g)
        if len(idx) < 2:
            continue
        ii, jj = np.triu_indices(len(idx), 1)
        a, b = idx[ii], idx[jj]
        wt = p_same[a, b][:, None] + 1e-6
        pi = (wt * q[a, b]).sum(0) / wt.sum()
        out[idx] = pi / pi.sum()
    return out


def decode_groups(same_logits: list[np.ndarray], type_logits: list[np.ndarray], t_same: float, t_type: float, samples_per_member: int,
                  rng: np.random.Generator) -> dict:
    """Explicit-group decoding of one sample (people already cropped to the real P).
    same_logits: M x [P, P]; type_logits: M x [P, P, 4].
    Returns the grouped 5-class marginal (v1 format), its entropy and the member mutual information, the MAP partition
    and its group types, and the raw factorised marginal (no grouping) for the ablation."""
    M = len(same_logits)
    P = same_logits[0].shape[0]
    ps = [1 / (1 + np.exp(-z.astype(np.float64) / t_same)) for z in same_logits]
    qs = []
    for z in type_logits:
        z = z.astype(np.float64) / t_type
        z = np.exp(z - z.max(-1, keepdims=True))
        qs.append(z / z.sum(-1, keepdims=True))
    iu = np.triu_indices(P, 1)
    grouped_m = []
    raw_m = []
    for m in range(M):
        r = np.zeros((P, P, 5))
        r[..., 0] = 1 - ps[m]
        r[..., 1:] = ps[m][..., None] * qs[m]
        raw_m.append(r)
        acc = np.zeros((P, P, 5))
        lg = np.log(ps[m].clip(1e-6, 1 - 1e-6)) - np.log1p(-ps[m].clip(1e-6, 1 - 1e-6))
        for _ in range(samples_per_member):
            u = rng.uniform(1e-6, 1 - 1e-6, len(iu[0]))
            noise = np.zeros((P, P))
            noise[iu] = np.log(u) - np.log1p(-u)  # logistic noise: sign(lg + noise) ~ Bernoulli(p) per pair
            noise = noise + noise.T
            lab = correlation_clustering(lg + noise)
            gt = group_types(lab, ps[m], qs[m])
            sm = (lab[:, None] == lab[None, :]) & ~np.eye(P, dtype=bool)
            acc[..., 0] += ~sm
            acc[..., 1:] += sm[..., None] * gt[:, None, :]
        grouped_m.append(acc / samples_per_member)

    def stats(arrs):
        a = np.stack(arrs)  # M, P, P, 5
        mean = a.mean(0)
        H = -(mean * np.log(mean.clip(1e-8))).sum(-1)
        He = -(a * np.log(a.clip(1e-8))).sum(-1).mean(0)
        return mean, H, np.clip(H - He, 0, None)

    g, gH, gMI = stats(grouped_m)
    r, rH, rMI = stats(raw_m)
    for x in (g, r):
        x[np.arange(P), np.arange(P)] = (1, 0, 0, 0, 0)
    p_bar, q_bar = np.mean(ps, 0), np.mean(qs, 0)
    lab = correlation_clustering(np.log(p_bar.clip(1e-6, 1 - 1e-6)) - np.log1p(-p_bar.clip(1e-6, 1 - 1e-6)))
    gt = group_types(lab, p_bar, q_bar)
    sizes = np.bincount(lab)[lab]
    map_type = np.where(sizes >= 2, gt.argmax(-1), -1)
    lab = np.where(sizes >= 2, lab, -1)
    return {"edge_p": g, "edge_entropy": gH, "edge_mi": gMI, "raw_edge_p": r, "raw_edge_entropy": rH, "raw_edge_mi": rMI,
            "map_group": lab, "map_type": map_type, "p_same": p_bar}
