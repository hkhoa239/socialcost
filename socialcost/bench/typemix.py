"""TypeMix: type-conditioned parametric social costmap, marginalised over predicted social types in closed form,
with a residual U-Net and a three-way uncertainty decomposition.

Social types and their cost primitives (normalised cost units, robot-centred 200 x 200 grid):
  personal space   per person, anisotropic Gaussian elongated along the motion direction (SocialTypeAwareCostmap,
                   age/human layers: sigma_front = sigma0 (1 + k_v |v|), sigma_back = sigma_lat = sigma0), amplitude a_ps.
                   Mobility-aid type: sigma0 and amplitude scaled by learned exp(d_sigma), exp(d_amp).
  halt gesture     per person, elliptical Gaussian centred m ahead along the body heading (sigma_long, sigma_lat, a_halt)
  pair classes     per unordered pair and class c in {talk, queue, group_walk, other}: capsule exp(-d_seg^2 / 2 w_c^2) * a_c
Every scale and amplitude is a learned scalar; one parameter set per type, shared over people and scenes.

Marginalisation over types (stage-1 probabilities, independence across primitives):
  Bernoulli cue with probability p and primitives K1 (present) / K0 (absent):  E = p K1 + (1-p) K0,  V = p (1-p) (K1 - K0)^2
  categorical pair with class probabilities p_c and primitives K_c (K_none = 0):  E = sum_c p_c K_c,  V = sum_c p_c K_c^2 - E^2
  C_mix = sum of E over primitives,  V_sem = sum of V   (semantic uncertainty: which social type is present)
Residual refinement (existing GridUNet): input [C_mix, sqrt(V_sem), position rasters] -> mean logit o0, log-variance o1
  mu = sigmoid(o0),  sigma^2 = exp(o1) (aleatoric) + softplus(lambda) * V_sem (propagated semantic uncertainty)
Epistemic uncertainty: seed ensemble (bench_post.py ensemble). Training: beta-NLL on (mu, sigma^2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..learn.models import GridUNet

N = 200
RES = 0.05


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    c = (torch.arange(N, dtype=torch.float32) + 0.5) * RES - N * RES / 2
    Y, X = torch.meshgrid(c, c, indexing="ij")  # array[j, i]: i along x, j along y
    return X, Y


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class TypedRenderer(nn.Module):
    """Differentiable renderer of type-specific cost primitives and their closed-form type marginalisation."""

    def __init__(self) -> None:
        super().__init__()
        X, Y = _grid()
        self.register_buffer("X", X, persistent=False)
        self.register_buffer("Y", Y, persistent=False)
        sp = lambda v: nn.Parameter(torch.tensor(_inv_softplus(v)))  # noqa: E731 - positive parameters through softplus
        # personal space (initialised near the Arena layer: radius 1 m, sigma 0.9 m, amplitude 0.45 after cost_scale)
        self.ps_sigma, self.ps_kv, self.ps_amp = sp(0.5), sp(0.5), sp(0.45)
        self.wc_dsigma, self.wc_damp = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
        # halt wedge-like ellipse
        self.h_ahead, self.h_long, self.h_lat, self.h_amp = sp(1.2), sp(0.9), sp(0.6), sp(0.8)
        # pair classes: talk, queue, group_walk, other
        self.pair_w = nn.Parameter(torch.tensor([_inv_softplus(0.5)] * 4))
        self.pair_amp = nn.Parameter(torch.tensor([_inv_softplus(0.7), _inv_softplus(0.8), _inv_softplus(0.05), _inv_softplus(0.6)]))

    def params(self) -> dict:
        s = F.softplus
        return {"personal_sigma0_m": float(s(self.ps_sigma)), "personal_k_velocity": float(s(self.ps_kv)), "personal_amp": float(s(self.ps_amp)),
                "wheelchair_sigma_scale": float(self.wc_dsigma.exp()), "wheelchair_amp_scale": float(self.wc_damp.exp()),
                "halt_ahead_m": float(s(self.h_ahead)), "halt_sigma_long_m": float(s(self.h_long)), "halt_sigma_lat_m": float(s(self.h_lat)),
                "halt_amp": float(s(self.h_amp)), **{f"pair_{n}_width_m": float(v) for n, v in zip(("talk", "queue", "group_walk", "other"), s(self.pair_w))},
                **{f"pair_{n}_amp": float(v) for n, v in zip(("talk", "queue", "group_walk", "other"), s(self.pair_amp))}}

    def forward(self, people: torch.Tensor, mask: torch.Tensor, edge_p: torch.Tensor, node_p: torch.Tensor,
                edge_u: torch.Tensor | None = None, node_u: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """people [B, P, 6] = (x, y, cos yaw, sin yaw, vx, vy) in the robot frame; mask [B, P]; edge_p [B, P, P, 5]; node_p [B, P, 2].
        edge_u [B, P, P], node_u [B, P, 2]: staleness uncertainty in [0, 1] from the temporal memory (memory.py); each adds
        u x E^2 of its primitive to V_sem (an unobserved belief may be wrong in full). None or 0: unchanged.
        Returns C_mix, V_sem [B, H, W]."""
        s = F.softplus
        B, P = mask.shape
        X, Y = self.X.view(1, 1, *self.X.shape), self.Y.view(1, 1, *self.Y.shape)  # any grid (cost_aware.py renders at 100 x 100)
        px, py = people[..., 0].view(B, P, 1, 1), people[..., 1].view(B, P, 1, 1)
        dx, dy = X - px, Y - py
        m = mask.float().view(B, P, 1, 1)
        # motion direction (body heading when slow)
        vx, vy = people[..., 4], people[..., 5]
        speed = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
        slow = (speed < 0.1).float()
        ux = (1 - slow) * vx / speed + slow * people[..., 2]
        uy = (1 - slow) * vy / speed + slow * people[..., 3]
        ux, uy, sp_ = ux.view(B, P, 1, 1), uy.view(B, P, 1, 1), speed.view(B, P, 1, 1)
        along = dx * ux + dy * uy
        lat = -dx * uy + dy * ux

        def personal(sig0, amp):
            sf = sig0 * (1 + s(self.ps_kv) * sp_)
            sa = torch.where(along > 0, sf, sig0)
            return amp * torch.exp(-0.5 * (along ** 2 / sa ** 2 + lat ** 2 / sig0 ** 2))

        sig0, amp = s(self.ps_sigma), s(self.ps_amp)
        k_def = personal(sig0, amp)
        k_wc = personal(sig0 * self.wc_dsigma.exp(), amp * self.wc_damp.exp())
        pw = node_p[..., 1].view(B, P, 1, 1)
        e_ps = (1 - pw) * k_def + pw * k_wc
        v_ps = pw * (1 - pw) * (k_wc - k_def) ** 2
        # halt: ellipse ahead along the body heading
        hx, hy = people[..., 2].view(B, P, 1, 1), people[..., 3].view(B, P, 1, 1)
        a2 = dx * hx + dy * hy - s(self.h_ahead)
        l2 = -dx * hy + dy * hx
        k_h = s(self.h_amp) * torch.exp(-0.5 * (a2 ** 2 / s(self.h_long) ** 2 + l2 ** 2 / s(self.h_lat) ** 2))
        ph = node_p[..., 0].view(B, P, 1, 1)
        e_h, v_h = ph * k_h, ph * (1 - ph) * k_h ** 2
        if node_u is not None:
            v_ps = v_ps + node_u[..., 1].view(B, P, 1, 1) * (pw * (k_wc - k_def)) ** 2  # only the wheelchair part is uncertain
            v_h = v_h + node_u[..., 0].view(B, P, 1, 1) * e_h ** 2
        C = ((e_ps + e_h) * m).sum(1)
        V = ((v_ps + v_h) * m).sum(1)
        # pairs (i < j): capsules per class
        iu = torch.triu_indices(P, P, 1, device=people.device)
        if iu.shape[1]:
            a_x, a_y = people[:, iu[0], 0].view(B, -1, 1, 1), people[:, iu[0], 1].view(B, -1, 1, 1)
            b_x, b_y = people[:, iu[1], 0].view(B, -1, 1, 1), people[:, iu[1], 1].view(B, -1, 1, 1)
            sx, sy = b_x - a_x, b_y - a_y
            L2 = (sx ** 2 + sy ** 2).clamp(min=1e-6)
            t = (((X - a_x) * sx + (Y - a_y) * sy) / L2).clamp(0, 1)
            d2 = (X - a_x - t * sx) ** 2 + (Y - a_y - t * sy) ** 2  # B, E, H, W
            w, amp_c = s(self.pair_w), s(self.pair_amp)
            pm = (mask[:, iu[0]] & mask[:, iu[1]]).float().view(B, -1, 1, 1)
            pc = 0.5 * (edge_p[:, iu[0], iu[1], 1:] + edge_p[:, iu[1], iu[0], 1:])  # B, E, 4
            E = torch.zeros_like(d2)
            S2 = torch.zeros_like(d2)
            for c in range(4):
                kc = amp_c[c] * torch.exp(-0.5 * d2 / w[c] ** 2)
                p = pc[..., c].view(B, -1, 1, 1)
                E = E + p * kc
                S2 = S2 + p * kc ** 2
            C = C + (E * pm).sum(1)
            Vp = (S2 - E ** 2).clamp(min=0)
            if edge_u is not None:
                Vp = Vp + edge_u[:, iu[0], iu[1]].view(B, -1, 1, 1) * E ** 2
            V = V + (Vp * pm).sum(1)
        return C, V


class TypeMixCost(nn.Module):
    """Typed renderer + residual GridUNet. Output [B, 2, H, W] = (mean logit, log total variance), so it plugs into the
    probabilistic runner (mu = sigmoid(o0), sigma^2 = exp(o1))."""

    def __init__(self, use_semantic_variance: bool = True, w: int = 32) -> None:
        super().__init__()
        self.render = TypedRenderer()
        self.use_sem = use_semantic_variance
        self.pre = nn.Sequential(nn.Conv2d(2 + 3, 16, 3, 2, 1), nn.ReLU(inplace=True))
        self.dec = GridUNet(16, w, out_ch=2)
        self.lam = nn.Parameter(torch.tensor(0.0))  # softplus(0) = 0.69

    def forward(self, x: dict) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            eu, nu = x.get("edge_u"), x.get("node_u")
            C, V = self.render(x["people"].float(), x["pmask"], x["edge_p"].float(), x["node_p"].float(),
                               None if eu is None else eu.float(), None if nu is None else nu.float())
        sem = torch.sqrt(V + 1e-8) if self.use_sem else torch.zeros_like(V)
        o = self.dec(self.pre(torch.cat([C.unsqueeze(1), sem.unsqueeze(1), x["raster"]], 1)))
        o0, o1 = o[:, 0].float(), o[:, 1].float().clamp(-12.0, 0.0)
        var = o1.exp() + (F.softplus(self.lam) * V if self.use_sem else 0.0)
        self._last = {"C_mix": C.detach(), "V_sem": V.detach()}
        return torch.stack([o0, torch.log(var + 1e-8)], 1)

    def interpretable(self) -> dict:
        return {**self.render.params(), "lambda_semantic": float(F.softplus(self.lam)) if self.use_sem else 0.0}
