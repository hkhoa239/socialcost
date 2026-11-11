"""Cost-aware training of stage 1 through the frozen, differentiable TypeMix renderer.

A classification loss treats every mistake alike; the costmap does not (a missed conversation band or raised-hand zone
costs far more than a wrong group-walk link, whose learned cost amplitude is ~0.03). Following task-based end-to-end
learning (Donti, Amos & Kolter, NeurIPS 2017) and planner-centric perception metrics (Philion et al., CVPR 2020), stage 1
is additionally trained on the downstream quantity: the cost its type probabilities produce.

  renderer   TypedRenderer with the parameters learned by TypeMix on true types (mean of the 3 seeds, membership
             label), frozen; robot-centred 10 m x 10 m grid at 0.125 m (80 x 80), first 24 samples of each batch
  prediction p(type) of every pair = sigmoid(same) * softmax(type), p(none) = 1 - sigmoid(same); halt / mobility aid
             = sigmoid(node). The renderer gives the expected cost E[C] and its type variance V (closed form)
  target     the same renderer on the true types: C*
  loss       expected squared cost error under the predicted type distribution, integrated over the grid (m^2):
                 E_types[(C - C*)^2] = (E[C] - C*)^2 + V
             a proper score in cost space (minimised by the true type distribution), so probabilities stay calibrated
             while errors are weighted by their cost impact.
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .typemix import TypedRenderer

GT_RUNS = "/home/robotics/arena_ws/data/costmap_bench/runs/main/typemix_gt_membership_s*/best.pt"


class CostCritic(nn.Module):
    def __init__(self, n: int = 80, res: float = 0.125, ckpts: str = GT_RUNS, max_samples: int = 24) -> None:
        super().__init__()
        self.r = TypedRenderer()
        sds = [torch.load(f, map_location="cpu")["model"] for f in sorted(glob.glob(ckpts))]
        assert sds, f"no TypeMix ground-truth checkpoints at {ckpts}"
        own = self.r.state_dict()
        for k in own:
            own[k] = torch.stack([sd[f"render.{k}"].float() for sd in sds]).mean(0)
        self.r.load_state_dict(own)
        c = (torch.arange(n, dtype=torch.float32) + 0.5) * res - n * res / 2
        Y, X = torch.meshgrid(c, c, indexing="ij")
        self.r.X, self.r.Y = X, Y  # replaces the non-persistent 200 x 200 buffers
        self.cell = res * res
        self.max_samples = max_samples  # the loss is a per-sample mean: a sub-batch keeps it unbiased and 3x cheaper
        self.n_ckpts = len(sds)
        for p in self.parameters():
            p.requires_grad_(False)

    def to(self, *a, **k):
        super().to(*a, **k)
        self.r.X, self.r.Y = self.r.X.to(*a, **k), self.r.Y.to(*a, **k)
        return self

    @staticmethod
    def people(b: dict) -> torch.Tensor:
        """(x, y, cos yaw, sin yaw, vx, vy) at t in the robot frame, from the last history step of the node features."""
        n = b["node"][:, -1].float()
        return torch.stack([n[..., 0] * 5, n[..., 1] * 5, n[..., 3], n[..., 2], n[..., 4], n[..., 5]], -1)

    def forward(self, b: dict, same: torch.Tensor, typ: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        k = self.max_samples
        b = {key: b[key][:k] for key in ("node", "mask", "ecls", "halt", "wheel")}
        same, typ, node = same[:k], typ[:k], node[:k]
        with torch.autocast("cuda", enabled=False):
            mask = b["mask"]
            P = mask.shape[1]
            ppl = self.people(b)
            ps = torch.sigmoid(same.float())
            q = torch.softmax(typ.float(), -1)
            edge = torch.cat([(1 - ps).unsqueeze(-1), ps.unsqueeze(-1) * q], -1)
            node_p = torch.sigmoid(node.float())
            y = b["ecls"].clamp(min=0)
            edge_gt = F.one_hot(y, 5).float()
            node_gt = torch.stack([b["halt"], b["wheel"]], -1).float()
            C, V = self.r(ppl, mask, edge, node_p)
            with torch.no_grad():
                Cg, _ = self.r(ppl, mask, edge_gt, node_gt)
            return (((C - Cg) ** 2 + V).sum((1, 2)) * self.cell).mean()
