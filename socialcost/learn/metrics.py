"""Costmap metrics, in label units (0..max_cost_value), over the full grid and the camera-visible region.

regions:
  all          every cell
  fov          cells inside the camera FOV (what an image model can observe)
  nonzero      cells where the target has cost
  high         cells with target >= 150
  social_extra cells where the target exceeds the personal-space-only rule on the same people:
               the interaction / gesture / membership / mobility part of the label
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

SCALE = 252.0
NZ = 12.0 / SCALE  # predicted "has cost" threshold (cost_bias 0.05 x 252)


class Accumulator:
    def __init__(self) -> None:
        self.s = defaultdict(lambda: defaultdict(float))

    def add(self, group: str, pred: torch.Tensor, target: torch.Tensor, rule: torch.Tensor, fov: torch.Tensor) -> None:
        p, t, r = pred * SCALE, target * SCALE, rule * SCALE
        err = (p - t).abs()
        regions = {"all": torch.ones_like(t, dtype=torch.bool), "fov": fov.bool(), "nonzero": t > 0, "high": t >= 150,
                   "social_extra": t > r + 0.5}
        regions["social_extra_fov"] = regions["social_extra"] & regions["fov"]
        regions["nonzero_fov"] = regions["nonzero"] & regions["fov"]
        g = self.s[group]
        for name, m in regions.items():
            n = float(m.sum())
            g[f"{name}_n"] += n
            g[f"{name}_abs"] += float(err[m].sum())
            g[f"{name}_sq"] += float((err[m] ** 2).sum())
        for name in ("all", "fov"):
            m = regions[name]
            pa, ta = (pred > NZ) & m, (target > 0) & m
            g[f"{name}_inter"] += float((pa & ta).sum())
            g[f"{name}_union"] += float((pa | ta).sum())
        # does the prediction raise cost where the social part of the label is (above the rule)?
        m = regions["social_extra"]
        g["social_extra_recall_hits"] += float(((p > r + 10) & m).sum())
        g["samples"] += float(t.shape[0])

    def result(self) -> dict:
        out = {}
        for group, g in self.s.items():
            r = {"samples": int(g["samples"])}
            for name in ("all", "fov", "nonzero", "nonzero_fov", "high", "social_extra", "social_extra_fov"):
                n = g[f"{name}_n"]
                r[f"mae_{name}"] = g[f"{name}_abs"] / n if n else None
                r[f"rmse_{name}"] = float(np.sqrt(g[f"{name}_sq"] / n)) if n else None
                r[f"cells_{name}"] = int(n)
            for name in ("all", "fov"):
                u = g[f"{name}_union"]
                r[f"iou_nonzero_{name}"] = g[f"{name}_inter"] / u if u else None
            n = g["social_extra_n"]
            r["social_extra_recall"] = g["social_extra_recall_hits"] / n if n else None
            out[group] = r
        return out
