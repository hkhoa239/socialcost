"""Benchmark metrics in label units (0..252), accumulated per group (split, split x family).

  mae / rmse            every cell                          mae_fov  cells inside the camera FOV
  mae_nonzero           cells where the target has cost
  hc_iou                IoU of {pred >= 100} and {target >= 100} (high cost)
  inter_iou             IoU of the predicted interaction region {pred - personal_space >= 12} and the true interaction
                        region (conversation component, plus queue for the membership label)
  fp_area_m2            area per sample where pred >= 12 and target == 0 (0.0025 m^2 per cell)
Probabilistic models (predicted sigma per cell, label units):
  nll                   Gaussian negative log-likelihood per cell (sigma floored at 1 cost unit)
  cov95 / cov95_nonzero fraction of cells whose target lies within mu +- 1.96 sigma (calibrated: 0.95)
  spearman_err_sigma    rank correlation between sigma and |error| on the evaluation region (nonzero target or prediction)
  ause                  area between the sparsification curve (drop most-uncertain cells first) and the oracle curve
                        (drop largest errors first), MAE normalised by the full-region MAE (lower is better)
Counterfactual (Test D): per pair and common slot, the OFF maps are first moved into the ON robot frame (the robot
pose differs between conditions, median 0.17 m), then delta = ON - OFF for prediction and target;
  cf_err                mean |delta_pred - delta_target| over cells where delta_target != 0
  cf_response           sum |delta_pred| / sum |delta_target| over those cells (1 = full response, 0 = no response)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

SCALE = 252.0
HIGH = 100.0
HAS = 12.0
CELL_M2 = 0.05 * 0.05


class Acc:
    def __init__(self) -> None:
        self.s = defaultdict(lambda: defaultdict(float))

    def add(self, groups: list[list[str]], pred: torch.Tensor, target: torch.Tensor, rule: torch.Tensor, fov: torch.Tensor,
            inter: torch.Tensor, sigma: torch.Tensor | None = None) -> None:
        """pred / target / rule in [0, 1]; groups[i] = group names of sample i."""
        p, t, r = pred * SCALE, target * SCALE, rule * SCALE
        err = (p - t).abs()
        for i, gs in enumerate(groups):
            e, pi, ti, ri = err[i], p[i], t[i], r[i]
            nz = ti > 0
            fv = fov[i].bool()
            ph, th = pi >= HIGH, ti >= HIGH
            pin, tin = (pi - ri) >= HAS, inter[i].bool()
            fp = ((pi >= HAS) & (ti == 0)).sum().item()
            vals = {"abs": e.sum().item(), "sq": (e ** 2).sum().item(), "n": e.numel(), "fov_abs": e[fv].sum().item(), "fov_n": fv.sum().item(),
                    "nz_abs": e[nz].sum().item(), "nz_n": nz.sum().item(), "hc_i": (ph & th).sum().item(), "hc_u": (ph | th).sum().item(),
                    "in_i": (pin & tin).sum().item(), "in_u": (pin | tin).sum().item(), "fp": fp, "samples": 1}
            if sigma is not None:
                sg = (sigma[i] * SCALE).clamp(min=1.0)
                vals["nll"] = float((0.5 * (e / sg) ** 2 + sg.log() + 0.5 * np.log(2 * np.pi)).sum())
                inside = e <= 1.96 * sg
                vals["cov"] = float(inside.sum())
                vals["cov_nz"] = float(inside[nz].sum())
                region = nz | (pi >= HAS)
                if region.sum() >= 50:
                    ee, ss = e[region].double().numpy(), sg[region].double().numpy()
                    rs = np.argsort(np.argsort(ss)).astype(float)
                    re = np.argsort(np.argsort(ee)).astype(float)
                    vals["spear"] = float(np.corrcoef(rs, re)[0, 1]) if ee.std() > 0 and ss.std() > 0 else 0.0
                    fr = np.linspace(0, 0.9, 10)
                    base = ee.mean() or 1.0
                    by_s = ee[np.argsort(-ss)]
                    by_e = np.sort(ee)[::-1]
                    n = len(ee)
                    cs = [by_s[int(f * n):].mean() / base for f in fr]
                    co = [by_e[int(f * n):].mean() / base for f in fr]
                    vals["ause"] = float(np.trapz(np.array(cs) - np.array(co), fr))
                    vals["unc_n"] = 1
            for g in gs:
                for k, v in vals.items():
                    self.s[g][k] += v

    def result(self) -> dict:
        out = {}
        for g, s in self.s.items():
            out[g] = {"samples": int(s["samples"]), "mae": s["abs"] / s["n"], "rmse": float(np.sqrt(s["sq"] / s["n"])),
                      "mae_fov": s["fov_abs"] / s["fov_n"] if s["fov_n"] else None, "mae_nonzero": s["nz_abs"] / s["nz_n"] if s["nz_n"] else None,
                      "hc_iou": s["hc_i"] / s["hc_u"] if s["hc_u"] else None, "inter_iou": s["in_i"] / s["in_u"] if s["in_u"] else None,
                      "fp_area_m2": s["fp"] * CELL_M2 / s["samples"]}
            if "nll" in s:
                out[g].update({"nll": s["nll"] / s["n"], "cov95": s["cov"] / s["n"], "cov95_nonzero": s["cov_nz"] / s["nz_n"] if s["nz_n"] else None,
                               "spearman_err_sigma": s["spear"] / s["unc_n"] if s.get("unc_n") else None,
                               "ause": s["ause"] / s["unc_n"] if s.get("unc_n") else None})
        return out


def warp_to(src: np.ndarray, pose_src: dict, pose_dst: dict, res: float = 0.05, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Resample a robot-centred grid (array[j, i], i along x) from pose_src into the frame of pose_dst
    (nearest cell). Returns the warped grid and the mask of destination cells that fall inside the source grid."""
    c = (np.arange(n) + 0.5) * res - n * res / 2
    X, Y = np.meshgrid(c, c)  # destination cell centres, X along i, Y along j
    cd, sd = np.cos(pose_dst["yaw"]), np.sin(pose_dst["yaw"])
    wx, wy = pose_dst["x"] + cd * X - sd * Y, pose_dst["y"] + sd * X + cd * Y
    cs_, ss_ = np.cos(pose_src["yaw"]), np.sin(pose_src["yaw"])
    dx, dy = wx - pose_src["x"], wy - pose_src["y"]
    lx, ly = cs_ * dx + ss_ * dy, -ss_ * dx + cs_ * dy
    i = np.floor((lx + n * res / 2) / res).astype(int)
    j = np.floor((ly + n * res / 2) / res).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    out = np.zeros_like(src)
    out[ok] = src[j[ok], i[ok]]
    return out, ok


def counterfactual(pairs: list[dict], preds: dict, targets: dict, poses: dict | None = None) -> dict:
    """pairs: {pair_id, family, on, off, observability}; preds / targets: {(episode, sample): uint8 array};
    poses: {(episode, sample): robot pose dict} - when given, OFF maps are warped into the ON robot frame first."""
    rows = defaultdict(lambda: defaultdict(float))
    for pr in pairs:
        ks = sorted({k for (e, k) in preds if e == pr["on"]} & {k for (e, k) in preds if e == pr["off"]})
        for k in ks:
            po, to = preds[(pr["off"], k)].astype(np.float32), targets[(pr["off"], k)].astype(np.float32)
            valid = np.ones_like(po, bool)
            if poses is not None:
                po, valid = warp_to(po, poses[(pr["off"], k)], poses[(pr["on"], k)])
                to, _ = warp_to(to, poses[(pr["off"], k)], poses[(pr["on"], k)])
            dp = preds[(pr["on"], k)].astype(np.float32) - po
            dt = targets[(pr["on"], k)].astype(np.float32) - to
            m = (dt != 0) & valid
            if not m.any():
                continue
            for g in ("all", f"family:{pr['family']}", f"obs:{pr['observability']}"):
                r = rows[g]
                r["err"] += float(np.abs(dp[m] - dt[m]).sum())
                r["n"] += float(m.sum())
                r["resp_p"] += float(np.abs(dp[m]).sum())
                r["resp_t"] += float(np.abs(dt[m]).sum())
                r["slots"] += 1
    return {g: {"cf_err": r["err"] / r["n"], "cf_response": r["resp_p"] / r["resp_t"], "slots": int(r["slots"])} for g, r in rows.items() if r["n"]}
