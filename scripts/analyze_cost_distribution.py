#!/usr/bin/env python3
"""Phase C/D of the RGB-D diagnosis: offline cost-distribution analysis and calibration ablations.

Reads the saved held-out predictions of each experiment (predictions.npz, raw model output in the
robot-centred label grid, 0..252) and the extended labels, and reports, inside the camera FOV (the
region the estimator publishes) and outside it:
  - fraction of cells with cost > 0/10/50/100/200, predicted vs label
  - precision / recall / IoU at cost thresholds, false-positive and false-negative area (m^2)
  - cost-weighted MAE (w = max(pred, label)), MAE by region, peak cost near each pedestrian
  - mean predicted and label cost by distance to the nearest pedestrian
  - background (label == 0) prediction histogram: low-amplitude noise vs high-cost false positives
  - semantic fidelity: recall of the social part of the label above the personal-space rule

Calibration (Phase D) is fitted on `val` only and evaluated on val and every test split. Pedestrian
states and labels are used for scoring only, never inside a post-processing step.

    python analyze_cost_distribution.py --out /opt/arena_ws/data/socialcost/outputs/rgbd_diagnosis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost import postprocess as pp  # noqa: E402
from socialcost.learn.data import EpisodeIndex, fov_mask  # noqa: E402
from socialcost.labels import Ped, paint  # noqa: E402

D = "/opt/arena_ws/data/socialcost"
EXP = f"{D}/experiments/pilot_v1"
CELL_AREA = 0.05 * 0.05
FRAC_T = [0, 10, 50, 100, 200]
PR_T = [10, 50, 100, 150]
DIST_BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 99]
BG_BINS = [(0, 0), (1, 9), (10, 49), (50, 99), (100, 252)]


class Sample:
    __slots__ = ("key", "family", "label", "rule", "dist", "fov", "ped_cells", "near", "depth_key")


def load_samples(split_keys: list[str], eps: dict) -> list[Sample]:
    out = []
    for ek in split_keys:
        e = eps[ek]
        fov = fov_mask(e.grid, e.K, e.T_bc)
        i, j = np.meshgrid(np.arange(e.grid.width), np.arange(e.grid.height))
        cx = e.grid.origin_x + (i + 0.5) * e.grid.resolution
        cy = e.grid.origin_y + (j + 0.5) * e.grid.resolution
        for k in range(len(e.robots)):
            s = Sample()
            s.key = (ek, k)
            s.family = e.meta.get("family") or e.meta.get("scenario")
            s.label = np.load(os.path.join(e.dir, "social_cost_extended", f"{k:06d}.npy")).astype(np.int16)
            pose = e.robots[k]["pose_map_base"]
            states = e.peds[k]["pedestrians"]
            s.rule = paint(e.params, [Ped(p["x"], p["y"], p["animation_state"]) for p in states], e.grid, (pose["x"], pose["y"], pose["yaw"])).astype(np.int16)
            c, sn = math.cos(pose["yaw"]), math.sin(pose["yaw"])
            d = np.full(cx.shape, 99.0, np.float32)
            cells = []
            for p in states:
                dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
                lx, ly = c * dx + sn * dy, -sn * dx + c * dy
                d = np.minimum(d, np.hypot(cx - lx, cy - ly))
                cells.append((lx, ly))
            s.dist = d.astype(np.float16)
            s.fov = fov
            s.ped_cells = cells
            s.near = [np.hypot(cx - lx, cy - ly) <= 0.5 for lx, ly in cells]
            s.depth_key = (e.dir, k)
            out.append(s)
    return out


class Acc:
    def __init__(self) -> None:
        self.s = defaultdict(float)
        self.dist = np.zeros((len(DIST_BINS) - 1, 3))  # n, sum pred, sum label (FOV)
        self.peaks = []  # (label peak, pred peak) per visible pedestrian

    def add(self, pred: np.ndarray, s: Sample, published: bool = True) -> None:
        p = pred.astype(np.int16)
        t = s.label
        f = s.fov
        S = self.s
        S["n"] += 1
        pub = np.where(f, p, 0) if published else p
        for name, m in (("fov", f), ("out", ~f)):
            pv = (p if name == "fov" else pub)[m]
            tv = t[m]
            S[f"{name}_cells"] += m.sum()
            S[f"{name}_abs"] += np.abs(pv - tv).sum()
            w = np.maximum(pv, tv).astype(np.float64)
            S[f"{name}_wabs"] += (w * np.abs(pv - tv)).sum()
            S[f"{name}_w"] += w.sum()
            S[f"{name}_label_mass"] += tv.sum()
            S[f"{name}_pred_mass"] += pv.sum()
        pf, tf = p[f], t[f]
        for th in FRAC_T:
            S[f"pred_gt{th}"] += (pf > th).sum()
            S[f"label_gt{th}"] += (tf > th).sum()
        for th in PR_T:
            a, b = pf >= th, tf >= th
            S[f"tp{th}"] += (a & b).sum()
            S[f"fp{th}"] += (a & ~b).sum()
            S[f"fn{th}"] += (~a & b).sum()
        bg = tf == 0
        S["bg_cells"] += bg.sum()
        S["bg_fp"] += (pf[bg] > 0).sum()
        for lo, hi in BG_BINS:
            S[f"bg_{lo}_{hi}"] += ((pf[bg] >= lo) & (pf[bg] <= hi)).sum()
        extra = (t > s.rule) & f
        S["extra_cells"] += extra.sum()
        S["extra_hits"] += ((p > s.rule + 10) & extra).sum()
        # strict: the social part above the rule that carries real cost (>= 30 and >= rule + 10)
        strict = (t >= s.rule + 10) & (t >= 30) & f
        S["strict_cells"] += strict.sum()
        S["strict_hits"] += ((p >= s.rule + 10) & (p >= 0.5 * t) & strict).sum()
        halo = (t > 0) & (t < 30) & f
        S["halo_cells"] += halo.sum()
        S["halo_hits"] += ((p > 0) & halo).sum()
        hi = (t >= 100) & f
        S["high_cells"] += hi.sum()
        S["high_abs"] += np.abs(p[hi] - t[hi]).sum()
        S["high_signed"] += (p[hi] - t[hi]).sum()
        db = np.digitize(s.dist[f].astype(np.float32), DIST_BINS) - 1
        np.add.at(self.dist, (db, 0), 1)
        np.add.at(self.dist, (db, 1), pf)
        np.add.at(self.dist, (db, 2), tf)
        for near in s.near:
            if (near & f).sum() > 20:
                self.peaks.append((int(t[near].max()), int(p[near].max())))

    def result(self) -> dict:
        S = self.s
        r = {"samples": int(S["n"])}
        r["mae_fov"] = S["fov_abs"] / S["fov_cells"]
        r["mae_outside_fov_published"] = S["out_abs"] / S["out_cells"]
        r["wmae_fov"] = S["fov_wabs"] / max(S["fov_w"], 1)
        r["label_mass_outside_fov_frac"] = S["out_label_mass"] / max(S["out_label_mass"] + S["fov_label_mass"], 1)
        r["pred_mass_over_label_mass_fov"] = S["fov_pred_mass"] / max(S["fov_label_mass"], 1)
        r["frac_fov_cells"] = {f">{th}": {"pred": S[f"pred_gt{th}"] / S["fov_cells"], "label": S[f"label_gt{th}"] / S["fov_cells"]} for th in FRAC_T}
        r["pr"] = {}
        for th in PR_T:
            tp, fp, fn = S[f"tp{th}"], S[f"fp{th}"], S[f"fn{th}"]
            r["pr"][f">={th}"] = {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                                  "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                                  "fp_area_m2_per_sample": fp * CELL_AREA / S["n"], "fn_area_m2_per_sample": fn * CELL_AREA / S["n"]}
        r["background_pred_hist"] = {f"{lo}-{hi}": S[f"bg_{lo}_{hi}"] / max(S["bg_cells"], 1) for lo, hi in BG_BINS}
        r["social_extra_recall_fov"] = S["extra_hits"] / S["extra_cells"] if S["extra_cells"] else None
        r["social_extra_recall_strict"] = S["strict_hits"] / S["strict_cells"] if S["strict_cells"] else None
        r["low_halo_recall"] = S["halo_hits"] / S["halo_cells"] if S["halo_cells"] else None
        r["bg_fp_area_m2_per_sample"] = S["bg_fp"] * CELL_AREA / S["n"]
        r["high_mae"] = S["high_abs"] / S["high_cells"] if S["high_cells"] else None
        r["high_bias"] = S["high_signed"] / S["high_cells"] if S["high_cells"] else None
        r["cost_by_distance"] = [{"bin": f"{DIST_BINS[k]}-{DIST_BINS[k + 1]}", "cells": int(self.dist[k, 0]),
                                  "pred": self.dist[k, 1] / self.dist[k, 0] if self.dist[k, 0] else None,
                                  "label": self.dist[k, 2] / self.dist[k, 0] if self.dist[k, 0] else None} for k in range(len(DIST_BINS) - 1)]
        if self.peaks:
            a = np.array(self.peaks, float)
            r["peak_near_ped"] = {"n": len(a), "label_mean": a[:, 0].mean(), "pred_mean": a[:, 1].mean(),
                                  "pred_below_half_label_frac": float((a[:, 1] < 0.5 * a[:, 0]).mean())}
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}


_VIS_CACHE: dict = {}


def vis_for(s: Sample, eps_by_dir: dict, margin: float) -> np.ndarray:
    k = (s.depth_key, margin)
    if k not in _VIS_CACHE:
        e = eps_by_dir[s.depth_key[0]]
        depth = np.load(os.path.join(e.dir, "depth", f"{s.depth_key[1]:06d}.npy"))
        _VIS_CACHE[k] = pp.visibility_mask(depth, e.K, e.T_bc, e.grid, margin_m=margin)
    return _VIS_CACHE[k]


def evaluate(preds: dict, samples: list[Sample], split: str, spec: dict, eps_by_dir: dict, by_family: bool = False) -> dict:
    acc = {"overall": Acc()}
    for s in samples:
        raw = preds[f"{split}|{s.key[0]}|{s.key[1]}"]
        vis = vis_for(s, eps_by_dir, spec.get("margin_m", 0.6)) if spec.get("visibility") else None
        p = pp.apply(raw, spec, vis) if spec else raw
        acc["overall"].add(p, s)
        if by_family:
            acc.setdefault(f"family:{s.family}", Acc()).add(p, s)
    return {k: a.result() for k, a in acc.items()}


def fit_lut(preds: dict, samples: list[Sample], split: str, base: dict, eps_by_dir: dict) -> list[int]:
    P, T = [], []
    for s in samples:
        raw = preds[f"{split}|{s.key[0]}|{s.key[1]}"]
        vis = vis_for(s, eps_by_dir, base.get("margin_m", 0.6)) if base.get("visibility") else None
        p = pp.apply(raw, base, vis) if base else raw
        P.append(p[s.fov])
        T.append(s.label[s.fov])
    return pp.fit_isotonic_lut(np.concatenate(P), np.concatenate(T))


def figures(out: str, preds_by_exp: dict, specs: dict, samples: list[Sample], split: str, eps_by_dir: dict, n: int = 8) -> list[str]:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    # spread over families and over time
    fams = defaultdict(list)
    for s in samples:
        fams[s.family].append(s)
    pick = []
    for f in sorted(fams):
        L = fams[f]
        pick += [L[len(L) // 3], L[2 * len(L) // 3]]
    pick = pick[:n]
    cols = [("label (extended)", None, None)] + [(f"{e[6:]} raw", e, {}) for e in preds_by_exp] + [(f"{e.strip()[6:]} {pp.describe(sp)}", e, sp) for e, sp in specs.items()]
    paths = []
    for s in pick:
        e = eps_by_dir[s.depth_key[0]]
        rgb = cv2.imread(os.path.join(e.dir, "rgb", f"{s.depth_key[1]:06d}.png"))[:, :, ::-1]
        fig, ax = plt.subplots(1, len(cols) + 1, figsize=(3.0 * (len(cols) + 1), 3.3))
        ax[0].imshow(rgb)
        ax[0].set_title(f"{s.family} {os.path.basename(s.key[0])}#{s.key[1]}", fontsize=8)
        ax[0].axis("off")
        for a, (title, exp, sp) in zip(ax[1:], cols):
            if exp is None:
                img = s.label
            else:
                raw = preds_by_exp[exp.strip()][f"{split}|{s.key[0]}|{s.key[1]}"]
                vis = vis_for(s, eps_by_dir, sp.get("margin_m", 0.6)) if sp and sp.get("visibility") else None
                img = np.where(s.fov, pp.apply(raw, sp, vis) if sp else raw, 0)
            # robot frame shown with x forward = up and y left = left: plot u = -y (right), v = x (up)
            # img[j, i]: j <-> y, i <-> x  ->  M[v, u] = img.T[:, ::-1]
            a.imshow(img.T[:, ::-1], vmin=0, vmax=252, cmap="magma", origin="lower", extent=(-5, 5, -5, 5))
            a.contour(s.fov.T[:, ::-1].astype(float), levels=[0.5], colors="cyan", linewidths=0.6, origin="lower", extent=(-5, 5, -5, 5))
            for lx, ly in s.ped_cells:
                a.plot(-ly, lx, "o", mfc="none", mec="lime", ms=6)
            a.plot(0, 0, "^", color="white", ms=5)
            a.set_xlim(-5, 5)
            a.set_ylim(-5, 5)
            a.set_title(title, fontsize=7)
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        p = os.path.join(out, f"{split}_{os.path.basename(s.key[0])}_{s.key[1]:03d}.png")
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_profiles(out: str, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    lab = None
    for name, r in results.items():
        cb = r["cost_by_distance"]
        x = [0.5 * (DIST_BINS[k] + min(DIST_BINS[k + 1], 6)) for k in range(len(cb))]
        ax[0].plot(x, [c["pred"] for c in cb], "o-", label=name, ms=3)
        lab = [c["label"] for c in cb]
    ax[0].plot(x, lab, "k--", label="label", lw=2)
    ax[0].set_xlabel("distance to nearest pedestrian [m] (FOV cells)")
    ax[0].set_ylabel("mean cost")
    ax[0].legend(fontsize=7)
    for name, r in results.items():
        th = list(r["pr"].keys())
        ax[1].plot([r["pr"][t]["recall"] for t in th], [r["pr"][t]["precision"] for t in th], "o-", label=name, ms=4)
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_title("cost >= 10 / 50 / 100 / 150", fontsize=8)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "profiles_test_A.png"), dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", nargs="+", default=["pilot_rgbd_extended", "pilot_depth_extended", "pilot_rgb_extended", "pilot_pos_extended", "pilot_pos_fov_extended"])
    ap.add_argument("--calibrate", nargs="+", default=["pilot_rgbd_extended", "pilot_depth_extended"])
    ap.add_argument("--test-splits", nargs="+", default=["test_A", "test_B", "test_C", "test_D"])
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from saved specs (post_*.json, specs/rgbd_threshold13.json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    split = json.load(open(f"{D}/experiments/splits_pilot_v1.json"))
    roots = [f"{D}/datasets/pilot_v1"]
    from socialcost.learn.data import episode_dirs
    eps = {e.key: e for e in (EpisodeIndex(d) for d in episode_dirs(roots))}
    eps_by_dir = {e.dir: e for e in eps.values()}
    samples = {sp: load_samples(split[sp], eps) for sp in ["val", *args.test_splits]}
    print({k: len(v) for k, v in samples.items()}, flush=True)
    preds = {e: dict(np.load(f"{EXP}/{e}/predictions.npz")) for e in args.experiments}
    if args.figures_only:
        figdir = os.path.join(args.out, "figures")
        os.makedirs(figdir, exist_ok=True)
        t13 = json.load(open(os.path.join(args.out, "specs", "rgbd_threshold13.json")))
        sel = json.load(open(os.path.join(args.out, "post_pilot_rgbd_extended.json")))
        cols = {"pilot_rgbd_extended": preds["pilot_rgbd_extended"], "pilot_depth_extended": preds["pilot_depth_extended"], "pilot_pos_extended": preds["pilot_pos_extended"]}
        figures(figdir, cols, {"pilot_rgbd_extended": t13, "pilot_rgbd_extended ": sel}, samples["test_A"], "test_A", eps_by_dir)
        return 0

    # ---- Phase C: raw distributions on every split
    phase_c = {e: {sp: evaluate(preds[e], samples[sp], sp, {}, eps_by_dir, by_family=(sp == "test_A")) for sp in samples} for e in args.experiments}
    json.dump(phase_c, open(os.path.join(args.out, "phaseC_cost_distribution.json"), "w"), indent=1)
    print("phase C done", flush=True)

    # ---- Phase D: ablations, fit on val, report val + tests
    phase_d = {}
    chosen = {}
    for e in args.calibrate:
        P = preds[e]
        rows = []

        def run(spec: dict, tag: str) -> dict:
            r = {"tag": tag, "spec": {k: v for k, v in spec.items() if k != "lut"}, "has_lut": "lut" in spec,
                 "val": evaluate(P, samples["val"], "val", spec, eps_by_dir)["overall"]}
            rows.append(r)
            v = r["val"]
            print(e, tag, "mae", round(v["mae_fov"], 2), "wmae", round(v["wmae_fov"], 2), "strict", round(v["social_extra_recall_strict"], 3),
                  "rec50", round(v["pr"][">=50"]["recall"], 3), "halo", round(v["low_halo_recall"], 3), "bgFP", round(v["bg_fp_area_m2_per_sample"], 2), flush=True)
            return r

        run({}, "raw")
        for t in (5, 10, 12, 13, 14, 16, 20, 30, 50):
            run({"threshold": t}, f"threshold={t}")
        lut = fit_lut(P, samples["val"], "val", {}, eps_by_dir)
        run({"lut": lut}, "isotonic")
        for sc in (0.8, 1.2, 1.4, 1.6):
            run({"scale": sc}, f"scale={sc}")
        for mp in (40, 80, 120):
            run({"min_peak": mp}, f"min_peak={mp}")
        for ma in (20, 80, 200):
            run({"min_area_cells": ma}, f"min_area={ma}")
        for m in (0.3, 0.6, 1.0):
            run({"visibility": True, "margin_m": m}, f"visibility(margin={m})")
        for bs in (1.0, 2.0, 3.0):
            run({"blur_sigma": bs}, f"blur={bs}")
        # pre-registered selection (val only): minimise MAE in the FOV subject to keeping >= 95 % of the raw
        # recall at cost >= 50 and of the strict social-extra recall; lut and scale enter only as candidates
        raw_v = rows[0]["val"]
        def feasible(v: dict) -> bool:
            return (v["pr"][">=50"]["recall"] >= 0.95 * raw_v["pr"][">=50"]["recall"]
                    and v["social_extra_recall_strict"] >= 0.95 * raw_v["social_extra_recall_strict"])
        for t in (12, 13, 14, 16):
            for bs in (0.0, 1.0, 2.0):
                for vis in (False, True):
                    cand = {"threshold": t}
                    if bs:
                        cand["blur_sigma"] = bs
                    if vis:
                        cand.update({"visibility": True, "margin_m": 1.0})
                    run(cand, "grid:" + pp.describe(cand))
        ok = [r for r in rows if feasible(r["val"])]
        best = min(ok, key=lambda r: r["val"]["mae_fov"]) if ok else rows[0]
        spec = dict(best["spec"])
        if best["has_lut"]:
            spec["lut"] = lut
        print(e, "selected", best["tag"], flush=True)
        run(spec, "selected")
        chosen[e] = spec
        # test every single-step best and the selected spec on held-out splits
        best_of = {}
        for r in rows[1:]:
            kind = r["tag"].split("=")[0].split("(")[0]
            if kind not in best_of or r["val"]["mae_fov"] < best_of[kind]["val"]["mae_fov"]:
                best_of[kind] = r
        test_tags = {"raw", "selected", "isotonic", "threshold=13", "visibility(margin=1.0)", "blur=2.0"} | {best_of[k]["tag"] for k in best_of}
        for r in rows:
            if r["tag"] in test_tags:
                sp_full = dict(r["spec"])
                if r["has_lut"]:
                    sp_full["lut"] = chosen[e]["lut"] if r["tag"] == "selected" and "lut" in chosen[e] else lut
                r["test"] = {ts: evaluate(P, samples[ts], ts, sp_full, eps_by_dir)["overall"] for ts in args.test_splits}
        phase_d[e] = rows
        json.dump(spec, open(os.path.join(args.out, f"post_{e}.json"), "w"))
    json.dump(phase_d, open(os.path.join(args.out, "phaseD_calibration.json"), "w"), indent=1)

    # ---- figures on test_A
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    plot_profiles(figdir,
                  {**{e: phase_c[e]["test_A"]["overall"] for e in args.experiments},
                   **{f"{e}+post": evaluate(preds[e], samples["test_A"], "test_A", chosen[e], eps_by_dir)["overall"] for e in chosen}})
    figures(figdir, {e: preds[e] for e in ["pilot_rgbd_extended", "pilot_pos_extended"] if e in preds}, {e: chosen[e] for e in chosen if e == "pilot_rgbd_extended"},
            samples["test_A"], "test_A", eps_by_dir)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
