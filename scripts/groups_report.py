#!/usr/bin/env python3
"""Report for stage 1 v2 (explicit groups) -> reports/stage1_groups_results.md

    $PY groups_report.py --root data/costmap_bench --out data/socialcost/reports

Stage 1: v1 (tracks, vis) re-scored with the v2 evaluator (v1 marginals; groups by post-hoc correlation clustering of
v1's P(interacting), types by the argmax of the group's pair classes) against v2 (2 s and 8 s history, with / without
grouping, tracks / wide crops). Stage 2: TypeMix fed by each stage 1, 3 seeds, both labels, Test A-D.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from cascade_stage1 import test_d  # noqa: E402
from cascade_stage1_v2 import evaluate  # noqa: E402
from socialcost.bench.data import split_ids  # noqa: E402
from socialcost.bench.group_net import EpisodeCache2, correlation_clustering  # noqa: E402

S1_ROWS = [("v1 tracks (2 s, NRI, 5-way)", "tracks"), ("v1 vision (2 s, 64x32 crops)", "vis"),
           ("v2 tracks, 2 s, grouped", "tracks2_h2s"), ("v2 tracks, 2 s, no grouping", "tracks2_h2sraw"),
           ("v2 tracks, 8 s, grouped", "tracks2"), ("v2 tracks, 8 s, no grouping", "tracks2raw"),
           ("v2 vision, 8 s, grouped", "vis2"), ("v2 vision, 8 s, no grouping", "vis2raw"),
           ("v3 tracks, 8 s, regularised, no grouping", "tracks3raw"), ("v3 vision, 2 s, regularised, no grouping", "vis3_h2sraw"),
           ("v3 vision, 8 s, regularised, grouped", "vis3"), ("v3 vision, 8 s, regularised, no grouping", "vis3raw")]
S2_ROWS = [("position only (pos_u)", "pos_u"), ("TypeMix + v1 tracks", "typemix_tracks"), ("TypeMix + v1 vision", "typemix_vis"),
           ("TypeMix + v2 tracks", "typemix_tracks2"), ("TypeMix + v2 vision, no grouping", "typemix_vis2raw"),
           ("TypeMix + v2 vision, grouped", "typemix_vis2"), ("TypeMix + v3 tracks (regularised)", "typemix_tracks3raw"),
           ("**TypeMix + v3 vision (regularised)**", "typemix_vis3raw"), ("*TypeMix + true types (bound)*", "typemix_gt")]


def v1_sink(root: str, var: str, ids: list[str], cache: dict) -> dict:
    sink = {}
    for e in ids:
        f = os.path.join(root, "stage1", var, "preds", f"{e}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        rows = {}
        for t in range(z["edge_p"].shape[0]):
            ep = z["edge_p"][t].astype(np.float64)
            P = ep.shape[0]
            pi = (1 - ep[..., 0]).clip(1e-6, 1 - 1e-6)
            pi = 0.5 * (pi + pi.T)
            lab = correlation_clustering(np.log(pi) - np.log1p(-pi))
            sizes = np.bincount(lab)[lab]
            mt = -np.ones(P, int)
            for g in np.unique(lab):
                idx = np.flatnonzero(lab == g)
                if len(idx) >= 2:
                    s = ep[np.ix_(idx, idx)][..., 1:].sum((0, 1))
                    mt[idx] = int(s.argmax())
            rows[t] = {"edge_p": ep, "node_p": z["node_p"][t].astype(np.float64), "map_group": np.where(sizes >= 2, lab, -1), "map_type": mt}
        sink[e] = rows
    return sink


def fmt(x, p=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root
    splits = {s: split_ids(s) for s in ("val", "test_a", "test_b", "test_c")}
    split_of = {e: s for s, v in splits.items() for e in v}
    ids = [e for v in splits.values() for e in v]

    class DS:
        cache = {}

    for e in ids:
        c = EpisodeCache2(e, False)
        if c.P:
            DS.cache[e] = c
    s1 = {}
    for _, key in S1_ROWS:
        if key in ("tracks", "vis"):
            if not os.path.exists(os.path.join(root, "stage1", key, "metrics.json")):
                continue
            sink = v1_sink(root, key, list(DS.cache), DS.cache)
            s1[key] = {"stage1": evaluate(sink, DS, split_of, raw=False), "test_d": json.load(open(os.path.join(root, "stage1", key, "metrics.json")))["test_d"]}
        else:
            f = os.path.join(root, "stage1", key, "metrics.json")
            if os.path.exists(f):
                s1[key] = json.load(open(f))
    L = ["# Stage 1 v2: explicit social groups, long history, wide crops", "",
         "Stage 1 predicts who interacts with whom (and how), raised stop hands and mobility aids; stage 2 (TypeMix) turns",
         "those probabilities into the social costmap. v2 changes (socialcost/bench/group_net.py, scripts/cascade_stage1_v2.py):",
         "", "- **history** 8 s (v1: 2 s) per person, plus a pair-level temporal encoder over 8 s of pairwise geometry",
         "  (distance, relative bearing, mutual facing, velocity difference) - the trajectory-level pair relation of",
         "  Jahangard et al. 2023 (*Real-time trajectory-based social group detection*), in metric space instead of image boxes",
         "- **relational layers** edge-aware graph transformer (Dwivedi & Bresson 2021, as in Jahangard et al.) instead of NRI message passing",
         "- **factorised output** P(same group) x P(type | group) instead of one 5-way pair softmax",
         "- **group loss** normalised-cut consistency of the true groups under the predicted affinity (eigen-free form of",
         "  the spectral grouping loss of Jahangard et al.)",
         "- **explicit groups** perturb-and-MAP correlation clustering: every sample is a consistent partition (no A-B, B-C",
         "  without A-C), averaged over samples and ensemble members into the v1 pair format, so stage 2 is unchanged",
         "- **crops** 128 x 64 of a wider box (+-0.55 m, 2.3 m tall: a raised / outstretched arm stays inside), frame",
         "  features plus their 0.8 s temporal difference",
         "- **v3 (regularised)** v2 overfitted (train loss ~0, temperatures 2.2-3.0, queue F1 below v1): dropout 0.2, weight",
         "  decay 0.01, 3000 steps, training-only augmentation (mirror about the robot x axis with flipped crops, 5 cm position /",
         "  velocity noise, 10 % of past samples dropped). Stage 2 uses the no-grouping export (grouping hurt stage 2 in v2).", "",
         "Difference to Jahangard et al.: their labels are untyped JRDB groups and their decoder is spectral clustering with a",
         "regressed group count; ours are typed groups (talk / queue / group walk / other) whose type sets the cost, so the",
         "type is predicted per pair, pooled per group, and the partition comes from correlation clustering (no group count).",
         "", "Same protocol as v1: 5-member ensembles, temperatures fitted on validation, 3-fold out-of-fold predictions for",
         "the train episodes; v2 trains 4000 steps per member (v1: 3000). v1 rows below are re-scored with the v2 evaluator",
         "(its groups: post-hoc correlation clustering of v1's P(interacting)).", ""]
    for sp in ("test_a", "test_b", "test_c"):
        L += [f"## Stage 1, {sp.replace('_', ' ').title()}", "",
              "| Stage 1 | talk F1 | queue F1 | group walk F1 | interacting F1 | group F1 (IoU>=0.5) | exact group F1 | typed group F1 | halt F1 | halt F1 (visible) | wheelchair F1 | ECE |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, key in S1_ROWS:
            r = s1.get(key, {}).get("stage1", {}).get(sp)
            if not r:
                continue
            L.append(f"| {name} | {fmt(r['f1_talk'])} | {fmt(r['f1_queue'])} | {fmt(r['f1_group_walk'])} | {fmt(r.get('f1_interacting'))} | "
                     f"{fmt(r.get('group_f1_g50'))} | {fmt(r.get('group_f1_gex'))} | {fmt(r.get('group_typed_f1_g50'))} | {fmt(r['f1_halt'])} | "
                     f"{fmt(r.get('f1_halt_visible'))} | {fmt(r['f1_wheelchair'])} | {fmt(r['ece_interacting'])} |")
        L.append("")
    L += ["No-grouping rows have no group metrics (they have no partition).", "",
          "## Stage 1, Test D (counterfactual pairs): mean predicted probability ON vs OFF", "",
          "| Stage 1 | conversation | queue | stop gesture | wheelchair |", "|---|---|---|---|---|"]
    for name, key in S1_ROWS:
        d = s1.get(key, {}).get("test_d")
        if not d:
            continue
        cell = lambda k: next((f"{v['p_on']:.2f} / {v['p_off']:.2f}" for kk, v in d.items() if kk.startswith(k)), "–")  # noqa: E731
        L.append(f"| {name} | {cell('conversation')} | {cell('queue')} | {cell('stop_gesture')} | {cell('wheelchair')} |")
    L += ["", "The no-OOF ablation (2 s) is scored on val / test episodes only, like every row.", ""]
    # stage 2
    for label in ("membership", "native"):
        L += [f"## Stage 2 (TypeMix), {label} label, Test A (mean ± std over 3 seeds)", "",
              "| Model | MAE nonzero | gap closed | interaction IoU | FP area m² | NLL | cov95 nonzero | Spearman | AUSE | Test B MAE nz | Test C MAE nz | Test D cf err | conv. response | queue response |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        res = {}
        for name, m in S2_ROWS:
            R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, "runs", "main", f"{m}_{label}_s[0-2]", "metrics.json")))]
            if R:
                res[m] = R

        def a(R, sp, k):
            return np.array([r[sp].get(sp, r[sp]).get(k, np.nan) if isinstance(r[sp].get(sp, r[sp]), dict) else np.nan for r in R], float)

        lo = np.mean(a(res["pos_u"], "test_a", "mae_nonzero")) if "pos_u" in res else np.nan
        hi = np.mean(a(res["typemix_gt"], "test_a", "mae_nonzero")) if "typemix_gt" in res else np.nan
        for name, m in S2_ROWS:
            if m not in res:
                continue
            R = res[m]
            nz = a(R, "test_a", "mae_nonzero")
            gap = (lo - nz.mean()) / (lo - hi) * 100 if m not in ("pos_u", "typemix_gt") else np.nan
            d = lambda fam, k: float(np.mean([r["test_d"].get(fam, {}).get(k, np.nan) for r in R]))  # noqa: E731
            L.append(f"| {name} | {nz.mean():.1f} ± {nz.std():.1f} | {'–' if np.isnan(gap) else f'{gap:.0f} %'} | "
                     f"{np.mean(a(R, 'test_a', 'inter_iou')):.2f} | {np.mean(a(R, 'test_a', 'fp_area_m2')):.1f} | {np.nanmean(a(R, 'test_a', 'nll')):.2f} | "
                     f"{np.nanmean(a(R, 'test_a', 'cov95_nonzero')):.2f} | {np.nanmean(a(R, 'test_a', 'spearman_err_sigma')):.2f} | {np.nanmean(a(R, 'test_a', 'ause')):.2f} | "
                     f"{np.mean(a(R, 'test_b', 'mae_nonzero')):.1f} | {np.mean(a(R, 'test_c', 'mae_nonzero')):.1f} | {d('all', 'cf_err'):.1f} | "
                     f"{d('family:conversation', 'cf_response'):.2f} | {d('family:queue', 'cf_response'):.2f} |")
        L += ["", "gap closed = (MAE nz of pos_u - model) / (pos_u - TypeMix with true types). Test D: OFF maps aligned to the ON robot",
              "frame; response 1.0 = the predicted ON - OFF change equals the true one.", ""]
        L += ["Learned TypeMix parameters (mean over seeds):", "", "| Model | talk amp | queue amp | halt amp | wheelchair σ scale | λ (semantic variance) |", "|---|---|---|---|---|---|"]
        for name, m in S2_ROWS:
            if m in res and "typemix_parameters" in res[m][0]:
                pp = lambda k: np.mean([r["typemix_parameters"][k] for r in res[m]])  # noqa: E731
                L.append(f"| {name} | {pp('pair_talk_amp'):.2f} | {pp('pair_queue_amp'):.2f} | {pp('halt_amp'):.2f} | {pp('wheelchair_sigma_scale'):.2f} | {pp('lambda_semantic'):.2f} |")
        L.append("")
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "stage1_groups_results.md")
    open(out, "w").write("\n".join(L) + "\n")
    json.dump({k: v for k, v in s1.items()}, open(os.path.join(root, "stage1", "groups_report_stage1.json"), "w"), indent=1, default=float)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
