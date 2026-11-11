#!/usr/bin/env python3
"""Report for counterfactual-supervised stage 1 -> reports/counterfactual_stage1_results.md

    $PY cf_report.py --root data/costmap_bench --out data/socialcost/reports
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
from cascade_stage1_v2 import evaluate  # noqa: E402
from socialcost.bench import counterfactual as cfm  # noqa: E402
from socialcost.bench.data import split_ids  # noqa: E402
from socialcost.bench.group_net import EpisodeCache2  # noqa: E402

S1 = [("v1 vision (2 s, NRI)", "vis"), ("v2 vision (8 s, no grouping)", "vis2raw"), ("v3 vision (regularised)", "vis3raw"),
      ("tracks, small backbone, no CF", "cftracksnoneraw"), ("tracks, small backbone, CF", "cftracksraw"),
      ("vision, small backbone, no CF (baseline)", "cfnoneraw"), ("CF without invariance", "cfnoinvraw"), ("CF without ranking", "cfnorankraw"),
      ("CF without hidden-state posterior", "cfnohiddenraw"), ("**vision, small backbone, CF (full)**", "cffullraw"),
      ("cost-aware only (no CF), weight 0.1", "costonlyraw"), ("CF + cost-aware, weight 0.3", "cfcost03raw"),
      ("**CF + cost-aware, weight 0.1**", "cfcostraw")]
S2 = [("position only (pos_u)", "pos_u"), ("TypeMix + v1 vision", "typemix_vis"), ("TypeMix + v2 vision", "typemix_vis2raw"),
      ("TypeMix + v3 vision", "typemix_vis3raw"), ("TypeMix + small backbone, no CF", "typemix_cfnoneraw"),
      ("**TypeMix + CF stage 1**", "typemix_cffullraw"),
      ("**TypeMix + CF + cost-aware stage 1**", "typemix_cfcostraw"), ("*TypeMix + true types (bound)*", "typemix_gt")]
FAMS = ("conversation", "stop_gesture", "wheelchair", "queue")


def sink_from_preds(d: str, ids: list[str]) -> dict:
    sink = {}
    for e in ids:
        f = os.path.join(d, f"{e}.npz")
        if os.path.exists(f):
            z = np.load(f)
            ep, npr = z["edge_p"].astype(np.float64), z["node_p"].astype(np.float64)
            sink[e] = {t: {"edge_p": ep[t], "raw_edge_p": ep[t], "node_p": npr[t]} for t in range(ep.shape[0])}
    return sink


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root
    splits = {s: split_ids(s) for s in ("val", "test_a", "test_b", "test_c")}
    split_of = {e: s for s, v in splits.items() for e in v}

    class DS:
        cache = {}

    for e in split_of:
        c = EpisodeCache2(e, False)
        if c.P:
            DS.cache[e] = c
    held = cfm.train_pairs(set(split_of))
    S = {}
    for _, key in S1:
        f = os.path.join(root, "stage1", key, "metrics.json")
        pdir = os.path.join(root, "stage1", key, "preds")
        if key == "vis" and os.path.isdir(pdir):
            sink = sink_from_preds(pdir, list(DS.cache))
            S[key] = {"stage1": evaluate(sink, DS, split_of, raw=True), "cf_test_d": cfm.cf_test_d(sink, DS.cache, held, raw=True)}
        elif os.path.exists(f):
            m = json.load(open(f))
            if "cf_test_d" not in m and os.path.isdir(pdir):
                m["cf_test_d"] = cfm.cf_test_d(sink_from_preds(pdir, list(DS.cache)), DS.cache, held, raw=True)
            S[key] = m
    L = ["# Counterfactual supervision of stage 1", "",
         "Stage 1 predicts who interacts with whom, raised stop hands and mobility aids; TypeMix (stage 2) renders the costmap",
         "from those probabilities. The dataset records every counterfactual unit twice (ON / OFF) with the same seed, roster",
         "and tracks (median position difference 0 m on the 126 train units); only the social cue or the hidden social state",
         "differs. Trained as independent samples, a model can fit both labels only by memorising nuisance details of an",
         "episode - the overfitting seen in stage 1 v2 / v3. Here the pair is the supervision (learning from counterfactual",
         "examples: Teney et al. ECCV 2020, Kaushik et al. ICLR 2020), with observability defined by the sensor:", "",
         "| entry of a matched (ON t, OFF t) pair | loss |", "|---|---|",
         "| labels differ, cue observable (crop model, cue holder visible > 0.3) | ranking: sign(y_on - y_off) (z_on - z_off) >= 2 |",
         "| labels differ, cue not observable (hidden queue state, tracks-only model, cue holder not visible) | targets -> pair mean 0.5, plus invariance |",
         "| labels equal | invariance: symmetric KL of the two predictions, (p - q)(a - b) for Bernoulli logits a, b |", "",
         "Entries: pair same-group logit, person halt / mobility-aid logits; type softmax invariance on pairs grouped in both.",
         "Only train pairs are used (val / test pairs are held out). Code: socialcost/bench/counterfactual.py,",
         "scripts/cascade_stage1_v2.py --cf. Backbone for the CF rows and their baseline: GroupNet with 2 s history, width 64,",
         "2 graph-transformer layers, 128 x 64 crops, dropout 0.1, no grouping; 5-member ensembles, temperature scaling,",
         "3-fold out-of-fold export (same protocol as v1-v3).", "",
         "Cost-aware rows add the expected squared cost error through the frozen TypeMix renderer (parameters learned on true",
         "types), E[(C - C*)^2] = (E[C] - C*)^2 + V, integrated over a 10 m grid (socialcost/bench/cost_aware.py; task-based",
         "end-to-end learning, Donti et al. 2017; planner-centric perception metrics, Philion et al. 2020).", "",
         "## Counterfactual response on held-out pairs (val + Test A-C units)", "",
         "shift = mean over label-differing entries of sign(y_on - y_off) (p_on - p_off): 1 = follows the cue, 0 = blind.",
         "spurious = mean |p_on - p_off| over label-equal entries: 0 = unaffected by changes that should not matter.",
         "For the hidden-state family (queue) the right shift is 0 and the right prediction is the posterior (~0.5).", "",
         "| Stage 1 | conv. shift | stop shift | wheelchair shift | queue shift (→0) | queue p (→0.5) | spurious (mean) |",
         "|---|---|---|---|---|---|---|"]
    for name, key in S1:
        d = S.get(key, {}).get("cf_test_d")
        if not d:
            continue
        g = lambda f, k: d.get(f, {}).get(k)  # noqa: E731
        sp = [g(f, "spurious") for f in FAMS if g(f, "spurious") is not None]
        L.append(f"| {name} | {f2(g('conversation', 'shift'))} | {f2(g('stop_gesture', 'shift'))} | {f2(g('wheelchair', 'shift'))} | "
                 f"{f2(g('queue', 'shift'))} | {f2(g('queue', 'p_on_differing'))} | {f2(float(np.mean(sp)) if sp else None)} |")
    L.append("")
    for sp in ("test_a", "test_b", "test_c"):
        L += [f"## Stage 1, {sp.replace('_', ' ').title()}", "",
              "| Stage 1 | talk F1 | queue F1 | group walk F1 | interacting F1 | halt F1 | halt F1 (visible) | wheelchair F1 | ECE |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, key in S1:
            r = S.get(key, {}).get("stage1", {}).get(sp)
            if r:
                L.append(f"| {name} | {f2(r['f1_talk'])} | {f2(r['f1_queue'])} | {f2(r['f1_group_walk'])} | {f2(r.get('f1_interacting'))} | "
                         f"{f2(r['f1_halt'])} | {f2(r.get('f1_halt_visible'))} | {f2(r['f1_wheelchair'])} | {f2(r['ece_interacting'])} |")
        L.append("")
    for label in ("membership", "native"):
        L += [f"## Stage 2 (TypeMix), {label} label (mean ± std over 3 seeds)", "",
              "| Model | Test A MAE nz | gap closed | Test B MAE nz | Test C MAE nz | inter. IoU | NLL | cov95 nz | Test D cf err | conv. | stop | wheelchair | queue |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        res = {m: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, "runs", "main", f"{m}_{label}_s[0-2]", "metrics.json")))] for _, m in S2}

        def a(R, sp, k):
            return np.array([r[sp][sp].get(k, np.nan) for r in R], float)

        lo = a(res["pos_u"], "test_a", "mae_nonzero").mean() if res.get("pos_u") else np.nan
        hi = a(res["typemix_gt"], "test_a", "mae_nonzero").mean() if res.get("typemix_gt") else np.nan
        for name, m in S2:
            R = res.get(m)
            if not R:
                continue
            nz = a(R, "test_a", "mae_nonzero")
            gap = "–" if m in ("pos_u", "typemix_gt") else f"{(lo - nz.mean()) / (lo - hi) * 100:.0f} %"
            d = lambda fam: float(np.mean([r["test_d"].get(f"family:{fam}", {}).get("cf_response", np.nan) for r in R]))  # noqa: E731
            L.append(f"| {name} | {nz.mean():.1f} ± {nz.std():.1f} | {gap} | {a(R, 'test_b', 'mae_nonzero').mean():.1f} | {a(R, 'test_c', 'mae_nonzero').mean():.1f} | "
                     f"{a(R, 'test_a', 'inter_iou').mean():.2f} | {np.nanmean(a(R, 'test_a', 'nll')):.2f} | {np.nanmean(a(R, 'test_a', 'cov95_nonzero')):.2f} | "
                     f"{np.mean([r['test_d']['all']['cf_err'] for r in R]):.1f} | {d('conversation'):.2f} | {d('stop_gesture'):.2f} | {d('wheelchair'):.2f} | {d('queue'):.2f} |")
        L += ["", "Test D columns: aligned counterfactual costmap error and the response ratio per family (1.0 = the predicted ON - OFF",
              "cost change equals the true one).", ""]
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "counterfactual_stage1_results.md")
    open(out, "w").write("\n".join(L) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
