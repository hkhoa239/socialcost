#!/usr/bin/env python3
"""Report for the interaction cascade with uncertainty -> reports/cascade_uncertainty_results.md (+ figures).

    $PY cascade_report.py --root data/costmap_bench --out data/socialcost/reports
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import LABELS, ROOT, test_d_pairs  # noqa: E402

SPLITS = ("test_a", "test_b", "test_c")
PROB = ["pos_u", "cascade_tracks", "cascade_vis", "pos_int_u", "typemix_tracks", "typemix_vis", "typemix_vis_nosem", "typemix_gt"]
NAME = {"pos_u": "Position (no interaction, lower bound)", "cascade_tracks": "Cascade U-Net: stage 1 from tracks",
        "cascade_vis": "Cascade U-Net: stage 1 from tracks + RGB-D crops", "pos_int_u": "U-Net with ground-truth interactions (oracle)",
        "typemix_tracks": "TypeMix: types from tracks", "typemix_vis": "TypeMix: types from tracks + RGB-D crops",
        "typemix_vis_nosem": "TypeMix (vision) without semantic variance (ablation)", "typemix_gt": "TypeMix with ground-truth types (oracle)"}
DET = ["rgb", "depth", "rgbd", "pos", "pos_int", "sngnn_sensor", "sngnn_priv"]
SURF, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2})


def fmt(v, d=2):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def runs_of(root, model, label):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "runs", "main", f"{model}_{label}_s[0-9]"))):
        if os.path.exists(os.path.join(d, "metrics.json")):
            out.append((d, json.load(open(os.path.join(d, "metrics.json")))))
    return out


def stage1_section(root) -> list[str]:
    L = ["## Stage 1: interaction predictor (ensemble of 5, temperature-scaled)", ""]
    for var, name in (("tracks", "tracks only"), ("vis", "tracks + RGB-D person crops")):
        f = os.path.join(root, "stage1", var, "metrics.json")
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        L += [f"**{name}** (temperatures: edges {m['info']['t_edge']}, nodes {m['info']['t_node']})", "",
              "| Split | F1 talk | F1 queue | F1 group walk | macro-F1 | pair accuracy | F1 halt | F1 wheelchair | ECE p(interacting) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for sp in ("val", *SPLITS):
            r = m["stage1"].get(sp)
            if r:
                L.append(f"| {sp} | {r['f1_talk']:.3f} | {r['f1_queue']:.3f} | {r['f1_group_walk']:.3f} | {r['macro_f1']:.3f} | {r['pair_accuracy']:.3f} | "
                         f"{r['f1_halt']:.3f} | {r['f1_wheelchair']:.3f} | {r['ece_interacting']:.3f} |")
        L += ["", "Test D: mean predicted probability of the manipulated state for the controlled people, ON vs OFF "
                  "(talk between participants, queue between members, halt for the gesturer, mobility aid for the wheelchair user)", "",
              "| Pair family (observability) | pairs | p ON | p OFF | ON − OFF |", "|---|---:|---:|---:|---:|"]
        for k, v in sorted(m["test_d"].items()):
            L.append(f"| {k} | {v['pairs']} | {v['p_on']:.3f} | {v['p_off']:.3f} | {v['p_on'] - v['p_off']:+.3f} |")
        L.append("")
    return L


def stage2_section(root) -> list[str]:
    L = ["## Stage 2: probabilistic costmap (per seed, mean ± std over 3 seeds)", ""]
    for label in ("native", "membership"):
        L += [f"### Label `{LABELS[label]}`", ""]
        for sp in SPLITS:
            L += [f"**{sp}**", "", "| Model | MAE | MAE nonzero | IoU ≥ 100 | interaction IoU | FP area (m²) | NLL | 95 % coverage | coverage (nonzero) | "
                  "Spearman ρ(σ, err) | AUSE |", "|---|" + "---:|" * 10]
            for m in PROB:
                rs = runs_of(root, m, label)
                if not rs:
                    continue
                g = lambda k, d=2: fmt([r[sp][sp].get(k) for _, r in rs], d)  # noqa: E731
                L.append(f"| {NAME[m]} | {g('mae')} | {g('mae_nonzero', 1)} | {g('hc_iou', 3)} | {g('inter_iou', 3)} | {g('fp_area_m2')} | {g('nll')} | "
                         f"{g('cov95', 3)} | {g('cov95_nonzero', 3)} | {g('spearman_err_sigma', 3)} | {g('ause', 3)} |")
            L.append("")
        # gap closed
        L += ["**Share of the oracle gap recovered** (MAE on nonzero cells): (lower bound − model) / (lower bound − U-Net oracle)", "",
              "| Split | " + " | ".join(NAME[m] for m in ("cascade_tracks", "cascade_vis", "typemix_tracks", "typemix_vis")) + " |", "|---|---:|---:|---:|---:|"]
        for sp in SPLITS:
            lo = np.mean([r[sp][sp]["mae_nonzero"] for _, r in runs_of(root, "pos_u", label)] or [np.nan])
            hi = np.mean([r[sp][sp]["mae_nonzero"] for _, r in runs_of(root, "pos_int_u", label)] or [np.nan])
            cells = []
            for m in ("cascade_tracks", "cascade_vis", "typemix_tracks", "typemix_vis"):
                v = np.mean([r[sp][sp]["mae_nonzero"] for _, r in runs_of(root, m, label)] or [np.nan])
                cells.append(f"{(lo - v) / (lo - hi):.2f}" if np.isfinite(v) and lo != hi else "–")
            L.append(f"| {sp} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def typemix_params_section(root) -> list[str]:
    """Learned type parameters (mean +- std over the 3 seeds) next to the hand-tuned SocialTypeAwareCostmap values and the
    Arena social layer the labels come from."""
    keys = None
    rows = {}
    for m in ("typemix_gt", "typemix_vis", "typemix_tracks"):
        for label in ("native", "membership"):
            ps = [r.get("typemix_parameters") for _, r in runs_of(root, m, label) if r.get("typemix_parameters")]
            if ps:
                keys = keys or list(ps[0])
                rows[(m, label)] = ps
    if not rows:
        return []
    ref = {"personal_sigma0_m": "0.8 (adult covariance), 1.3 (child) / σ 0.9, radius 1.0", "personal_k_velocity": "5.0 (velocity factor) / none (isotropic)",
           "wheelchair_sigma_scale": "– / 1.0 (labels: no wheelchair term in native; 1.5 in the separate hypothesis channel)",
           "pair_group_walk_amp": "1.5 x group size (group layer) / weight 0.0 (GROUP_WALK)", "pair_talk_amp": "– / weight 0.9 (TALK_TO), 1.2 (GROUP_CONVERSATION)",
           "pair_queue_amp": "– / weight 1.0 (QUEUE_USE, membership label only)", "halt_amp": "– / halt-gesture wedge (extended label only)"}
    L = ["## TypeMix: learned social-type parameters", "",
         "Each value is learned end to end (softplus-positive scalars shared across people and scenes). Reference column: "
         "SocialTypeAwareCostmap hand-tuned value / Arena social layer that generated the labels.", "",
         "| Parameter | " + " | ".join(f"{NAME[m].split(':')[0].replace('TypeMix with ground-truth types (oracle)', 'TypeMix GT')} {m.split('_')[1]} / {label}"
                                        for (m, label) in rows) + " | reference |", "|---|" + "---:|" * len(rows) + "---|"]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(fmt([p[k] for p in ps], 2) for ps in rows.values()) + f" | {ref.get(k, '')} |")
    return L + [""]


def ensemble_section(root) -> list[str]:
    L = ["## Seed ensembles (3 seeds): aleatoric + epistemic", "",
         "σ²_total = mean_s σ_s² + Var_s μ_s. Rows compare the full predictive σ with its aleatoric and epistemic parts alone.", ""]
    for label in ("native", "membership"):
        L += [f"### Label `{LABELS[label]}`", "", "| Model | split | σ used | MAE nonzero | NLL | 95 % coverage | coverage (nonzero) | Spearman ρ | AUSE |",
              "|---|---|---|" + "---:|" * 6]
        for m in PROB:
            f = os.path.join(root, "runs", "ensemble", f"{m}_{label}.json")
            if not os.path.exists(f):
                continue
            e = json.load(open(f))
            for sp in SPLITS:
                for kind in ("ensemble", "aleatoric_only", "epistemic_only"):
                    r = e[kind].get(sp)
                    if r:
                        L.append(f"| {NAME[m]} | {sp} | {kind} | {r['mae_nonzero']:.1f} | {r['nll']:.2f} | {r['cov95']:.3f} | {r['cov95_nonzero']:.3f} | "
                                 f"{r['spearman_err_sigma']:.3f} | {r['ause']:.3f} |")
        L.append("")
    return L


def testd_section(root) -> list[str]:
    L = ["## Test D, robot-frame aligned (all models)", "",
         "The OFF maps are moved into the ON robot frame before ON − OFF (the robot pose differs between conditions, median 0.17 m). "
         "cf error: MAE of the predicted change on cells where the true change is nonzero (lower is better); response: |predicted change| / |true change|.", ""]
    for label in ("native", "membership"):
        L += [f"### Label `{LABELS[label]}`", "", "| Model | seeds | cf error | response | response: stop gesture | conversation | wheelchair | queue (hidden state) | "
              "cf error before alignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in DET + PROB:
            dirs = sorted(glob.glob(os.path.join(root, "runs", "main", f"{m}_{label}_s[0-9]")))
            al, un = [], []
            for d in dirs:
                if os.path.exists(os.path.join(d, "test_d_aligned.json")):
                    j = json.load(open(os.path.join(d, "test_d_aligned.json")))
                    al.append(j["aligned"])
                    un.append(j["unaligned"])
                elif os.path.exists(os.path.join(d, "metrics.json")) and "test_d_unaligned" in json.load(open(os.path.join(d, "metrics.json"))):
                    j = json.load(open(os.path.join(d, "metrics.json")))
                    al.append(j["test_d"])
                    un.append(j["test_d_unaligned"])
            if not al:
                continue
            g = lambda key, f, src=al: fmt([x.get(key, {}).get(f) for x in src], 2)  # noqa: E731
            L.append(f"| {NAME.get(m, m)} | {len(al)} | {g('all', 'cf_err')} | {g('all', 'cf_response')} | {g('obs:visual', 'cf_response')} | "
                     f"{g('obs:visual_temporal', 'cf_response')} | {g('obs:visual_appearance', 'cf_response')} | {g('obs:hidden_state', 'cf_response')} | "
                     f"{g('all', 'cf_err', un)} |")
        L.append("")
    return L


def figures(root, out) -> list[str]:
    figdir = os.path.join(out, "cascade_figures")
    os.makedirs(figdir, exist_ok=True)
    L = ["## Figures", ""]
    pairs = [p for p in test_d_pairs() if p["family"] in ("conversation", "stop_gesture", "queue")]
    rows = []
    for m in PROB:
        d = os.path.join(root, "runs", "main", f"{m}_membership_s0")
        if os.path.exists(os.path.join(d, "sigma.npz")):
            rows.append((m, np.load(os.path.join(d, "predictions.npz")), np.load(os.path.join(d, "sigma.npz"))))
    if not rows:
        return L
    for pr in pairs[:3]:
        k = 40
        fig, axs = plt.subplots(len(rows), 4, figsize=(9, 2.3 * len(rows)))
        tgt = np.load(os.path.join(ROOT, "episodes", pr["on"], LABELS["membership"], f"{k:06d}.npy")).astype(float)
        for r, (m, P, S) in enumerate(rows):
            key = f"{pr['on']}|{k}"
            if key not in P.files:
                continue
            mu, sg = P[key].astype(float), S[key].astype(float)
            for c, (img, title, cmap, vmax) in enumerate(((tgt, "ground truth", "Blues", 252), (mu, "μ", "Blues", 252), (sg, "σ", "Oranges", 80),
                                                          (np.abs(mu - tgt), "|error|", "Reds", 120))):
                ax = axs[r, c] if len(rows) > 1 else axs[c]
                ax.imshow(np.flipud(np.fliplr(img.T)), cmap=cmap, vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(NAME[m].split(":")[0].split("(")[0][:22], fontsize=7)
        fig.suptitle(f"pair {pr['pair_id']} ({pr['family']}) ON, sample {k}, membership label, seed 0", fontsize=9)
        fig.tight_layout()
        f = os.path.join(figdir, f"mu_sigma_{pr['family']}_{pr['pair_id']}.png")
        fig.savefig(f, dpi=90)
        plt.close(fig)
        L += [f"![{pr['family']}]({os.path.relpath(f, out)})", ""]
    # stage-1 Test D probabilities
    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.2
    fams = ["conversation|visual_temporal", "stop_gesture|visual", "wheelchair|visual_appearance", "queue|hidden_state"]
    for vi, (var, col) in enumerate((("tracks", "#2a78d6"), ("vis", "#eb6834"))):
        f = os.path.join(root, "stage1", var, "metrics.json")
        if not os.path.exists(f):
            continue
        td = json.load(open(f))["test_d"]
        x = np.arange(len(fams))
        ax.bar(x + (2 * vi - 1) * width - width / 2, [td.get(k, {}).get("p_on", 0) for k in fams], width, color=col, label=f"{var} ON")
        ax.bar(x + (2 * vi - 1) * width + width / 2, [td.get(k, {}).get("p_off", 0) for k in fams], width, color=col, alpha=0.4, label=f"{var} OFF")
    ax.set_xticks(np.arange(len(fams)), [k.split("|")[0] for k in fams])
    ax.set_ylabel("mean predicted probability")
    ax.set_title("Stage 1 on Test D: manipulated social state, ON vs OFF")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    f = os.path.join(figdir, "stage1_test_d.png")
    fig.savefig(f, dpi=90)
    plt.close(fig)
    L += [f"![stage 1 Test D]({os.path.relpath(f, out)})", ""]
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    L = ["# Interaction cascade with uncertainty: results", "", "Design, losses and literature: [cascade_uncertainty_design.md](cascade_uncertainty_design.md). "
         "Same splits, budget (48k samples), checkpoint selection and seeds as [costmap_prediction_benchmark.md](costmap_prediction_benchmark.md). "
         "Positions come from simulator tracks in every model here.", ""]
    L += stage1_section(root) + stage2_section(root) + typemix_params_section(root) + ensemble_section(root) + testd_section(root)
    ex = sorted(glob.glob(os.path.join(root, "runs", "extra_seed", "*", "metrics.json")))
    if ex:
        L += ["## Extra seed for the collapsed image runs", "", "| Run | Test A MAE | MAE nonzero | FP area (m²) |", "|---|---:|---:|---:|"]
        for f in ex:
            r = json.load(open(f))["test_a"]["test_a"]
            L.append(f"| {os.path.basename(os.path.dirname(f))} | {r['mae']:.2f} | {r['mae_nonzero']:.1f} | {r['fp_area_m2']:.2f} |")
        L.append("")
    L += figures(root, args.out)
    L += ["## Reproduce", "", "```bash", "cd /home/robotics/arena_ws/data/costmap_bench && source env.sh",
          "$PY -m socialcost.bench.cascade_data 12                                   # stage-1 cache (tracks, labels, crops)",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/cascade_pipeline.py --root $PWD   # everything below, resumable",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/cascade_stage1.py --variant vis --out stage1/vis --members 5 --folds 3 --steps 3000",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/bench_train.py --model cascade_vis --label membership --seed 0 --lr 0.001 --samples 48000 --val-every 6000 --out runs/main/cascade_vis_membership_s0",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/bench_post.py ensemble --root $PWD",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/cascade_report.py --root $PWD --out /home/robotics/arena_ws/data/socialcost/reports", "```"]
    open(os.path.join(args.out, "cascade_uncertainty_results.md"), "w").write("\n".join(L) + "\n")
    print("written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
