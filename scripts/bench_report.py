#!/usr/bin/env python3
"""Costmap prediction benchmark report: tables (mean +- std over seeds), figures, failure cases, commands.

    $PY bench_report.py --root data/costmap_bench --out data/socialcost/reports

Writes reports/costmap_prediction_benchmark.md and reports/costmap_benchmark_figures/*.png.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import ROOT, Episode, test_d_pairs  # noqa: E402

MODELS = ["rgb", "depth", "rgbd", "pos", "pos_int", "sngnn_sensor", "sngnn_priv"]
NAMES = {"rgb": "SOCIALCOST RGB", "depth": "SOCIALCOST Depth", "rgbd": "SOCIALCOST RGB-D", "pos": "SOCIALCOST Position",
         "pos_int": "SOCIALCOST Position+Interaction", "sngnn_sensor": "SNGNN2D-v2 (tracks, no interaction graph)",
         "sngnn_priv": "SNGNN2D-v2 (privileged interaction graph)"}
INPUTS = {"rgb": "sensor (RGB + calibration)", "depth": "sensor (depth + calibration)", "rgbd": "sensor (RGB-D + calibration)",
          "pos": "privileged tracks (positions, headings)", "pos_int": "privileged tracks + interaction / gesture / mobility metadata (oracle)",
          "sngnn_sensor": "privileged tracks (pos, heading, velocity, 3 frames) + walls, objects, goal, command",
          "sngnn_priv": "same + privileged interaction links (oracle)"}
SPLITS = ["test_a", "test_b", "test_c"]
SPLIT_NAME = {"test_a": "Test A (in-distribution)", "test_b": "Test B (unseen world)", "test_c": "Test C (unseen bodies)"}
METRICS = [("mae", "MAE", 2, False), ("rmse", "RMSE", 2, False), ("mae_nonzero", "MAE nonzero", 1, False), ("hc_iou", "IoU >= 100", 3, True),
           ("inter_iou", "interaction IoU", 3, True), ("fp_area_m2", "FP area (m^2)", 2, False)]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"rgb": "#2a78d6", "depth": "#1baf7a", "rgbd": "#eb6834", "pos": "#eda100", "pos_int": "#e87ba4", "sngnn_sensor": "#6250d6", "sngnn_priv": "#e34948"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_runs(root: str) -> dict:
    runs = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "runs", "main", "*"))):
        f = os.path.join(d, "metrics.json")
        if not os.path.exists(f):
            continue
        name = os.path.basename(d)
        model, label, seed = name.rsplit("_", 2)[0], name.rsplit("_", 2)[1], name.rsplit("_", 2)[2]
        runs[(model, label)].append({"dir": d, "seed": seed, "m": json.load(open(f)), "cfg": json.load(open(os.path.join(d, "config.json")))})
    return runs


def ms(vals: list) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"


def fmt(vals, digits):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if len(vals) > 1 else f"{m:.{digits}f}"


def tables(runs: dict) -> list[str]:
    L = []
    for label, lname in (("native", "`social_cost` (arena-native/1.1)"), ("membership", "`social_cost_membership` (extended-membership/1.1)")):
        L += [f"### Label {lname}", ""]
        for sp in SPLITS:
            L += [f"**{SPLIT_NAME[sp]}**", "", "| Model | Inputs | Seeds | " + " | ".join(m[1] for m in METRICS) + " |",
                  "|---|---|---:|" + "---:|" * len(METRICS)]
            for model in MODELS:
                rs = runs.get((model, label), [])
                if not rs:
                    continue
                cells = [fmt([r["m"][sp][sp].get(k) for r in rs if sp in r["m"]], d) for k, _, d, _ in METRICS]
                L.append(f"| {NAMES[model]} | {INPUTS[model].split(' (')[0]} | {len(rs)} | " + " | ".join(cells) + " |")
            L.append("")
        L += ["**Test D (counterfactual pairs inside Test A/B/C)**: paired-response error on cells where the ON-OFF target differs "
              "(lower is better) and response ratio (1 = predicted change as large as the true change)", "",
              "| Model | Seeds | cf error (all) | response (all) | response (visually observable) | response (hidden-state queues) |",
              "|---|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            rs = runs.get((model, label), [])
            if not rs:
                continue
            g = lambda key, f: fmt([r["m"]["test_d"].get(key, {}).get(f) for r in rs], 2)  # noqa: E731
            L.append(f"| {NAMES[model]} | {len(rs)} | {g('all', 'cf_err')} | {g('all', 'cf_response')} | "
                     f"{g('obs:visual', 'cf_response')} / {g('obs:visual_temporal', 'cf_response')} / {g('obs:visual_appearance', 'cf_response')} | "
                     f"{g('obs:hidden_state', 'cf_response')} |")
        L += ["", "Visually observable response is given per design class: stop gesture / conversation / wheelchair.", ""]
    return L


def cost_table(runs: dict) -> list[str]:
    L = ["| Model | Parameters | Latency, batch 1 (ms, GPU forward) | Extra input cost |", "|---|---:|---:|---|"]
    for model in MODELS:
        rs = runs.get((model, "native"), []) or runs.get((model, "membership"), [])
        if not rs:
            continue
        lat = [r["m"].get("latency_ms_batch1") for r in rs]
        extra = f"graph build {np.mean([r['m'].get('graph_build_ms_cpu', 0) for r in rs]):.0f} ms CPU" if "sngnn" in model else (
            "depth lifting indices (CPU)" if model in ("rgb", "depth", "rgbd") else "rasterisation (CPU)")
        L.append(f"| {NAMES[model]} | {rs[0]['m']['params'] / 1e6:.2f} M | {fmt(lat, 1)} | {extra} |")
    return L


def curves(runs: dict, out: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axs, ("native", "membership")):
        for model in MODELS:
            for i, r in enumerate(runs.get((model, label), [])):
                L = [json.loads(line) for line in open(os.path.join(r["dir"], "log.jsonl"))]
                ax.plot([x["seen"] / 1000 for x in L], [x["val"]["mae_nonzero"] for x in L], color=COLORS[model], lw=1.5, alpha=0.9,
                        label=NAMES[model] if i == 0 else None)
        ax.set_title(f"validation MAE on nonzero cells, label {label}")
        ax.set_xlabel("training samples seen (k)")
        ax.set_ylabel("MAE (cost units)")
    axs[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=90)
    plt.close(fig)


def load_pred(run_dir: str) -> dict:
    z = np.load(os.path.join(run_dir, "predictions.npz"))
    return {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files}


def lab(a):
    return np.flipud(np.fliplr(a.T))


def sample_figure(runs: dict, label: str, eid: str, k: int, out: str, title: str) -> None:
    ep = Episode(eid)
    rgb = plt.imread(os.path.join(ep.dir, "rgb", f"{k:06d}.png"))
    tgt = np.load(os.path.join(ep.dir, {"native": "social_cost", "membership": "social_cost_membership"}[label], f"{k:06d}.npy"))
    cols = [m for m in MODELS if runs.get((m, label))]
    fig, axs = plt.subplots(1, 2 + len(cols), figsize=(2.3 * (2 + len(cols)), 2.6))
    axs[0].imshow(rgb); axs[0].set_title("RGB", fontsize=8)
    axs[1].imshow(lab(tgt), cmap="Blues", vmin=0, vmax=252); axs[1].set_title("ground truth", fontsize=8)
    for ax, m in zip(axs[2:], cols):
        p = load_pred(runs[(m, label)][0]["dir"]).get((eid, k))
        if p is not None:
            ax.imshow(lab(p), cmap="Blues", vmin=0, vmax=252)
        ax.set_title(NAMES[m].replace("SOCIALCOST ", "").replace("SNGNN2D-v2 ", "SNGNN "), fontsize=7)
    for ax in axs:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def cf_figure(runs: dict, label: str, pair: dict, k: int, out: str) -> None:
    cols = [m for m in MODELS if runs.get((m, label))]
    fig, axs = plt.subplots(3, 1 + len(cols), figsize=(2.2 * (1 + len(cols)), 6.6))
    lbl = {"native": "social_cost", "membership": "social_cost_membership"}[label]
    T = {c: np.load(os.path.join(ROOT, "episodes", pair[c], lbl, f"{k:06d}.npy")).astype(float) for c in ("on", "off")}
    rows = [("on", T["on"]), ("off", T["off"]), ("ON - OFF", T["on"] - T["off"])]
    for r, (name, img) in enumerate(rows):
        axs[r, 0].imshow(lab(img), cmap="RdBu_r" if r == 2 else "Blues", vmin=-252 if r == 2 else 0, vmax=252)
        axs[r, 0].set_ylabel(name)
        axs[r, 0].set_title("ground truth" if r == 0 else "", fontsize=8)
    for j, m in enumerate(cols, 1):
        P = load_pred(runs[(m, label)][0]["dir"])
        po, pf = P.get((pair["on"], k)), P.get((pair["off"], k))
        if po is None or pf is None:
            continue
        for r, img in enumerate((po.astype(float), pf.astype(float), po.astype(float) - pf.astype(float))):
            axs[r, j].imshow(lab(img), cmap="RdBu_r" if r == 2 else "Blues", vmin=-252 if r == 2 else 0, vmax=252)
        axs[0, j].set_title(NAMES[m].replace("SOCIALCOST ", "").replace("SNGNN2D-v2 ", "SNGNN "), fontsize=7)
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Counterfactual pair {pair['pair_id']} ({pair['family']}, {pair['observability']}), sample {k}, label {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def failures(runs: dict, label: str, model: str, n: int = 6) -> list[tuple]:
    """Worst test samples (MAE on nonzero cells) for a model, seed 0."""
    rs = runs.get((model, label))
    if not rs:
        return []
    P = load_pred(rs[0]["dir"])
    lbl = {"native": "social_cost", "membership": "social_cost_membership"}[label]
    errs = []
    for (e, k), p in list(P.items())[::3]:
        t = np.load(os.path.join(ROOT, "episodes", e, lbl, f"{k:06d}.npy")).astype(float)
        m = t > 0
        if m.sum() > 200:
            errs.append((float(np.abs(p.astype(float) - t)[m].mean()), e, k))
    return sorted(errs, reverse=True)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = load_runs(args.root)
    figdir = os.path.join(args.out, "costmap_benchmark_figures")
    os.makedirs(figdir, exist_ok=True)
    curves(runs, figdir)
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "manifest_v1.json")))["episodes"]}
    pairs = test_d_pairs()
    figs = []
    for fam in ("conversation", "stop_gesture", "queue", "wheelchair"):
        pr = next((p for p in pairs if p["family"] == fam and p["observability"] != "not_visually_observable"), None)
        if pr:
            k = 40
            f = os.path.join(figdir, f"cf_{fam}_{pr['pair_id']}.png")
            cf_figure(runs, "membership", pr, k, f)
            figs.append((f"Counterfactual {fam}", f))
    for sp, eid in (("test_a", next(e for e, v in man.items() if v["split"] == "test_a" and v["family"] == "conversation" and v["condition"] == "on")),
                    ("test_b", next(e for e, v in man.items() if v["split"] == "test_b" and v["family"] == "queue" and v["condition"] == "on")),
                    ("test_c", next(e for e, v in man.items() if v["split"] == "test_c" and v["family"] == "partial_observability"))):
        f = os.path.join(figdir, f"pred_{sp}_{eid}.png")
        sample_figure(runs, "membership", eid, 40, f, f"{SPLIT_NAME[sp]}: episode {eid} ({man[eid]['family']}/{man[eid]['subtype']}), sample 40")
        figs.append((f"Prediction vs ground truth, {SPLIT_NAME[sp]}", f))
    fail = {m: failures(runs, "membership", m) for m in ("rgbd", "pos_int", "sngnn_priv")}
    for m, rows in fail.items():
        for i, (err, e, k) in enumerate(rows[:2]):
            f = os.path.join(figdir, f"failure_{m}_{i}.png")
            sample_figure(runs, "membership", e, k, f, f"Failure case for {NAMES[m]}: episode {e} ({man[e]['family']}/{man[e]['subtype']}, {man[e]['world_key']}), "
                                                        f"sample {k}, MAE nonzero {err:.0f}")
            figs.append((f"Failure, {NAMES[m]}", f))
    sel = json.load(open(os.path.join(args.root, "selected_lr.json"))) if os.path.exists(os.path.join(args.root, "selected_lr.json")) else {}
    seeds_done = {k: len(v) for k, v in runs.items()}
    L = ["# Costmap prediction benchmark on SOCIALCOST dataset_v1", "",
         f"Runs found: {sum(seeds_done.values())} of {len(MODELS) * 2 * 3} (7 models x 2 labels x 3 seeds). Tables report mean ± std over the seeds available.", "",
         "## Setup", "",
         "| Item | Value |", "|---|---|",
         "| Data | dataset_v1, 600 episodes x 125 samples; train 360 / val 60 / test_a 60 / test_b 60 (unseen world D) / test_c 60 (unseen bodies); Test D = 63 counterfactual pairs inside the test splits |",
         "| Sampling | train every 2nd sample (22.5k), validation every 5th, test every 2nd |",
         "| Targets | 200 x 200, 0.05 m, robot-centred; `social_cost` and `social_cost_membership` trained and evaluated separately |",
         "| Budget | 48,000 training samples per run for every model; checkpoint selected on validation (MAE on nonzero cells + 50 x (1 - interaction IoU)); mixed precision for all |",
         f"| Learning rates (selected on validation, seed 0, native label, 16k samples) | {', '.join(f'{k} {v:g}' for k, v in sel.items()) or 'pending'} |",
         "| Seeds | 0, 1, 2 |", "",
         "### Models and the information they use", "", "| Model | Inputs | Information class |", "|---|---|---|"]
    for m in MODELS:
        cls = "sensor-observable" if m in ("rgb", "depth", "rgbd") else ("privileged + oracle interaction" if m in ("pos_int", "sngnn_priv") else "privileged tracks")
        L.append(f"| {NAMES[m]} | {INPUTS[m]} | {cls} |")
    L += ["", "Privileged-state models (position, position + interaction, both SNGNN2D variants) read simulator ground-truth tracks; they bound what "
          "a perfect tracker (and, for the interaction variants, a perfect social-state oracle) would allow. They are not comparable to the "
          "RGB / depth / RGB-D models as equal-information competitors.", "",
          "### SNGNN2D-v2 adaptation", "",
          "The official implementation in `external/SNGNN2D-v2` is used unmodified (graph alternative 8, GAT with hidden sizes [95, 71, 62, 57, 45, 35] and "
          "heads [34, 28, 22, 15, 13, 10], 9-block ResNet generator, tanh output, MSE loss on [-1, 1] labels, Adam with weight decay 1e-11). "
          "The adapter `socialcost/bench/sngnn_adapter.py` changes, from outside the package:", "",
          "1. grid spacing `area_width` 700 mm → 526.3 mm, so the 19 x 19 grid graph spans the 10 m SOCIALCOST label;",
          "2. the 256 x 256 output is resized bilinearly to 200 x 200 (loss and metrics on the label grid);",
          "3. batch size 5 → 16, learning rate selected on validation, training budget matched to the other models (the original trains up to 2,300 epochs with early stopping);",
          "4. inputs built from SOCIALCOST: people (position, heading, velocity, angular velocity) at t, t-1 s, t-2 s in the robot frame; scenario and world "
          "objects within 7 m (bounding boxes); wall segments from the map within 7 m; the navigation goal; the executed command. "
          "Interaction links (privileged variant only): every pair of members of an active interaction (membership-completed), and queue members to "
          "their desk. Halt gestures and wheelchairs have no node feature in the original graph, so SNGNN2D cannot represent the gesture or "
          "mobility parts of the label.", "",
          "### SOCIALCOST models", "",
          "The existing estimator architecture (`socialcost/learn/models.py`) is reused: `ImageToGrid` (image encoder, depth- or ground-plane lifting "
          "to a 100 x 100 grid, grid U-Net) for RGB / depth / RGB-D and `PositionToGrid` (rasterised people, grid U-Net) for the position variants. "
          "The only change is an input-channel argument on `PositionToGrid` (3 → 9 channels for Position + Interaction: 4 interaction-class segment "
          "channels, halt gesture, mobility aid). Loss: the original weighted smooth-L1 on the full grid.", "",
          "## Overfitting test", "", "| Model | Train loss first → last | Nonzero MAE first → last |", "|---|---|---|"]
    for m in MODELS:
        f = os.path.join(args.root, "runs", "overfit", f"{m}_membership", "log.jsonl")
        if os.path.exists(f):
            R = [json.loads(line) for line in open(f)]
            L.append(f"| {NAMES[m]} | {R[0]['train_loss']:.4f} → {R[-1]['train_loss']:.4f} | {R[0]['val']['mae_nonzero']:.1f} → {R[-1]['val']['mae_nonzero']:.1f} |")
    L += ["", "## Results", ""] + tables(runs) + ["## Model cost", ""] + cost_table(runs) + ["", "## Figures", ""]
    L += [f"![{n}]({os.path.relpath(f, args.out)})" + "\n" for n, f in figs]
    L += ["", "![training curves](costmap_benchmark_figures/training_curves.png)", "", "## Failure cases (membership label, seed 0, worst test samples by nonzero MAE)", ""]
    for m, rows in fail.items():
        L.append(f"- {NAMES[m]}: " + "; ".join(f"{e}/{k} {man[e]['family']}/{man[e]['subtype']} world {man[e]['world_key']} (MAE {err:.0f})" for err, e, k in rows))
    L += ["", "## Reproduce", "", "```bash", "cd /home/robotics/arena_ws/data/costmap_bench && source env.sh",
          "# environment (once): uv venv --python 3.11 .venv; torch 2.1.2+cu121, dgl 2.0.0+cu121, nvidia-cusparse-cu12 12.1.0.106 (see env.sh)",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/bench_pipeline.py --root /home/robotics/arena_ws/data/costmap_bench   # everything, resumable",
          "# one run, e.g. RGB-D, membership label, seed 1:",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/bench_train.py --model rgbd --label membership --seed 1 --lr <selected> --samples 48000 --val-every 6000 --out runs/main/rgbd_membership_s1",
          "$PY /home/robotics/arena_ws/src/deps/socialcost/scripts/bench_report.py --root /home/robotics/arena_ws/data/costmap_bench --out /home/robotics/arena_ws/data/socialcost/reports",
          "```", "", "Every run directory holds `cmd.txt` (the exact command), `config.json`, `log.jsonl`, `best.pt`, `metrics.json` and `predictions.npz` "
          "(test predictions, every 2nd sample, uint8)."]
    open(os.path.join(args.out, "costmap_prediction_benchmark.md"), "w").write("\n".join(L) + "\n")
    print("report written", sum(seeds_done.values()), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
