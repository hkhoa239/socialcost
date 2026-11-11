#!/usr/bin/env python3
"""Markdown tables (+ navigation plots) for the RGB-D diagnosis report, from the saved result files.

    python make_rgbd_report_tables.py --dir /opt/arena_ws/data/socialcost/outputs/rgbd_diagnosis > tables.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def f(x, nd=2):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def ci(e, nd=1):
    if not e or e.get("mean") is None:
        return "–"
    s = f(e["mean"], nd)
    if e.get("ci95_t"):
        s += f" [{e['ci95_t'][0]:.{nd}f}, {e['ci95_t'][1]:.{nd}f}]"
    return s + f" (n={e['n']})"


def phase_c(d: str) -> str:
    c = json.load(open(os.path.join(d, "phaseC_cost_distribution.json")))
    out = ["### Phase C: held-out cost distribution (test_A, camera-FOV cells, raw predictions)\n",
           "| model | FOV MAE | cells >0 pred / label | cells >10 | cells >50 | cells >100 | bg cells predicted 10–49 | P / R @≥50 | IoU @≥100 | FP / FN area @≥10 (m²/sample) | high-cost bias (label ≥100) | strict social recall |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, v in c.items():
        r = v["test_A"]["overall"]
        fr = r["frac_fov_cells"]
        pr = r["pr"]
        out.append(f"| {e.replace('pilot_', '')} | {f(r['mae_fov'], 1)} | {f(fr['>0']['pred'])} / {f(fr['>0']['label'])} | {f(fr['>10']['pred'])} / {f(fr['>10']['label'])} | "
                   f"{f(fr['>50']['pred'], 3)} / {f(fr['>50']['label'], 3)} | {f(fr['>100']['pred'], 3)} / {f(fr['>100']['label'], 3)} | {f(r['background_pred_hist']['10-49'])} | "
                   f"{f(pr['>=50']['precision'])} / {f(pr['>=50']['recall'])} | {f(pr['>=100']['iou'])} | {f(pr['>=10']['fp_area_m2_per_sample'], 1)} / {f(pr['>=10']['fn_area_m2_per_sample'], 2)} | "
                   f"{f(r['high_bias'], 1)} | {f(r.get('social_extra_recall_strict'))} |")
    out.append("\nCost by distance to the nearest pedestrian (test_A, FOV cells; mean predicted / label):\n")
    bins = [b["bin"] for b in next(iter(c.values()))["test_A"]["overall"]["cost_by_distance"]]
    out.append("| model | " + " | ".join(bins) + " |")
    out.append("|---|" + "---|" * len(bins))
    for e, v in c.items():
        cb = v["test_A"]["overall"]["cost_by_distance"]
        out.append(f"| {e.replace('pilot_', '')} | " + " | ".join(f"{f(b['pred'], 0)} / {f(b['label'], 0)}" for b in cb) + " |")
    return "\n".join(out)


def phase_d(d: str) -> str:
    dd = json.load(open(os.path.join(d, "phaseD_calibration.json")))
    out = ["### Phase D: calibration ablations (fitted on val; held-out test splits)\n"]
    for e, rows in dd.items():
        out += [f"**{e.replace('pilot_', '')}**\n", "| step | val MAE | val strict recall | val R@≥50 | val halo recall | val bg FP m² | test_A MAE | test_A P/R@≥50 | test_B MAE | test_C MAE | test_D MAE | test_A bg FP m² |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if "test" not in r:
                continue
            v, t = r["val"], r["test"]
            out.append(f"| {r['tag']} | {f(v['mae_fov'])} | {f(v['social_extra_recall_strict'])} | {f(v['pr']['>=50']['recall'])} | {f(v['low_halo_recall'])} | {f(v['bg_fp_area_m2_per_sample'])} | "
                       f"{f(t['test_A']['mae_fov'])} | {f(t['test_A']['pr']['>=50']['precision'])}/{f(t['test_A']['pr']['>=50']['recall'])} | {f(t['test_B']['mae_fov'])} | {f(t['test_C']['mae_fov'])} | {f(t['test_D']['mae_fov'])} | {f(t['test_A']['bg_fp_area_m2_per_sample'])} |")
        out.append("")
    return "\n".join(out)


def gate5(d: str) -> str:
    g = os.path.join(d, "gate5_v2")
    out = ["### Phase A/B/E: Gate 5 v2 per estimator (conversation, seed 1, DWB; merged costmap vs the grid actually published)\n",
           "| estimator | verdict | geometry | timestamps | cost preserved | aligned | latency bounded | moving | static | global costmap | local updates checked (moving / static) | coverage mean / p05 / min | map age p50 / p95 / max (ms) | mask rate (Hz) | dropped | input→publish p50 (ms) | compute p50 (ms) | excess non-lethal cells in footprint p50/p95 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ("oracle_ext", "oracle_ext_replay", "rgbd", "rgbd_t13", "depth", "pos"):
        p = os.path.join(g, a, "audit_summary.json")
        if not os.path.exists(p):
            out.append(f"| {a} | not run | | | | | | | | | | | | | | | | |")
            continue
        s = json.load(open(p))
        c = s["criteria"]
        L = s["local"]
        cov, age = L["all"]["coverage"] or {}, L["all"]["age_ms"] or {}
        iv = s["mask_stream"]["interval_ms"] or {}
        est = s.get("estimator", {})
        ex = L.get("excess_in_footprint") or {}
        out.append(f"| {a} | **{s['verdict']}** | {f(c['geometry'])} | {f(c['timestamps'])} | {f(c['cost_preserved'])} | {f(c['aligned'])} | {f(c['latency_bounded'])} | "
                   f"{f(c['planner_receives_while_moving'])} | {f(c['static_robot'])} | {f(c['global_costmap_cost_preserved'])} | {L['moving']['n']} / {L['static']['n']} | "
                   f"{f(cov.get('mean'), 4)} / {f(cov.get('p05'), 3)} / {f(cov.get('min'), 3)} | {f(age.get('p50'), 0)} / {f(age.get('p95'), 0)} / {f(age.get('max'), 0)} | "
                   f"{f(1000 / iv['mean'], 2) if iv.get('mean') else '–'} | {f(est.get('dropped_frac'), 3)} | {f((est.get('input_to_publish_ms') or {}).get('p50'), 0)} | "
                   f"{f((est.get('compute_ms') or {}).get('p50'), 1)} | {f(ex.get('p50'), 0)} / {f(ex.get('p95'), 0)} |")
    out.append("\nOnline agreement with the extended oracle during the Gate 5 drive (shadow, robot grid, FOV; not published):\n")
    out.append("| estimator | ticks | MAE (FOV) | P@≥50 | R@≥50 | P@≥100 | R@≥100 | predicted / label nonzero FOV cells |")
    out.append("|---|---|---|---|---|---|---|---|")
    for a in ("rgbd", "rgbd_t13", "depth", "pos"):
        p = os.path.join(g, a, "audit_summary.json")
        if not os.path.exists(p):
            continue
        sh = json.load(open(p)).get("estimator", {}).get("shadow_vs_oracle_ext")
        if sh:
            out.append(f"| {a} | {sh['ticks']} | {f(sh['mae_fov'], 1)} | {f(sh['precision@50'])} | {f(sh['recall@50'])} | {f(sh['precision@100'])} | {f(sh['recall@100'])} | "
                       f"{f(sh['pred_nonzero_fov_mean'], 0)} / {f(sh['label_nonzero_fov_mean'], 0)} |")
    sc = os.path.join(d, "stage_consistency.json")
    if os.path.exists(sc):
        r = json.load(open(sc))
        out += ["\nStage consistency of the estimator's intermediate grids (dumps every 10th tick):\n",
                "| estimator | ticks | FOV mask applied | resample reproduced exactly (independent cv2) | max mismatching cells | encode error max (cost units) | clipped cells | cells ≥100 nearest / max-pool | frames | origins on lattice |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for a, v in r.items():
            nv = v.get("nearest_vs_maxpool") or {}
            out.append(f"| {a} | {v['ticks']} | {v['fov_mask_ok']} | {v['resample_exact']} | {v['resample_mismatch_cells_max']} | {v['encode_err_max']} | {v['clipped_max']} | "
                       f"{f(nv.get('hi_cells_nearest_mean'), 0)} / {f(nv.get('hi_cells_maxpool_mean'), 0)} | {', '.join(v['frames'])} | {v['origin_on_lattice']}/{v['ticks']} |")
    return "\n".join(out)


def nav(d: str, figdir: str) -> str:
    p = os.path.join(d, "navigation_v2.json")
    if not os.path.exists(p):
        return "### Phase F: navigation v2\n\nnot available yet\n"
    n = json.load(open(p))
    S = n["summary"]
    methods = ["bodies", "pos", "oracle_ext", "oracle_ext_replay", "rgbd", "rgbd_t13"]
    out = ["### Phase F: controlled navigation re-evaluation (nav_v2)\n",
           "Success: k/n with Wilson 95 % interval. Other cells: mean [t-based 95 % CI] (n). Time-to-goal over successful runs only.\n"]
    keys = sorted({tuple(k.split("|")[:2]) for k in S})
    for sc, ctrl in keys:
        out += [f"**{sc.replace('socialcost_nav_', '')} / {ctrl}**\n",
                "| arm | success | timeouts | time to goal (s) | stop time (s) | personal-zone time (s) | min clearance (m) | pass-inside | halt-wedge time (s) | ped deflection vs robot-free ref (m) | estimator compute (ms) |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            e = S.get(f"{sc}|{ctrl}|{m}")
            if not e:
                continue
            su = e["success"]
            w = su["ci95_wilson"]
            out.append(f"| {m} | {su['k']}/{su['n']} [{w[0]:.2f}, {w[1]:.2f}] | {e['timeouts']} | {ci(e['time_to_goal_s'])} | {ci(e['stop_time_s'])} | {ci(e['personal_zone_time_s'])} | "
                       f"{ci(e['min_clearance_m'], 2)} | {e['pass_inside']['k']}/{e['pass_inside']['n']} | {ci(e['halt_wedge_time_s'])} | {ci(e['ped_deflection_m'], 2)} | {ci(e['compute_ms_mean'], 1)} |")
        out.append("")
    # paired: rgbd_t13 - rgbd and oracle_ext_replay - oracle_ext
    P = n["paired"]
    out += ["**Paired differences (same scenario, controller and seed): mean [t 95 % CI] (n)**\n",
            "| comparison | scenario / controller | Δ success | Δ time to goal (s) | Δ stop time (s) | Δ personal-zone time (s) | Δ min clearance (m) |", "|---|---|---|---|---|---|---|"]
    for base, arm in (("rgbd", "rgbd_t13"), ("bodies", "rgbd"), ("bodies", "rgbd_t13"), ("bodies", "oracle_ext_replay"), ("bodies", "oracle_ext"), ("bodies", "pos")):
        for sc, ctrl in keys:
            e = P.get(base, {}).get(f"{sc}|{ctrl}|{arm}")
            if not e:
                continue
            out.append(f"| {arm} − {base} | {sc.replace('socialcost_nav_', '')} / {ctrl} | {ci(e.get('success'), 2)} | {ci(e.get('time_to_goal_s'))} | {ci(e.get('stop_time_s'))} | "
                       f"{ci(e.get('personal_zone_time_s'))} | {ci(e.get('min_clearance_m'), 2)} |")
    if n.get("infra_failures"):
        out += ["\nInfrastructure failures (not scored): " + ", ".join(f"{x['tag']} ({x['status']})" for x in n["infra_failures"])]
    try:
        plot_nav(n, figdir, methods, keys)
        out.append(f"\n![navigation v2]({os.path.relpath(os.path.join(figdir, 'navigation_v2.png'), d)})")
    except Exception as ex:  # noqa: BLE001
        out.append(f"\n(plot failed: {ex})")
    return "\n".join(out)


def plot_nav(n, figdir, methods, keys) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    S = n["summary"]
    mets = [("time_to_goal_s", "time to goal [s]"), ("stop_time_s", "stop time [s]"), ("personal_zone_time_s", "personal-zone time [s]"), ("min_clearance_m", "min clearance [m]")]
    fig, ax = plt.subplots(len(keys), len(mets), figsize=(4.2 * len(mets), 2.6 * len(keys)), squeeze=False)
    colors = ["#888888", "#4c72b0", "#55a868", "#8fd18f", "#c44e52", "#dd8452"]
    for i, (sc, ctrl) in enumerate(keys):
        for j, (k, lab) in enumerate(mets):
            a = ax[i][j]
            for mi, m in enumerate(methods):
                e = S.get(f"{sc}|{ctrl}|{m}", {}).get(k)
                if not e or e.get("mean") is None:
                    continue
                err = None
                if e.get("ci95_t"):
                    err = [[e["mean"] - e["ci95_t"][0]], [e["ci95_t"][1] - e["mean"]]]
                a.bar(mi, e["mean"], color=colors[mi], yerr=err, capsize=3)
                runs = [p[k] for p in n["per_run"] if p["scenario"] == sc and p["controller"] == ctrl and p["method"] == m and p.get(k) is not None]
                a.plot(np.full(len(runs), mi) + np.linspace(-0.15, 0.15, len(runs)) if runs else [], runs, "k.", ms=3)
            a.set_xticks(range(len(methods)))
            a.set_xticklabels(methods, rotation=40, fontsize=6, ha="right")
            a.set_title(f"{sc.replace('socialcost_nav_', '')}/{ctrl}: {lab}", fontsize=8)
    fig.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "navigation_v2.png"), dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir
    print(phase_c(d))
    print()
    print(phase_d(d))
    print()
    print(gate5(d))
    print()
    print(nav(d, os.path.join(d, "figures")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
