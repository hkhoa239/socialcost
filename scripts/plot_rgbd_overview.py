#!/usr/bin/env python3
"""Overview figures for the RGB-D diagnosis (from saved JSON only).

    python plot_rgbd_overview.py --dir <outputs/rgbd_diagnosis> [--nav navigation_v2_interim.json]
writes figures/summary_overview.png and figures/navigation_v2_<scenario>_<ctrl>.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ["bodies", "pos", "oracle_ext", "oracle_ext_replay", "rgbd", "rgbd_t13"]
COL = {"bodies": "#8c8c8c", "pos": "#4c72b0", "oracle_ext": "#2ca02c", "oracle_ext_replay": "#98df8a", "rgbd": "#d62728", "rgbd_t13": "#ff9896"}


def overview(d: str, nav: dict | None, out: str) -> None:
    c = json.load(open(os.path.join(d, "phaseC_cost_distribution.json")))
    dd = json.load(open(os.path.join(d, "phaseD_calibration.json")))
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # A: background floor — fraction of label-free FOV cells per predicted cost band
    a = ax[0][0]
    bands = ["0-0", "1-9", "10-49", "50-99", "100-252"]
    models = ["pilot_rgbd_extended", "pilot_depth_extended", "pilot_rgb_extended", "pilot_pos_extended"]
    w = 0.2
    for k, m in enumerate(models):
        h = c[m]["test_A"]["overall"]["background_pred_hist"]
        a.bar(np.arange(len(bands)) + (k - 1.5) * w, [h[b] for b in bands], w, label=m.replace("pilot_", "").replace("_extended", ""))
    a.set_xticks(range(len(bands)))
    a.set_xticklabels(bands)
    a.set_xlabel("predicted cost on cells where label = 0 (FOV, test_A)")
    a.set_ylabel("fraction of background cells")
    a.set_title("A. RGB-D/depth paint a ~12-cost floor on 73 % of empty FOV cells", fontsize=9)
    a.legend(fontsize=7)

    # B: cost vs distance to nearest pedestrian
    a = ax[0][1]
    for m in models:
        cb = c[m]["test_A"]["overall"]["cost_by_distance"]
        x = [0.125, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 4.0, 5.5]
        a.plot(x, [b["pred"] for b in cb], "o-", ms=3, label=m.replace("pilot_", "").replace("_extended", ""))
    a.plot(x, [b["label"] for b in c[models[0]]["test_A"]["overall"]["cost_by_distance"]], "k--", lw=2, label="label")
    a.set_xlabel("distance to nearest pedestrian [m]")
    a.set_ylabel("mean cost (FOV cells, test_A)")
    a.set_title("B. core under-predicted (0.5–1 m), floor ~10 out to 5 m", fontsize=9)
    a.legend(fontsize=7)

    # C: calibration on held-out splits
    a = ax[0][2]
    rows = {r["tag"]: r for r in dd["pilot_rgbd_extended"] if "test" in r}
    tags = [("raw", "raw"), ("threshold=13", "thr 13 (used)"), ("selected", "thr 50 (pre-reg. pick)"), ("isotonic", "isotonic"), ("blur=2.0", "blur σ2")]
    splits = ["test_A", "test_B", "test_C", "test_D"]
    for k, (t, lab) in enumerate(tags):
        if t not in rows:
            continue
        a.bar(np.arange(len(splits)) + (k - 2) * 0.16, [rows[t]["test"][s]["mae_fov"] for s in splits], 0.16, label=lab)
    a.set_xticks(range(len(splits)))
    a.set_xticklabels(splits)
    a.set_ylabel("FOV MAE (cost units)")
    a.set_title("C. RGB-D calibration (fit on val): threshold removes the floor", fontsize=9)
    a.set_ylim(0, 19)
    a.legend(fontsize=7, loc="upper right", ncol=2)
    tx = rows["raw"]["test"]["test_A"], rows["threshold=13"]["test"]["test_A"]
    a.text(0.02, 0.97, f"test_A bg FP area {tx[0]['bg_fp_area_m2_per_sample']:.1f} → {tx[1]['bg_fp_area_m2_per_sample']:.1f} m²/sample\n"
                       f"P/R@≥50 {tx[0]['pr']['>=50']['precision']:.2f}/{tx[0]['pr']['>=50']['recall']:.2f} → {tx[1]['pr']['>=50']['precision']:.2f}/{tx[1]['pr']['>=50']['recall']:.2f}",
           transform=a.transAxes, va="top", fontsize=7)

    # D: Gate 5 v2 per estimator
    a = ax[1][0]
    names, cov, age95, rate = [], [], [], []
    for e in ARMS[1:] + ["depth"]:
        p = os.path.join(d, "gate5_v2", e, "audit_summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        names.append(f"{e}\n{s['verdict']}" + ("*" if s.get("not_evaluable") else ""))
        cov.append(s["local"]["all"]["coverage"]["mean"])
        age95.append(s["local"]["all"]["age_ms"]["p95"])
        rate.append(1000 / s["mask_stream"]["interval_ms"]["mean"])
    xs = np.arange(len(names))
    a.bar(xs, age95, color="#4c72b0", label="map age p95 [ms]")
    a.axhline(500, color="r", ls=":", lw=1)
    a2 = a.twinx()
    a2.plot(xs, cov, "ko", label="cost preserved (mean frac.)")
    a2.set_ylim(0.9, 1.01)
    a.set_xticks(xs)
    a.set_xticklabels(names, fontsize=7)
    a.set_ylabel("map age p95 [ms] (limit 500)")
    a2.set_ylabel("merged ≥ published, mean fraction")
    a.set_title("D. Gate 5 v2: Nav2 receives every estimator's grid intact\n(* pos static check not evaluable: no cost painted)", fontsize=9)

    # E/F: navigation (conversation / DWB) time and clearance per seed
    if nav:
        for a, key, lab in ((ax[1][1], "time_to_goal_s", "time to goal [s]"), (ax[1][2], "min_clearance_m", "min clearance [m]")):
            for k, m in enumerate(ARMS):
                runs = [p for p in nav["per_run"] if p["scenario"] == "socialcost_nav_conversation" and p["controller"] == "dwb" and p["method"] == m and p.get(key) is not None]
                v = [p[key] for p in runs]
                if not v:
                    continue
                e = nav["summary"]["socialcost_nav_conversation|dwb|" + m][key]
                err = None if not e.get("ci95_t") else [[e["mean"] - e["ci95_t"][0]], [e["ci95_t"][1] - e["mean"]]]
                a.bar(k, e["mean"], color=COL[m], yerr=err, capsize=3)
                a.plot(k + np.linspace(-0.2, 0.2, len(v)), v, "k.", ms=5)
            a.set_xticks(range(len(ARMS)))
            a.set_xticklabels(ARMS, rotation=30, ha="right", fontsize=7)
            a.set_ylabel(lab)
        ax[1][1].set_title("E. conversation / DWB, 5 seeds: oracle through the RGB-D path\nis as slow as RGB-D; removing the floor does not help", fontsize=9)
        ax[1][2].set_title("F. clearance: perfect labels lose 1.1 m once limited to\nthe camera FOV + camera timing (oracle_ext vs replay)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def nav_panels(nav: dict, figdir: str) -> list[str]:
    paths = []
    keys = sorted({tuple(k.split("|")[:2]) for k in nav["summary"]})
    mets = [("success", "success rate"), ("time_to_goal_s", "time to goal [s]"), ("stop_time_s", "stop time [s]"),
            ("personal_zone_time_s", "personal-zone time [s]"), ("min_clearance_m", "min clearance [m]"), ("ped_deflection_m", "ped deflection vs ref [m]")]
    for sc, ctrl in keys:
        arms = [m for m in ARMS if f"{sc}|{ctrl}|{m}" in nav["summary"]]
        fig, ax = plt.subplots(1, len(mets), figsize=(3.3 * len(mets), 3.4))
        for a, (key, lab) in zip(ax, mets):
            for k, m in enumerate(arms):
                S = nav["summary"][f"{sc}|{ctrl}|{m}"]
                if key == "success":
                    s = S["success"]
                    lo, hi = s["ci95_wilson"]
                    a.bar(k, s["rate"], color=COL[m], yerr=[[s["rate"] - lo], [hi - s["rate"]]], capsize=3)
                    a.text(k, 0.02, f"{s['k']}/{s['n']}", ha="center", fontsize=7)
                    continue
                e = S.get(key)
                if not e or e["mean"] is None:
                    continue
                err = None if not e.get("ci95_t") else [[e["mean"] - e["ci95_t"][0]], [e["ci95_t"][1] - e["mean"]]]
                a.bar(k, e["mean"], color=COL[m], yerr=err, capsize=3)
                v = [p[key] for p in nav["per_run"] if p["scenario"] == sc and p["controller"] == ctrl and p["method"] == m and p.get(key) is not None]
                a.plot(k + np.linspace(-0.2, 0.2, len(v)), v, "k.", ms=4)
            a.set_xticks(range(len(arms)))
            a.set_xticklabels(arms, rotation=35, ha="right", fontsize=7)
            a.set_title(lab, fontsize=8)
        fig.suptitle(f"{sc.replace('socialcost_nav_', '')} / {ctrl}  (bars: mean, whiskers: t 95 % CI or Wilson for success, dots: seeds)", fontsize=9)
        fig.tight_layout()
        p = os.path.join(figdir, f"navigation_v2_{sc.replace('socialcost_nav_', '')}_{ctrl}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--nav", default="navigation_v2_interim.json")
    args = ap.parse_args()
    figdir = os.path.join(args.dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    np_ = os.path.join(args.dir, args.nav)
    nav = json.load(open(np_)) if os.path.exists(np_) else None
    overview(args.dir, nav, os.path.join(figdir, "summary_overview.png"))
    print(os.path.join(figdir, "summary_overview.png"))
    if nav:
        for p in nav_panels(nav, figdir):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
