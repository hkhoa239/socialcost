#!/usr/bin/env python3
"""Run the automated QA checks on a dataset and write validation.json + validation.md.

    python validate_dataset.py <dataset_root> [--repro <root_of_repeat_run>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.dataset import read_jsonl  # noqa: E402,F401
from socialcost.labels import GridSpec, LayerParams  # noqa: E402
from socialcost.qa import check_episode, episodes, trajectory_repro  # noqa: E402


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--repro", default=None, help="a second dataset recorded with the same scenarios and seeds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.root, "metadata.json")))
    params = LayerParams.from_mapping(meta["label_params"])
    grid = GridSpec(**meta["label_grid"])
    results = [check_episode(ep, params, grid) for ep in episodes(args.root)]

    repro = []
    if args.repro:
        other = {}
        for ep in episodes(args.repro):
            m = json.load(open(os.path.join(ep, "episode_metadata.json")))
            other[(m["scenario"], m["run_seed"])] = ep
        for ep in episodes(args.root):
            m = json.load(open(os.path.join(ep, "episode_metadata.json")))
            twin = other.get((m["scenario"], m["run_seed"]))
            if twin:
                repro.append({"scenario": m["scenario"], "run_seed": m["run_seed"], "a": os.path.basename(ep), "b": twin,
                              "seed_a": m.get("episode_seed"), "seed_b": json.load(open(os.path.join(twin, "episode_metadata.json"))).get("episode_seed"),
                              **trajectory_repro(ep, twin)})

    total_samples = sum(r["samples"] for r in results)
    summary = {
        "episodes": len(results), "episodes_passing": sum(r["pass"] for r in results),
        "valid_samples": total_samples, "expected_samples": sum(r["expected_samples"] for r in results),
        "missing_frames": sum(r["missing_frames"] for r in results),
        "rgb_depth_sync_max_s": max((r["sync"]["rgb_depth_max_s"] for r in results), default=None),
        "frame_to_slot_lag_max_s": max((r["sync"]["frame_to_slot_lag_max_s"] or 0 for r in results), default=None),
        "peds_interp_bracket_max_s": max((r["sync"]["peds_interp_bracket_max_s"] or 0 for r in results), default=None),
        "disk_bytes": int(subprocess.run(["du", "-sb", args.root], capture_output=True, text=True).stdout.split()[0]),
    }
    ag = [r["live_agreement"] for r in results if "live_agreement" in r]
    if ag:
        n = sum(a["samples"] for a in ag)
        summary["live_agreement"] = {
            "samples": n,
            "exact_fraction_mean": sum(a["exact_fraction_mean"] * a["samples"] for a in ag) / n,
            "mae_mean": sum(a["mae_mean"] * a["samples"] for a in ag) / n,
            "iou_nonzero_mean": sum(a["iou_nonzero_mean"] * a["samples"] for a in ag) / n,
            "fully_identical_samples": sum(a["fully_identical_samples"] for a in ag),
        }
    rates = [r["recording"]["samples_per_wall_s"] for r in results if r["recording"]["samples_per_wall_s"]]
    rtfs = [r["recording"].get("real_time_factor") for r in results if r["recording"].get("real_time_factor")]
    summary["samples_per_wall_s_mean"] = sum(rates) / len(rates) if rates else None
    summary["real_time_factor_mean"] = sum(rtfs) / len(rtfs) if rtfs else None
    report = {"summary": summary, "episodes": results, "reproducibility": repro}
    out = args.out or os.path.join(args.root, "validation")
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "validation.json"), "w"), indent=2)

    lines = ["# Validation", "", "| episode | scenario | seed | samples | missing | depth invalid (mean) | ped reproj. agree / occluded / missing | personal-space ok | interaction ok | cost outside reach | live exact (mean) | live IoU | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        la = r.get("live_agreement", {})
        pr = r["ped_reprojection"]
        lb = r["label"]
        lines.append(f"| {r['episode']} | {r['scenario']} | {r['run_seed']} | {r['samples']}/{r['expected_samples']} | {r['missing_frames']} | "
                     f"{fmt(r['depth_invalid_fraction_mean'])} | {pr["depth_agrees"]}/{pr["occluded"]}/{pr["missing"]} of {pr["projected"]} | {lb['personal_space_ok']}/{lb['personal_space_cells_checked']} | "
                     f"{lb['interaction_midpoints_ok']}/{lb['interaction_midpoints_checked']} | {lb['cost_outside_reach']} | {fmt(la.get('exact_fraction_mean'), 4)} | "
                     f"{fmt(la.get('iou_nonzero_mean'))} | {'PASS' if r['pass'] else 'FAIL: ' + ', '.join(r['failures'])} |")
    lines += ["", "```json", json.dumps(summary, indent=2), "```"]
    if repro:
        lines += ["", "## Reproducibility (same scenario and seed, separate launches)", "",
                  "| scenario | seed | samples | episode seed equal | ped pos err mean / max (m) | robot pos err mean / max (m) | label MAE |", "|---|---|---|---|---|---|---|"]
        for p in repro:
            lines.append(f"| {p['scenario']} | {p['run_seed']} | {p['samples']} | {p['seed_a'] == p['seed_b']} | {fmt(p['ped_pos_err_mean_m'])} / {fmt(p['ped_pos_err_max_m'])} | "
                         f"{fmt(p['robot_pos_err_mean_m'])} / {fmt(p['robot_pos_err_max_m'])} | {fmt(p['label_mae_mean'])} |")
    open(os.path.join(out, "validation.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["episodes_passing"] == summary["episodes"] else 1


if __name__ == "__main__":
    sys.exit(main())
