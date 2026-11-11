#!/usr/bin/env python3
"""Verify recorded SOCIALCOST datasets and write outputs/verification/.

    python verify_dataset.py --dataset <root> [--dataset <root> ...] --output <dir> [--replay 8]

--replay N feeds N recorded rosters per episode through the real social layer plugin (needs a
free ROS domain, e.g. ROS_DOMAIN_ID=9) for the identical-state offline/live comparison (test G).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from socialcost import verify as V  # noqa: E402
from socialcost.dataset import peds_for_layer  # noqa: E402


def _viz():
    spec = importlib.util.spec_from_file_location("viz", os.path.join(HERE, "visualize_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dump(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def replay_cases(eps: list[V.Ep], per_episode: int) -> list:
    cases = []
    for e in eps:
        n = len(e.robots)
        if not n:
            continue
        for k in sorted({int(round(x)) for x in np.linspace(0, n - 1, per_episode)}):
            cases.append((f"{e.name}#{k}", e.pose(k), peds_for_layer(e.states(k))))
            if e.scenario == "socialcost_queue":
                cases.append((f"{e.name}#{k}+membership", e.pose(k), peds_for_layer(e.states(k, membership=True))))
    return cases


def status_line(name: str, r: dict) -> str:
    m = r.get("metrics", {})
    short = json.dumps(m, default=str)
    short = short if len(short) < 400 else short[:400] + " …"
    return f"| {name} | **{r['status']}** | {r.get('n_failing', 0)} | {r.get('note', '')} | `{short}` |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--replay", type=int, default=0)
    args = ap.parse_args()
    out = args.output
    for d in ("failure_samples", "visualizations"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    eps = V.load_episodes(*args.dataset)

    summary = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "datasets": args.dataset,
               "episodes": len(eps), "samples": sum(len(e.robots) for e in eps),
               "per_episode": [V.inventory(e) for e in eps]}
    summary["problems_total"] = {k: sum(len(p["problems"][k]) if isinstance(p["problems"][k], (list, dict)) else p["problems"][k] for p in summary["per_episode"])
                                 for k in ("empty_files", "unreadable", "nan_only_depth_samples", "inf_depth_values", "count_mismatch")}
    dump(os.path.join(out, "dataset_summary.json"), summary)

    sync = [V.synchronization(e) for e in eps]
    dump(os.path.join(out, "synchronization_report.json"), {"episodes": sync, "all_pass": all(s["status"] == V.PASS for s in sync)})

    geometry = [V.geometry(e) for e in eps]
    costmap = {"geometry": geometry,
               "A_empty_world": V.test_empty(eps), "B_single_pedestrian": V.test_single(eps), "C_conversation": V.test_conversation(eps),
               "D_queue": V.test_queue(eps), "E_group_walking": V.test_group_walk(eps), "F_interaction_lifecycle": V.test_lifecycle(eps),
               "G_offline_vs_live_in_simulation": V.test_live_recorded(eps)}
    if args.replay:
        from socialcost.replay import replay
        res = replay(eps[0].params, replay_cases(eps, args.replay), size_m=eps[0].grid.width * eps[0].grid.resolution, resolution=eps[0].grid.resolution)
        bad = [r for r in res if "error" in r or r["max_abs"] > 0]
        ok = [r for r in res if "error" not in r]
        costmap["G_offline_vs_live_identical_states"] = V.result(V.PASS if res and not bad else V.FAIL, {
            "cases": len(res), "identical": len(ok) - sum(r["max_abs"] > 0 for r in ok),
            "mae_max": max((r["mae"] for r in ok), default=None), "max_abs": max((r["max_abs"] for r in ok), default=None),
            "mismatched_cells_total": sum(r["mismatched_cells"] for r in ok), "mismatched_fraction_max": max((r["mismatched_fraction"] for r in ok), default=None),
            "iou_nonzero_min": min((r["iou_nonzero"] for r in ok), default=None),
            "centroid_offset_max_m": max((r["centroid_offset_m"] or 0 for r in ok), default=None), "per_case": res},
            [(r["case"].split("#")[0], int(r["case"].split("#")[1].split("+")[0])) for r in bad if "#" in r["case"]],
            note="recorded rosters through the real plugin (fixed map-aligned grid around the recorded robot pose, same parameters) vs. the offline painter on that grid")
    else:
        costmap["G_offline_vs_live_identical_states"] = V.result(V.NOT_TESTED, note="run with --replay N")
    dump(os.path.join(out, "costmap_validation.json"), costmap)

    scen = {}
    for e in eps:
        s = scen.setdefault(e.scenario, {"episodes": [], "samples": 0, "samples_with_cost": 0, "membership_samples_differing": 0, "interaction_types_seen": set()})
        s["episodes"].append(e.name)
        s["samples"] += len(e.robots)
        s["samples_with_cost"] += e.meta["label"]["labels_with_cost"]
        s["membership_samples_differing"] += e.meta["label"].get("membership_labels_differing", 0)
        for f in e.frames:
            for i in f["interactions"]:
                s["interaction_types_seen"].add(i["interaction_type"])
    for s in scen.values():
        s["interaction_types_seen"] = sorted(s["interaction_types_seen"])
    dump(os.path.join(out, "scenario_validation.json"), scen)

    # failure samples and per-episode snapshots
    viz = _viz()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(19, 4.6))
    by_name = {e.name: e for e in eps}
    fails = V.failing_samples({"sync": sync, "costmap": costmap})
    written = []
    for name, k, test in fails[:60]:
        e = by_name.get(name)
        if e is None:
            continue
        ve = viz.Episode(e.dir, e.params, e.grid, "social_cost")
        viz.draw(fig, ve, k)
        p = os.path.join(out, "failure_samples", f"{test.replace('.', '_')}__{name.replace('/', '_')}__{k:06d}.png")
        fig.savefig(p, dpi=70)
        written.append(os.path.relpath(p, out))
    for e in eps:
        ve = viz.Episode(e.dir, e.params, e.grid, "social_cost_membership" if e.scenario == "socialcost_queue" else "social_cost")
        for k in sorted({int(round(x)) for x in np.linspace(0, len(e.robots) - 1, 3)}):
            viz.draw(fig, ve, k)
            fig.savefig(os.path.join(out, "visualizations", f"{e.name.replace('/', '_')}_{e.scenario}_{k:06d}.png"), dpi=70)
        viz.temporal_plot(ve, os.path.join(out, "visualizations", f"{e.name.replace('/', '_')}_{e.scenario}_temporal.png"))
    plt.close(fig)

    # report
    L = ["# SOCIALCOST dataset verification", "", f"Generated {summary['generated']} from {', '.join(args.dataset)}.",
         f"{summary['episodes']} episodes, {summary['samples']} samples. Problems: `{json.dumps(summary['problems_total'])}`.", "",
         "Status: PASS = tested and verified, FAIL = tested and incorrect, NOT_TESTED = missing data or functionality.", "",
         "## Synchronization (per episode, simulator stamps, ms)", "", "| episode | scenario | status | rgb−depth max | label state − image max | pose − image max | nearest peds msg max | image − slot max | live costmap − image mean | missing slots |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, e in zip(sync, eps):
        m = s["metrics"]
        L.append(f"| {e.name} | {e.scenario} | **{s['status']}** | {m['rgb_minus_depth_ms']['max_abs']:.1f} | {m['label_state_minus_image_ms']['max_abs']:.1f} | {m['robot_pose_minus_image_ms']['max_abs']:.1f} | "
                 f"{m['bracketing_peds_msgs_ms']['nearest_max']:.1f} | {m['image_minus_slot_ms']['max']:.1f} | {m['live_costmap_minus_image_ms']['mean'] if m['live_costmap_minus_image_ms']['mean'] is not None else float('nan'):.1f} | {len(m['missing_slots'])} |")
    L += ["", "## Geometry (per episode)", "", "| episode | TF base→camera | grid / regeneration | orientation from depth | metric units | values & invalid |", "|---|---|---|---|---|---|"]
    for g in geometry:
        L.append(f"| {g['episode']} | {g['tf_base_camera']['status']} | {g['grid_definition']['status']} | {g['label_orientation_from_depth']['status']} "
                 f"({g['label_orientation_from_depth']['metrics']['inside_cost']}/{g['label_orientation_from_depth']['metrics']['visible_person_pixels']}) | {g['metric_units']['status']} | {g['values_and_invalid']['status']} |")
    L += ["", "## Social costmap semantics", "", "| test | status | failing samples | note | metrics |", "|---|---|---|---|---|"]
    for k, v in costmap.items():
        if k != "geometry":
            L.append(status_line(k, v))
    L += ["", "## Failure samples", ""] + ([f"- [{w}]({w})" for w in written] or ["none"])
    open(os.path.join(out, "verification_report.md"), "w").write("\n".join(L) + "\n")
    statuses = [s["status"] for s in sync] + [v["status"] for k, v in costmap.items() if k != "geometry"] + [c["status"] for g in geometry for k, c in g.items() if isinstance(c, dict)]
    print("\n".join(L[:6]))
    print({s: statuses.count(s) for s in (V.PASS, V.FAIL, V.NOT_TESTED)})
    return 0 if V.FAIL not in statuses else 1


if __name__ == "__main__":
    sys.exit(main())
