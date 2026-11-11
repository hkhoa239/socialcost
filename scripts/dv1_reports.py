#!/usr/bin/env python3
"""Write reports/dataset_v1_validation.md, dataset_v1_distribution.md and navigation_challenge_v1.md from the
validation, distribution, visibility and navigation-challenge outputs.

    python dv1_reports.py --base /opt/arena_ws/data/socialcost
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f} %" if d else "-"


def validation_md(base: str) -> str:
    v = json.load(open(os.path.join(base, "outputs", "dataset_v1_validation", "all", "validation.json")))
    s = v["summary"]
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(base, "dataset_v1", "manifests", "manifest_v1.json")))["episodes"]}
    vis = json.load(open(os.path.join(base, "dataset_v1", "metadata", "visibility_achieved.json")))
    L = [json.loads(line) for line in open(os.path.join(base, "dataset_v1", "metadata", "collection_log.jsonl"))]
    att = [x for x in L if "wall_s" in x]
    ok_wall = [x["wall_s"] for x in att if x["result"] == "ok"]
    groups = defaultdict(lambda: [0, 0])
    for ep in v["episodes"].values():
        for g, cs in ep["groups"].items():
            for c in cs:
                groups[f"{g}/{c['check']}"][0] += 1
                groups[f"{g}/{c['check']}"][1] += int(c["ok"])
    lines = ["# SOCIALCOST dataset_v1: validation report", "",
             f"Validation run {v['created']} over all {len(man)} manifest episodes (`scripts/dv1_validate.py --stage all`). "
             "Machine-readable output: `outputs/dataset_v1_validation/all/validation.json`, pair results in `pairs.json`.", "",
             "## Summary", "", "| Item | Result |", "|---|---|",
             f"| Episodes recorded and checked | {s['episodes_checked']} of {len(man)} ({s['missing']} missing) |",
             f"| Episodes passing every check | {s['episodes_ok']} |",
             f"| Counterfactual pairs checked / passing | {s['pairs_checked']} / {s['pairs_ok']} |",
             f"| Leakage (episode, pair, template, seed and geometry keys; held-out world; held-out bodies) | {'none' if s['leakage_ok'] else 'FOUND'} |",
             f"| Pair observability | {', '.join(f'{k} {n}' for k, n in sorted(s['observability'].items()))} |",
             f"| Collection attempts / valid | {len(att)} / {sum(x['result'] == 'ok' for x in att)} (median {statistics.median(ok_wall):.0f} s per valid episode) |", ""]
    if v["missing"]:
        lines += ["Missing episodes (shortfall): " + ", ".join(f"{m} ({man[m]['family']}/{man[m]['subtype']}/{man[m]['condition']})" for m in v["missing"]), ""]
    lines += ["## Checks", "", "| Check | Passing / checked |", "|---|---:|"]
    lines += [f"| {k} | {ok} / {n} |" for k, (n, ok) in sorted(groups.items())]
    pf = Counter(c["check"] for p in v["pairs"] for c in p["checks"] if not c["ok"])
    lines += ["", "## Counterfactual pairs", "",
              "Controlled people (primary, participants, queue members, targets) must agree within 2 cm and 2 deg in every sample where they "
              "are stationary in both conditions, until the robot first passes within 1.0 m of them. Pairs whose interaction starts, stops or "
              "moves (approach-and-stop, lifecycle, advancing queues, moving wheelchair users) are dynamic: their divergence is recorded, "
              "not tested. Robot and bystander trajectories are not controlled: bystanders react to the robot and to each other.", ""]
    static = [p for p in v["pairs"] if p.get("controlled_static_design")]
    robot = [p["robot"]["robot_pos_max_m"] for p in v["pairs"] if p["robot"]["robot_pos_max_m"] is not None]
    lines += [f"- Static-design pairs: {len(static)}, matched within 2 cm / 2 deg: {sum(any(c['check'] == 'matched_geometry_2cm_2deg' and c['ok'] for c in p['checks']) for p in static)}.",
              f"- Robot position difference between conditions (max per pair): median {statistics.median(robot):.2f} m, 90th percentile "
              f"{sorted(robot)[int(0.9 * len(robot))]:.2f} m, max {max(robot):.2f} m.",
              f"- Failing pair checks: {dict(pf) if pf else 'none'}.", ""]
    obs = Counter((p["family"], p["observability_class"]) for p in v["pairs"])
    lines += ["| Family | Visually observable | Not visually observable | Hidden state |", "|---|---:|---:|---:|"]
    for fam in ("conversation", "stop_gesture", "wheelchair", "queue"):
        lines.append(f"| {fam} | {obs[(fam, 'visually_observable')]} | {obs[(fam, 'not_visually_observable')]} | {obs[(fam, 'hidden_state')]} |")
    nvo = [p for p in v["pairs"] if p["observability_class"] == "not_visually_observable"]
    if nvo:
        lines += ["", "Not visually observable pairs (cue never rendered visibly, kept and labelled as such in Test D): "
                  + ", ".join(f"{p['pair_id']} ({p['subtype']})" for p in nvo) + "."]
    pc = vis["per_designed_category"]
    lines += ["", "## Partial-observability categories", "",
              "Each episode keeps its designed category and gains `achieved_visibility_category` (episode_metadata.json `dv1`, "
              "`metadata/visibility_achieved.json`).", "", "| Designed | Met | Not met |", "|---|---:|---:|"]
    lines += [f"| {c} | {pc[c].get('met', 0)} | {pc[c].get('not_met', 0)} |" for c in sorted(pc)]
    lines += ["", "Achieved categories over the 80 episodes: " + ", ".join(f"{k} {n}" for k, n in sorted(vis["achieved_counts"].items())) + ".", ""]
    bad = {k: [f"{g}/{c['check']}" for g, cs in e["groups"].items() for c in cs if not c["ok"]] for k, e in v["episodes"].items() if not e["ok"]}
    if bad:
        lines += ["## Episodes with a failing check", "", "| Episode | Family | Failing |", "|---|---|---|"]
        lines += [f"| {k} | {man[k]['family']}/{man[k]['subtype']} | {', '.join(b)} |" for k, b in sorted(bad.items())]
    lines += ["", "## Rule changes made during collection", "",
              "- Robot health: upright (median tilt < 3 deg, <= 5 % of samples above 5 deg, max < 12 deg), path >= 1 m, start delay <= 1.5 s after the reset.",
              "- One skipped 5 Hz slot tolerated (124 / 125 samples).",
              "- Gesture term required only when the gesturing person is inside the 10 m label grid.",
              "- Empty control episodes exempt from the body-split rule.",
              "- FOV exit / re-entry: seen, outside, seen again, with unseen samples allowed in between.",
              "- Approach-and-stop conversations re-recorded with scenario compiler v1.1 (participant k seeks k x 1.0 s after arriving), "
              "because simultaneous symmetric seeks left 3 of 15 pairs with two one-member interactions and no conversation.", ""]
    return "\n".join(lines)


def distribution_md(base: str) -> str:
    d = json.load(open(os.path.join(base, "outputs", "dataset_v1_visualizations", "all", "distributions.json")))
    man = json.load(open(os.path.join(base, "dataset_v1", "manifests", "manifest_v1.json")))
    E = man["episodes"]
    pools = man["body_pools"]
    lines = ["# SOCIALCOST dataset_v1: distribution audit", "",
             "Figures: `outputs/dataset_v1_visualizations/all/distributions.png`, `cost_magnitude.png`; numbers: `distributions.json`.", "",
             "## Episodes", "", "| Family | train | val | test_a | test_b | test_c | total |", "|---|---:|---:|---:|---:|---:|---:|"]
    fam = sorted({e["family"] for e in E})
    for f in fam:
        c = Counter(e["split"] for e in E if e["family"] == f)
        lines.append(f"| {f} | {c['train']} | {c['val']} | {c['test_a']} | {c['test_b']} | {c['test_c']} | {sum(c.values())} |")
    w = Counter(e["world_key"] for e in E)
    lines += ["", "Worlds: " + ", ".join(f"{k} ({man['worlds'][k]['name']}) {w[k]}" for k in sorted(w)) + ". World D appears only in test_b.", "",
              "## Pedestrian count", "", "| Bin | Target | Recorded episodes | Share |", "|---|---:|---:|---:|"]
    pc = d["ped_count"]
    n = sum(pc["recorded"].values()) or 1
    for b in ("0-1", "2", "3-4", "5-6", "7-10"):
        lines.append(f"| {b} | {pc['target_pct'][b]:.0f} % | {pc['recorded'].get(b, 0)} | {100 * pc['recorded'].get(b, 0) / n:.1f} % |")
    lines += ["", "The 5-6 bin is short by design: only queue, group walking and partial observability may exceed 4 people under the family ranges.", ""]
    r = d.get("recorded", {})
    if r:
        vs = r["visibility_states"]
        tot = sum(vs.values())
        lines += ["## Recorded measures (every 5th sample)", "",
                  f"- Camera-visible pedestrians per sample: {dict(sorted((int(k), v) for k, v in r['visible_per_sample'].items()))}.",
                  "- Visibility states (pedestrian samples): " + ", ".join(f"{k} {pct(v, tot)}" for k, v in sorted(vs.items())) + ".",
                  f"- Mean camera distance of in-FOV pedestrians {r['camera_distance_mean']:.2f} m, mean pedestrian speed {r['speed_mean']:.2f} m/s, "
                  f"mean social-cost area {r['cost_area_m2_mean']:.2f} m^2 per sample (extended/2.1).", ""]
    by = Counter(h.get("trajectory_class") for e in E for h in e["humans"] if h["role"] == "bystander")
    lines += ["## Pedestrian roles and motion", "",
              "- Bystander motion classes: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])) + ".",
              f"- Body identities: {len(pools['known_walking'])} known, {len(pools['heldout_walking'])} held out (test_c only); "
              f"{len(d['body_asset_use'])} bodies used.", "",
              "| Interaction condition | Episodes |", "|---|---:|"]
    for k, v in sorted(Counter(str(e["interaction_condition"]) for e in E).items()):
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def nav_md(base: str) -> str:
    root = os.path.join(base, "navigation_challenge_v1")
    doc = json.load(open(os.path.join(root, "navigation_challenge_v1.json")))
    idx_p = os.path.join(root, "references", "index.json")
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    C = doc["configs"]
    lines = ["# SOCIALCOST navigation_challenge_v1", "",
             "36 fixed configurations with verified route alternatives, generated by `scripts/dv1_nav_challenge.py` "
             "(config per scenario in `navigation_challenge_v1/scenarios/<id>/config.json` with a route figure). No controller "
             "evaluation has been run.", "",
             f"Validity rule: {doc['validity_rule']}. Route lengths come from 8-connected A* on the map with furniture and pedestrian "
             "bodies (0.30 m robot margin); the social planner weights each metre by 1 + 4 x cost/252 of extended/2.1.", "",
             "| Kind | Configs | Worlds |", "|---|---:|---|"]
    for k in doc["quota"]:
        cs = [c for c in C if c["kind"] == k]
        lines.append(f"| {k} | {len(cs)} | {dict(Counter(c['world_key'] for c in cs))} |")
    lines += ["", "World coverage is uneven (" + ", ".join(f"{k} {v}" for k, v in sorted(Counter(c["world_key"] for c in C).items())) +
              "): the 3-4 m corridors of A and B rarely leave a detour around a social region, so configurations fell back to the open worlds.", "",
              "| Config | Kind | World | Site | Route A (m) | Route B (m) | Detour | Horizon (s) | Reference |", "|---|---|---|---|---:|---:|---:|---:|---|"]
    for c in C:
        a = c["alternative_routes"]
        ref = idx.get(c["config_id"], {})
        rs = "ok" if ref.get("ok") else ("not recorded" if not ref else "FAILED")
        lines.append(f"| {c['config_id']} | {c['kind']} | {c['world_key']} | {c['site']} | {a['route_a_m']} | {a['route_b_m']} | x{a['detour_ratio']} | "
                     f"{c['evaluation_horizon_s']} | {rs} |")
    if idx:
        okn = sum(v.get("ok", False) for v in idx.values())
        d = [v["robot_min_dist_to_ped_m"] for v in idx.values() if v.get("robot_min_dist_to_ped_m") is not None]
        lines += ["", f"Robot-free references: {okn} / {len(C)} recorded over the full horizon with the robot parked "
                      f"(minimum robot-pedestrian distance {min(d):.1f} m) in `navigation_challenge_v1/references/<id>/`." if d else ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    rep = os.path.join(args.base, "reports")
    open(os.path.join(rep, "dataset_v1_validation.md"), "w").write(validation_md(args.base))
    open(os.path.join(rep, "dataset_v1_distribution.md"), "w").write(distribution_md(args.base))
    open(os.path.join(rep, "navigation_challenge_v1.md"), "w").write(nav_md(args.base))
    print("reports written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
