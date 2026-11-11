#!/usr/bin/env python3
"""Pick the Nav2 controller setup for the full re-run, from the controller diagnosis (ground-truth arm only).

Rule (fixed before the diagnosis ran; widened 2026-11-07 before any diagnosis run: 4 variants incl. global-only social cost, bodies arm added):
  1. per variant: ground-truth-arm (oracle_ext) and bodies-arm success on the 24 diagnosis configurations
     (18 the ground-truth arm failed in challenge_v2 + 6 controls)
  2. a variant qualifies if the ground-truth arm solves >= 18 / 24 and >= 5 / 6 controls, and the bodies arm solves at
     least as many of the 24 as it did in challenge_v2 (the setup must not make plain navigation worse)
  3. among qualifying variants: most ground-truth + bodies successes, then lowest ground-truth region exposure on successes
If one qualifies, writes configs/navigation/challenge_v3.yaml (all six arms, every arm gets the same controller, overlay and
publish transform; 1 repeat) and exits 0; otherwise exits 3 and nothing is re-run.

    python ctrl_select.py --diag <ctrl_diag dir> --baseline <challenge_v2.json> --report <md>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.realpath(__file__))
CFG = os.path.join(os.path.dirname(HERE), "configs", "navigation")
# order = run order; scale50 / mppi_scale50 dropped 2026-11-07 before any run (N09 stalled on cost 26-29: scaling cannot help)
VARIANTS = ["global", "obs01_patient", "obs01", "mppi"]
# camera-only arms (benchmark RGB / RGB-D image U-Nets on the extended label, seed ensemble, in view only), added to v3
CAM_ARMS = yaml.safe_load(open(os.path.join(CFG, "challenge_v2cam.yaml")))["method_specs"]
MIN_SUCCESS, MIN_CONTROLS = 18, 5


def summarise(runs: list[dict], only: list[str], controls: list[str], method: str = "oracle_ext") -> dict:
    by = {r["config_id"]: r for r in runs if r["method"] == method and r.get("repeat", 0) == 0 and r["config_id"] in only}
    succ = [c for c in only if c in by and by[c]["success"]]
    ok_runs = [by[c] for c in succ]
    outc = {}
    for r in by.values():
        k = "goal" if r["success"] else {-6: "nav2_aborted", -1: "timeout"}.get(r.get("outcome"), str(r.get("outcome")))
        outc[k] = outc.get(k, 0) + 1
    m = lambda k, rs: float(np.mean([r.get(k) or 0 for r in rs])) if rs else None  # noqa: E731
    return {"n": len(by), "success": len(succ), "controls": sum(c in succ for c in controls), "solved": succ, "outcomes": outc,
            "region_time_s": m("social_region_time_s", list(by.values())), "exposure_success": m("region_exposure", ok_runs),
            "time_to_goal_s": m("time_to_goal_s", ok_runs), "path_m": m("path_length_m", ok_runs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    spec = {v: yaml.safe_load(open(os.path.join(CFG, f"challenge_ctrl_{v}.yaml"))) for v in VARIANTS}
    only = spec[VARIANTS[0]]["only"]
    controls = only[18:]
    BASE = "baseline (challenge_v2: DWB, cost as is)"
    b_runs = json.load(open(args.baseline))["runs"]
    res = {BASE: summarise(b_runs, only, controls)}
    bod = {BASE: summarise(b_runs, only, controls, "bodies")}
    for v in VARIANTS:
        p = os.path.join(args.diag, f"{v}.json")
        if os.path.exists(p):
            runs = json.load(open(p))["runs"]
            res[v], bod[v] = summarise(runs, only, controls), summarise(runs, only, controls, "bodies")
    qual = [v for v in VARIANTS if v in res and res[v]["success"] >= MIN_SUCCESS and res[v]["controls"] >= MIN_CONTROLS
            and bod[v]["success"] >= bod[BASE]["success"]]
    qual.sort(key=lambda v: (-(res[v]["success"] + bod[v]["success"]), res[v]["exposure_success"] if res[v]["exposure_success"] is not None else 1e9))
    pick = qual[0] if qual else None
    f = lambda x, p=1: "–" if x is None else f"{x:.{p}f}"  # noqa: E731
    L = ["# Controller diagnosis: can Nav2 drive the ground-truth social costmap?", "",
         "Ground-truth arm (oracle_ext, extended label) and bodies arm, repeat 0, 24 configurations: the 18 it failed in challenge_v2 and 6 it",
         f"solved (controls: {', '.join(controls)}). Rule fixed in advance: a variant qualifies with >= {MIN_SUCCESS}/24 and >= {MIN_CONTROLS}/6",
         "controls, and the bodies arm no worse than in challenge_v2; the pick has the most ground-truth + bodies successes, then the lowest region exposure.", "",
         "| Setup | runs | success / 24 | controls / 6 | outcomes | time in region (s, all runs) | region exposure (successes) | time to goal (s) | path (m) |",
         "|---|---|---|---|---|---|---|---|---|"]
    desc = {"scale50": "DWB, published cost x 0.5", "obs01": "DWB, BaseObstacle.scale 0.1 (was 0.3)", "mppi": "MPPI",
            "obs01_patient": "DWB, BaseObstacle.scale 0.1 + progress checker 60 s",
            "global": "social cost in the global costmap only (path shaping), Nav2-default local controller", "mppi_scale50": "MPPI, published cost x 0.5"}
    for k, r in res.items():
        L.append(f"| {desc.get(k, k)} | {r['n']} | {r['success']} | {r['controls']} | {', '.join(f'{a} {b}' for a, b in sorted(r['outcomes'].items()))} | "
                 f"{f(r['region_time_s'])} | {f(r['exposure_success'], 2)} | {f(r['time_to_goal_s'])} | {f(r['path_m'])} |")
    L += ["", "Bodies-only arm (no social cost) on the same 24 configurations:", "",
          "| Setup | runs | success / 24 | outcomes | time to goal (s) |", "|---|---|---|---|---|"]
    for k, r in bod.items():
        L.append(f"| {desc.get(k, k)} | {r['n']} | {r['success']} | {', '.join(f'{a} {b}' for a, b in sorted(r['outcomes'].items()))} | {f(r['time_to_goal_s'])} |")
    L += ["", f"**Pick: {desc.get(pick, pick)}**" if pick else "**No variant qualified; the full re-run was not started.**", ""]
    for k, r in res.items():
        L.append(f"- {desc.get(k, k)} solved: {', '.join(r['solved']) or 'none'}")
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    open(args.report, "w").write("\n".join(L) + "\n")
    json.dump({"results": res, "bodies": bod, "pick": pick}, open(os.path.join(args.diag, "selection.json"), "w"), indent=1)
    print("\n".join(L))
    if not pick:
        return 3
    v2 = yaml.safe_load(open(os.path.join(CFG, "challenge_v2.yaml")))
    s = spec[pick]
    v3 = {"output": "/opt/arena_ws/data/socialcost/experiments/navigation/challenge_v3",
          "challenge": v2["challenge"], "controllers": s["controllers"], "overlay": s["overlay"], "estimator_args_all": s["estimator_args_all"],
          "repeats": 1, "methods": v2["methods"][:1] + list(CAM_ARMS) + v2["methods"][1:], "method_specs": {**v2["method_specs"], **CAM_ARMS}, "wall_budget_s": v2["wall_budget_s"]}
    with open(os.path.join(CFG, "challenge_v3.yaml"), "w") as fh:
        fh.write(f"# Full re-run of all arms with the controller setup picked by ctrl_select.py: {desc[pick]}.\n"
                 "# Same arms, thresholds and budget as challenge_v2; the controller / overlay / publish transform apply to every arm.\n")
        yaml.safe_dump(v3, fh, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
