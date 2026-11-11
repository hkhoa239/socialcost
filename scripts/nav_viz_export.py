#!/usr/bin/env python3
"""Export navigation-challenge runs for the bird's-eye replay page (one JSON per configuration + index.json).

Per configuration: occupancy map crop, start / goal, and for every arm (repeat 0 by default) the robot path, the
pedestrian tracks it saw, the outcome and the run metrics, plus the ground-truth extended social cost (extended/2.1:
arena-native + queue membership + halt wedge, no mobility hypothesis) repainted on a map-frame grid once per second.
The cost uses each arm's own pedestrian positions and the per-person attributes (heading, interaction membership,
gestures) of the configuration's reference recording at the aligned time; configurations without a reference get no
cost field. For arms that run stage 1 (typemix*), the predicted pair interaction and halt probabilities are exported.

    python nav_viz_export.py --runs <challenge_v2 dir> --metrics <metrics.json> --out <dir>
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import glob
import io
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.dataset import EXTENDED_CONFIG  # noqa: E402
from socialcost.extensions import ExtParams, paint_extended  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402

CHALLENGE = "/opt/arena_ws/data/socialcost/navigation_challenge_v1"
WORLDS = "/opt/arena_ws/src/Arena/arena_simulation_setup/worlds"
ARMS = ["bodies", "pos_u", "typemix", "typemix_ucb", "typemix_gt", "oracle_ext"]
RES = 0.1
ROBOT_HZ, PED_HZ, PRED_HZ = 5.0, 2.0, 2.0


def png_b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def thin(rows, hz, key=lambda r: r[0]):
    out, last = [], -1e18
    for r in rows:
        if key(r) - last >= 1e9 / hz - 1:
            out.append(r)
            last = key(r)
    return out


def jsonl(path: str) -> list:
    out = []
    for x in open(path):
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:  # a truncated last line
            pass
    return out


def load_reference(cid: str):
    d = os.path.join(CHALLENGE, "references", cid)
    if not os.path.exists(os.path.join(d, "pedestrian_states.jsonl")):
        return None
    samples = jsonl(os.path.join(d, "pedestrian_states.jsonl"))
    frames = jsonl(os.path.join(d, "interactions.jsonl"))
    if not samples:
        return None
    t0 = samples[0]["stamp_ns"]
    return {"t": np.array([(s["stamp_ns"] - t0) / 1e9 for s in samples]), "samples": samples, "frames": frames, "t0": t0}


def ref_at(ref, t_rel: float):
    k = int(np.clip(np.searchsorted(ref["t"], t_rel), 0, len(ref["t"]) - 1))
    s = ref["samples"][k]
    fr = None
    for f in ref["frames"]:
        if f["stamp_ns"] <= s["stamp_ns"]:
            fr = f
    return s, fr


def align(ref, ep) -> float:
    """Time shift d (s) so that reference(t + d) matches the episode's pedestrians at t (episode-relative)."""
    peds = thin(ep["peds"], 1.0)
    if not peds:
        return 0.0
    best, best_d = 1e18, 0.0
    names = {p["name"]: i for i, p in enumerate(ref["samples"][0]["pedestrians"])}
    pos = np.full((len(ref["samples"]), len(names), 2), np.nan)
    for k, s in enumerate(ref["samples"]):
        for p in s["pedestrians"]:
            if p["name"] in names:
                pos[k, names[p["name"]]] = (p["x"], p["y"])
    for d in sorted(np.arange(-30, 30.01, 0.4), key=abs):  # ties (static scenes) keep the smallest shift
        err, n = 0.0, 0
        for t_ns, ps in peds:
            t = (t_ns - ep["start_ns"]) / 1e9 + d
            k = int(np.clip(np.searchsorted(ref["t"], t), 0, len(ref["t"]) - 1))
            for name, x, y, _ in ps:
                j = names.get(name)
                if j is not None and np.isfinite(pos[k, j, 0]):
                    err += np.hypot(pos[k, j, 0] - x, pos[k, j, 1] - y)
                    n += 1
        if n and err / n < best - 1e-3:
            best, best_d = err / n, float(d)
    return best_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    doc = json.load(open(os.path.join(CHALLENGE, "navigation_challenge_v1.json")))
    met = json.load(open(args.metrics))
    mruns = {r["tag"]: r for r in met["runs"]}
    valid = set(met["valid_configs"])
    params = LayerParams.from_overlay()
    ext = dataclasses.replace(ExtParams.load(EXTENDED_CONFIG), mobility=False)
    index = []
    for c in doc["configs"]:
        cid = c["config_id"]
        arms = {}
        for arm in ARMS:
            tag = f"{cid}__{arm}__dwb" + (f"__r{args.repeat}" if args.repeat else "")
            d = os.path.join(args.runs, tag)
            rj = os.path.join(d, "run.json")
            if not os.path.exists(rj):
                continue
            meta = json.load(open(rj))
            epf = os.path.join(d, "episode.jsonl")
            if meta["status"].startswith("failed") or not os.path.exists(epf):
                continue
            arms[arm] = (d, meta, json.loads(open(epf).readline()))
        if not arms:
            continue
        # frames: map = config (world) + offset, from the robot's start pose
        any_ep = next(iter(arms.values()))[2]
        off = np.round((np.array(any_ep["start_pose"][:2]) - np.array(c["robot"]["start"][:2])) * 4) / 4  # scenario -> map frame (5.25 m)
        start = (np.array(c["robot"]["start"][:2]) + off).tolist()
        goal = (np.array(c["robot"]["goal"][:2]) + off).tolist()
        pts = [start, goal]
        for _, _, ep in arms.values():
            pts += [[r[1], r[2]] for r in ep["robot"][::10]]
            pts += [[p[1], p[2]] for _, ps in ep["peds"][::20] for p in ps]
        pts = np.array(pts)
        lo = np.floor((np.percentile(pts, 0, 0) - 2.0) / RES) * RES
        hi = np.ceil((np.percentile(pts, 100, 0) + 2.0) / RES) * RES
        span = hi - lo
        if span.max() > 40:  # keep the view on the scene: centre on route midpoint
            mid = (np.array(start) + np.array(goal)) / 2
            lo, hi = mid - 20, mid + 20
            span = hi - lo
        W, H = int(round(span[0] / RES)), int(round(span[1] / RES))
        grid = GridSpec(resolution=RES, width=W, height=H, origin_x=float(lo[0]), origin_y=float(lo[1]), frame="map")
        # occupancy map crop (rows bottom-up like the grid)
        my = yaml.safe_load(open(os.path.join(WORLDS, c["world"], "map", "0.yaml")))
        img = np.array(Image.open(os.path.join(WORLDS, c["world"], "map", my["image"])).convert("L"))
        mres, (mox, moy) = my["resolution"], my["origin"][:2]
        occ = np.zeros((H, W), np.uint8)
        jj, ii = np.mgrid[0:H, 0:W]
        for sx in (0.25, 0.75):
            for sy in (0.25, 0.75):
                wx = lo[0] + (ii + sx) * RES - off[0]  # the occupancy map is in the scenario frame
                wy = lo[1] + (jj + sy) * RES - off[1]
                pc = np.floor((wx - mox) / mres).astype(int)
                pr = img.shape[0] - 1 - np.floor((wy - moy) / mres).astype(int)
                ok = (pc >= 0) & (pc < img.shape[1]) & (pr >= 0) & (pr < img.shape[0])
                v = np.zeros_like(occ)
                v[ok] = (img[pr[ok], pc[ok]] < 100).astype(np.uint8)
                occ = np.maximum(occ, v)
        ref = load_reference(cid)
        out_arms = {}
        for arm, (d, meta, ep) in arms.items():
            s0 = ep["start_ns"]
            robot = [[round((r[0] - s0) / 1e9, 2), round(r[1], 3), round(r[2], 3), round(r[3], 3)] for r in thin(ep["robot"], ROBOT_HZ)]
            names = sorted({p[0] for _, ps in ep["peds"] for p in ps})
            nidx = {n: i for i, n in enumerate(names)}
            peds = []
            for t, ps in thin(ep["peds"], PED_HZ):
                row = [None] * len(names)
                for n, x, y, _ in ps:
                    row[nidx[n]] = [round(x, 2), round(y, 2)]
                peds.append([round((t - s0) / 1e9, 2), row])
            m = mruns.get(meta["tag"], {})
            arm_out = {"robot": robot, "ped_names": [n.split("/")[-1] for n in names], "peds": peds, "status": meta["status"],
                       "outcome": ep.get("outcome"), "outcome_info": ep.get("outcome_info"),
                       "metrics": {k: m.get(k) for k in ("success", "time_to_goal_s", "path_length_m", "social_region_time_s", "region_exposure",
                                                         "social_exposure", "min_clearance_m", "personal_zone_time_s")}}
            # ground-truth cost once per second
            if ref is not None:
                shift = align(ref, ep)
                frames, fidx, seen = [], [], {}
                T = robot[-1][0] if robot else 0
                ped_by_t = thin(ep["peds"], 1.0)
                for sec in range(int(np.ceil(T)) + 1):
                    tns = s0 + sec * 1e9
                    k = min(range(len(ped_by_t)), key=lambda i: abs(ped_by_t[i][0] - tns)) if ped_by_t else None
                    live = {n: (x, y) for n, x, y, _ in ped_by_t[k][1]} if k is not None else {}
                    s, fr = ref_at(ref, sec + shift)
                    states = []
                    for p in s["pedestrians"]:
                        q = dict(p)
                        if p["name"] in live:
                            q["x"], q["y"] = live[p["name"]]
                        states.append(q)
                    cost, _ = paint_extended(params, ext, states, fr, grid, (0.0, 0.0, 0.0))
                    key = cost.tobytes()
                    if key not in seen:
                        seen[key] = len(frames)
                        frames.append(png_b64(cost[::-1]))
                    fidx.append(seen[key])
                arm_out["cost"] = {"frames": frames, "index": fidx, "shift_s": shift}
            # stage-1 predictions
            el = os.path.join(d, "estimator.jsonl")
            if arm.startswith("typemix") and arm != "typemix_gt" and os.path.exists(el):
                preds = []
                rows = [json.loads(x) for x in open(el) if x.strip()]
                rows = [r for r in rows if "stage1" in r and r["t_ns"] >= s0]
                for r in thin(rows, PRED_HZ, key=lambda r: r["t_ns"]):
                    st = r["stage1"]
                    nm = st["names"]
                    pairs = [[nm[i], nm[j], round(st["p_inter"][i][j], 2)] for i in range(len(nm)) for j in range(i + 1, len(nm))
                             if st["p_inter"][i][j] >= 0.05]
                    halt = {nm[i]: round(v, 2) for i, v in enumerate(st["p_halt"]) if v >= 0.05}
                    preds.append([round((r["t_ns"] - s0) / 1e9, 2), pairs, halt])
                arm_out["stage1"] = preds
            out_arms[arm] = arm_out
        rec = {"config_id": cid, "kind": c["kind"], "world": c["world"], "valid": cid in valid, "has_cost": ref is not None,
               "grid": {"x0": float(lo[0]), "y0": float(lo[1]), "res": RES, "w": W, "h": H}, "map": png_b64(occ[::-1] * 255),
               "start": start, "goal": goal, "route_a_m": c["alternative_routes"]["route_a_m"], "route_b_m": c["alternative_routes"]["route_b_m"],
               "scene": {k: c["scene"].get(k) for k in ("interaction", "core_participants", "formation", "mode")}, "arms": out_arms}
        json.dump(rec, open(os.path.join(args.out, f"{cid}.json"), "w"), separators=(",", ":"))
        index.append({"config_id": cid, "kind": c["kind"], "world": c["world"], "valid": cid in valid, "has_cost": ref is not None,
                      "arms": {a: {"success": v["metrics"]["success"], "status": v["status"]} for a, v in out_arms.items()}})
        print(cid, c["kind"], len(out_arms), "arms", "cost" if ref else "no-cost", f"{os.path.getsize(os.path.join(args.out, f'{cid}.json')) / 1e3:.0f} kB",
              flush=True)
    json.dump({"configs": index, "repeat": args.repeat, "res": RES}, open(os.path.join(args.out, "index.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
