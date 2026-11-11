#!/usr/bin/env python3
"""navigation_challenge_v1: 36 fixed navigation configurations with verified route alternatives.

    python dv1_nav_challenge.py --out /opt/arena_ws/data/socialcost/navigation_challenge_v1 [--build]

Per configuration (scene built with the dataset_v1 scene builders, fixed seed):
  * Route A: shortest bodies-only path (map + furniture + pedestrian bodies, robot margin);
  * expected social region: cells where an interaction term (conversation, queue membership, halt wedge)
    of extended/2.1 is non-zero, painted in the world frame from the planned pedestrian states;
  * Route B: shortest path that avoids the social region;
  * validity: both routes exist, Route A crosses the region, Route B is 2-60 % longer than A, and an A*
    weighted by the social costmap (weight 4 per unit cost per metre) takes a path that does not cross the region,
    so the social layer changes the decision and geometry alone does not force it.
Writes scenarios/<id>/config.json + route figure, the Arena scenario scv1_nav_<id> and the robot-free
reference scenario scv1_nav_<id>_ref (robot parked >= 10 m away), and navigation_challenge_v1.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from socialcost.dv1 import scenes, spec  # noqa: E402
from socialcost.dv1.geometry import Route, World  # noqa: E402
from socialcost.dv1.scenario import write_episode  # noqa: E402
from socialcost.extensions import ExtParams, halt_term  # noqa: E402
from socialcost.labels import GridSpec, LayerParams, Ped, paint  # noqa: E402

ARENA = os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena")
OVERLAY = os.path.join(ARENA, "arena_robots/arena_robots/config/nav2/overlays/social_cost_layer.yaml")
QUOTA = {"conversation": 8, "halt": 6, "queue": 8, "group_walking": 6, "occlusion_mixed": 8}
WORLD_CYCLE = ["C", "A", "B", "D"]
ROBOT_MARGIN = 0.30
SOCIAL_WEIGHT = 4.0
# parking routes for robot-free references (world frame), far from every challenge site
PARK = {"A": [(40.0, 17.5), (42.0, 17.5)], "B": [(10.0, 32.5), (10.0, 33.5)], "C": [(43.0, 2.0), (43.0, 5.0)], "D": [(3.0, 21.0), (6.0, 21.0)]}
SITE_KINDS = {"conversation": ("open", "corridor", "aisle"), "halt": ("corridor", "open", "aisle"), "queue": ("open", "corridor", "aisle"),
              "group_walking": ("open", "aisle", "corridor"), "occlusion_mixed": ("open", "corridor", "aisle")}


def world_grid(w: World) -> tuple[GridSpec, tuple[float, float, float]]:
    H, W = w.occ.shape
    return GridSpec(resolution=w.resolution, width=W, height=H, origin_x=w.origin[0], origin_y=w.origin[1], frame="world"), (0.0, 0.0, 0.0)


def social_region(w: World, params: LayerParams, humans: list[dict], cond: str, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(interaction-term cost, full extended cost) in png orientation (row 0 = top), from planned states at time t."""
    grid, gp = world_grid(w)
    states = planned_states(humans, cond, t)
    nop = __import__("dataclasses").replace(params, personal_space_radius=1e-6)
    inter = [Ped(s["x"], s["y"], 0, s["iid"], s["itype"]) for s in states if s["iid"] >= 0]
    term = paint(nop, inter, grid, gp)
    full = paint(params, [Ped(s["x"], s["y"], 0, s["iid"], s["itype"]) for s in states], grid, gp)
    ext = ExtParams(membership=True, halt=True, mobility=False)
    for s in states:
        if s["halt"]:
            p = {"x": s["x"], "y": s["y"], "yaw": s["yaw"], "gestures": [{"slot": "halt", "at": {"x": s["x"] + math.cos(s["yaw"]), "y": s["y"] + math.sin(s["yaw"])}}]}
            h = halt_term(params, ext, p, grid, gp)
            term = np.maximum(term, h)
            full = np.maximum(full, h)
    return np.flipud(term), np.flipud(full)


def meeting_time(sc: scenes.Scene, route: Route) -> float:
    """Time at which the planned group leader is closest to the robot's planned position (0.29 m/s, s = 2.3 m at the window start)."""
    best = (1e9, spec.WARMUP_S)
    for t in np.arange(spec.WARMUP_S, spec.WARMUP_S + 50, 0.5):
        s_r = spec.ROBOT_S_WINDOW[0] + spec.ROBOT_SPEED * (t - spec.WARMUP_S)
        rx, ry, _ = route.at(min(s_r, route.length))
        lead = planned_states([h for h in sc.humans if h["role"] == "leader"], "on", t)[0]
        d = math.dist((rx, ry), (lead["x"], lead["y"]))
        if d < best[0]:
            best = (d, float(t))
    return best[1]


def group_hull(w: World, states: list[dict], inflate: float) -> np.ndarray:
    """Benchmark region of a walking group (GROUP_WALK has native weight 0): member convex hull inflated by `inflate` m."""
    import cv2
    m = np.zeros(w.occ.shape, np.uint8)
    pts = np.array([[w.to_px(s["x"], s["y"])[1], w.to_px(s["x"], s["y"])[0]] for s in states], np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    k = int(inflate / w.resolution)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return m.astype(np.uint8) * 100


def planned_states(humans: list[dict], cond: str, t: float) -> list[dict]:
    """Planned pedestrian states at time t (static poses, straight-line motion along go_to legs, interaction flags)."""
    out = []
    for h in humans:
        b = h["behavior"][cond]
        x, y, yaw = h["spawn"]
        clock, iid, itype, halt = 0.0, -1, 0, False
        sp = h.get("speed") or 1.0
        for st in b["steps"]:
            if "wait" in st:
                clock += st["wait"]
            elif "goto" in st:
                tx, ty, th = st["goto"]
                d = math.dist((x, y), (tx, ty))
                if t < clock + d / sp:
                    f = (t - clock) * sp / d if d > 0 and t > clock else 0.0
                    x, y = x + (tx - x) * f, y + (ty - y) * f
                    break
                clock += d / sp
                x, y, yaw = tx, ty, th
            elif "interact" in st and t >= clock:
                iid, itype = 900, (0 if st["interact"] == "TALK_TO" else 1)
                if t > clock + st["duration"]:
                    iid = -1
                clock += st["duration"]
            elif "queue" in st and t >= clock:
                iid, itype = 901, 6
            elif "halt" in st and t >= clock:
                halt = t <= clock + st["duration"]
                clock += st["duration"]
            if clock > t:
                break
        out.append({"name": h["name"], "x": x, "y": y, "yaw": yaw, "iid": iid, "itype": itype, "halt": halt})
    return out


def body_world(w: World, humans: list[dict], cond: str, t: float, radius: float) -> World:
    ww = World(w.name, w.origin, w.resolution, w.occ.copy(), w.objects)
    for s in planned_states(humans, cond, t):
        r0, c0 = ww.to_px(s["x"], s["y"])
        rr = int(radius / w.resolution) + 1
        for dr in range(-rr, rr + 1):
            for dc in range(-rr, rr + 1):
                if dr * dr + dc * dc <= rr * rr and 0 <= r0 + dr < ww.occ.shape[0] and 0 <= c0 + dc < ww.occ.shape[1]:
                    ww.occ[r0 + dr, c0 + dc] = True
    return ww


def crosses(path: list[tuple[float, float]], w: World, region: np.ndarray) -> int:
    n = 0
    for x, y in path:
        r, c = w.to_px(x, y)
        if 0 <= r < region.shape[0] and 0 <= c < region.shape[1] and region[r, c] > 0:
            n += 1
    return n


def build_scene(kind: str, world_key: str, worlds: dict, rng: random.Random) -> tuple[str, scenes.Scene, dict] | None:
    sites = [s for s, (wk, k, _) in spec.SITES.items() if wk == world_key and k in SITE_KINDS[kind]]
    site = rng.choice(sites)
    wk, sk, pts = spec.SITES[site]
    route = Route(pts)
    sc = scenes.Scene(worlds[wk], route, rng, ("on",))
    info = {"site": site, "site_kind": sk}
    if kind == "conversation":
        n = rng.choice([2, 2, 3, 4])
        scenes.build_conversation(sc, n, "static")
        if n == 2 and not sc.meta["gap_on_route"]:
            raise scenes.Reject("needs_gap")
        if n > 2 and abs(route_lateral(route, sc.meta["group_centre"])) > 0.6:
            raise scenes.Reject("group_off_route")
    elif kind == "halt":
        scenes.build_stop(sc, rng.choice([1, 1, 2]), rng.choice(["frontal", "oblique_30"]), "active_from_start")
    elif kind == "queue":
        scenes.build_queue(sc, rng.randint(4, 6), "sparse", "perpendicular", False)
        # robot route must cross the queue line: tail beyond the route centre line
        tail = sc.humans[-1]["spawn"]
        if route_lateral(route, tail[:2]) * route_lateral(route, sc.meta["desk"][:2]) > 0:
            raise scenes.Reject("queue_not_across")
    elif kind == "group_walking":
        scenes.build_group_walk(sc, rng.randint(3, 5), "opposite", rng.choice(["side_by_side", "loose", "v_formation"]), safe_goal=True)
    else:
        sub = rng.choice(["occluded_conversation", "conversation_plus_halt", "occluded_queue_tail"])
        info["mixed"] = sub
        if sub == "occluded_conversation":
            scenes.build_conversation(sc, 2, "static")
            if not sc.meta["gap_on_route"]:
                raise scenes.Reject("needs_gap")
            s_g = sc.meta["group_s"]
            side = rng.choice([-1, 1])
            lo, hi = sc.lat(s_g - 2.5, 0.8)
            v = min(max(side * 1.0, lo), hi)
            ox, oy = sc.w(s_g - 2.5, v)
            sc.add_object("occluder", rng.choice(spec.OCCLUDERS), ox, oy, route.heading(s_g))
        elif sub == "conversation_plus_halt":
            scenes.build_conversation(sc, 2, "static")
            if not sc.meta["gap_on_route"]:
                raise scenes.Reject("needs_gap")
            s_h = sc.meta["group_s"] + rng.uniform(2.5, 3.5)
            lo, hi = sc.lat(s_h)
            x, y = sc.w(s_h, min(max(rng.choice([-1, 1]) * 1.2, lo), hi))
            sc.add("primary", (x, y, sc.toward_robot(s_h)), {"on": {"steps": [{"halt": {"azimuth": 0, "elevation": 10}, "duration": 1000.0}]}})
        else:
            scenes.build_queue(sc, rng.randint(4, 5), "sparse", "perpendicular", False)
            tail = sc.humans[-1]["spawn"]
            if route_lateral(route, tail[:2]) * route_lateral(route, sc.meta["desk"][:2]) > 0:
                raise scenes.Reject("queue_not_across")
            s_q = sc.meta["desk"]
            lo, hi = sc.lat(4.0, 0.8)
            ox, oy = sc.w(4.5, min(max(rng.choice([-1, 1]) * 1.1, lo), hi))
            sc.add_object("occluder", rng.choice(spec.OCCLUDERS), ox, oy, route.heading(4.5))
    info.update(sc.meta)
    return site, sc, info


def route_lateral(route: Route, p) -> float:
    best = (1e9, 0.0)
    for s in np.arange(0, route.length, 0.1):
        x, y, h = route.at(s)
        d = math.dist((x, y), p)
        if d < best[0]:
            best = (d, -math.sin(h) * (p[0] - x) + math.cos(h) * (p[1] - y))
    return best[1]


def evaluate(w: World, params: LayerParams, sc: scenes.Scene, route: Route, kind: str) -> dict | None:
    # evaluation time: when the robot (0.29 m/s) reaches the scene, or the static state
    t_eval = spec.WARMUP_S + 12.0
    if kind == "group_walking":
        t_eval = meeting_time(sc, route)
    term, full = social_region(sc._occ, params, sc.humans, "on", t_eval)
    if kind == "group_walking":
        term = group_hull(sc._occ, planned_states(sc.humans, "on", t_eval), 0.8)
    if not term.any():
        return None
    occ_w = body_world(sc._occ, sc.humans, "on", t_eval, 0.30)
    a, b = route.pts[0], route.pts[-1]
    pa = occ_w.path_length(a, b, ROBOT_MARGIN)
    if pa is None:
        return None
    blocked = World(occ_w.name, occ_w.origin, occ_w.resolution, occ_w.occ | (term > 0), occ_w.objects)
    pb = blocked.path_length(a, b, ROBOT_MARGIN)
    cost = full.astype(float) / 252.0
    ps = occ_w.path_length(a, b, ROBOT_MARGIN, cost=cost, cost_weight=SOCIAL_WEIGHT)
    if pb is None or ps is None:
        return None
    ca, cb, cs = crosses(pa[1], occ_w, term), crosses(pb[1], occ_w, term), crosses(ps[1], occ_w, term)
    ratio = pb[0] / pa[0]

    def clearance(path):
        return round(min(occ_w.clearance(x, y) for x, y in path), 3)
    res = {"t_eval_s": t_eval, "route_a_m": round(pa[0], 2), "route_b_m": round(pb[0], 2), "detour_ratio": round(ratio, 3),
           "route_a_region_cells": ca, "route_b_region_cells": cb, "social_planner_region_cells": cs, "social_planner_m": round(ps[0], 2),
           "route_a_min_clearance_m": clearance(pa[1]), "route_b_min_clearance_m": clearance(pb[1]),
           "region_area_m2": round(float((term > 0).sum()) * w.resolution ** 2, 2),
           "route_a": [[round(x, 2), round(y, 2)] for x, y in pa[1]], "route_b": [[round(x, 2), round(y, 2)] for x, y in pb[1]],
           "social_planner_path": [[round(x, 2), round(y, 2)] for x, y in ps[1]]}
    social_ok = cs == 0 if kind != "group_walking" else cs < ca
    lo = 1.02 if kind != "group_walking" else 1.005  # passing an oncoming group close vs giving room: small detours in open space
    res["valid"] = bool(ca > 0 and cb == 0 and lo <= ratio <= 1.6 and social_ok)
    res["_term"], res["_world"] = term, occ_w
    return res


def figure(path: str, w: World, cfg: dict, ev: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    term, ow = ev["_term"], ev["_world"]
    pts = np.array(ev["route_a"] + ev["route_b"])
    x0, y0 = pts.min(0) - 3
    x1, y1 = pts.max(0) + 3
    fig, ax = plt.subplots(figsize=(6, 5))
    ext = [w.origin[0], w.origin[0] + w.occ.shape[1] * w.resolution, w.origin[1], w.origin[1] + w.occ.shape[0] * w.resolution]
    ax.imshow(ow.occ, cmap="Greys", extent=ext, vmin=0, vmax=1.5)
    ax.imshow(np.ma.masked_where(term == 0, term), cmap="Oranges", extent=ext, alpha=0.7, vmin=0, vmax=252)
    for key, col, lab in (("route_a", "#2a78d6", "Route A (bodies-only shortest)"), ("route_b", "#1baf7a", "Route B (avoids social region)")):
        p = np.array(ev[key])
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2, label=f"{lab} {ev[key + '_m']} m")
    for h in cfg["humans"]:
        ax.plot(h["spawn"][0], h["spawn"][1], "o", color="#0b0b0b", ms=5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"{cfg['config_id']} {cfg['kind']} ({cfg['world']}) detour x{ev['detour_ratio']}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed", type=int, default=7701)
    args = ap.parse_args()
    params = LayerParams.from_overlay(OVERLAY, "local_costmap")
    worlds = {k: World.load(v["name"]) for k, v in spec.WORLDS.items()}
    pools = json.load(open(os.path.join(os.path.dirname(args.out.rstrip("/")), "dataset_v1", "metadata", "body_pools.json")))
    bodies = pools["known_walking"]
    os.makedirs(os.path.join(args.out, "scenarios"), exist_ok=True)
    configs, rejected = [], {}
    idx = 0
    for kind, q in QUOTA.items():
        for i in range(q):
            idx += 1
            cid = f"N{idx:02d}"
            planned_wk = WORLD_CYCLE[(idx - 1) % 4]
            order = WORLD_CYCLE[(idx - 1) % 4:] + WORLD_CYCLE[:(idx - 1) % 4]
            rng = random.Random(args.seed * 100 + idx)
            for attempt in range(1200):
                wk = order[min(attempt // 300, 3)]
                try:
                    built = build_scene(kind, wk, worlds, rng)
                except scenes.Reject as e:
                    rejected[str(e)] = rejected.get(str(e), 0) + 1
                    continue
                site, sc, info = built
                ev = evaluate(worlds[wk], params, sc, sc.route, kind)
                if ev and ev["valid"]:
                    break
                rejected["route_test"] = rejected.get("route_test", 0) + 1
            else:
                raise RuntimeError(f"{cid} {kind} in {wk}: no valid configuration")
            for j, h in enumerate(sc.humans):
                h["body"] = bodies[(idx * 7 + j * 3) % len(bodies)]
                h["identity"] = h["body"]
                h["behavior"] = {"on": h["behavior"]["on"]}
                h["agent_type"] = {"on": h["agent_type"]["on"]}
            r = sc.route
            horizon = round(spec.WARMUP_S + 1.6 * r.length / spec.ROBOT_SPEED, 1)
            cfg = {"config_id": cid, "kind": kind, "world_key": wk, "planned_world_key": planned_wk, "world": spec.WORLDS[wk]["name"], "site": site, "run_seed": 5000 + idx,
                   "robot": {"start": [*r.pts[0], round(r.heading(0), 4)], "goal": [*r.pts[-1], round(r.heading(r.length), 4)], "route": [list(p) for p in r.pts]},
                   "humans": sc.humans, "objects": sc.objects, "scene": info,
                   "interaction_states": sorted({str(s.get("interact") or ("QUEUE_USE" if "queue" in s else "halt" if "halt" in s else "GROUP_WALK" if "group_form" in s else ""))
                                                 for h in sc.humans for s in h["behavior"]["on"]["steps"]} - {""}),
                   "alternative_routes": {k: v for k, v in ev.items() if not k.startswith("_")},
                   "expected_social_region": {"definition": ("convex hull of the walking group inflated by 0.8 m at the planned meeting time "
                                                             "(benchmark region; GROUP_WALK has native weight 0 and no label term)") if kind == "group_walking" else
                                                            "cells where an interaction term of extended/2.1 (conversation, queue membership, halt wedge) "
                                                            "is non-zero at t_eval, world frame", "area_m2": ev["region_area_m2"]},
                   "evaluation_horizon_s": horizon,
                   "reference": {"scenario": f"scv1_nav_{cid}_ref", "robot_parked": PARK[wk], "note": "same pedestrians, robot parked >= 10 m away: "
                                 "robot-free pedestrian trajectories for disturbance metrics"}}
            d = os.path.join(args.out, "scenarios", cid)
            os.makedirs(d, exist_ok=True)
            json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
            figure(os.path.join(d, "routes.png"), worlds[wk], cfg, ev)
            ep = {"episode_id": f"nav_{cid}", "world": cfg["world"], "condition": "on", "family": f"nav_{kind}", "humans": sc.humans, "objects": sc.objects,
                  "robot": cfg["robot"]}
            write_episode(ep)
            ref = copy.deepcopy(ep)
            ref["episode_id"] = f"nav_{cid}_ref"
            ref["robot"] = {"start": [*PARK[wk][0], 0.0], "goal": [*PARK[wk][1], 0.0]}
            write_episode(ref)
            configs.append(cfg)
            print(f"{cid} {kind:16s} {wk} {site:12s} A {ev['route_a_m']:5.1f} m  B {ev['route_b_m']:5.1f} m  x{ev['detour_ratio']:.2f}  attempts {attempt + 1}")
    json.dump({"version": "navigation_challenge_v1", "configs": configs, "quota": QUOTA, "rejections": rejected,
               "validity_rule": "Route A crosses the social region, Route B avoids it and is 2-60 % longer (group walking: 0.5-60 %), the social-cost planner "
                                    "avoids the region (group walking: crosses fewer region cells than Route A)",
               "park_check": {k: min((math.dist(PARK[k][0], c["robot"]["start"][:2]) for c in configs if c["world_key"] == k), default=None) for k in PARK},
               "per_world": {k: sum(c["world_key"] == k for c in configs) for k in PARK}},
              open(os.path.join(args.out, "navigation_challenge_v1.json"), "w"), indent=1)
    if args.build:
        from dv1_make_scenarios import build
        build()
    print(f"{len(configs)} configurations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
