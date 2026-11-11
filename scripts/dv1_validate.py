#!/usr/bin/env python3
"""dataset_v1 automatic validation (spec section 16): integrity, synchronization, counterfactual validity,
social semantics, visibility, generalization. Also writes the counterfactual pair classification.

    python dv1_validate.py --root /opt/arena_ws/data/socialcost/dataset_v1 --stage subset|pilot|remaining|all --out <dir>

Writes <out>/validation.json, <out>/validation.md, <out>/pairs.json, <out>/rejected.json (episodes to re-record).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from socialcost.dataset import read_jsonl  # noqa: E402
from socialcost.dv1 import spec  # noqa: E402
from socialcost.dv1.post import _body_id  # noqa: E402
from socialcost.pairs import compare_pair  # noqa: E402
from socialcost.verify import Ep  # noqa: E402

TALK = {0, 1}
WEIGHTED_NONQUEUE = {0: 0.9, 1: 1.2, 3: 0.8, 4: 0.8, 5: 0.8, 7: 0.6, 8: 0.8, 9: 0.8, 10: 1.5, 11: 0.8}
STATIONARY_SPEED = 0.05  # m/s
ROBOT_CONTACT_M = 1.0  # matched windows end once the robot passes this close to a person


def check(ok: bool, name: str, detail=None) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def ped_index(states: list[dict]) -> dict:
    return {p["name"].split("/")[-1]: p for p in states}


# ------------------------------------------------------------------------------------------------ episode checks

def integrity(ep_dir: str, meta: dict, entry: dict) -> list[dict]:
    out = []
    n = meta["samples"]
    for d in ("rgb", "depth", "social_cost", "social_cost_membership", "social_cost_extended", "components"):
        p = os.path.join(ep_dir, d)
        c = len(os.listdir(p)) if os.path.isdir(p) else 0
        out.append(check(c == n, f"count_{d}", {"files": c, "samples": n}))
    for f in ("robot_states.jsonl", "pedestrian_states.jsonl", "sync.jsonl", "dv1/visibility.jsonl"):
        c = len(read_jsonl(os.path.join(ep_dir, f)))
        out.append(check(c == n, f"rows_{f}", {"rows": c, "samples": n}))
    bad_depth, bad_cost, bad_comp = [], [], []
    for k in range(0, n, 5):
        try:
            d = np.load(os.path.join(ep_dir, "depth", f"{k:06d}.npy"))
            fin = d[np.isfinite(d)]
            if d.shape != (240, 320) or fin.size < 0.2 * d.size or fin.min() < 0.29 or fin.max() > 101:
                bad_depth.append(k)
            for ch in ("social_cost", "social_cost_membership", "social_cost_extended"):
                c = np.load(os.path.join(ep_dir, ch, f"{k:06d}.npy"))
                if c.shape != (200, 200) or c.dtype != np.uint8 or c.max() > 252:
                    bad_cost.append((ch, k))
            z = np.load(os.path.join(ep_dir, "components", f"{k:06d}.npz"))
            if set(z.files) != {"personal_space", "conversation", "queue", "gesture", "appearance_or_geometry_extension", "combined_native",
                                "combined_extended", "body_occupancy"}:
                bad_comp.append(k)
        except Exception as e:  # noqa: BLE001
            bad_depth.append((k, str(e)))
    out.append(check(not bad_depth, "depth_valid", bad_depth[:10]))
    out.append(check(not bad_cost, "costmap_valid", bad_cost[:10]))
    out.append(check(not bad_comp, "components_valid", bad_comp[:10]))
    # metadata vs manifest
    first = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))[:1]
    rec_bodies = sorted({_body_id(p.get("model_uri", "")) for p in first[0]["pedestrians"]}) if first else []
    out.append(check(meta.get("scenario") == entry["scenario"] and meta.get("world") == entry["world"] and str(meta.get("run_seed")) == str(entry["run_seed"]),
                     "metadata_matches_manifest", {"scenario": meta.get("scenario"), "world": meta.get("world"), "seed": meta.get("run_seed")}))
    out.append(check(rec_bodies == sorted(entry["bodies"]), "bodies_match_manifest", {"recorded": rec_bodies, "manifest": entry["bodies"]}))
    out.append(check(len(first[0]["pedestrians"]) == entry["n_peds"] if first else entry["n_peds"] == 0, "ped_count_matches", {"recorded": len(first[0]["pedestrians"]) if first else 0, "manifest": entry["n_peds"]}))
    out.append(check(set(meta.get("label_definitions", {})) >= {"social_cost", "social_cost_membership"}, "label_definitions_present", meta.get("label_definitions")))
    return out


def synchronization(ep_dir: str, meta: dict, entry: dict, summary: dict) -> list[dict]:
    out = []
    sync = read_jsonl(os.path.join(ep_dir, "sync.jsonl"))
    robots = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))
    peds = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))
    out.append(check(all(s["rgb_stamp_ns"] == s["depth_stamp_ns"] == s["stamp_ns"] for s in sync), "rgb_depth_aligned",
                     max((abs(s["rgb_stamp_ns"] - s["depth_stamp_ns"]) for s in sync), default=0)))
    out.append(check(all(r["stamp_ns"] == s["stamp_ns"] for r, s in zip(robots, sync)), "robot_pose_at_image_stamp"))
    br = [max(abs(p["stamp_ns"] - p[k]) for k in ("peds_before_ns", "peds_after_ns") if p.get(k) is not None) for p in peds
          if p.get("peds_before_ns") is not None or p.get("peds_after_ns") is not None]
    out.append(check(all(b <= 100_000_000 for b in br), "pedestrian_states_bracketed_100ms", max(br) / 1e6 if br else None))
    slots = [r["slot"] for r in robots]
    gaps = sum(b - a - 1 for a, b in zip(slots, slots[1:])) if slots else 99
    out.append(check(bool(slots) and gaps <= 1, "consecutive_5hz_slots", {"skipped_slots": gaps}))  # one dropped frame tolerated
    st = meta["stats"]
    out.append(check(st.get("time_jumps", 0) == 0 and st.get("end_reason") == "duration" and meta["samples"] >= meta["expected_samples"] - 1, "episode_boundaries",
                     {"end_reason": st.get("end_reason"), "samples": meta["samples"], "expected": meta["expected_samples"], "time_jumps": st.get("time_jumps")}))
    # interaction-state timing against the manifest plan
    fam, cond, sc = entry["family"], entry["condition"], entry["scene"]
    if fam == "stop_gesture" and cond == "on":
        iv = [g for g in summary["gesture_intervals"]]
        plan_on = sc["gesture_t_on_s"]
        got = iv[0]["on_s"] if iv else None
        out.append(check(bool(iv) and (plan_on == 0.0 or abs(got - plan_on) <= 1.5), "gesture_onset_vs_plan", {"planned_s": plan_on, "observed_s": got, "intervals": iv}))
        if sc.get("gesture_t_off_s"):
            off = iv[0].get("off_s") if iv else None
            out.append(check(off is not None and abs(off - sc["gesture_t_off_s"]) <= 1.5, "gesture_offset_vs_plan", {"planned_s": sc["gesture_t_off_s"], "observed_s": off}))
    if fam == "stop_gesture" and cond == "off":
        out.append(check(not summary["gesture_intervals"], "no_gesture_in_off", summary["gesture_intervals"]))
    if fam == "conversation" and cond == "on" and entry["subtype"] == "lifecycle":
        tl = summary["interaction_timeline"]
        starts = [t["t_s"] for t in tl if any(a["type"] in TALK for a in t["active"])]
        out.append(check(bool(starts) and abs(starts[0] - sc["t_on"]) <= 2.0, "conversation_onset_vs_plan", {"planned_s": sc["t_on"], "observed_s": starts[0] if starts else None}))
    return out


def semantics(ep_dir: str, entry: dict, summary: dict) -> list[dict]:
    out = []
    out.append(check(summary["components"]["combined_native_mismatch_samples"] == 0, "combined_native_equals_social_cost",
                     summary["components"]["combined_native_mismatch_samples"]))
    out.append(check(summary["components"]["combined_extended_ge_native_samples"] == summary["components"]["samples"], "extended_ge_native"))
    peds = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))
    robots = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))
    peak_bad, stale, missing, gw_bad, halt_bad, queue_bad = [], [], [], [], [], []
    for k in range(0, len(peds), 2):
        states, r = peds[k]["pedestrians"], robots[k]["pose_map_base"]
        z = np.load(os.path.join(ep_dir, "components", f"{k:06d}.npz"))
        c, s = math.cos(r["yaw"]), math.sin(r["yaw"])

        def cell(x, y):
            dx, dy = x - r["x"], y - r["y"]
            u, v = c * dx + s * dy, -s * dx + c * dy
            i, j = int((u + 5) / 0.05), int((v + 5) / 0.05)
            return (j, i) if 2 <= i < 198 and 2 <= j < 198 else None

        for p in states:
            cc = cell(p["x"], p["y"])
            if cc and p["animation_state"] in (0, 1, 2) and int(z["personal_space"][cc]) not in (112, 113):
                peak_bad.append((k, p["name"].split("/")[-1], int(z["personal_space"][cc])))
        groups = defaultdict(list)
        for p in states:
            if p["interaction_id"] >= 0 and p["interaction_type"] != 6:
                groups[p["interaction_id"]].append(p)
        painted = [g for g in groups.values() if len(g) >= 2 and WEIGHTED_NONQUEUE.get(g[0]["interaction_type"], 0) > 0]
        conv_any = bool(z["conversation"].any())
        if conv_any and not painted:
            stale.append(k)
        for g in painted:
            a, b = g[0], g[1]
            cc = cell((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
            if cc and int(z["conversation"][cc]) < 150:
                missing.append((k, int(z["conversation"][cc])))
        if any(p["interaction_type"] == 12 and p["interaction_id"] >= 0 for p in states) and not painted and conv_any:
            gw_bad.append(k)
        halters = [p for p in states if any(isinstance(gg, dict) and str(gg.get("slot", "")).startswith("halt") for gg in p.get("gestures") or [])]
        # a halt wedge can only appear when the gesturing person is inside the 10 m robot-centred grid
        if any(cell(p["x"], p["y"]) for p in halters) and not z["gesture"].any():
            halt_bad.append((k, "halt inside grid, no gesture term"))
        if not halters and z["gesture"].any():
            halt_bad.append((k, "gesture term without halt"))
        if entry["family"] == "queue" and entry["condition"] == "off" and z["queue"].any():
            queue_bad.append(k)
    out.append(check(not peak_bad, "personal_space_peak_113", peak_bad[:8]))
    out.append(check(not stale, "no_stale_interaction_cost", stale[:8]))
    out.append(check(not missing, "interaction_capsule_present", missing[:8]))
    out.append(check(not gw_bad, "group_walk_no_capsule", gw_bad[:8]))
    out.append(check(not halt_bad, "gesture_term_iff_halt", halt_bad[:8]))
    out.append(check(not queue_bad, "no_queue_cost_in_off", queue_bad[:8]))
    fam, cond = entry["family"], entry["condition"]
    swc = summary["components"]["samples_with_cost"]
    if fam == "queue" and cond == "on":
        out.append(check(swc.get("queue", 0) > 0, "queue_membership_term_present", swc.get("queue")))
    if fam == "conversation" and cond == "on":
        out.append(check(swc.get("conversation", 0) > 0, "conversation_term_present", swc.get("conversation")))
    if fam == "conversation" and cond == "off":
        out.append(check(swc.get("conversation", 0) == 0, "no_conversation_term_in_off", swc.get("conversation")))
    if fam == "controls":
        out.append(check(swc.get("conversation", 0) == 0 and swc.get("queue", 0) == 0 and swc.get("gesture", 0) == 0, "controls_no_interaction_cost", swc))
        if entry["subtype"] == "empty":
            out.append(check(swc.get("combined_extended", 0) == 0, "empty_world_label_empty", swc.get("combined_extended")))
    return out


def visibility(ep_dir: str, entry: dict) -> list[dict]:
    out = []
    rows = read_jsonl(os.path.join(ep_dir, "dv1", "visibility.jsonl"))
    half = spec.CAMERA_HFOV_DEG / 2
    fov_bad, hidden_bad = [], []
    for r in rows:
        for p in r["pedestrians"]:
            b = p["bearing_deg"]
            if p["state"] == "visible" and b is not None and abs(b) > half + 12:
                fov_bad.append((r["sample"], p["name"], b))
            if p["state"] in ("outside_fov", "fully_occluded") and (p["visibility_fraction"] > 0 or p["visible"]):
                hidden_bad.append((r["sample"], p["name"]))
            if p["state"] == "outside_fov" and b is not None and abs(b) < half - 12 and p["distance_m"] > 1.0 and p["in_image_fraction"] > 0:
                fov_bad.append((r["sample"], p["name"], b, "outside_but_central"))
    out.append(check(not fov_bad, "fov_membership_consistent", fov_bad[:8]))
    out.append(check(not hidden_bad, "hidden_not_marked_visible", hidden_bad[:8]))
    cat = entry["visibility_category"]
    if cat != "natural":
        tgt = next(h["name"] for h in entry["humans"] if h["role"] == "target")
        seq = [next((p for p in r["pedestrians"] if p["name"] == tgt), None) for r in rows]
        states = [p["state"] if p else None for p in seq]
        seen = [s in ("visible", "partially_occluded") for s in states]
        first_seen = next((i for i, v in enumerate(seen) if v), None)
        exit_i = next((i for i in range(first_seen + 1, len(states)) if states[i] == "outside_fov"), None) if first_seen is not None else None
        if cat == "fov_exit":
            out.append(check(exit_i is not None, "fov_exit_observed", {"first_seen": first_seen, "exit_sample": exit_i, "states": dict(Counter(states))}))
        elif cat == "fov_reentry":
            # seen, then outside the FOV, then seen again; fully occluded / unseen samples may sit in between
            re_i = next((i for i in range(exit_i + 1, len(states)) if seen[i]), None) if exit_i is not None else None
            out.append(check(exit_i is not None and re_i is not None, "fov_reentry_observed",
                             {"first_seen": first_seen, "exit_sample": exit_i, "reentry_sample": re_i, "states": dict(Counter(states))}))
        elif cat == "static_occlusion":
            occ = sum(s in ("fully_occluded", "partially_occluded") for s in states)
            out.append(check(occ >= 3, "static_occlusion_observed", {"occluded_samples": occ, "states": dict(Counter(states))}))
        else:
            occ = sum(s in ("fully_occluded", "partially_occluded") for s in states)
            out.append(check(occ >= 2, "dynamic_occlusion_observed", {"occluded_samples": occ, "states": dict(Counter(states))}))
    return out


# ------------------------------------------------------------------------------------------------ pair checks

def pair_checks(root: str, on: dict, off: dict, pools: dict) -> dict:
    d_on, d_off = os.path.join(root, "episodes", on["episode_id"]), os.path.join(root, "episodes", off["episode_id"])
    A, B = Ep(d_on), Ep(d_off)
    cmp = compare_pair(A, B)
    res = {"pair_id": on["pair_id"], "family": on["family"], "subtype": on["subtype"], "split": on["split"], "world": on["world"], "on": on["episode_id"],
           "off": off["episode_id"], "design_observability": on["observability"], "checks": []}
    ck = res["checks"]
    ck.append(check(on["robot"] == off["robot"] and on["world"] == off["world"] and on["run_seed"] == off["run_seed"], "same_robot_start_goal_world_seed"))
    ck.append(check(A.cam.get("K") == B.cam.get("K") and A.cam.get("T_base_camera") == B.cam.get("T_base_camera"), "same_camera"))
    ck.append(check(on["objects"] == off["objects"], "same_world_objects"))
    if on["family"] == "wheelchair":
        wheel = {**pools["wheelchair_known"], **pools["wheelchair_heldout"]}
        prim_on = next(h for h in on["humans"] if h["body_slot"] == "wheelchair")
        prim_off = next(h for h in off["humans"] if h["body_slot"] == "wheelchair")
        ck.append(check(wheel.get(prim_off["body"]) == prim_on["body"], "wheelchair_variant_of_same_identity", [prim_on["body"], prim_off["body"]]))
        ck.append(check([h["body"] for h in on["humans"] if h["body_slot"] != "wheelchair"] == [h["body"] for h in off["humans"] if h["body_slot"] != "wheelchair"],
                        "same_other_bodies"))
    else:
        ck.append(check([h["body"] for h in on["humans"]] == [h["body"] for h in off["humans"]], "same_bodies"))
    # matched stationary windows, per person: slots where that person is still in both conditions. Controlled people
    # (primary / participants / members / targets) must agree within 2 cm / 2 deg there; bystanders are reported.
    controlled = {h["name"] for h in on["humans"] if h["role"] in ("primary", "participant", "member", "target")}
    static_ctrl = all(not any("goto" in st or "group_form" in st or "group_join" in st or ("queue" in st and st["service"] < 1000)
                              for st in h["behavior"]["on"]["steps"]) for h in on["humans"] if h["name"] in controlled)
    static_ctrl = static_ctrl and all(not any("goto" in st for st in h["behavior"]["off"]["steps"]) for h in off["humans"] if h["name"] in controlled)
    # an interaction that starts after a wait or ends mid-episode turns the participants toward each other: dynamic by design
    static_ctrl = static_ctrl and not any(any("interact" in st and (i > 0 and "wait" in h["behavior"]["on"]["steps"][i - 1] or st["duration"] < 1000)
                                              for i, st in enumerate(h["behavior"]["on"]["steps"])) for h in on["humans"] if h["name"] in controlled)
    slots_on = {r["slot"]: i for i, r in enumerate(A.robots)}
    slots_off = {r["slot"]: i for i, r in enumerate(B.robots)}
    common = sorted(set(slots_on) & set(slots_off))
    per = {}
    contact = {}  # first slot where the robot comes within ROBOT_CONTACT_M of the person in either condition
    for s in common:
        a, b = ped_index(A.states(slots_on[s])), ped_index(B.states(slots_off[s]))
        ra, rb = A.robots[slots_on[s]]["pose_map_base"], B.robots[slots_off[s]]["pose_map_base"]
        for n in set(a) & set(b):
            if n not in contact and min(math.dist((ra["x"], ra["y"]), (a[n]["x"], a[n]["y"])), math.dist((rb["x"], rb["y"]), (b[n]["x"], b[n]["y"]))) < ROBOT_CONTACT_M:
                contact[n] = s
            d = per.setdefault(n, {"still": 0, "matched": 0, "pos_max": 0.0, "head_max": 0.0, "traj_max": 0.0, "eligible": 0})
            if n in contact:
                continue  # after a close robot pass, people may be pushed or step aside: reactive, recorded not matched
            d["eligible"] += 1
            dp = math.dist((a[n]["x"], a[n]["y"]), (b[n]["x"], b[n]["y"]))
            d["traj_max"] = max(d["traj_max"], dp)
            if math.hypot(a[n]["vx"], a[n]["vy"]) < STATIONARY_SPEED and math.hypot(b[n]["vx"], b[n]["vy"]) < STATIONARY_SPEED:
                dh = abs(math.degrees(math.atan2(math.sin(a[n]["yaw"] - b[n]["yaw"]), math.cos(a[n]["yaw"] - b[n]["yaw"]))))
                d["still"] += 1
                d["pos_max"], d["head_max"] = max(d["pos_max"], dp), max(d["head_max"], dh)
                d["matched"] += int(dp <= spec.MATCH_POS_M and dh <= spec.MATCH_HEADING_DEG)
    ctrl = {n: v for n, v in per.items() if n in controlled}
    still = sum(v["still"] for v in ctrl.values())
    eligible = sum(v["eligible"] for v in ctrl.values())
    matched = sum(v["matched"] for v in ctrl.values())
    win = {"slots": len(common), "controlled": sorted(controlled), "stationary_person_slots": still, "matched_person_slots": matched,
           "eligible_person_slots": eligible, "robot_contact_slot": {n: contact[n] for n in controlled if n in contact},
           "pos_max_m": round(max((v["pos_max"] for v in ctrl.values()), default=0.0), 4), "heading_max_deg": round(max((v["head_max"] for v in ctrl.values()), default=0.0), 3),
           "matched_fraction_of_stationary": round(matched / still, 3) if still else None,
           "bystander_traj_max_m": {n: round(v["traj_max"], 3) for n, v in per.items() if n not in controlled}}
    res["matched_window"] = win
    if static_ctrl and controlled:
        ck.append(check(still >= 0.8 * eligible and matched == still, "matched_geometry_2cm_2deg", win))
    else:
        ck.append(check(True, "dynamic_pair_divergence_recorded", win))
    res["controlled_static_design"] = bool(static_ctrl)
    res["geometry_matched_whole_episode"] = bool(static_ctrl and controlled and still == len(common) * len(ctrl) and matched == still)
    # robot trajectory agreement (closed loop on bodies-only Nav2)
    res["robot"] = {k: cmp["geometry"][k] for k in ("robot_pos_max_m", "robot_pos_mean_m", "robot_heading_p95_deg")}
    # ON/OFF semantics
    s_on = json.load(open(os.path.join(d_on, "dv1", "summary.json")))
    s_off = json.load(open(os.path.join(d_off, "dv1", "summary.json")))
    fam = on["family"]
    if fam == "conversation":
        ck.append(check(s_on["components"]["samples_with_cost"].get("conversation", 0) > 0 and s_off["components"]["samples_with_cost"].get("conversation", 0) == 0,
                        "on_has_conversation_off_not"))
    elif fam == "stop_gesture":
        ck.append(check(bool(s_on["gesture_intervals"]) and not s_off["gesture_intervals"], "on_has_gesture_off_not"))
    elif fam == "queue":
        ck.append(check(s_on["components"]["samples_with_cost"].get("queue", 0) > 0 and s_off["components"]["samples_with_cost"].get("queue", 0) == 0,
                        "on_has_queue_membership_off_not"))
    elif fam == "wheelchair":
        ck.append(check(True, "on_wheelchair_body_off_walking_twin"))
    # observability classification from rendered frames
    cue = s_on.get("cue", {})
    frac, n_vis = cue.get("cue_visible_fraction"), cue.get("samples_cue_visible", 0)
    if on["observability"] == "hidden_state":
        cls = "hidden_state"
    elif frac is not None and n_vis >= 3 and frac >= 0.1:
        cls = "visually_observable"
    else:
        cls = "not_visually_observable"
    res["observability_class"] = cls
    res["cue"] = cue
    res["visual"] = cmp["visual"]
    res["labels"] = {c: v for c, v in cmp["labels"].items()}
    res["ok"] = all(c["ok"] for c in ck)
    return res


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root
    os.makedirs(args.out, exist_ok=True)
    man = json.load(open(os.path.join(root, "manifests", "manifest_v1.json")))
    pools = json.load(open(os.path.join(root, "metadata", "body_pools.json")))
    eps = man["episodes"]
    sel = {"subset": [e for e in eps if e["subset"]], "pilot": [e for e in eps if e["pilot"]], "remaining": [e for e in eps if not e["pilot"]]}.get(args.stage, eps)
    report = {"stage": args.stage, "created": datetime.datetime.now().isoformat(timespec="seconds"), "episodes": {}, "pairs": [], "missing": []}
    held = set(pools["heldout_walking"]) | set(pools["wheelchair_heldout"].values())
    for e in sel:
        d = os.path.join(root, "episodes", e["episode_id"])
        if not os.path.exists(os.path.join(d, "DONE")):
            report["missing"].append(e["episode_id"])
            continue
        meta = json.load(open(os.path.join(d, "episode_metadata.json")))
        summ = json.load(open(os.path.join(d, "dv1", "summary.json")))
        groups = {"integrity": integrity(d, meta, e), "synchronization": synchronization(d, meta, e, summ), "semantics": semantics(d, e, summ),
                  "visibility": visibility(d, e)}
        rec_ids = set(e["bodies"])
        groups["generalization"] = [
            check(not rec_ids or ((e["split"] == "test_c") == bool(rec_ids & held) and (e["split"] != "test_c" or rec_ids <= held)), "body_split_rule",
                  sorted(rec_ids & held)),
            check((e["world_key"] == "D") == (e["split"] == "test_b"), "world_split_rule", e["world_key"]),
            check(meta["dv1"]["robot_health"]["ok"], "robot_upright_and_moving", meta["dv1"]["robot_health"]),
        ]
        report["episodes"][e["episode_id"]] = {"family": e["family"], "condition": e["condition"], "world": e["world_key"], "split": e["split"],
                                               "groups": groups, "ok": all(c["ok"] for g in groups.values() for c in g)}
    # pairs and leakage
    by_pair = defaultdict(dict)
    for e in sel:
        if e["pair_id"]:
            by_pair[e["pair_id"]][e["condition"]] = e
    for pid, pr in sorted(by_pair.items()):
        if all(os.path.exists(os.path.join(root, "episodes", pr[c]["episode_id"], "DONE")) for c in ("on", "off") if c in pr) and len(pr) == 2:
            report["pairs"].append(pair_checks(root, pr["on"], pr["off"], pools))
    split_of = defaultdict(set)
    for e in eps:
        for key in (e["episode_id"], e["pair_id"], e["template_id"], (e["world"], e["site"], json.dumps(e["humans"][0]["spawn"]) if e["humans"] else e["unit_id"])):
            if key:
                split_of[key if isinstance(key, str) else json.dumps(key)].add(e["split"])
    leaks = [k for k, v in split_of.items() if len(v) > 1]
    report["leakage"] = {"keys_in_multiple_splits": leaks[:20], "ok": not leaks,
                         "heldout_world_in_training": any(e["world_key"] == "D" and e["split"] != "test_b" for e in eps),
                         "heldout_bodies_in_training": any(set(e["identities"]) & set(pools["heldout_walking"]) and e["split"] != "test_c" for e in eps)}
    # summary
    fails = Counter()
    for v in report["episodes"].values():
        for g, cs in v["groups"].items():
            for c in cs:
                if not c["ok"]:
                    fails[f"{g}/{c['check']}"] += 1
    pair_fails = Counter(c["check"] for p in report["pairs"] for c in p["checks"] if not c["ok"])
    report["summary"] = {"episodes_checked": len(report["episodes"]), "episodes_ok": sum(v["ok"] for v in report["episodes"].values()),
                         "missing": len(report["missing"]), "failing_checks": dict(fails), "pairs_checked": len(report["pairs"]),
                         "pairs_ok": sum(p["ok"] for p in report["pairs"]), "pair_failing_checks": dict(pair_fails),
                         "observability": dict(Counter(p["observability_class"] for p in report["pairs"])), "leakage_ok": report["leakage"]["ok"]}
    rejected = sorted([k for k, v in report["episodes"].items() if not v["ok"]] + [x for p in report["pairs"] if not p["ok"] for x in (p["on"], p["off"])])
    json.dump(report, open(os.path.join(args.out, "validation.json"), "w"), indent=1)
    json.dump(report["pairs"], open(os.path.join(args.out, "pairs.json"), "w"), indent=1)
    json.dump(sorted(set(rejected)), open(os.path.join(args.out, "rejected.json"), "w"), indent=1)
    write_md(report, os.path.join(args.out, "validation.md"))
    print(json.dumps(report["summary"], indent=1))
    return 0


def write_md(r: dict, path: str) -> None:
    s = r["summary"]
    lines = [f"# dataset_v1 validation: stage `{r['stage']}`", "", f"Created {r['created']}.", "",
             f"Episodes checked: {s['episodes_checked']}, passing every check: {s['episodes_ok']}, not recorded / not done: {s['missing']}.",
             f"Counterfactual pairs checked: {s['pairs_checked']}, passing: {s['pairs_ok']}. Leakage checks pass: {s['leakage_ok']}.", ""]
    if s["failing_checks"]:
        lines += ["## Failing episode checks", "", "| check | episodes |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(s["failing_checks"].items())] + [""]
    if s["pair_failing_checks"]:
        lines += ["## Failing pair checks", "", "| check | pairs |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(s["pair_failing_checks"].items())] + [""]
    lines += ["## Pairs", "", "| pair | family | subtype | world | matched / stationary person-slots | pos max (m) | heading max (deg) | robot pos max (m) | observability | ok |",
              "|---|---|---|---|---|---:|---:|---:|---|---|"]
    for p in r["pairs"]:
        w = p["matched_window"]
        lines.append(f"| {p['pair_id']} | {p['family']} | {p['subtype']} | {p['world']} | {w['matched_person_slots']} / {w['stationary_person_slots']} ({w['slots']} slots) | "
                     f"{w['pos_max_m']} | {w['heading_max_deg']} | {p['robot']['robot_pos_max_m']:.3f} | {p['observability_class']} | {p['ok']} |")
    lines += ["", "## Episodes", "", "| episode | family | cond | world | split | ok | failing |", "|---|---|---|---|---|---|---|"]
    for k, v in sorted(r["episodes"].items()):
        bad = [f"{g}/{c['check']}" for g, cs in v["groups"].items() for c in cs if not c["ok"]]
        lines.append(f"| {k} | {v['family']} | {v['condition']} | {v['world']} | {v['split']} | {v['ok']} | {', '.join(bad)} |")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
