"""dataset_v1 per-episode post-processing: cost components, body occupancy, visibility labels, cue visibility.

Outputs inside the episode directory (the existing channels are not touched):

  components/NNNNNN.npz      uint8 [200, 200] per key: personal_space, conversation, queue, gesture,
                             appearance_or_geometry_extension, combined_native, combined_extended, body_occupancy
  dv1/visibility.jsonl       per sample: per pedestrian visibility state, fraction, pixel estimate, time since seen
  dv1/summary.json           component QA, visibility and cue statistics, gesture intervals, interaction timeline
"""

from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np

from ..dataset import interaction_frame_at, peds_for_layer, read_jsonl, with_membership
from ..extensions import ExtParams, halt_direction, halt_term, is_mobility_aid
from ..labels import GridSpec, LayerParams, Ped, paint
from ..qa import pose_matrix
from ..reference import cell_centres

TALK_TYPES = {0, 1}  # TALK_TO, GROUP_CONVERSATION
QUEUE_TYPE = 6
COMPOSITION = {"operator": "per-cell maximum (the layer's own compositing)",
               "combined_native": "max(personal_space, native interaction term over every published interaction) == social_cost",
               "combined_extended": "extended/2.1: max(arena-native with membership completion, halt-gesture/1.0); mobility hypothesis excluded",
               "appearance_or_geometry_extension": "mobility-aid/1.0-hypothesis term alone (personal space radius and sigma x1.5 for wheelchair users); "
                                                   "not part of combined_extended, evaluated separately",
               "body_occupancy": "physical footprint of each body (asset bounding box, oriented by pedestrian yaw), 1 inside, 0 outside"}
VIS_TOL = 0.35  # m, depth consistency for a body sample point
VISIBLE_SEEN = 0.1  # fraction counted as "seen" for time_since_last_seen


def _ext21() -> ExtParams:
    return ExtParams(membership=True, halt=True, mobility=False)


def _no_personal(params: LayerParams) -> LayerParams:
    return dataclasses.replace(params, personal_space_radius=1e-6)


def components(params: LayerParams, grid: GridSpec, states: list[dict], frame: dict | None, pose, bboxes: dict) -> dict[str, np.ndarray]:
    ext = _ext21()
    full = with_membership(states, frame)
    peds_native = peds_for_layer(states)
    out = {}
    out["personal_space"] = paint(params, [Ped(p.x, p.y, p.animation_state) for p in peds_native], grid, pose)
    talk = [Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in states
            if p["interaction_id"] >= 0 and p["interaction_type"] != QUEUE_TYPE]
    out["conversation"] = paint(_no_personal(params), talk, grid, pose)
    queue = [Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in full
             if p["interaction_id"] >= 0 and p["interaction_type"] == QUEUE_TYPE]
    out["queue"] = paint(_no_personal(params), queue, grid, pose)
    g = np.zeros((grid.height, grid.width), np.uint8)
    for p in states:
        g = np.maximum(g, halt_term(params, ext, p, grid, pose))
    out["gesture"] = g
    mob_params = dataclasses.replace(params, personal_space_radius=params.personal_space_radius * 1.5, personal_space_sigma=params.personal_space_sigma * 1.5)
    aids = [Ped(p["x"], p["y"], p["animation_state"]) for p in states if is_mobility_aid(p, ExtParams(mobility=True))]
    out["appearance_or_geometry_extension"] = paint(mob_params, aids, grid, pose) if aids else np.zeros_like(g)
    out["combined_native"] = paint(params, peds_native, grid, pose)
    out["combined_extended"] = np.maximum(paint(params, peds_for_layer(full), grid, pose), g)
    out["body_occupancy"] = body_occupancy(grid, pose, states, bboxes)
    return out


def body_occupancy(grid: GridSpec, pose, states: list[dict], bboxes: dict) -> np.ndarray:
    wx, wy = cell_centres(grid, pose)
    occ = np.zeros((grid.height, grid.width), np.uint8)
    for p in states:
        bb = bboxes.get(_body_id(p.get("model_uri", ""))) or [[-0.2, 0.2], [-0.25, 0.25]]
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        rx, ry = wx - p["x"], wy - p["y"]
        u, v = c * rx + s * ry, -s * rx + c * ry
        occ |= ((u >= bb[0][0]) & (u <= bb[0][1]) & (v >= bb[1][0]) & (v <= bb[1][1])).astype(np.uint8)
    return occ


def _body_id(model_uri: str) -> str:
    parts = model_uri.replace("\\", "/").split("/")
    if "Human" in parts:
        i = parts.index("Human")
        dom, name = parts[i - 1], parts[i + 1]
        return name if dom == "Common" else f"{dom}/{name}"
    return os.path.splitext(os.path.basename(model_uri))[0]


# ---------------------------------------------------------------------------------------------- visibility

def _project(T_cam_map: np.ndarray, K: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph = np.c_[pts, np.ones(len(pts))] @ T_cam_map.T
    z = ph[:, 2]
    uv = (ph[:, :3] @ K.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def body_points(p: dict, cam_xy: tuple[float, float], height: float, radius: float = 0.2) -> np.ndarray:
    """Points on the camera-facing half of a body cylinder: 5 heights x (front, left edge, right edge)."""
    dx, dy = cam_xy[0] - p["x"], cam_xy[1] - p["y"]
    n = math.hypot(dx, dy) or 1.0
    fx, fy = dx / n, dy / n
    lx, ly = -fy, fx
    out = []
    for z in np.linspace(0.15 * height, 0.9 * height, 5):
        out.append((p["x"] + radius * fx, p["y"] + radius * fy, z))
        out.append((p["x"] + 0.8 * radius * lx + 0.4 * radius * fx, p["y"] + 0.8 * radius * ly + 0.4 * radius * fy, z))
        out.append((p["x"] - 0.8 * radius * lx + 0.4 * radius * fx, p["y"] - 0.8 * radius * ly + 0.4 * radius * fy, z))
    return np.array(out)


def point_states(depth: np.ndarray, uv: np.ndarray, z: np.ndarray) -> list[str]:
    """Per point: outside (not in image), visible (depth agrees within VIS_TOL in a 3x3 window),
    occluded (something nearer), miss (only farther surfaces: the ray passes the body sample)."""
    H, W = depth.shape
    out = []
    for (u, v), zz in zip(uv, z):
        if zz <= 0.3 or not (0 <= u < W and 0 <= v < H) or not np.isfinite(u):
            out.append("outside")
            continue
        iu, iv = int(u), int(v)
        win = depth[max(0, iv - 1):iv + 2, max(0, iu - 1):iu + 2]
        win = win[np.isfinite(win)]
        if win.size == 0:
            out.append("miss")
        elif np.min(np.abs(win - zz)) <= VIS_TOL:
            out.append("visible")
        elif np.min(win) < zz - VIS_TOL:
            out.append("occluded")
        else:
            out.append("miss")
    return out


def classify(states: list[str]) -> tuple[str, float, float]:
    n_in = sum(s != "outside" for s in states)
    vis = sum(s == "visible" for s in states)
    occl = sum(s == "occluded" for s in states)
    if n_in == 0:
        return "outside_fov", 0.0, 0.0
    frac = vis / len(states)  # fraction of the body samples seen (truncation by the image border lowers it)
    in_frac = n_in / len(states)
    if vis == 0:
        return ("fully_occluded" if occl > 0 or in_frac > 0.5 else "outside_fov"), 0.0, in_frac
    if frac >= 0.6 and occl == 0:
        return "visible", frac, in_frac
    return "partially_occluded", frac, in_frac


def hand_point(p: dict, height: float) -> tuple[float, float, float] | None:
    d = halt_direction(p)
    if d is None:
        return None
    return p["x"] + 0.5 * d[0], p["y"] + 0.5 * d[1], 0.83 * height


def process(ep_dir: str, bboxes: dict, heights: dict, dv1_entry: dict | None = None) -> dict:
    meta = json.load(open(os.path.join(ep_dir, "episode_metadata.json")))
    params = LayerParams.from_mapping(meta["label"]["layer_params"])
    grid = GridSpec(**meta["label"]["grid"])
    cam = json.load(open(os.path.join(ep_dir, "camera_info.json")))
    K = np.array(cam["K"]).reshape(3, 3)
    robots = {r["sample"]: r for r in read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))}
    peds = {p["sample"]: p for p in read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))}
    frames = read_jsonl(os.path.join(ep_dir, "interactions.jsonl"))
    os.makedirs(os.path.join(ep_dir, "components"), exist_ok=True)
    os.makedirs(os.path.join(ep_dir, "dv1"), exist_ok=True)
    native_mismatch, ext_ge_native, comp_nonzero = 0, 0, {}
    last_seen: dict[str, float] = {}
    vis_rows = []
    cue = {"samples_active": 0, "samples_cue_visible": 0, "role": None}
    primary_roles = ("primary", "participant", "member", "target")
    for k in sorted(robots):
        r = robots[k]
        pose = (r["pose_map_base"]["x"], r["pose_map_base"]["y"], r["pose_map_base"]["yaw"])
        states = peds[k]["pedestrians"]
        fr = interaction_frame_at(frames, r["stamp_ns"])
        comp = components(params, grid, states, fr, pose, bboxes)
        np.savez_compressed(os.path.join(ep_dir, "components", f"{k:06d}.npz"), **comp)
        sc = np.load(os.path.join(ep_dir, "social_cost", f"{k:06d}.npy"))
        native_mismatch += int(not np.array_equal(sc, comp["combined_native"]))
        ext_ge_native += int(np.all(comp["combined_extended"] >= comp["combined_native"]))
        for key, a in comp.items():
            comp_nonzero[key] = comp_nonzero.get(key, 0) + int(a.any())
        # visibility
        Tmc = pose_matrix(r["pose_map_camera"])
        Tcm = np.linalg.inv(Tmc)
        depth = np.load(os.path.join(ep_dir, "depth", f"{k:06d}.npy"))
        cam_xy = (r["pose_map_camera"]["x"], r["pose_map_camera"]["y"])
        t = r["stamp_ns"] / 1e9
        row = {"sample": k, "stamp_ns": r["stamp_ns"], "pedestrians": []}
        for p in states:
            name = p["name"].split("/")[-1]
            hgt = heights.get(_body_id(p.get("model_uri", "")), 1.7)
            pts = body_points(p, cam_xy, hgt)
            uv, z = _project(Tcm, K, pts)
            ps = point_states(depth, uv, z)
            state, frac, in_frac = classify(ps)
            if frac >= VISIBLE_SEEN:
                last_seen[name] = t
            inside = uv[[s != "outside" for s in ps]]
            area = 0.0
            if len(inside) >= 2:
                area = float((np.ptp(np.clip(inside[:, 0], 0, cam["width"])) + 4) * (np.ptp(np.clip(inside[:, 1], 0, cam["height"])) + 4))
            dist = math.dist(cam_xy, (p["x"], p["y"]))
            pc = Tcm @ np.array([p["x"], p["y"], 0.9, 1.0])
            bearing = math.degrees(math.atan2(-pc[0], pc[2])) if pc[2] > -50 else None
            cue_vis = None
            hp = hand_point(p, hgt)
            if hp is not None:
                # a raised hand covers a few pixels at 4-8 m, so depth at the projected point is usually the background
                # ("miss"); the cue counts as visible when the hand projects into the image, nothing nearer covers it,
                # and the gesturing person is at least 30 % visible
                huv, hz = _project(Tcm, K, np.array([hp]))
                cue_vis = point_states(depth, huv, hz)[0] in ("visible", "miss") and frac >= 0.3
            row["pedestrians"].append({
                "name": name, "state": state, "visible": state == "visible", "outside_fov": state == "outside_fov",
                "partially_occluded": state == "partially_occluded", "fully_occluded": state == "fully_occluded",
                "visibility_fraction": round(frac, 3), "in_image_fraction": round(in_frac, 3),
                "visible_pixel_count": int(area * frac), "time_since_last_seen": round(t - last_seen[name], 3) if name in last_seen else None,
                "distance_m": round(dist, 3), "bearing_deg": round(bearing, 1) if bearing is not None else None,
                "point_states": {s: ps.count(s) for s in set(ps)}, "halt_active": hp is not None, "halt_hand_visible": cue_vis,
                "interaction_id": p["interaction_id"], "interaction_type": p["interaction_type"],
            })
        vis_rows.append(row)
    with open(os.path.join(ep_dir, "dv1", "visibility.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in vis_rows)
    summary = {
        "components": {"composition": COMPOSITION, "samples_with_cost": comp_nonzero, "combined_native_mismatch_samples": native_mismatch,
                       "combined_extended_ge_native_samples": ext_ge_native, "samples": len(vis_rows),
                       "versions": {"combined_extended": "extended/2.1", "personal_space": "component/personal-space/1.0",
                                    "conversation": "component/interaction-native/1.0", "queue": "component/queue-membership/1.0",
                                    "gesture": "component/halt-gesture/1.0", "appearance_or_geometry_extension": "component/mobility-aid/1.0-hypothesis",
                                    "body_occupancy": "component/body-footprint/1.0"}},
        "visibility": visibility_summary(vis_rows),
        "gesture_intervals": gesture_intervals(ep_dir, meta),
        "cue": cue_summary(vis_rows, dv1_entry),
        "interaction_timeline": interaction_timeline(frames, meta),
    }
    json.dump(summary, open(os.path.join(ep_dir, "dv1", "summary.json"), "w"), indent=1)
    return summary


def visibility_summary(rows: list[dict]) -> dict:
    from collections import Counter
    st = Counter(p["state"] for r in rows for p in r["pedestrians"])
    per = {}
    for r in rows:
        for p in r["pedestrians"]:
            d = per.setdefault(p["name"], Counter())
            d[p["state"]] += 1
    transitions = {}
    for name in per:
        seq = [next((p["state"] for p in r["pedestrians"] if p["name"] == name), None) for r in rows]
        ev = []
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            if a != b and a is not None and b is not None:
                ev.append({"sample": i, "from": a, "to": b})
        transitions[name] = ev
    counts = [sum(p["state"] == "visible" or p["state"] == "partially_occluded" for p in r["pedestrians"]) for r in rows]
    return {"state_counts": dict(st), "per_pedestrian": {k: dict(v) for k, v in per.items()}, "transitions": transitions,
            "camera_visible_mean": float(np.mean(counts)) if counts else 0.0}


def gesture_intervals(ep_dir: str, meta: dict) -> list[dict]:
    """Halt gesture on/off times from the full arena_peds stream, seconds from the episode start."""
    t0 = meta.get("episode_start_ns", 0)
    on: dict[str, float] = {}
    out = []
    last_t = None
    for m in read_jsonl(os.path.join(ep_dir, "peds_stream.jsonl")):
        t = m.get("stamp_ns", 0)
        last_t = t
        for p in m.get("pedestrians", []):
            name = p["name"].split("/")[-1]
            active = any(isinstance(g, dict) and str(g.get("slot", "")).startswith("halt") for g in p.get("gestures") or [])
            if active and name not in on:
                on[name] = t
            elif not active and name in on:
                out.append({"name": name, "on_s": round((on.pop(name) - t0) / 1e9, 3), "off_s": round((t - t0) / 1e9, 3)})
    for name, t in on.items():
        out.append({"name": name, "on_s": round((t - t0) / 1e9, 3), "off_s": None, "still_active_at_s": round(((last_t or t) - t0) / 1e9, 3)})
    return out


def cue_summary(rows: list[dict], entry: dict | None) -> dict:
    if not entry:
        return {}
    fam = entry["family"]
    primary = [h["name"] for h in entry["humans"] if h["role"] in ("primary", "participant", "member")]
    active = vis = 0
    for r in rows:
        for p in r["pedestrians"]:
            if p["name"] not in primary:
                continue
            if fam == "stop_gesture":
                if p["halt_active"]:
                    active += 1
                    vis += int(bool(p["halt_hand_visible"]))
            elif fam in ("conversation", "wheelchair"):
                active += 1
                vis += int(p["visibility_fraction"] >= 0.3)
    return {"family": fam, "cue": {"stop_gesture": "raised hand point visible (depth-consistent)", "conversation": "participant >= 30 % visible",
                                   "wheelchair": "primary >= 30 % visible"}.get(fam, "n/a"),
            "samples_active": active, "samples_cue_visible": vis, "cue_visible_fraction": round(vis / active, 3) if active else None}


def interaction_timeline(frames: list[dict], meta: dict) -> list[dict]:
    t0 = meta.get("episode_start_ns", 0)
    out, prev = [], None
    for f in frames:
        act = sorted((it["interaction_type"], tuple(sorted(it["member_ids"]))) for it in f["interactions"] if it["outcome"] == 1)
        if act != prev:
            out.append({"t_s": round((f["stamp_ns"] - t0) / 1e9, 3), "active": [{"type": a, "members": list(m)} for a, m in act]})
            prev = act
    return out
