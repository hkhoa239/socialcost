"""Automated checks on a recorded SOCIALCOST dataset."""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess

import cv2
import numpy as np

from .dataset import interaction_frame_at, read_jsonl, with_membership
from .labels import GridSpec, LayerParams


def quat_to_rot(q: dict) -> np.ndarray:
    x, y, z, w = q["qx"], q["qy"], q["qz"], q["qw"]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(p: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(p)
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


def to_cell_cost(params: LayerParams, normalized: float) -> int:
    return int(min(max(normalized * params.cost_scale + params.cost_bias, 0.0), 1.0) * params.max_cost_value)


def robot_cell(grid: GridSpec, pose: dict, x: float, y: float) -> tuple[int, int] | None:
    """Label cell (i, j) holding map point (x, y) for the robot at `pose`, None outside the grid."""
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    dx, dy = x - pose["x"], y - pose["y"]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    i = int(math.floor((lx - grid.origin_x) / grid.resolution))
    j = int(math.floor((ly - grid.origin_y) / grid.resolution))
    if 0 <= i < grid.width and 0 <= j < grid.height:
        return i, j
    return None


def _groups(states: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for p in states:
        if p["interaction_id"] >= 0:
            groups.setdefault(p["interaction_id"], []).append(p)
    return groups


def _midpoint_check(lab: np.ndarray, groups: dict[int, list[dict]], params: LayerParams, grid: GridSpec, pose: dict) -> tuple[int, int]:
    """Cells at the midpoint of every weighted pair must hold at least the group term there."""
    checked = ok = 0
    # the cell centre is within res/sqrt(2) of the midpoint; allow the falloff over that distance
    near_peak = math.exp(-((grid.resolution ** 2 / 2) / (2 * params.group_sigma ** 2)) ** params.falloff_shape)
    for members in groups.values():
        kind = members[0]["interaction_type"]
        w = params.interaction_weights[kind] if kind < len(params.interaction_weights) else 0.0
        if w <= 0 or len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                c = robot_cell(grid, pose, (members[a]["x"] + members[b]["x"]) / 2, (members[a]["y"] + members[b]["y"]) / 2)
                if c is not None:
                    checked += 1
                    ok += int(lab[c[1], c[0]] >= to_cell_cost(params, w * near_peak))
    return checked, ok


def check_episode(ep_dir: str, params: LayerParams, grid: GridSpec, person_height: float = 1.0) -> dict:
    meta = json.load(open(os.path.join(ep_dir, "episode_metadata.json")))
    cam = json.load(open(os.path.join(ep_dir, "camera_info.json")))
    robots = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))
    peds = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))
    syncs = read_jsonl(os.path.join(ep_dir, "sync.jsonl"))
    K = np.array(cam["K"]).reshape(3, 3)
    W, H = cam["width"], cam["height"]
    r = {"episode": os.path.basename(ep_dir), "scenario": meta.get("scenario"), "run_seed": meta.get("run_seed"),
         "samples": len(robots), "expected_samples": meta["expected_samples"], "valid_flag": meta.get("valid")}
    fails: list[str] = []

    # 1. dimensions and encodings
    enc_ok = all(s["rgb_encoding"] == "rgb8" and s["depth_encoding"] == "32FC1" for s in syncs)
    shape_bad = 0
    invalid_frac = []
    inf_found = 0
    for k in range(len(robots)):
        img = cv2.imread(os.path.join(ep_dir, "rgb", f"{k:06d}.png"))
        d = np.load(os.path.join(ep_dir, "depth", f"{k:06d}.npy"))
        if img is None or img.shape != (H, W, 3) or d.shape != (H, W) or d.dtype != np.float32:
            shape_bad += 1
        inf_found += int(np.isinf(d).sum())
        invalid_frac.append(float(np.isnan(d).mean()))
    r["encodings_ok"] = enc_ok
    r["shape_mismatches"] = shape_bad
    r["depth_invalid_fraction_mean"] = float(np.mean(invalid_frac)) if invalid_frac else None
    r["depth_invalid_fraction_max"] = float(np.max(invalid_frac)) if invalid_frac else None
    r["depth_inf_values"] = inf_found
    if not enc_ok or shape_bad:
        fails.append("encoding/shape")
    if inf_found:
        fails.append("inf depth stored")

    # 3. timestamps
    st = [s["stamp_ns"] for s in syncs]
    rgb_depth = max((abs(s["rgb_stamp_ns"] - s["depth_stamp_ns"]) for s in syncs), default=0)
    monotonic = all(b > a for a, b in zip(st, st[1:]))
    slots = [rb["slot"] for rb in robots]
    slot_dt = np.diff(st) / 1e9 if len(st) > 1 else np.array([])
    t0 = meta.get("record_start_ns") or (st[0] if st else 0)
    slot_err = [abs(s - (t0 + sl * 1e9 / meta["sample_hz"])) / 1e9 for s, sl in zip(st, slots)]
    in_window = all(t0 <= s < t0 + meta["duration_s"] * 1e9 for s in st)
    peds_dt = [max(abs(p["stamp_ns"] - p["peds_before_ns"]), abs((p["peds_after_ns"] or p["peds_before_ns"]) - p["stamp_ns"])) / 1e9 for p in peds]
    pose_dt = [abs(s["stamp_ns"] - rb["stamp_ns"]) for s, rb in zip(syncs, robots)]
    r["sync"] = {"rgb_depth_max_s": rgb_depth / 1e9, "monotonic": monotonic, "in_episode_window": in_window,
                 "sample_dt_mean_s": float(slot_dt.mean()) if slot_dt.size else None,
                 "sample_dt_max_s": float(slot_dt.max()) if slot_dt.size else None,
                 "frame_to_slot_lag_max_s": float(max(slot_err)) if slot_err else None,
                 "frame_to_slot_lag_mean_s": float(np.mean(slot_err)) if slot_err else None,
                 "peds_interp_bracket_max_s": float(max(peds_dt)) if peds_dt else None,
                 "pose_stamp_dt_max_ns": int(max(pose_dt)) if pose_dt else 0}
    r["missing_frames"] = meta["expected_samples"] - len(robots)
    if rgb_depth != 0 or not monotonic or not in_window:
        fails.append("timestamps")

    # 4. transforms: rigid camera mount, optical axis forward, peds reprojected onto depth
    T_bc = [np.linalg.inv(pose_matrix(rb["pose_map_base"])) @ pose_matrix(rb["pose_map_camera"]) for rb in robots]
    mount_dev = max((np.abs(T - T_bc[0]).max() for T in T_bc), default=0.0)
    optical_forward = T_bc[0][:3, :3] @ np.array([0, 0, 1.0]) if T_bc else np.zeros(3)
    r["camera_mount_max_deviation"] = float(mount_dev)
    r["optical_z_in_base"] = [round(float(v), 4) for v in optical_forward]
    r["camera_height_m"] = float(robots[0]["pose_map_camera"]["z"]) if robots else None
    proj_total = proj_agree = proj_occluded = proj_missing = 0
    proj_errors = []
    for k, (rb, ps) in enumerate(zip(robots, peds)):
        T_cm = np.linalg.inv(pose_matrix(rb["pose_map_camera"]))
        d = None
        for p in ps["pedestrians"]:
            pc = T_cm @ np.array([p["x"], p["y"], person_height, 1.0])
            if pc[2] < 0.5 or pc[2] > 8.0:
                continue
            u, v = (K @ pc[:3])[:2] / pc[2]
            if not (8 <= u < W - 8 and 8 <= v < H - 8):
                continue
            if d is None:
                d = np.load(os.path.join(ep_dir, "depth", f"{k:06d}.npy"))
            win = d[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4]
            if np.isnan(win).all():
                continue
            measured = float(np.nanmin(win))
            proj_total += 1
            err = pc[2] - measured  # person surface is in front of its centre
            proj_errors.append(err)
            if -0.3 <= err <= 0.8:
                proj_agree += 1
            elif err > 0.8:
                proj_occluded += 1  # a nearer surface (another person) hides this one
            else:
                proj_missing += 1  # depth sees past the spot the person should occupy
    r["ped_reprojection"] = {"projected": proj_total, "depth_agrees": proj_agree, "occluded": proj_occluded, "missing": proj_missing,
                             "agree_fraction": proj_agree / proj_total if proj_total else None,
                             "median_centre_minus_surface_m": float(np.median(proj_errors)) if proj_errors else None}
    if mount_dev > 1e-3 or optical_forward[0] < 0.9:
        fails.append("camera extrinsics")
    # Occlusion is expected (conversation partners line up); seeing past a person is not.
    if proj_total and proj_missing / proj_total > 0.05:
        fails.append("pedestrians not where depth sees them")

    # 5-8. labels
    peak_ps = to_cell_cost(params, 0.5)  # base amplitude, the minimum personal-space peak
    vmax = []
    nonzero_samples = 0
    ps_checked = ps_ok = 0
    grp_checked = grp_ok = 0
    mem_checked = mem_ok = mem_not_superset = mem_differs = 0
    frames = read_jsonl(os.path.join(ep_dir, "interactions.jsonl"))
    outside_bad = 0
    shapes_ok = True
    for k, (rb, ps) in enumerate(zip(robots, peds)):
        lab = np.load(os.path.join(ep_dir, "social_cost", f"{k:06d}.npy"))
        shapes_ok &= lab.shape == (grid.height, grid.width) and lab.dtype == np.uint8
        vmax.append(int(lab.max()))
        nonzero_samples += int(lab.any())
        pose = rb["pose_map_base"]
        for p in ps["pedestrians"]:
            c = robot_cell(grid, pose, p["x"], p["y"])
            if c is not None:
                ps_checked += 1
                ps_ok += int(lab[c[1], c[0]] >= peak_ps)
        groups = _groups(ps["pedestrians"])
        n, ok = _midpoint_check(lab, groups, params, grid, pose)
        grp_checked += n
        grp_ok += ok
        full = np.load(os.path.join(ep_dir, "social_cost_membership", f"{k:06d}.npy"))
        completed = with_membership(ps["pedestrians"], interaction_frame_at(frames, rb["stamp_ns"]))
        n, ok = _midpoint_check(full, _groups(completed), params, grid, pose)
        mem_checked += n
        mem_ok += ok
        mem_not_superset += int((full < lab).sum())
        mem_differs += int((full != lab).any())
        # cells far from every pedestrian and every weighted pair must be zero
        reach = max(params.personal_space_radius, params.group_extra_radius) + grid.resolution
        ii, jj = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
        lx = grid.origin_x + (ii + 0.5) * grid.resolution
        ly = grid.origin_y + (jj + 0.5) * grid.resolution
        cy, sy = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        wx, wy = pose["x"] + cy * lx - sy * ly, pose["y"] + sy * lx + cy * ly
        near = np.zeros_like(lab, dtype=bool)
        pts = [(p["x"], p["y"]) for p in ps["pedestrians"]]
        for groups_members in groups.values():
            if len(groups_members) >= 2:
                xs = [m["x"] for m in groups_members]
                ys = [m["y"] for m in groups_members]
                near |= (wx >= min(xs) - reach) & (wx <= max(xs) + reach) & (wy >= min(ys) - reach) & (wy <= max(ys) + reach)
        for x, y in pts:
            near |= (wx - x) ** 2 + (wy - y) ** 2 <= reach ** 2
        outside_bad += int((lab[~near] > 0).sum())
    r["label"] = {"shape_dtype_ok": bool(shapes_ok), "max_value": max(vmax) if vmax else None, "max_allowed": params.max_cost_value,
                  "samples_with_cost": nonzero_samples, "personal_space_cells_checked": ps_checked, "personal_space_ok": ps_ok,
                  "interaction_midpoints_checked": grp_checked, "interaction_midpoints_ok": grp_ok, "cost_outside_reach": outside_bad,
                  "membership_midpoints_checked": mem_checked, "membership_midpoints_ok": mem_ok,
                  "membership_cells_below_exact": mem_not_superset, "membership_samples_differing": mem_differs}
    if not shapes_ok or (vmax and max(vmax) > params.max_cost_value):
        fails.append("label range/shape")
    if nonzero_samples == 0:
        fails.append("no social cost")
    if ps_checked and ps_ok != ps_checked:
        fails.append("personal space")
    if grp_checked and grp_ok != grp_checked:
        fails.append("interaction region")
    if outside_bad:
        fails.append("cost outside reach")
    if mem_not_superset or (mem_checked and mem_ok != mem_checked):
        fails.append("membership label")

    # 10. live agreement (recorded during collection)
    ag = meta.get("live_agreement") or []
    if ag:
        r["live_agreement"] = {
            "samples": len(ag),
            "exact_fraction_mean": float(np.mean([a["exact_fraction"] for a in ag])),
            "exact_fraction_min": float(np.min([a["exact_fraction"] for a in ag])),
            "mae_mean": float(np.mean([a["mae"] for a in ag])),
            "iou_nonzero_mean": float(np.mean([a["iou_nonzero"] for a in ag])),
            "fully_identical_samples": int(sum(a["max_abs"] == 0 for a in ag)),
            "live_stamp_minus_image_s_mean": float(np.mean([a["live_minus_image_stamp_s"] for a in ag])),
        }
    stats = meta["stats"]
    wall = stats["wall_end"] - stats["wall_start"] if stats.get("wall_end") else None
    r["recording"] = {"wall_s": wall, "samples_per_wall_s": len(robots) / wall if wall else None,
                      "sim_s": (st[-1] - st[0]) / 1e9 if len(st) > 1 else None,
                      "rgb_without_depth": stats.get("rgb_without_depth"), "tf_failures": stats.get("tf_failures"),
                      "peds_gap_violations": stats.get("peds_gap_violations"), "end_reason": stats.get("end_reason")}
    if r["recording"]["sim_s"] and wall:
        r["recording"]["real_time_factor"] = (meta["duration_s"]) / wall
    r["disk_bytes"] = int(subprocess.run(["du", "-sb", ep_dir], capture_output=True, text=True).stdout.split()[0])
    r["failures"] = fails
    r["pass"] = not fails
    return r


def trajectory_repro(ep_a: str, ep_b: str) -> dict:
    """Same scenario and seed recorded twice: per-sample pedestrian and robot position differences."""
    pa = read_jsonl(os.path.join(ep_a, "pedestrian_states.jsonl"))
    pb = read_jsonl(os.path.join(ep_b, "pedestrian_states.jsonl"))
    ra = read_jsonl(os.path.join(ep_a, "robot_states.jsonl"))
    rb = read_jsonl(os.path.join(ep_b, "robot_states.jsonl"))
    n = min(len(pa), len(pb))
    ped_err, robot_err, label_diff = [], [], []
    for k in range(n):
        A = {p["name"]: p for p in pa[k]["pedestrians"]}
        B = {p["name"]: p for p in pb[k]["pedestrians"]}
        for name in A.keys() & B.keys():
            ped_err.append(math.hypot(A[name]["x"] - B[name]["x"], A[name]["y"] - B[name]["y"]))
        qa, qb = ra[k]["pose_map_base"], rb[k]["pose_map_base"]
        robot_err.append(math.hypot(qa["x"] - qb["x"], qa["y"] - qb["y"]))
        la = np.load(os.path.join(ep_a, "social_cost", f"{k:06d}.npy")).astype(int)
        lb = np.load(os.path.join(ep_b, "social_cost", f"{k:06d}.npy")).astype(int)
        label_diff.append(float(np.abs(la - lb).mean()))
    return {"samples": n, "ped_pos_err_mean_m": float(np.mean(ped_err)) if ped_err else None,
            "ped_pos_err_max_m": float(np.max(ped_err)) if ped_err else None,
            "robot_pos_err_mean_m": float(np.mean(robot_err)) if robot_err else None,
            "robot_pos_err_max_m": float(np.max(robot_err)) if robot_err else None,
            "label_mae_mean": float(np.mean(label_diff)) if label_diff else None}


def episodes(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "episodes", "episode_*")))
