"""Dataset verification: inventory, synchronization, geometry and social-cost semantics.

Every check returns {"status": PASS | FAIL | NOT_TESTED, "metrics": ..., "failing": [(episode, sample), ...]}.
Coordinates are compared in declared frames: map (world), base_link (label grid), camera optical.
Times are simulator stamps from the recorded files, never file times.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter

import cv2
import numpy as np

from .dataset import interaction_frame_at, peds_for_layer, read_jsonl, with_membership
from .labels import GridSpec, LayerParams, Ped, paint
from .qa import pose_matrix, robot_cell
from .reference import reference_paint, to_cell_cost

PASS, FAIL, NOT_TESTED = "PASS", "FAIL", "NOT_TESTED"
LABEL_CHANNELS = ("social_cost", "social_cost_membership")


def result(status: str, metrics: dict | None = None, failing: list | None = None, note: str = "") -> dict:
    return {"status": status, "metrics": metrics or {}, "failing": (failing or [])[:20], "n_failing": len(failing or []), "note": note}


class Ep:
    """One recorded episode, loaded lazily."""

    def __init__(self, ep_dir: str) -> None:
        self.dir = ep_dir
        self.name = f"{os.path.basename(os.path.dirname(os.path.dirname(ep_dir)))}/{os.path.basename(ep_dir)}"
        self.meta = json.load(open(os.path.join(ep_dir, "episode_metadata.json")))
        self.cam = json.load(open(os.path.join(ep_dir, "camera_info.json"))) if os.path.exists(os.path.join(ep_dir, "camera_info.json")) else {}
        self.robots = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))
        self.peds = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))
        self.sync = read_jsonl(os.path.join(ep_dir, "sync.jsonl"))
        self.frames = read_jsonl(os.path.join(ep_dir, "interactions.jsonl"))
        self.stream = None
        self.params = LayerParams.from_mapping(self.meta["label"]["layer_params"])
        self.grid = GridSpec(**self.meta["label"]["grid"])
        self.scenario = self.meta.get("scenario")

    def label(self, k: int, channel: str = "social_cost") -> np.ndarray:
        return np.load(os.path.join(self.dir, channel, f"{k:06d}.npy"))

    def depth(self, k: int) -> np.ndarray:
        return np.load(os.path.join(self.dir, "depth", f"{k:06d}.npy"))

    def pose(self, k: int) -> tuple[float, float, float]:
        p = self.robots[k]["pose_map_base"]
        return p["x"], p["y"], p["yaw"]

    def states(self, k: int, membership: bool = False) -> list[dict]:
        ps = self.peds[k]["pedestrians"]
        return with_membership(ps, interaction_frame_at(self.frames, self.robots[k]["stamp_ns"])) if membership else ps

    def peds_stream(self) -> list[dict]:
        if self.stream is None:
            self.stream = read_jsonl(os.path.join(self.dir, "peds_stream.jsonl"))
        return self.stream


def load_episodes(*roots: str) -> list[Ep]:
    eps = []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "episodes", "episode_*"))):
            if os.path.exists(os.path.join(d, "episode_metadata.json")):
                eps.append(Ep(d))
    return eps


# ---------------------------------------------------------------- Phase 1
def inventory(ep: Ep) -> dict:
    n = len(ep.robots)
    counts = {d: len(glob.glob(os.path.join(ep.dir, d, "*"))) for d in ("rgb", "depth", *LABEL_CHANNELS, "live_social_cost")}
    lines = {f: len(read_jsonl(os.path.join(ep.dir, f))) for f in ("robot_states.jsonl", "pedestrian_states.jsonl", "sync.jsonl", "interactions.jsonl", "peds_stream.jsonl")}
    empty_files = [os.path.relpath(f, ep.dir) for d in ("rgb", "depth", *LABEL_CHANNELS) for f in glob.glob(os.path.join(ep.dir, d, "*")) if os.path.getsize(f) == 0]
    shapes = Counter()
    unreadable, nan_only_depth, inf_depth = [], [], 0
    for k in range(n):
        img = cv2.imread(os.path.join(ep.dir, "rgb", f"{k:06d}.png"))
        if img is None:
            unreadable.append(f"rgb/{k:06d}.png")
        else:
            shapes[f"rgb {img.shape[1]}x{img.shape[0]}x{img.shape[2]} {img.dtype}"] += 1
        try:
            d = ep.depth(k)
            shapes[f"depth {d.shape[1]}x{d.shape[0]} {d.dtype}"] += 1
            inf_depth += int(np.isinf(d).sum())
            if np.isnan(d).all():
                nan_only_depth.append(k)
        except Exception:  # noqa: BLE001
            unreadable.append(f"depth/{k:06d}.npy")
        for ch in LABEL_CHANNELS:
            try:
                lab = ep.label(k, ch)
                shapes[f"{ch} {lab.shape[1]}x{lab.shape[0]} {lab.dtype}"] += 1
            except Exception:  # noqa: BLE001
                unreadable.append(f"{ch}/{k:06d}.npy")
    st = [r["stamp_ns"] for r in ep.robots]
    m = ep.meta
    return {
        "episode": ep.name, "scenario": ep.scenario, "run_seed": m.get("run_seed"), "episode_id": m.get("episode_id"), "episode_seed": m.get("episode_seed"),
        "samples": n, "expected_samples": m.get("expected_samples"), "valid_flag": m.get("valid"),
        "file_counts": counts, "jsonl_lines": lines, "shapes": dict(shapes),
        "camera": {k: ep.cam.get(k) for k in ("frame_id", "width", "height", "K", "distortion_model", "depth_units", "optical_frame_convention")},
        "frames": {"map": m["frames"]["map"], "base": m["frames"]["base"], "camera": m["frames"]["camera"],
                   "rgb_frame_ids": sorted({s["rgb_frame"] for s in ep.sync}), "depth_frame_ids": sorted({s["depth_frame"] for s in ep.sync}),
                   "pedestrian_frame_ids": sorted({s.get("frame_id", "?") for s in ep.peds_stream()[:200]})},
        "grid": ep.meta["label"]["grid"], "layer_params": ep.meta["label"]["layer_params"],
        "time_range_s": [st[0] / 1e9, st[-1] / 1e9] if st else None, "episode_start_s": (m.get("episode_start_ns") or 0) / 1e9,
        "bodies": m.get("bodies"), "pedestrians": sorted({p["name"] for ps in ep.peds for p in ps["pedestrians"]}),
        "problems": {"empty_files": empty_files, "unreadable": unreadable, "nan_only_depth_samples": nan_only_depth, "inf_depth_values": inf_depth,
                     "count_mismatch": {k: v for k, v in {**counts, **lines}.items() if k in ("rgb", "depth", "social_cost", "social_cost_membership", "robot_states.jsonl", "pedestrian_states.jsonl", "sync.jsonl") and v != n}},
    }


# ---------------------------------------------------------------- Phase 3: synchronization
def ms(v_ns: float) -> float:
    return v_ns / 1e6


def synchronization(ep: Ep) -> dict:
    n = len(ep.robots)
    m = ep.meta
    hz = m["sample_hz"]
    t0 = m["record_start_ns"]
    st = np.array([r["stamp_ns"] for r in ep.robots], dtype=np.int64)
    rgb_depth = np.array([s["rgb_stamp_ns"] - s["depth_stamp_ns"] for s in ep.sync], dtype=np.int64)
    img_vs_sample = np.array([s["stamp_ns"] - r["stamp_ns"] for s, r in zip(ep.sync, ep.robots)], dtype=np.int64)
    peds_vs_img = np.array([p["stamp_ns"] - r["stamp_ns"] for p, r in zip(ep.peds, ep.robots)], dtype=np.int64)
    before = np.array([r["stamp_ns"] - p["peds_before_ns"] for p, r in zip(ep.peds, ep.robots)], dtype=np.int64)
    after = np.array([(p["peds_after_ns"] - r["stamp_ns"]) if p["peds_after_ns"] else -1 for p, r in zip(ep.peds, ep.robots)], dtype=np.int64)
    nearest = np.minimum(before, np.where(after >= 0, after, before))
    odom = np.array([abs(r["odom_twist"]["stamp_ns"] - r["stamp_ns"]) for r in ep.robots if r.get("odom_twist")], dtype=np.int64)
    live = np.array([s["live_costmap"]["stamp_ns"] - s["stamp_ns"] for s in ep.sync if "live_costmap" in s], dtype=np.int64)
    slots = np.array([r["slot"] for r in ep.robots])
    slot_lag = st - (t0 + np.round(slots * 1e9 / hz).astype(np.int64))
    dt = np.diff(st)
    end_ns = t0 + int(m["duration_s"] * 1e9)
    in_window = bool(((st >= t0) & (st < end_ns)).all())
    monotonic = bool((dt > 0).all()) if n > 1 else True
    expected = m["expected_samples"]
    missing_slots = sorted(set(range(expected)) - set(slots.tolist()))
    failing = [(ep.name, int(k)) for k in np.nonzero(rgb_depth != 0)[0]] + [(ep.name, int(k)) for k in np.nonzero(peds_vs_img != 0)[0]]
    failing += [(ep.name, int(k)) for k in np.nonzero(before < 0)[0]]
    metrics = {
        "samples": n, "expected": expected, "missing_slots": missing_slots,
        "rgb_minus_depth_ms": {"max_abs": ms(np.abs(rgb_depth).max()) if n else None},
        "label_state_minus_image_ms": {"max_abs": ms(np.abs(peds_vs_img).max()) if n else None, "note": "pedestrian states are interpolated to the image stamp"},
        "robot_pose_minus_image_ms": {"max_abs": ms(np.abs(img_vs_sample).max()) if n else None, "note": "TF looked up at the image stamp"},
        "bracketing_peds_msgs_ms": {"before_max": ms(before.max()), "after_max": ms(after[after >= 0].max()) if (after >= 0).any() else None,
                                    "nearest_mean": ms(nearest.mean()), "nearest_max": ms(nearest.max()),
                                    "brackets_image_stamp": bool((before >= 0).all() and (after[after >= 0] >= 0).all())},
        "odom_twist_minus_image_ms": {"mean": ms(odom.mean()) if odom.size else None, "max": ms(odom.max()) if odom.size else None},
        "live_costmap_minus_image_ms": {"mean": ms(live.mean()) if live.size else None, "min": ms(live.min()) if live.size else None, "max": ms(live.max()) if live.size else None},
        "sample_interval_ms": {"mean": ms(dt.mean()) if dt.size else None, "min": ms(dt.min()) if dt.size else None, "max": ms(dt.max()) if dt.size else None, "nominal": 1000.0 / hz},
        "image_minus_slot_ms": {"mean": ms(slot_lag.mean()), "max": ms(slot_lag.max()), "min": ms(slot_lag.min())},
        "monotonic": monotonic, "inside_episode_window": in_window,
        "window_ms": [ms(t0), ms(end_ns)], "episode_start_ms": ms(m["episode_start_ns"]),
        "rgb_frames_without_depth_dropped": m["stats"]["rgb_without_depth"], "end_reason": m["stats"]["end_reason"], "time_jumps": m["stats"]["time_jumps"],
    }
    ok = (n > 0 and not failing and monotonic and in_window and metrics["bracketing_peds_msgs_ms"]["brackets_image_stamp"]
          and metrics["bracketing_peds_msgs_ms"]["nearest_max"] <= 50 and not missing_slots)
    return {"episode": ep.name, **result(PASS if ok else FAIL, metrics, failing)}


# ---------------------------------------------------------------- Phase 3: geometry
def _rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def geometry(ep: Ep) -> dict:
    n = len(ep.robots)
    out: dict = {"episode": ep.name}
    fails: list = []
    # TF: base->camera from the per-sample map poses equals the recorded static mount
    Tbc = pose_matrix(ep.cam["T_base_camera"]) if "T_base_camera" in ep.cam else None
    dev_t, dev_r = [], []
    for k, rb in enumerate(ep.robots):
        T = np.linalg.inv(pose_matrix(rb["pose_map_base"])) @ pose_matrix(rb["pose_map_camera"])
        if Tbc is not None:
            dev_t.append(float(np.linalg.norm(T[:3, 3] - Tbc[:3, 3])))
            dev_r.append(_rot_angle_deg(T[:3, :3].T @ Tbc[:3, :3]))
    optical_z = (Tbc[:3, :3] @ [0, 0, 1.0]).tolist() if Tbc is not None else None
    optical_x = (Tbc[:3, :3] @ [1.0, 0, 0]).tolist() if Tbc is not None else None
    tf_ok = Tbc is not None and max(dev_t) < 1e-3 and max(dev_r) < 0.05 and optical_z[0] > 0.99 and optical_x[1] < -0.99
    out["tf_base_camera"] = result(PASS if tf_ok else FAIL, {
        "translation_dev_max_mm": max(dev_t) * 1e3 if dev_t else None, "rotation_dev_max_deg": max(dev_r) if dev_r else None,
        "T_base_camera_xyz_m": Tbc[:3, 3].round(4).tolist() if Tbc is not None else None,
        "optical_z_in_base": [round(v, 4) for v in optical_z] if optical_z else None, "optical_x_in_base": [round(v, 4) for v in optical_x] if optical_x else None},
        note="inv(T_map_base) * T_map_camera per sample vs. the static mount; optical z must be base +x, optical x base -y (REP-103)")

    # grid: declared origin / resolution / shape, labels regenerate identically, orientation by an independent painter
    g = ep.grid
    declared_ok = (g.width, g.height) == (200, 200) and abs(g.resolution - 0.05) < 1e-12 and abs(g.origin_x + 5) < 1e-9 and abs(g.origin_y + 5) < 1e-9
    regen_bad, ref_maxdiff, ref_frac = [], 0, []
    for k in range(n):
        lab = ep.label(k)
        if not np.array_equal(paint(ep.params, peds_for_layer(ep.states(k)), g, ep.pose(k)), lab):
            regen_bad.append((ep.name, k))
        ref = reference_paint(ep.params, peds_for_layer(ep.states(k)), g, ep.pose(k))
        d = np.abs(ref.astype(int) - lab.astype(int))
        ref_maxdiff = max(ref_maxdiff, int(d.max()))
        ref_frac.append(float((d > 0).mean()))
    ref_ok = ref_maxdiff <= 1 and max(ref_frac, default=0) < 1e-3
    out["grid_definition"] = result(PASS if declared_ok and not regen_bad and ref_ok else FAIL, {
        "declared": ep.meta["label"]["grid"], "labels_regenerated_identically": n - len(regen_bad), "samples": n,
        "independent_reference_max_abs_diff": ref_maxdiff, "independent_reference_mismatch_fraction_max": max(ref_frac, default=0.0)},
        regen_bad, note="independent numpy painter on the same rotated grid; <=1 unit allowed for float rounding at the truncation boundary")

    # orientation from sensors only: depth pixels on visible people -> base frame -> label cell must carry cost
    K = np.array(ep.cam["K"]).reshape(3, 3)
    W, H = ep.cam["width"], ep.cam["height"]
    hits = total = 0
    orient_fail = []
    for k, rb in enumerate(ep.robots):
        T_cm = np.linalg.inv(pose_matrix(rb["pose_map_camera"]))
        d = None
        lab = None
        for p in ep.states(k):
            pc = T_cm @ np.array([p["x"], p["y"], 1.0, 1.0])
            if not 0.5 < pc[2] < 6.0:
                continue
            u, v = (K @ pc[:3])[:2] / pc[2]
            if not (8 <= u < W - 8 and 8 <= v < H - 8):
                continue
            d = ep.depth(k) if d is None else d
            z = d[int(v), int(u)]
            if not np.isfinite(z) or not (-0.3 <= pc[2] - z <= 0.8):
                continue  # occluded or not on this person
            # back-project the measured depth pixel, independent of the pedestrian state
            pcam = np.array([(int(u) + 0.5 - K[0, 2]) * z / K[0, 0], (int(v) + 0.5 - K[1, 2]) * z / K[1, 1], z, 1.0])
            pb = Tbc @ pcam
            c = robot_cell(g, {"x": 0.0, "y": 0.0, "yaw": 0.0}, pb[0], pb[1])
            if c is None:
                continue
            lab = ep.label(k) if lab is None else lab
            total += 1
            if lab[c[1], c[0]] > 0:
                hits += 1
            else:
                orient_fail.append((ep.name, k))
    out["label_orientation_from_depth"] = result(PASS if total and hits == total else (NOT_TESTED if not total else FAIL),
                                                 {"visible_person_pixels": total, "inside_cost": hits},
                                                 orient_fail, note="a depth pixel on a visible person, back-projected through K and T_base_camera, must fall on a cell with cost")

    # metric units: speeds from poses vs recorded twists, depth vs geometric range
    rv, odv, pv_pos, pv_rec = [], [], [], []
    for k in range(1, n):
        dt = (ep.robots[k]["stamp_ns"] - ep.robots[k - 1]["stamp_ns"]) / 1e9
        a, b = ep.robots[k - 1]["pose_map_base"], ep.robots[k]["pose_map_base"]
        rv.append(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt)
        odv.append(abs(ep.robots[k]["odom_twist"]["vx"]) if ep.robots[k].get("odom_twist") else float("nan"))
        A = {p["name"]: p for p in ep.peds[k - 1]["pedestrians"]}
        for p in ep.peds[k]["pedestrians"]:
            if p["name"] in A:
                pv_pos.append(math.hypot(p["x"] - A[p["name"]]["x"], p["y"] - A[p["name"]]["y"]) / dt)
                pv_rec.append(math.hypot(p["vx"], p["vy"]))
    rv, odv = np.array(rv), np.array(odv)
    speed_err = float(np.nanmedian(np.abs(rv - odv))) if rv.size else None
    units_ok = speed_err is not None and speed_err < 0.05 and (not pv_pos or max(pv_pos) < 2.5)
    out["metric_units"] = result(PASS if units_ok else FAIL, {
        "robot_speed_from_poses_median_mps": float(np.median(rv)) if rv.size else None, "odom_vx_median_mps": float(np.nanmedian(odv)) if odv.size else None,
        "robot_speed_pose_vs_odom_median_abs_err_mps": speed_err,
        "ped_speed_from_states_max_mps": max(pv_pos) if pv_pos else None, "ped_speed_recorded_max_mps": max(pv_rec) if pv_rec else None,
        "label_resolution_m": g.resolution, "depth_units": ep.cam.get("depth_units")},
        note="poses in metres and stamps in ns must give the odometry speed; pedestrian speeds must be human-scale")

    # values and invalid data
    vmax, bad = 0, []
    depth_invalid, depth_min, depth_max = [], math.inf, 0.0
    for k in range(n):
        for ch in LABEL_CHANNELS:
            lab = ep.label(k, ch)
            vmax = max(vmax, int(lab.max()))
            if lab.dtype != np.uint8 or lab.shape != (g.height, g.width) or lab.max() > ep.params.max_cost_value:
                bad.append((ep.name, k))
        d = ep.depth(k)
        fin = d[np.isfinite(d)]
        depth_invalid.append(float(np.isnan(d).mean()))
        if np.isinf(d).any() or fin.size == 0:
            bad.append((ep.name, k))
        if fin.size:
            depth_min, depth_max = min(depth_min, float(fin.min())), max(depth_max, float(fin.max()))
    near = ep.cam.get("clip_near", 0.3)
    out["values_and_invalid"] = result(PASS if not bad and depth_min >= 0.29 else FAIL, {
        "label_max": vmax, "label_max_allowed": ep.params.max_cost_value, "depth_min_m": depth_min, "depth_max_m": depth_max,
        "depth_invalid_fraction_mean": float(np.mean(depth_invalid)) if depth_invalid else None,
        "depth_invalid_fraction_max": float(np.max(depth_invalid)) if depth_invalid else None,
        "depth_near_clip_m": near, "inf_values": 0 if not bad else "see failing"}, bad,
        note="labels uint8 in [0, max_cost_value]; depth float32 metres, NaN = no surface (sky; the world has no ceiling), no Inf, nothing closer than the 0.3 m near clip")
    return out


# ---------------------------------------------------------------- Phase 4: semantics
def _blob_centroid(lab: np.ndarray, g: GridSpec) -> tuple[float, float] | None:
    if not lab.any():
        return None
    j, i = np.nonzero(lab)
    w = lab[j, i].astype(float)
    return float(g.origin_x + ((i + 0.5) * w).sum() / w.sum() * g.resolution), float(g.origin_y + ((j + 0.5) * w).sum() / w.sum() * g.resolution)


def _local(pose: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return c * (x - pose[0]) + s * (y - pose[1]), -s * (x - pose[0]) + c * (y - pose[1])


def _midpoint_cost(ep: Ep, k: int, a: dict, b: dict, channel: str = "social_cost") -> int | None:
    c = robot_cell(ep.grid, ep.robots[k]["pose_map_base"], (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
    return None if c is None else int(ep.label(k, channel)[c[1], c[0]])


def _personal_only(ep: Ep, k: int, a: dict, b: dict) -> int:
    """The midpoint cost the layer would paint with the interaction removed (same grid, same states)."""
    stripped = [Ped(p["x"], p["y"], p["animation_state"]) for p in ep.states(k)]
    lab = paint(ep.params, stripped, ep.grid, ep.pose(k))
    c = robot_cell(ep.grid, ep.robots[k]["pose_map_base"], (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
    return int(lab[c[1], c[0]])


def test_empty(eps: list[Ep]) -> dict:
    sel = [e for e in eps if e.scenario == "socialcost_verify_empty"]
    if not sel:
        return result(NOT_TESTED, note="no socialcost_verify_empty episode recorded")
    fail, n, peds_seen, live_nonzero, live_seen = [], 0, 0, 0, 0
    for e in sel:
        for k in range(len(e.robots)):
            n += 1
            peds_seen += len(e.states(k))
            if any(e.label(k, ch).any() for ch in LABEL_CHANNELS):
                fail.append((e.name, k))
            lp = os.path.join(e.dir, "live_social_cost", f"{k:06d}.npy")
            live_seen += int(os.path.exists(lp))
            if os.path.exists(lp) and np.load(lp).any():
                live_nonzero += 1
                fail.append((e.name, k))
    return result(PASS if n and not fail and peds_seen == 0 else FAIL,
                  {"samples": n, "pedestrians_in_states": peds_seen, "nonzero_labels": len(fail), "live_costmaps_recorded": live_seen,
                   "nonzero_live_costmaps": live_nonzero}, fail,
                  note="offline labels only when live_costmaps_recorded is 0: with nothing to paint the plugin publishes no costmap; "
                       "the live side is covered by the primed replay in G_offline_vs_live_identical_states")


def test_single(eps: list[Ep]) -> dict:
    sel = [e for e in eps if e.scenario == "socialcost_verify_single"]
    if not sel:
        return result(NOT_TESTED, note="no socialcost_verify_single episode recorded")
    fail, cent_err, peak_ok, radius_ok, checked, moved = [], [], 0, 0, 0, 0.0
    for e in sel:
        prm, g = e.params, e.grid
        expect_peak = to_cell_cost(prm, 0.5)
        first = None
        for k in range(len(e.robots)):
            ps = e.states(k)
            if len(ps) != 1:
                fail.append((e.name, k))
                continue
            p = ps[0]
            first = first or (p["x"], p["y"])
            moved = max(moved, math.dist(first, (p["x"], p["y"])))
            lx, ly = _local(e.pose(k), p["x"], p["y"])
            lab = e.label(k)
            # fully inside the grid only, so the blob is complete
            if not (g.origin_x + prm.personal_space_radius < lx < -g.origin_x - prm.personal_space_radius and g.origin_y + prm.personal_space_radius < ly < -g.origin_y - prm.personal_space_radius):
                continue
            checked += 1
            c = _blob_centroid(lab, g)
            err = math.dist(c, (lx, ly)) if c else math.inf
            cent_err.append(err)
            j, i = np.nonzero(lab)
            far = np.hypot(g.origin_x + (i + 0.5) * g.resolution - lx, g.origin_y + (j + 0.5) * g.resolution - ly).max()
            pk = int(lab.max())
            peak_ok += int(pk == expect_peak)
            radius_ok += int(far <= prm.personal_space_radius + 1e-9)
            if err > g.resolution or pk != expect_peak or far > prm.personal_space_radius + 1e-9:
                fail.append((e.name, k))
    ok = checked > 0 and not fail
    return result(PASS if ok else FAIL, {"samples_with_full_disk": checked, "centroid_error_max_m": max(cent_err) if cent_err else None,
                                         "centroid_error_mean_m": float(np.mean(cent_err)) if cent_err else None,
                                         "peak_value_expected": to_cell_cost(sel[0].params, 0.5), "peak_ok": peak_ok,
                                         "max_cost_cell_distance_within_radius": radius_ok, "pedestrian_drift_m": moved}, fail,
                  note="isotropic disk: centroid on the person (< 1 cell), peak = toCellCost(base amplitude 0.5), no cost beyond personal_space_radius")


def test_conversation(eps: list[Ep]) -> dict:
    on = [e for e in eps if e.scenario == "socialcost_conversation"]
    off = [e for e in eps if e.scenario == "socialcost_verify_conversation_off"]
    if not on:
        return result(NOT_TESTED, note="no conversation episode")

    def collect(sel: list[Ep]) -> dict:
        mids, persons, ids, types, extra = [], [], Counter(), Counter(), []
        for e in sel:
            for k in range(len(e.robots)):
                ps = {p["name"].split("/")[-1]: p for p in e.states(k)}
                if "talker_a" not in ps or "talker_b" not in ps:
                    continue
                a, b = ps["talker_a"], ps["talker_b"]
                ids[(a["interaction_id"], b["interaction_id"])] += 1
                types[(a["interaction_type"], b["interaction_type"])] += 1
                mc = _midpoint_cost(e, k, a, b)
                if mc is None:
                    continue
                mids.append(mc)
                extra.append(mc - _personal_only(e, k, a, b))
                for p in (a, b):
                    c = robot_cell(e.grid, e.robots[k]["pose_map_base"], p["x"], p["y"])
                    if c:
                        persons.append(int(e.label(k)[c[1], c[0]]))
        return {"midpoint_samples": len(mids), "midpoint_cost_min": min(mids) if mids else None, "midpoint_cost_max": max(mids) if mids else None,
                "interaction_cost_above_personal_only_min": min(extra) if extra else None, "interaction_cost_above_personal_only_max": max(extra) if extra else None,
                "person_cell_cost_min": min(persons) if persons else None, "interaction_ids": {str(k): v for k, v in ids.items()},
                "interaction_types": {str(k): v for k, v in types.items()}}

    m_on = collect(on)
    prm = on[0].params
    w = prm.interaction_weights[0]
    peak = to_cell_cost(prm, w * math.exp(-((on[0].grid.resolution ** 2 / 2) / (2 * prm.group_sigma ** 2)) ** prm.falloff_shape))
    metrics = {"with_interaction": m_on, "expected_midpoint_min": peak, "talk_to_weight": w}
    fails = []
    ok_on = m_on["midpoint_samples"] > 0 and m_on["midpoint_cost_min"] >= peak and m_on["interaction_cost_above_personal_only_min"] > 0 and m_on["person_cell_cost_min"] >= to_cell_cost(prm, 0.5)
    if not ok_on:
        fails.append("interaction region missing or weak in the conversation")
    if off:
        m_off = collect(off)
        metrics["interaction_disabled"] = m_off
        ok_off = m_off["midpoint_samples"] > 0 and m_off["interaction_cost_above_personal_only_max"] == 0 and set(m_off["interaction_ids"]) == {"(-1, -1)"}
        if not ok_off:
            fails.append("control has interaction cost")
        status = PASS if ok_on and ok_off else FAIL
    else:
        status = FAIL if not ok_on else PASS
        metrics["interaction_disabled"] = "NOT_TESTED (no socialcost_verify_conversation_off episode)"
    return result(status, metrics, note="; ".join(fails) or "midpoint between the talkers: capsule with TALK_TO weight vs personal space only")


def test_queue(eps: list[Ep]) -> dict:
    sel = [e for e in eps if e.scenario == "socialcost_queue"]
    if not sel:
        return result(NOT_TESTED, note="no queue episode")
    roles, published, pairs_exact, pairs_full, n_samples = Counter(), Counter(), [], [], 0
    peak = None
    for e in sel:
        prm = e.params
        peak = to_cell_cost(prm, prm.interaction_weights[6] * math.exp(-((e.grid.resolution ** 2 / 2) / (2 * prm.group_sigma ** 2)) ** prm.falloff_shape))
        for k in range(len(e.robots)):
            f = interaction_frame_at(e.frames, e.robots[k]["stamp_ns"])
            q = [i for i in (f["interactions"] if f else []) if i["interaction_type"] == 6 and i["outcome"] == 1]
            if not q:
                continue
            n_samples += 1
            for r in q[0]["member_roles"]:
                roles["participant" if r == 0 else "queued"] += 1
            ps = e.states(k)
            published[sum(1 for p in ps if p["interaction_type"] == 6 and p["interaction_id"] >= 0)] += 1
            members = [p for p in e.states(k, membership=True) if p["interaction_id"] == q[0]["interaction_id"]]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    me = _midpoint_cost(e, k, members[a], members[b], "social_cost")
                    mf = _midpoint_cost(e, k, members[a], members[b], "social_cost_membership")
                    if me is not None:
                        pairs_exact.append(me - _personal_only(e, k, members[a], members[b]))
                        pairs_full.append(mf)
    queue_term_in_layer = bool(pairs_exact) and max(pairs_exact) > 0
    membership_ok = bool(pairs_full) and min(pairs_full) >= peak
    metrics = {"samples_with_active_queue": n_samples, "member_roles_counted": dict(roles),
               "published_queue_members_per_sample": {str(k): v for k, v in published.items()},
               "pairs_checked": len(pairs_exact), "arena_layer_extra_cost_at_pair_midpoints_max": max(pairs_exact) if pairs_exact else None,
               "membership_channel_midpoint_min": min(pairs_full) if pairs_full else None, "expected_queue_capsule_min": peak}
    note = ("humansim marks 1 participant + queued members in InteractionFrame, but publishes Pedestrian.interaction_id only for the "
            "active participant, so the Arena layer paints no queue region (QUEUE_USE weight never fires). "
            "social_cost_membership adds it with the layer's own QUEUE_USE term.")
    status = PASS if n_samples and not queue_term_in_layer and membership_ok and roles.get("queued") else FAIL
    return result(status, metrics, note=note)


def test_group_walk(eps: list[Ep]) -> dict:
    sel = [e for e in eps if e.scenario == "socialcost_group_walk"]
    if not sel:
        return result(NOT_TESTED, note="no group-walk episode")
    meta_bad, track_err, fails, moved, n = [], [], [], [], 0
    for e in sel:
        prev = None
        for k in range(len(e.robots)):
            ps = e.states(k)
            ids = {(p["interaction_id"], p["interaction_type"]) for p in ps}
            if len(ps) != 3 or len(ids) != 1 or next(iter(ids))[1] != 12 or next(iter(ids))[0] < 0:
                meta_bad.append((e.name, k))
            pose = e.pose(k)
            lab = e.label(k)
            # each walker fully inside the grid: label must match the layer at the walker's current position
            inside = all(abs(c) < 5 - e.params.personal_space_radius for p in ps for c in _local(pose, p["x"], p["y"]))
            if inside:
                n += 1
                want = reference_paint(e.params, peds_for_layer(ps), e.grid, pose)
                d = int(np.abs(want.astype(int) - lab.astype(int)).max())
                track_err.append(d)
                if d > 1:
                    fails.append((e.name, k))
                cen = _blob_centroid(lab, e.grid)
                exp = np.mean([_local(pose, p["x"], p["y"]) for p in ps], axis=0)
                if cen is not None and math.dist(cen, exp) > 0.1:
                    fails.append((e.name, k))
            if prev is not None:
                moved.append(math.dist(prev, (ps[0]["x"], ps[0]["y"])))
            prev = (ps[0]["x"], ps[0]["y"])
    return result(PASS if n and not fails and not meta_bad else FAIL, {
        "samples_group_fully_in_grid": n, "label_vs_current_positions_max_abs_diff": max(track_err) if track_err else None,
        "group_metadata_bad_samples": len(meta_bad), "leader_step_per_sample_mean_m": float(np.mean(moved)) if moved else None,
        "group_walk_weight": sel[0].params.interaction_weights[12]}, fails + meta_bad,
        note="labels follow the walkers sample by sample (independent painter at the sample's positions, blob centroid on the group centroid); GROUP_WALK weight is 0 so no capsule")


def test_lifecycle(eps: list[Ep]) -> dict:
    sel = [e for e in eps if e.scenario == "socialcost_verify_lifecycle"]
    if not sel:
        return result(NOT_TESTED, note="no socialcost_verify_lifecycle episode recorded")
    phases, fails, lingering, live_linger = [], [], 0, 0
    for e in sel:
        prm = e.params
        for k in range(len(e.robots)):
            ps = {p["name"].split("/")[-1]: p for p in e.states(k)}
            a, b = ps.get("talker_a"), ps.get("talker_b")
            if a is None or b is None:
                continue
            active = a["interaction_id"] >= 0 and a["interaction_id"] == b["interaction_id"]
            f = interaction_frame_at(e.frames, e.robots[k]["stamp_ns"])
            frame_active = bool(f) and any(i["outcome"] == 1 for i in f["interactions"])
            phase = "during" if active else ("before" if not phases or all(p[1] == "before" for p in phases) else "after")
            mc = _midpoint_cost(e, k, a, b)
            base = _personal_only(e, k, a, b) if mc is not None else None
            extra = None if mc is None else mc - base
            full = _midpoint_cost(e, k, a, b, "social_cost_membership")
            phases.append((k, phase, active, frame_active, extra, (full - base) if full is not None and base is not None else None, math.dist((a["x"], a["y"]), (b["x"], b["y"]))))
            if extra is None:
                continue
            if active and extra <= 0:
                fails.append((e.name, k))
            if not active and (extra > 0 or (full is not None and full - base > 0)):
                lingering += 1
                fails.append((e.name, k))
            lp = os.path.join(e.dir, "live_social_cost", f"{k:06d}.npy")
            sync = e.sync[k]
            if not active and os.path.exists(lp) and "live_costmap" in sync:
                lc = sync["live_costmap"]
                g = GridSpec(resolution=round(lc["resolution"], 6), width=lc["size_x"], height=lc["size_y"], origin_x=lc["origin_x"], origin_y=lc["origin_y"])
                i = int((((a["x"] + b["x"]) / 2) - g.origin_x) / g.resolution)
                j = int((((a["y"] + b["y"]) / 2) - g.origin_y) / g.resolution)
                if 0 <= i < g.width and 0 <= j < g.height:
                    live_mid = int(np.load(lp)[j, i])
                    if live_mid > base + 1:
                        live_linger += 1
    seen = Counter(p[1] for p in phases)
    trans = [(phases[i][0], phases[i - 1][1], phases[i][1]) for i in range(1, len(phases)) if phases[i][1] != phases[i - 1][1]]
    ok = all(seen.get(ph) for ph in ("before", "during", "after")) and not fails and live_linger == 0
    status = PASS if ok else (NOT_TESTED if not all(seen.get(ph) for ph in ("before", "during", "after")) and not fails else FAIL)
    return result(status, {"samples_per_phase": dict(seen), "transitions_at_sample": trans,
                           "lingering_cost_samples_offline": lingering, "lingering_cost_samples_live": live_linger,
                           "trace": [{"sample": p[0], "phase": p[1], "published_active": p[2], "frame_active": p[3],
                                      "midpoint_extra_cost": p[4], "membership_extra_cost": p[5], "separation_m": round(p[6], 2)} for p in phases[::5]]},
                  fails, note="interaction cost present iff the interaction is active on that sample; none before activation or after release (offline labels and the live layer)")


def test_live_recorded(eps: list[Ep]) -> dict:
    """In-simulation live layer vs offline repaint on the live grid, pedestrians at the live stamp."""
    recs = []
    for e in eps:
        for a in e.meta.get("live_agreement") or []:
            recs.append({**a, "episode": e.name, "scenario": e.scenario})
    if not recs:
        return result(NOT_TESTED, note="no live costmaps recorded")
    by = {}
    for r in recs:
        by.setdefault(r["scenario"], []).append(r)
    summary = {s: {"samples": len(v), "identical_samples": sum(r["max_abs"] == 0 for r in v), "mae_mean": float(np.mean([r["mae"] for r in v])),
                   "max_abs": max(r["max_abs"] for r in v), "mismatched_fraction_mean": float(np.mean([1 - r["exact_fraction"] for r in v])),
                   "iou_nonzero_mean": float(np.mean([r["iou_nonzero"] for r in v])), "live_minus_image_s_mean": float(np.mean([r["live_minus_image_stamp_s"] for r in v]))}
               for s, v in by.items()}
    return result(PASS, summary, note="not identical states: the plugin paints the newest arena_peds at its own update; residuals appear only with moving people")


def failing_samples(report: dict) -> list[tuple[str, int, str]]:
    out = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("status") == FAIL:
                for f in node.get("failing", []):
                    if isinstance(f, (list, tuple)) and len(f) == 2:
                        out.append((f[0], int(f[1]), path))
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for v in node:
                walk(v, path)
    walk(report, "")
    return out
