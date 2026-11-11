"""Checks on paired ON/OFF episodes: matched geometry, visual difference, label difference."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

from .qa import pose_matrix
from .verify import Ep

CHANNELS = ("social_cost", "social_cost_membership", "social_cost_extended")
# Pedestrian geometry is the controlled quantity. The robot drives closed loop on bodies-only Nav2, so
# its heading jitters by a few degrees when the pedestrian body differs physically (wheelchair).
LIMITS = {"ped_pos_m": 0.10, "ped_heading_deg": 5.0, "robot_pos_m": 0.15, "robot_heading_p95_deg": 5.0, "robot_heading_deg": 10.0}


def cue_state(p: dict) -> list[str]:
    """Visual cue names carried by a pedestrian record (gesture slots and body clips)."""
    cues = []
    for g in p.get("gestures") or []:
        if isinstance(g, dict):
            if str(g.get("slot", "")).startswith("halt"):
                cues.append("halt")
            elif g.get("slot") == "body" and g.get("clip"):
                cues.append(f"clip:{g['clip']}")
            elif g.get("slot") in ("arm", "arm_l", "arm_r"):
                cues.append("point")
    if "wheelchair" in os.path.basename(str(p.get("model_uri", ""))).lower():
        cues.append("wheelchair")
    return sorted(set(cues))


def person_roi(ep: Ep, k: int, states: list[dict]) -> np.ndarray:
    """Image mask of boxes around every projected person (0.6 m wide, 0 to 1.9 m tall)."""
    K = np.array(ep.cam["K"]).reshape(3, 3)
    W, H = ep.cam["width"], ep.cam["height"]
    T = np.linalg.inv(pose_matrix(ep.robots[k]["pose_map_camera"]))
    mask = np.zeros((H, W), bool)
    for p in states:
        uv = []
        for dx, dy, z in ((-0.3, -0.3, 0.0), (0.3, 0.3, 1.9), (-0.3, 0.3, 1.9), (0.3, -0.3, 0.0)):
            pc = T @ np.array([p["x"] + dx, p["y"] + dy, z, 1.0])
            if pc[2] > 0.2:
                uv.append((K @ pc[:3])[:2] / pc[2])
        if len(uv) < 2:
            continue
        uv = np.array(uv)
        u0, v0 = np.clip(uv.min(0), 0, [W, H]).astype(int)
        u1, v1 = np.clip(uv.max(0), 0, [W, H]).astype(int)
        mask[v0:v1, u0:u1] = True
    return mask


def compare_pair(on: Ep, off: Ep) -> dict:
    slots_on = {r["slot"]: i for i, r in enumerate(on.robots)}
    slots_off = {r["slot"]: i for i, r in enumerate(off.robots)}
    common = sorted(set(slots_on) & set(slots_off))
    ped_pos, ped_head, rob_pos, rob_head = [], [], [], []
    rgb_all, rgb_roi, depth_roi = [], [], []
    lab = {c: {"l1": [], "l2": [], "cells": [], "roi_l1": []} for c in CHANNELS}
    cues_on, cues_off = set(), set()
    rows = []
    for s in common:
        i, j = slots_on[s], slots_off[s]
        A = {p["name"]: p for p in on.states(i)}
        B = {p["name"]: p for p in off.states(j)}
        for n in A.keys() & B.keys():
            ped_pos.append(math.dist((A[n]["x"], A[n]["y"]), (B[n]["x"], B[n]["y"])))
            ped_head.append(abs(math.degrees(math.atan2(math.sin(A[n]["yaw"] - B[n]["yaw"]), math.cos(A[n]["yaw"] - B[n]["yaw"])))))
            cues_on.update(cue_state(A[n]))
            cues_off.update(cue_state(B[n]))
        ra, rb = on.robots[i]["pose_map_base"], off.robots[j]["pose_map_base"]
        rob_pos.append(math.dist((ra["x"], ra["y"]), (rb["x"], rb["y"])))
        rob_head.append(abs(math.degrees(math.atan2(math.sin(ra["yaw"] - rb["yaw"]), math.cos(ra["yaw"] - rb["yaw"])))))
        ia = cv2.imread(os.path.join(on.dir, "rgb", f"{i:06d}.png")).astype(np.float32)
        ib = cv2.imread(os.path.join(off.dir, "rgb", f"{j:06d}.png")).astype(np.float32)
        da, db = on.depth(i), off.depth(j)
        roi = person_roi(on, i, list(A.values()))
        d = np.abs(ia - ib).mean(axis=2)
        rgb_all.append(float(d.mean()))
        if roi.any():
            rgb_roi.append(float(d[roi].mean()))
            dd = np.abs(da - db)[roi]
            dd = dd[np.isfinite(dd)]
            if dd.size:
                depth_roi.append(float(dd.mean()))
        for c in CHANNELS:
            if not os.path.exists(os.path.join(on.dir, c, f"{i:06d}.npy")):
                continue
            x, y = on.label(i, c).astype(float), off.label(j, c).astype(float)
            diff = x - y
            lab[c]["l1"].append(float(np.abs(diff).mean()))
            lab[c]["l2"].append(float(np.sqrt((diff ** 2).mean())))
            lab[c]["cells"].append(int((diff != 0).sum()))
            region = (x > 0) | (y > 0)
            if region.any():
                lab[c]["roi_l1"].append(float(np.abs(diff[region]).mean()))
        rows.append({"slot": s, "rgb_roi_absdiff": rgb_roi[-1] if roi.any() else None})

    def mx(v):
        return float(max(v)) if v else None

    def mean(v):
        return float(np.mean(v)) if v else None

    geom = {"ped_pos_max_m": mx(ped_pos), "ped_pos_mean_m": mean(ped_pos), "ped_heading_max_deg": mx(ped_head), "robot_pos_max_m": mx(rob_pos),
            "robot_pos_mean_m": mean(rob_pos), "robot_heading_max_deg": mx(rob_head),
            "robot_heading_p95_deg": float(np.percentile(rob_head, 95)) if rob_head else None}
    violations = [k for k, lim in (("ped_pos_max_m", LIMITS["ped_pos_m"]), ("ped_heading_max_deg", LIMITS["ped_heading_deg"]),
                                   ("robot_pos_max_m", LIMITS["robot_pos_m"]), ("robot_heading_p95_deg", LIMITS["robot_heading_p95_deg"]),
                                   ("robot_heading_max_deg", LIMITS["robot_heading_deg"])) if geom[k] is not None and geom[k] > lim]
    return {
        "pair_id": on.meta.get("pair_id"), "family": on.meta.get("family"), "on": on.name, "off": off.name, "slots_compared": len(common),
        "geometry": geom, "geometry_violations": violations, "matched_geometry": not violations,
        "visual": {"rgb_absdiff_mean": mean(rgb_all), "rgb_person_roi_absdiff_mean": mean(rgb_roi), "depth_person_roi_absdiff_mean_m": mean(depth_roi),
                   "samples_with_person_in_view": len(rgb_roi)},
        "cues": {"on": sorted(cues_on), "off": sorted(cues_off), "differ": cues_on != cues_off},
        "labels": {c: {"l1_mean": mean(v["l1"]), "l2_mean": mean(v["l2"]), "cells_changed_mean": mean(v["cells"]), "cells_changed_max": mx(v["cells"]),
                       "region_l1_mean": mean(v["roi_l1"])} for c, v in lab.items()},
        "per_slot": rows[::10],
    }
