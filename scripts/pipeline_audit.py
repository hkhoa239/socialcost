#!/usr/bin/env python3
"""Gate 5 (v2): does Nav2 consume the published social cost grid as intended, for ANY estimator?

Compares each merged costmap against the grid the estimator actually published (not against an
oracle label), so it isolates integration from prediction quality. Per local (and global) costmap
update:
  - geometry: mask frame/resolution/size/origin lattice vs the costmap
  - which recently received mask explains the costmap best (lag 0 = latest), and its age
    (costmap stamp - mask stamp, simulator time)
  - cost preservation under the configured MAX merge: merged >= v * 254 / 100 - tol on every cell
    the mask paints (v = published OccupancyGrid value), sampled exactly as nav2_mask_overlay_layer
    does (cell centre -> floor((x - origin) / res)); and the same for the mask shifted by +-1 cell
    (a better shifted fit means misalignment)
  - robot speed at the costmap stamp (static vs moving)
Mask stream: inter-arrival gaps, receive latency (receive sim time - stamp), dropped updates
(estimator seq gaps), clipping (v == 99).

    python pipeline_audit.py --out audit.jsonl --summary audit.json [--duration 600] [--until-episode-end]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import rclpy
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from task_generator_msgs.msg import EpisodeRecord
from tf2_ros import Buffer, TransformListener

ENV = "/arena/env_0"
NS = f"{ENV}/task_generator_node/turtlebot"
RUNNING = 1
TOL = 3  # cost units: occupancy quantisation (254 / 100 = 2.54)


def sns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def mask_cost(v: np.ndarray) -> np.ndarray:
    """nav2_mask_overlay_layer interpretMaskValue: 0 -> 0, 1..99 -> v * 254 / 100 (int), 100 -> 254, <0 -> no info (-1)."""
    out = (v.astype(np.int32) * 254) // 100
    out[v == 100] = 254
    out[(v < 0) | (v > 100)] = -1
    return out


class Audit(Node):
    def __init__(self, keep: int) -> None:
        super().__init__("socialcost_pipeline_audit", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.tf = Buffer(cache_time=Duration(seconds=30))
        TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.masks: deque = deque(maxlen=keep)
        self.mask_log: list = []
        self.local = None
        self.glob = None
        self.odom = deque(maxlen=400)
        self.episode = None
        self.create_subscription(OccupancyGrid, f"{ENV}/socialcost/estimate", self.on_mask, latched)
        self.create_subscription(Costmap, f"{NS}/local_costmap/costmap_raw", lambda m: setattr(self, "local", m), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Costmap, f"{NS}/global_costmap/costmap_raw", lambda m: setattr(self, "glob", m), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Odometry, f"{NS}/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(EpisodeRecord, f"{ENV}/task_generator_node/state/episode", lambda m: setattr(self, "episode", m),
                                 QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def on_mask(self, m: OccupancyGrid) -> None:
        now = self.get_clock().now().nanoseconds
        v = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        rec = {"stamp_ns": sns(m.header.stamp), "recv_ns": now, "frame": m.header.frame_id, "res": m.info.resolution, "w": m.info.width, "h": m.info.height,
               "ox": m.info.origin.position.x, "oy": m.info.origin.position.y, "v": v}
        self.masks.append(rec)
        self.mask_log.append({k: rec[k] for k in ("stamp_ns", "recv_ns")} | {"nonzero": int((v > 0).sum()), "clipped": int((v >= 99).sum()), "vmax": int(v.max()),
                                                                         "frame": rec["frame"], "res": rec["res"], "w": rec["w"], "h": rec["h"], "ox": rec["ox"], "oy": rec["oy"]})

    def on_odom(self, m: Odometry) -> None:
        self.odom.append((sns(m.header.stamp), math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y), abs(m.twist.twist.angular.z)))

    def speed_at(self, t_ns: int):
        if not self.odom:
            return None, None
        best = min(self.odom, key=lambda o: abs(o[0] - t_ns))
        return best[1], best[2]


def expected_from_mask(mk: dict, md, frame: str, T_mask_from_cm: tuple | None, shift: tuple = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """Expected social cost on the costmap's cells from one mask (and whether the mask covers each cell)."""
    i, j = np.meshgrid(np.arange(md.size_x), np.arange(md.size_y))
    wx = md.origin.position.x + (i + 0.5) * md.resolution
    wy = md.origin.position.y + (j + 0.5) * md.resolution
    if T_mask_from_cm is not None:
        tx, ty, yaw = T_mask_from_cm
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = tx + c * wx - s * wy, ty + s * wx + c * wy
    rx = wx - mk["ox"]
    ry = wy - mk["oy"]
    col = np.floor(rx / mk["res"]).astype(np.int64) + shift[0]
    row = np.floor(ry / mk["res"]).astype(np.int64) + shift[1]
    inside = (rx >= 0) & (ry >= 0) & (col >= 0) & (row >= 0) & (col < mk["w"]) & (row < mk["h"])
    exp = np.zeros(wx.shape, np.int32)
    exp[inside] = mask_cost(mk["v"][row[inside], col[inside]])
    exp[exp < 0] = 0
    return exp, inside


def check(cm, masks: list, T_by_frame: dict, window: float | None = None, robot_xy=None) -> dict | None:
    md = cm.metadata
    merged = np.frombuffer(bytes(cm.data), np.uint8).reshape(md.size_y, md.size_x).astype(np.int32)
    cm_stamp = sns(cm.header.stamp)
    best = None
    first_eligible = True
    for lag, mk in enumerate(reversed(masks)):
        if mk["stamp_ns"] > cm_stamp + 50_000_000:  # a mask stamped after the costmap cannot be in it
            continue
        T = None if mk["frame"] == cm.header.frame_id else T_by_frame.get((mk["frame"], cm.header.frame_id))
        if mk["frame"] != cm.header.frame_id and T is None:
            continue
        exp, inside = expected_from_mask(mk, md, cm.header.frame_id, T)
        sel = exp > 0
        if window and robot_xy is not None:
            i, j = np.meshgrid(np.arange(md.size_x), np.arange(md.size_y))
            cx = md.origin.position.x + (i + 0.5) * md.resolution
            cy = md.origin.position.y + (j + 0.5) * md.resolution
            sel &= (np.abs(cx - robot_xy[0]) <= window / 2) & (np.abs(cy - robot_xy[1]) <= window / 2)
        n = int(sel.sum())
        if n == 0:
            cov = None
        else:
            cov = float((merged[sel] >= exp[sel] - TOL).mean())
        r = {"lag": lag, "mask_stamp_ns": mk["stamp_ns"], "cells": n, "coverage": cov, "mask": mk, "T": T, "sel": sel, "exp": exp}
        if first_eligible and n == 0:
            # the newest mask Nav2 can have paints nothing: the costmap has no social cost to check (do not fall
            # back to older masks, whose cells are correctly gone)
            best = r
            break
        first_eligible = False
        if best is None or (cov is not None and (best["coverage"] is None or cov > best["coverage"] + 1e-9)):
            best = r
        if cov is not None and cov >= 0.9999:
            break
    if best is None:
        return None
    out = {"cm_stamp_ns": cm_stamp, "cm_frame": cm.header.frame_id, "cm_res": md.resolution, "cm_size": [md.size_x, md.size_y],
           "cm_origin": [md.origin.position.x, md.origin.position.y], "lag": best["lag"], "mask_stamp_ns": best["mask_stamp_ns"],
           "age_ms": (cm_stamp - best["mask_stamp_ns"]) / 1e6, "cells": best["cells"], "coverage": best["coverage"]}
    mk = best["mask"]
    out["mask_frame"] = mk["frame"]
    out["same_frame"] = mk["frame"] == cm.header.frame_id
    out["same_res"] = abs(mk["res"] - md.resolution) < 1e-9
    off = ((mk["ox"] - md.origin.position.x) / md.resolution, (mk["oy"] - md.origin.position.y) / md.resolution)
    out["lattice_offset_cells"] = [round(off[0] - round(off[0]), 6), round(off[1] - round(off[1]), 6)]
    if best["cells"]:
        sel, exp = best["sel"], best["exp"]
        out["exact_fraction"] = float((merged[sel] == exp[sel]).mean())
        out["exp_max"] = int(exp[sel].max())
        out["merged_max_in_mask"] = int(merged[sel].max())
        out["shortfall_max"] = int(np.max(exp[sel] - merged[sel]))
        # alignment test: does the same mask fit better shifted by one cell?
        shifted = {}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            e2, _ = expected_from_mask(mk, md, cm.header.frame_id, best["T"], (dx, dy))
            s2 = e2 > 0
            shifted[f"{dx},{dy}"] = float((merged[s2] >= e2[s2] - TOL).mean()) if s2.any() else None
        out["shifted_coverage"] = shifted
    # excess: non-lethal cost where the mask paints nothing (inside the footprint) and outside the footprint
    _, inside = expected_from_mask(mk, md, cm.header.frame_id, best["T"])
    nl = (merged >= 1) & (merged <= 252)
    out["excess_in_footprint"] = int((nl & inside & (best["exp"] == 0)).sum())
    out["nonlethal_outside_footprint"] = int((nl & ~inside).sum())
    return out


def summarise(rows: list[dict], masks: list[dict], est_log: str | None) -> dict:
    def stats(vals):
        v = np.array([x for x in vals if x is not None], float)
        if not len(v):
            return None
        return {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "p05": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    out = {}
    def is_static(r):
        return r.get("v") is not None and r["v"] <= 0.02 and (r.get("w") or 0) <= 0.05
    # the first mask ever published can be stamped at (or just before) a costmap that was computed before Nav2
    # received it; with no earlier mask to compare, such a costmap is a start-up artifact, reported separately
    first_stamp = masks[0]["stamp_ns"] if masks else None
    def startup(r):
        return r["mask_stamp_ns"] == first_stamp and (r["coverage"] or 0) == 0
    out["startup_first_mask_costmaps"] = sum(1 for r in rows if r.get("cells") and startup(r))
    # rows recorded with the pre-fix matcher: if an empty mask newer than the matched one was eligible for this
    # costmap, the costmap had nothing to check (the matcher fell back to an older, superseded mask)
    empty = sorted(m["stamp_ns"] for m in masks if m["nonzero"] == 0)
    import bisect
    def superseded_by_empty(r):
        k = bisect.bisect_right(empty, r["mask_stamp_ns"])
        return k < len(empty) and empty[k] <= r["cm_stamp_ns"] + 50_000_000
    corrected = 0
    for r in rows:
        if r.get("cells") and r.get("lag", 0) > 0 and superseded_by_empty(r):
            r["cells"] = 0
            r["corrected_empty_mask"] = True
            corrected += 1
    out["rows_corrected_empty_newest_mask"] = corrected
    for which in ("local", "global"):
        A = [r for r in rows if r["costmap"] == which and r.get("cells") and not startup(r)]
        R = [r for r in A if r.get("phase", "scored") in ("scored", "static_test")]
        o = {"updates_with_social_cells": len(R), "updates_total": sum(1 for r in rows if r["costmap"] == which),
             "transients": {ph: {"n": len([r for r in A if r.get("phase") == ph]),
                                 "coverage_min": min((r["coverage"] for r in A if r.get("phase") == ph and r["coverage"] is not None), default=None)}
                            for ph in ("pre", "transient")}}
        for phase, sel in (("all", R), ("moving", [r for r in R if r.get("phase", "scored") == "scored" and not is_static(r)]),
                           ("static", [r for r in R if r.get("phase") == "static_test" and is_static(r)]),
                           ("static_any", [r for r in R if is_static(r)])):
            o[phase] = {"n": len(sel), "coverage": stats([r["coverage"] for r in sel]), "exact_fraction": stats([r.get("exact_fraction") for r in sel]),
                        "age_ms": stats([r["age_ms"] for r in sel]), "lag": stats([r["lag"] for r in sel]),
                        "best_shift_beats_unshifted": int(sum(1 for r in sel if r.get("shifted_coverage") and
                                                             max(x for x in r["shifted_coverage"].values() if x is not None) > r["coverage"] + 0.02))}
        o["excess_in_footprint"] = stats([r.get("excess_in_footprint") for r in R])
        o["nonlethal_outside_footprint"] = stats([r.get("nonlethal_outside_footprint") for r in R])
        o["frames"] = sorted({(r["mask_frame"], r["cm_frame"]) for r in R})
        o["same_res_all"] = all(r["same_res"] for r in R) if R else None
        o["lattice_offset_max"] = max((max(abs(x) for x in r["lattice_offset_cells"]) for r in R), default=None)
        out[which] = o
    st = np.array([m["stamp_ns"] for m in masks], float)
    rv = np.array([m["recv_ns"] for m in masks], float)
    out["mask_stream"] = {"received": len(masks), "interval_ms": stats(list(np.diff(st) / 1e6)) if len(st) > 1 else None,
                          "receive_latency_ms": stats(list((rv - st) / 1e6)), "non_monotonic_stamps": int((np.diff(st) <= 0).sum()) if len(st) > 1 else 0,
                          "clipped_cells_max": max((m["clipped"] for m in masks), default=0), "nonzero": stats([m["nonzero"] for m in masks]),
                          "geometry": sorted({(m["frame"], round(m["res"], 6), m["w"], m["h"]) for m in masks})}
    if est_log and os.path.exists(est_log):
        E = [json.loads(line) for line in open(est_log)]
        seqs = [e["seq"] for e in E if "seq" in e]
        out["estimator"] = {"published": len(E), "seq_gaps": int(sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)),
                            "compute_ms": stats([e["compute_ms"] for e in E]),
                            "input_to_publish_ms": stats([(e["t_ns"] - e["input_stamp_ns"]) / 1e6 for e in E]),
                            "stamp_matches_input": True}
        # published stamps should equal input stamps: compare the received mask stamps against the logged input stamps
        logged = {e["input_stamp_ns"] for e in E}
        out["estimator"]["received_stamps_in_log_frac"] = float(np.mean([m["stamp_ns"] in logged for m in masks])) if masks else None
        out["estimator"]["dropped_frac"] = 1.0 - len(masks) / len(E) if E else None
        sh = [e["shadow"] for e in E if "shadow" in e]
        if sh:
            agg = {k: sum(s[k] for s in sh) for k in sh[0] if k != "mae_fov"}
            out["estimator"]["shadow_vs_oracle_ext"] = {"ticks": len(sh), "mae_fov": float(np.mean([s["mae_fov"] for s in sh])),
                                                        **{f"precision@{t}": agg[f"tp{t}"] / max(agg[f"tp{t}"] + agg[f"fp{t}"], 1) for t in (10, 50, 100)},
                                                        **{f"recall@{t}": agg[f"tp{t}"] / max(agg[f"tp{t}"] + agg[f"fn{t}"], 1) for t in (10, 50, 100)},
                                                        "pred_nonzero_fov_mean": float(np.mean([s["pred_nonzero_fov"] for s in sh])),
                                                        "label_nonzero_fov_mean": float(np.mean([s["label_nonzero_fov"] for s in sh]))}
    # verdict (thresholds fixed before the runs)
    L = out["local"]
    crit = {
        "geometry": bool(L["updates_with_social_cells"] > 0 and L["same_res_all"] and (L["lattice_offset_max"] or 0) < 1e-3
                         and all(a == b for a, b in L["frames"]) and len(out["mask_stream"]["geometry"]) == 1
                         and out["mask_stream"]["non_monotonic_stamps"] == 0),
        "timestamps": bool(out.get("estimator", {}).get("received_stamps_in_log_frac", 1.0) >= 0.999 and
                           (L["all"]["age_ms"] or {}).get("min", 0) >= -50),
        "cost_preserved": bool(L["all"]["coverage"] and L["all"]["coverage"]["mean"] >= 0.999 and L["all"]["coverage"]["p05"] >= 0.99),
        "aligned": bool(L["all"]["n"] and L["all"]["best_shift_beats_unshifted"] <= 0.01 * L["all"]["n"]),
        "latency_bounded": bool(L["all"]["age_ms"] and L["all"]["age_ms"]["p95"] <= 500 and L["all"]["age_ms"]["max"] <= 1000
                                and out["mask_stream"]["interval_ms"] and out["mask_stream"]["interval_ms"]["p95"] <= 400),
        "planner_receives_while_moving": bool(L["moving"]["n"] >= 50 and L["moving"]["coverage"] and L["moving"]["coverage"]["mean"] >= 0.999
                                              and L["moving"]["coverage"]["p05"] >= 0.99),
        # None = not evaluable: the robot stood still but the estimator painted no social cost to check
        "static_robot": (None if L["static"]["n"] == 0 and any(r.get("phase") == "static_test" for r in rows)
                         else bool(L["static"]["n"] >= 50 and L["static"]["coverage"] and L["static"]["coverage"]["mean"] >= 0.999
                                   and L["static"]["coverage"]["p05"] >= 0.99)),
    }
    G = out["global"]
    crit["global_costmap_cost_preserved"] = bool(G["all"]["coverage"] and G["all"]["coverage"]["mean"] >= 0.99) if G["updates_with_social_cells"] else None
    out["criteria"] = crit
    out["verdict"] = "PASS" if all(v for k, v in crit.items() if k != "global_costmap_cost_preserved" and v is not None) else "FAIL"
    out["not_evaluable"] = [k for k, v in crit.items() if v is None]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--estimator-log", default=None)
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--until-episode-end", action="store_true", help="stop when the first episode that started after launch has ended (+5 s)")
    ap.add_argument("--min-episode-id", type=int, default=0)
    ap.add_argument("--snapshots", default=None, help="dir for every 25th local costmap + expected grid (npz)")
    ap.add_argument("--static-test", type=float, default=0.0, help="after the scored episode: at the next episode start freeze the cmd_vel relay and audit N s of a standing robot")
    ap.add_argument("--resummarise", action="store_true", help="recompute the summary from --out, <out>.masks.jsonl and --estimator-log")
    args = ap.parse_args()
    if args.resummarise:
        rows = [json.loads(x) for x in open(args.out)]
        masks = [json.loads(x) for x in open(args.out + ".masks.jsonl")]
        summ = summarise(rows, masks, args.estimator_log)
        json.dump(summ, open(args.summary, "w"), indent=1, default=str)
        print(json.dumps({"verdict": summ["verdict"], "criteria": summ["criteria"]}, indent=1))
        return 0
    rclpy.init()
    n = Audit(keep=25)
    f = open(args.out, "w")
    rows = []
    last_local = last_glob = None
    t_end = time.monotonic() + args.duration
    seen_running = None
    end_at = None
    static_from = static_until = None
    phase = "pre"
    k = 0
    if args.snapshots:
        os.makedirs(args.snapshots, exist_ok=True)
    while time.monotonic() < t_end and (end_at is None or time.monotonic() < end_at):
        rclpy.spin_once(n, timeout_sec=0.02)
        ep = n.episode
        if args.until_episode_end and ep is not None:
            if seen_running is None and ep.outcome_state == RUNNING and ep.episode_id >= args.min_episode_id:
                seen_running = ep.episode_id
                phase = "scored"
            elif seen_running is not None and phase == "scored" and (ep.episode_id != seen_running or ep.outcome_state != RUNNING):
                phase = "transient"
                if not args.static_test:
                    end_at = time.monotonic() + 5.0
            if phase == "transient" and args.static_test and static_from is None and ep.episode_id > (seen_running or 0) and ep.outcome_state == RUNNING:
                # stop the robot without stopping navigation: freeze the cmd_vel relay (twist_stamper) so no velocity
                # command reaches the base; planner, controller and costmaps keep running (SIGCONT at the end)
                import subprocess
                subprocess.run(["pkill", "-STOP", "-f", "twist_stamper/twist_stamper"])
                static_from = time.monotonic() + 3.0
                static_until = static_from + args.static_test
            if static_from is not None and phase != "static_test" and time.monotonic() >= static_from:
                phase = "static_test"
            if static_until is not None and time.monotonic() >= static_until and end_at is None:
                end_at = time.monotonic()
                phase = "transient"
                import subprocess
                subprocess.run(["pkill", "-CONT", "-f", "twist_stamper/twist_stamper"])
        for which in ("local", "global"):
            cm = n.local if which == "local" else n.glob
            last = last_local if which == "local" else last_glob
            if cm is None or not n.masks or (last is not None and cm.header.stamp == last):
                continue
            if which == "local":
                last_local = cm.header.stamp
            else:
                last_glob = cm.header.stamp
            T_by = {}
            mk_frame = n.masks[-1]["frame"]
            if mk_frame != cm.header.frame_id:
                try:
                    t = n.tf.lookup_transform(mk_frame, cm.header.frame_id, Time())
                    q = t.transform.rotation
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    T_by[(mk_frame, cm.header.frame_id)] = (t.transform.translation.x, t.transform.translation.y, yaw)
                except Exception:  # noqa: BLE001
                    pass
            try:
                tr = n.tf.lookup_transform(cm.header.frame_id, "env_0/turtlebot/base_link", Time())
                rxy = (tr.transform.translation.x, tr.transform.translation.y)
            except Exception:  # noqa: BLE001
                rxy = None
            r = check(cm, list(n.masks), T_by, window=10.0 if which == "global" else None, robot_xy=rxy)
            if r is None:
                continue
            v, w = n.speed_at(r["cm_stamp_ns"])
            r.update({"costmap": which, "v": v, "w": w, "episode_id": int(ep.episode_id) if ep else None,
                      "episode_running": bool(ep and ep.outcome_state == RUNNING),
                      "phase": phase})
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            k += 1
            if args.snapshots and which == "local" and k % 25 == 0:
                md = cm.metadata
                mk = n.masks[-1 - r["lag"]]
                exp, _ = expected_from_mask(mk, md, cm.header.frame_id, None)
                np.savez_compressed(os.path.join(args.snapshots, f"local_{r['cm_stamp_ns']}.npz"),
                                    merged=np.frombuffer(bytes(cm.data), np.uint8).reshape(md.size_y, md.size_x), expected=exp,
                                    mask=mk["v"], mask_origin=np.array([mk["ox"], mk["oy"]]), cm_origin=np.array(r["cm_origin"]), res=md.resolution)
    f.close()
    with open(args.out + ".masks.jsonl", "w") as mf:
        for m in n.mask_log:
            mf.write(json.dumps(m) + "\n")
    time.sleep(2.0)
    summ = summarise(rows, n.mask_log, args.estimator_log)
    json.dump(summ, open(args.summary, "w"), indent=1, default=str)
    print(json.dumps({"verdict": summ["verdict"], "criteria": summ["criteria"]}, indent=1))
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
