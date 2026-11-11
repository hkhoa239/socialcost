#!/usr/bin/env python3
"""Nav2 integration check of the social layer on a running env, for one controller.

Reports which controller plugin runs, whether the merged local costmap carries the social
layer (merged >= offline social label on every cell the label paints), the episode outcome,
and whether the robot passed through an interaction (crossed the segment between two members).

    python controller_check.py --out result.json [--episodes 1]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from nav2_msgs.msg import Costmap
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from task_generator_msgs.msg import EpisodeRecord
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.labels import GridSpec, LayerParams, paint, peds_from_msg  # noqa: E402

NS = "/arena/env_0/task_generator_node/turtlebot"


def segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and orient(q1, q2, p1) * orient(q1, q2, p2) < 0


class Check(Node):
    def __init__(self) -> None:
        super().__init__("socialcost_controller_check", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.tf = Buffer(cache_time=Duration(seconds=60))
        TransformListener(self.tf, self)
        self.peds = None
        self.episode = None
        self.costmap = None
        latched = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Pedestrians, "/arena/env_0/arena_peds", lambda m: setattr(self, "peds", m), qos_profile_sensor_data)
        self.create_subscription(EpisodeRecord, "/arena/env_0/task_generator_node/state/episode", lambda m: setattr(self, "episode", m), latched)
        self.create_subscription(Costmap, f"{NS}/local_costmap/costmap_raw", lambda m: setattr(self, "costmap", m), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))


def param(node_path: str, name: str) -> str:
    r = subprocess.run(["ros2", "param", "get", node_path, name], capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--tolerance", type=int, default=0, help="cost units a merged cell may fall short of the label (mask quantisation: 3)")
    ap.add_argument("--window", type=float, default=0.0, help="compare only cells within +-window/2 of the robot (the estimator's grid; 0 = whole costmap)")
    args = ap.parse_args()
    result = {
        "controller_plugin": param(f"{NS}/controller_server", "FollowPath.plugin"),
        "local_plugins": param(f"{NS}/local_costmap/local_costmap", "plugins"),
        "global_plugins": param(f"{NS}/local_costmap/local_costmap", "plugins").replace("local", "local") and param(f"{NS}/global_costmap/global_costmap", "plugins"),
    }
    params = LayerParams.from_overlay(costmap="local_costmap")
    rclpy.init()
    n = Check()
    deadline = time.monotonic() + args.timeout
    # wait for a fresh RUNNING episode
    first = None
    while time.monotonic() < deadline:
        rclpy.spin_once(n, timeout_sec=0.1)
        if n.episode is not None:
            if first is None:
                first = n.episode.episode_id
            if n.episode.outcome_state == 1 and n.episode.episode_id != first:
                break
    ep_id = n.episode.episode_id
    path, cover, painted_cells, checked = [], [], 0, 0
    last_cm_stamp = None
    pairs_crossed = set()
    min_dist = math.inf
    while time.monotonic() < deadline and n.episode.episode_id == ep_id and n.episode.outcome_state == 1:
        rclpy.spin_once(n, timeout_sec=0.05)
        try:
            t = n.tf.lookup_transform("map", "env_0/turtlebot/base_link", Time())
        except Exception:  # noqa: BLE001
            continue
        xy = (t.transform.translation.x, t.transform.translation.y)
        if path and path[-1] != xy and n.peds is not None:
            groups: dict[int, list] = {}
            for p in n.peds.pedestrians:
                min_dist = min(min_dist, math.hypot(p.pose.position.x - xy[0], p.pose.position.y - xy[1]))
                if p.interaction_id >= 0:
                    groups.setdefault(p.interaction_id, []).append((p.pose.position.x, p.pose.position.y))
            for gid, mem in groups.items():
                for a in range(len(mem)):
                    for b in range(a + 1, len(mem)):
                        if segments_cross(path[-1], xy, mem[a], mem[b]):
                            pairs_crossed.add(gid)
        if not path or path[-1] != xy:
            path.append(xy)
        cm = n.costmap
        if cm is not None and n.peds is not None and cm.header.stamp != last_cm_stamp:
            last_cm_stamp = cm.header.stamp
            md = cm.metadata
            grid = GridSpec(resolution=md.resolution, width=md.size_x, height=md.size_y, origin_x=md.origin.position.x, origin_y=md.origin.position.y)
            merged = np.frombuffer(bytes(cm.data), np.uint8).reshape(md.size_y, md.size_x)
            social = paint(params, peds_from_msg(n.peds), grid)
            mask = social > 0
            if args.window > 0:
                ii, jj = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
                cx = grid.origin_x + (ii + 0.5) * grid.resolution
                cy = grid.origin_y + (jj + 0.5) * grid.resolution
                h = args.window / 2 - grid.resolution
                mask &= (np.abs(cx - xy[0]) <= h) & (np.abs(cy - xy[1]) <= h)
            if mask.any():
                checked += 1
                painted_cells += int(mask.sum())
                cover.append(float((merged[mask].astype(int) >= social[mask].astype(int) - args.tolerance).mean()))
    ep = n.episode
    result.update({
        "episode_id": int(ep_id), "outcome_state": int(ep.outcome_state), "outcome_info": ep.outcome_info,
        "path_length_m": float(sum(math.dist(a, b) for a, b in zip(path, path[1:]))),
        "min_distance_to_pedestrian_m": min_dist,
        "crossed_interaction_ids": sorted(pairs_crossed),
        "costmaps_checked": checked, "social_cells_checked": painted_cells,
        "merged_ge_social_fraction_mean": float(np.mean(cover)) if cover else None,
        "merged_ge_social_fraction_min": float(np.min(cover)) if cover else None,
    })
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
