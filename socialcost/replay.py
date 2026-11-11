"""Recorded pedestrian states through the real arena_social_cost_layer plugin, offline.

For identical states, grids and parameters: each recorded roster is published (static, for long
enough that the layer paints it) to a Nav2 costmap holding only the social layer, on a fixed
map-aligned grid around the recorded robot pose, and the published grid is compared with
labels.paint on that grid. Unlike the in-simulation comparison this has no time offset.
The plugin only paints axis-aligned grids; the yaw-aligned label uses the same per-cell
function (checked by tests/test_labels.py::test_rotated_grid_is_the_same_field).
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrian, Pedestrians
from geometry_msgs.msg import TransformStamped
from nav2_msgs.msg import Costmap
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from tf2_ros import StaticTransformBroadcaster

from .labels import GridSpec, LayerParams, Ped, paint
from .live import LiveSocialCostmap, costmap_params

TOPIC = "/socialcost_replay/arena_peds"


class _Replay(Node):
    def __init__(self) -> None:
        super().__init__("socialcost_replay")
        self.pub = self.create_publisher(Pedestrians, TOPIC, qos_profile_sensor_data)
        self.tf = StaticTransformBroadcaster(self)
        self.latest: Costmap | None = None

    def place_robot(self, x: float, y: float, yaw: float) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "map", "replay_base"
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf.sendTransform(t)

    def publish(self, peds: list[Ped]) -> None:
        msg = Pedestrians()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for k, p in enumerate(peds):
            m = Pedestrian()
            m.name, m.id = f"p{k}", k
            m.pose.position.x, m.pose.position.y = float(p.x), float(p.y)
            m.pose.orientation.w = 1.0
            m.animation_state, m.interaction_id, m.interaction_type = int(p.animation_state), int(p.interaction_id), int(p.interaction_type)
            msg.pedestrians.append(m)
        self.pub.publish(msg)

    def spin_for(self, seconds: float, peds: list[Ped]) -> None:
        end, nxt = time.monotonic() + seconds, 0.0
        while time.monotonic() < end:
            if time.monotonic() >= nxt:
                self.publish(peds)
                nxt = time.monotonic() + 0.1
            rclpy.spin_once(self, timeout_sec=0.02)


def replay(params: LayerParams, cases: list[tuple[str, tuple[float, float, float], list[Ped]]], size_m: float = 10.0, resolution: float = 0.05) -> list[dict]:
    """cases: (id, robot pose in map, roster). Returns one comparison record per case."""
    own_init = not rclpy.ok()
    if own_init:
        rclpy.init()
    node = _Replay()
    out = []
    try:
        for cid, (x, y, yaw), peds in cases:
            node.place_robot(x, y, yaw)
            origin = (round((x - size_m / 2) / resolution) * resolution, round((y - size_m / 2) / resolution) * resolution)
            doc = costmap_params(params, TOPIC, "map", "replay_base", size_m, resolution, rolling_window=False, origin=origin, use_sim_time=False, name="replay_costmap")
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            with LiveSocialCostmap(doc, namespace="/socialcost_replay/cm") as live:
                sub = node.create_subscription(Costmap, live.raw_topic, lambda m: setattr(node, "latest", m), qos)
                if not peds:
                    # An empty roster on a fixed grid gives the layer no update bounds, so Nav2 never
                    # publishes. Paint one person first: the layer must then clear that area to zero.
                    node.spin_for(1.5, [Ped(x, y)])
                node.spin_for(1.5, peds)
                node.latest = None
                node.spin_for(1.0, peds)
                msg = node.latest
                node.destroy_subscription(sub)
            if msg is None:
                out.append({"case": cid, "error": "no costmap_raw received"})
                continue
            md = msg.metadata
            grid = GridSpec(resolution=round(md.resolution, 6), width=md.size_x, height=md.size_y, origin_x=md.origin.position.x, origin_y=md.origin.position.y, frame="map")
            live_grid = np.frombuffer(bytes(msg.data), np.uint8).reshape(md.size_y, md.size_x)
            off = paint(params, peds, grid)
            out.append(compare(cid, live_grid, off, grid))
    finally:
        node.destroy_node()
        if own_init:
            rclpy.try_shutdown()
    return out


def compare(cid: str, live: np.ndarray, off: np.ndarray, grid: GridSpec) -> dict:
    diff = np.abs(live.astype(int) - off.astype(int))
    a, b = live > 0, off > 0
    union = int((a | b).sum())

    def centroid(m: np.ndarray, w: np.ndarray) -> list[float] | None:
        if not m.any():
            return None
        j, i = np.nonzero(m)
        ww = w[m].astype(float)
        return [float(grid.origin_x + ((i + 0.5) * ww).sum() / ww.sum() * grid.resolution), float(grid.origin_y + ((j + 0.5) * ww).sum() / ww.sum() * grid.resolution)]

    cl, co = centroid(a, live), centroid(b, off)
    return {
        "case": cid, "cells": int(diff.size), "mae": float(diff.mean()), "max_abs": int(diff.max()),
        "mismatched_cells": int((diff > 0).sum()), "mismatched_fraction": float((diff > 0).mean()),
        "nonzero_live": int(a.sum()), "nonzero_offline": int(b.sum()), "iou_nonzero": float((a & b).sum() / union) if union else 1.0,
        "centroid_offset_m": float(math.dist(cl, co)) if cl and co else (0.0 if cl is None and co is None else None),
    }
