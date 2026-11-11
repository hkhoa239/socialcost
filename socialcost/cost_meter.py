"""Ground-truth social cost actually incurred by the robot, logged during closed-loop navigation runs.

Independent of the arm under test (every arm runs the same meter): at 10 Hz (sim time) it paints the extended/2.0
ground-truth cost (arena-native layer + queue membership + halt wedge + mobility aid, from the full simulator state)
on a 1.2 m x 1.2 m grid around the robot and logs, under the robot footprint (radius 0.22 m):
  cost_max      max full extended cost (personal space included)
  inter_max     max of the interaction terms only (conversation / queue / halt wedge: personal space removed)
  centre        full cost at the robot centre
  v2n_cost / v2n_inter, v2e_cost / v2e_inter   the same max under the footprint for the socialcost_v2 native and
                extended labels (labels_v2.py), full and interaction-only (personal space removed)

    python -m socialcost.cost_meter --log meter.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .labels import GridSpec, LayerParams, yaw_from_quat

ENV = "/arena/env_0"
BASE = "env_0/turtlebot/base_link"
FOOTPRINT_R = 0.22


class Meter(Node):
    def __init__(self, log: str, rate: float) -> None:
        super().__init__("socialcost_meter", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        from arena_humansim_msgs.msg import InteractionFrame

        from .dataset import EXTENDED_CONFIG
        from .extensions import ExtParams
        self.params = LayerParams.from_overlay()
        self.inter_params = dataclasses.replace(self.params, personal_space_radius=1e-6)
        self.ext = ExtParams.load(EXTENDED_CONFIG)
        self.grid = GridSpec.centered(1.2, 0.05)
        i, j = np.meshgrid(np.arange(self.grid.width), np.arange(self.grid.height))
        cx = self.grid.origin_x + (i + 0.5) * self.grid.resolution
        cy = self.grid.origin_y + (j + 0.5) * self.grid.resolution
        self.fp = np.hypot(cx, cy) <= FOOTPRINT_R
        self.c = (self.grid.height // 2, self.grid.width // 2)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.peds = None
        self.frames = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Pedestrians, f"{ENV}/arena_peds", lambda m: setattr(self, "peds", m), qos_profile_sensor_data)
        self.create_subscription(InteractionFrame, f"{ENV}/task_generator_node/interactions", self._frames, latched)
        self.out = open(log, "a")
        self.create_timer(1.0 / rate, self.tick)

    def _frames(self, msg) -> None:
        offs = list(msg.member_offset) + [len(msg.member_id)]
        self.frames = {"interactions": [{"interaction_id": int(i), "interaction_type": int(msg.interaction_type[k]), "outcome": int(msg.outcome[k]),
                                         "member_ids": [int(x) for x in msg.member_id[offs[k]:offs[k + 1]]]} for k, i in enumerate(msg.interaction_id)]}

    def _states(self) -> list[dict]:
        from rosidl_runtime_py.convert import message_to_ordereddict
        out = []
        for p in self.peds.pedestrians:
            q = p.pose.orientation
            out.append({"name": p.name, "id": int(p.id), "x": p.pose.position.x, "y": p.pose.position.y, "yaw": yaw_from_quat(q.x, q.y, q.z, q.w),
                        "animation_state": int(p.animation_state), "interaction_id": int(p.interaction_id), "interaction_type": int(p.interaction_type),
                        "agent_type": p.agent_type, "model_uri": p.model_uri,
                        "gestures": [json.loads(json.dumps(message_to_ordereddict(g))) for g in p.gestures]})
        return out

    def tick(self) -> None:
        if self.peds is None:
            return
        try:
            t = self.tf.lookup_transform("map", BASE, Time())
        except Exception:  # noqa: BLE001
            return
        q = t.transform.rotation
        pose = (t.transform.translation.x, t.transform.translation.y, yaw_from_quat(q.x, q.y, q.z, q.w))
        from .extensions import paint_extended
        states = self._states()
        full, fired = paint_extended(self.params, self.ext, states, self.frames, self.grid, pose)
        inter, _ = paint_extended(self.inter_params, dataclasses.replace(self.ext, mobility=False), states, self.frames, self.grid, pose)
        from .labels_v2 import SPECS, labels_and_components
        v2 = labels_and_components(SPECS["v2"], self.params, self.grid, states, self.frames, pose)
        v2i = labels_and_components(SPECS["v2"], self.inter_params, self.grid, states, self.frames, pose)
        stamp = t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec
        self.out.write(json.dumps({"t_ns": stamp, "x": pose[0], "y": pose[1], "yaw": pose[2], "cost_max": int(full[self.fp].max()),
                                   "inter_max": int(inter[self.fp].max()), "centre": int(full[self.c]), "halt_fired": fired["halt"],
                                   "v2n_cost": int(v2["native"][self.fp].max()), "v2n_inter": int(v2i["native"][self.fp].max()),
                                   "v2e_cost": int(v2["extended"][self.fp].max()), "v2e_inter": int(v2i["extended"][self.fp].max())}) + "\n")
        self.out.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--rate", type=float, default=10.0)
    args = ap.parse_args()
    rclpy.init()
    n = Meter(args.log, args.rate)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
