#!/usr/bin/env python3
"""Per-episode timing and trajectories of a running env, in simulator time, for reproducibility runs.

Writes one JSON line per finished episode to --out:
  episode_id, seed, start_ns (EpisodeRecord.start_time), outcome, and relative to the start:
  goal_ns (first goal on task_pose/goal_pose), first_cmd_ns (first non-zero velocity command),
  first_motion_ns (robot 2 cm from its start pose), interaction activations, and the robot and
  pedestrian trajectories (every TF / arena_peds sample).

--end-on-abort: a Nav2 navigate_to_pose goal of this episode reported ABORTED ends the record with outcome -6
("nav2_aborted": the controller gave up, Arena keeps the episode running without a goal). A wall timeout writes the
partial record with outcome -1 ("probe_timeout"), so failed runs keep their trajectories.

    python repro_probe.py --out probe.jsonl --episodes 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import rclpy
from arena_people_msgs.msg import Pedestrians
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from task_generator_msgs.msg import EpisodeRecord
from tf2_ros import Buffer, TransformListener

NS = "/arena/env_0/task_generator_node/turtlebot"
RUNNING = 1


def sns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class Probe(Node):
    def __init__(self, out: str, episodes: int, min_episode_id: int = 1, end_on_abort: bool = False) -> None:
        super().__init__("socialcost_repro_probe", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.out, self.left, self.min_episode_id = out, episodes, min_episode_id
        self.tf = Buffer()
        TransformListener(self.tf, self)
        latched = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(EpisodeRecord, "/arena/env_0/task_generator_node/state/episode", self.on_episode, latched)
        self.create_subscription(Pedestrians, "/arena/env_0/arena_peds", self.on_peds, qos_profile_sensor_data)
        self.subs_pending = {f"{NS}/cmd_vel": self.on_cmd, f"{NS}/task_pose": self.on_goal, f"{NS}/goal_pose": self.on_goal}
        self.create_timer(1.0, self.attach, clock=rclpy.clock.Clock())
        self.create_timer(0.05, self.sample_tf)
        self.ep = None
        self.seen_first = None
        self.end_on_abort = end_on_abort
        self.status_topic = None

    def on_status(self, m) -> None:
        if self.ep is None or not self.end_on_abort:
            return
        for s in m.status_list:
            if s.status == 6 and sns(s.goal_info.stamp) >= self.ep["start_ns"]:  # ABORTED, goal sent in this episode
                self.finish(-6, "nav2_aborted")
                return

    def attach(self) -> None:
        types = dict(self.get_topic_names_and_types())
        if self.end_on_abort and self.status_topic is None:
            for topic, ty in types.items():
                if topic.endswith("navigate_to_pose/_action/status") and topic.startswith("/arena/env_0"):
                    from action_msgs.msg import GoalStatusArray
                    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                    self.create_subscription(GoalStatusArray, topic, self.on_status, latched)
                    self.status_topic = topic
                    self.get_logger().info(f"ending on Nav2 abort: {topic}")
        for topic, cb in list(self.subs_pending.items()):
            if topic in types:
                self.create_subscription(get_message(types[topic][0]), topic, cb, 10)
                del self.subs_pending[topic]

    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def on_episode(self, m) -> None:
        if self.ep is not None and (m.episode_id != self.ep["episode_id"] or m.outcome_state != RUNNING):
            self.finish(m.outcome_state, m.outcome_info)
        if m.outcome_state == RUNNING and m.episode_id >= self.min_episode_id and (self.ep is None or self.ep["episode_id"] != m.episode_id):
            if self.seen_first is None:
                self.seen_first = m.episode_id
            self.ep = {"episode_id": int(m.episode_id), "seed": int(m.seed), "start_ns": sns(m.start_time), "wall_start": time.time(),
                       "first_after_attach": m.episode_id == self.seen_first,
                       "goal_ns": None, "first_cmd_ns": None, "first_motion_ns": None, "start_pose": None,
                       "robot": [], "peds": [], "interactions": {}}

    def on_goal(self, m) -> None:
        if self.ep is not None and self.ep["goal_ns"] is None:
            self.ep["goal_ns"] = self.now_ns()

    def on_cmd(self, m) -> None:
        tw = getattr(m, "twist", m)
        if self.ep is not None and self.ep["first_cmd_ns"] is None and (abs(tw.linear.x) > 1e-3 or abs(tw.angular.z) > 1e-3):
            self.ep["first_cmd_ns"] = self.now_ns()

    def on_peds(self, m) -> None:
        if self.ep is None:
            return
        t = sns(m.header.stamp)
        self.ep["peds"].append([t, [[p.name, p.pose.position.x, p.pose.position.y, p.interaction_id] for p in m.pedestrians]])
        for p in m.pedestrians:
            if p.interaction_id >= 0:
                self.ep["interactions"].setdefault(p.name, t)

    def sample_tf(self) -> None:
        if self.ep is None:
            return
        try:
            tr = self.tf.lookup_transform("map", "env_0/turtlebot/base_link", Time())
        except Exception:  # noqa: BLE001
            return
        t = sns(tr.header.stamp)
        x, y = tr.transform.translation.x, tr.transform.translation.y
        q = tr.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if t < self.ep["start_ns"]:
            return
        if not self.ep["robot"] or self.ep["robot"][-1][0] != t:
            self.ep["robot"].append([t, x, y, yaw])
        if self.ep["start_pose"] is None:
            self.ep["start_pose"] = [x, y, yaw]
        elif self.ep["first_motion_ns"] is None and math.hypot(x - self.ep["start_pose"][0], y - self.ep["start_pose"][1]) > 0.02:
            self.ep["first_motion_ns"] = t

    def finish(self, outcome: int, info: str) -> None:
        ep, self.ep = self.ep, None
        ep.update({"outcome": int(outcome), "outcome_info": info, "wall_end": time.time()})
        for k in ("goal_ns", "first_cmd_ns", "first_motion_ns"):
            ep[k.replace("_ns", "_rel_s")] = (ep[k] - ep["start_ns"]) / 1e9 if ep[k] is not None else None
        with open(self.out, "a") as f:
            f.write(json.dumps(ep) + "\n")
        self.get_logger().info(f"episode {ep['episode_id']}: goal {ep['goal_rel_s']} cmd {ep['first_cmd_rel_s']} motion {ep['first_motion_rel_s']} outcome {outcome}")
        self.left -= 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--min-episode-id", type=int, default=1)
    ap.add_argument("--end-on-abort", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rclpy.init()
    n = Probe(args.out, args.episodes, args.min_episode_id, args.end_on_abort)
    end = time.monotonic() + args.timeout
    while rclpy.ok() and n.left > 0 and time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    if n.left > 0 and n.ep is not None and args.end_on_abort:
        n.finish(-1, "probe_timeout")
    rclpy.try_shutdown()
    return 0 if n.left <= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
