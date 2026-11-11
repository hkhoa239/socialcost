"""Warm-up episode: wait until the robot actually moves in the first episode of a launch, then reset.

The first episode's robot start depends on Nav2 / ros2_control bring-up (0.4-5.5 s of simulator time
in docs/REPRODUCIBILITY.md); later episodes start within 0.02 s. After the robot has moved 5 cm the
whole chain (planner, controller, velocity controller) is live, and lifecycle/reset_episode with
seed -1 starts the next episode with Arena's derived seed for that episode index.

    python -m socialcost.warmup [--timeout 300]   -> prints the id of the episode now running
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from task_generator_msgs.msg import EpisodeRecord
from task_generator_msgs.srv import ResetEpisode

TG = "/arena/env_0/task_generator_node"
ODOM = f"{TG}/turtlebot/odom"
RUNNING = 1


class Warmup(Node):
    def __init__(self) -> None:
        super().__init__("socialcost_warmup", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        latched = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ep = None
        self.start = None
        self.moved = False
        self.create_subscription(EpisodeRecord, f"{TG}/state/episode", self.on_ep, latched)
        self.create_subscription(Odometry, ODOM, self.on_odom, qos_profile_sensor_data)
        self.reset = self.create_client(ResetEpisode, f"{TG}/lifecycle/reset_episode")

    def on_ep(self, m) -> None:
        if self.ep is None or m.episode_id != self.ep.episode_id:
            self.start = None
            self.moved = False
        self.ep = m

    def on_odom(self, m) -> None:
        if self.ep is None or self.ep.outcome_state != RUNNING:
            return
        p = m.pose.pose.position
        if self.start is None:
            self.start = (p.x, p.y)
        elif math.hypot(p.x - self.start[0], p.y - self.start[1]) > 0.05:
            self.moved = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    rclpy.init()
    n = Warmup()
    end = time.monotonic() + args.timeout
    while time.monotonic() < end and not (n.ep is not None and n.ep.outcome_state == RUNNING and n.moved):
        rclpy.spin_once(n, timeout_sec=0.05)
    if not n.moved:
        print("warmup: robot never moved", file=sys.stderr)
        return 1
    warm_id = n.ep.episode_id
    if not n.reset.wait_for_service(timeout_sec=30):
        print("warmup: reset_episode unavailable", file=sys.stderr)
        return 1
    fut = n.reset.call_async(ResetEpisode.Request(world="", seed=-1))
    rclpy.spin_until_future_complete(n, fut, timeout_sec=60)
    if not fut.done() or not fut.result().success:
        print(f"warmup: reset failed {fut.result() if fut.done() else 'timeout'}", file=sys.stderr)
        return 1
    while time.monotonic() < end and not (n.ep.episode_id > warm_id and n.ep.outcome_state == RUNNING):
        rclpy.spin_once(n, timeout_sec=0.05)
    ok = n.ep.episode_id > warm_id and n.ep.outcome_state == RUNNING
    print(n.ep.episode_id if ok else -1)
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
