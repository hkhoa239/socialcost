"""Switch a running Arena env to another scenario of the same world and start a fresh seeded episode.

task_generator re-reads `task.scenario.file` on every episode reset (TM_Scenario.reset), so setting the
parameter and calling lifecycle/reset_episode loads the next scenario (people, objects, robot start/goal)
without relaunching Isaac.

    python -m socialcost.switch --scenario scv1_000123 --seed 1061 [--timeout 120]   -> prints the new episode id
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from task_generator_msgs.msg import EpisodeRecord
from task_generator_msgs.srv import ResetEpisode

TG = "/arena/env_0/task_generator_node"
RUNNING = 1


class Switch(Node):
    def __init__(self) -> None:
        super().__init__("socialcost_switch", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        latched = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ep = None
        self.create_subscription(EpisodeRecord, f"{TG}/state/episode", self.on_ep, latched)
        self.setp = self.create_client(SetParameters, f"{TG}/set_parameters")
        self.reset = self.create_client(ResetEpisode, f"{TG}/lifecycle/reset_episode")

    def on_ep(self, m) -> None:
        self.ep = m


def call(n: Node, client, req, timeout: float):
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError(f"{client.srv_name} unavailable")
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=timeout)
    if not fut.done():
        raise RuntimeError(f"{client.srv_name} timed out")
    return fut.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    rclpy.init()
    n = Switch()
    end = time.monotonic() + args.timeout
    while n.ep is None and time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    if n.ep is None:
        print("switch: no episode state", file=sys.stderr)
        return 1
    old = n.ep.episode_id
    p = Parameter(name="task.scenario.file", value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=args.scenario))
    r = call(n, n.setp, SetParameters.Request(parameters=[p]), 30)
    if not r.results or not r.results[0].successful:
        print(f"switch: parameter rejected {r.results[0].reason if r.results else ''}", file=sys.stderr)
        return 1
    r = call(n, n.reset, ResetEpisode.Request(world="", seed=args.seed), 60)
    if not r.success:
        print(f"switch: reset failed {r.error_msg}", file=sys.stderr)
        return 1
    while time.monotonic() < end and not (n.ep.episode_id > old and n.ep.outcome_state == RUNNING):
        rclpy.spin_once(n, timeout_sec=0.05)
    ok = n.ep.episode_id > old and n.ep.outcome_state == RUNNING
    print(n.ep.episode_id if ok else -1)
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
