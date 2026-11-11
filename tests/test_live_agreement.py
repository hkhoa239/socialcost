"""Offline labels against the running arena_social_cost_layer plugin, cell for cell.

Starts a Nav2 costmap holding only the social layer, publishes fixed pedestrian
rosters on it, and compares each published costmap_raw with labels.paint on the
same grid. Run inside the Arena container on a free ROS domain, e.g.

    ROS_DOMAIN_ID=9 python -m pytest tests/test_live_agreement.py -q
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from arena_people_msgs.msg import Pedestrian, Pedestrians  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav2_msgs.msg import Costmap  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from tf2_ros import StaticTransformBroadcaster  # noqa: E402

from socialcost.labels import GridSpec, LayerParams, Ped, paint  # noqa: E402
from socialcost.live import LiveSocialCostmap, costmap_params  # noqa: E402

PEDS_TOPIC = "/socialcost_test/arena_peds"
ROBOT = (3.0, -2.0, 0.6)


def _scene(rng: np.random.Generator, n_weights: int) -> list[Ped]:
    """A random roster around the robot: loose people, plus groups of 2-4 with random kinds."""
    peds: list[Ped] = []
    for _ in range(int(rng.integers(1, 4))):
        peds.append(Ped(ROBOT[0] + rng.uniform(-4, 4), ROBOT[1] + rng.uniform(-4, 4), int(rng.integers(0, 7))))
    for gid in range(int(rng.integers(1, 3))):
        kind = int(rng.integers(0, n_weights))
        cx, cy = ROBOT[0] + rng.uniform(-3, 3), ROBOT[1] + rng.uniform(-3, 3)
        for _ in range(int(rng.integers(2, 5))):
            peds.append(Ped(cx + rng.uniform(-1.2, 1.2), cy + rng.uniform(-1.2, 1.2), int(rng.integers(0, 7)), 100 + gid, kind))
    order = rng.permutation(len(peds))
    return [peds[k] for k in order]


def _msg(peds: list[Ped], node: Node) -> Pedestrians:
    msg = Pedestrians()
    msg.header.frame_id = "map"
    msg.header.stamp = node.get_clock().now().to_msg()
    for k, p in enumerate(peds):
        m = Pedestrian()
        m.name, m.id = f"p{k}", k
        m.pose.position.x, m.pose.position.y = p.x, p.y
        m.pose.orientation.w = 1.0
        m.animation_state = p.animation_state
        m.interaction_id = p.interaction_id
        m.interaction_type = p.interaction_type
        msg.pedestrians.append(m)
    return msg


class _Harness(Node):
    def __init__(self) -> None:
        super().__init__("socialcost_agreement_test")
        self.pub = self.create_publisher(Pedestrians, PEDS_TOPIC, qos_profile_sensor_data)
        self.latest: Costmap | None = None
        self.tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = "map", "test_base"
        t.transform.translation.x, t.transform.translation.y = ROBOT[0], ROBOT[1]
        t.transform.rotation.z, t.transform.rotation.w = math.sin(ROBOT[2] / 2), math.cos(ROBOT[2] / 2)
        self.tf.sendTransform(t)

    def listen(self, topic: str) -> None:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Costmap, topic, lambda m: setattr(self, "latest", m), qos)

    def spin_for(self, seconds: float, peds: list[Ped] | None = None) -> None:
        end = time.monotonic() + seconds
        next_pub = 0.0
        while time.monotonic() < end:
            if peds is not None and time.monotonic() >= next_pub:
                self.pub.publish(_msg(peds, self))
                next_pub = time.monotonic() + 0.1
            rclpy.spin_once(self, timeout_sec=0.02)


@pytest.fixture(scope="module")
def harness():
    rclpy.init()
    node = _Harness()
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize("block", ["local_costmap", "global_costmap"])
@pytest.mark.parametrize("rolling", [False, True])
def test_offline_labels_match_live_layer(harness: _Harness, block: str, rolling: bool) -> None:
    params = LayerParams.from_overlay(costmap=block)
    size, res = 10.0, 0.05
    doc = costmap_params(
        params, PEDS_TOPIC, "map", "test_base", size, res, rolling_window=rolling,
        origin=(ROBOT[0] - size / 2, ROBOT[1] - size / 2), use_sim_time=False,
        name=f"social_costmap_{block}_{int(rolling)}",
    )
    rng = np.random.default_rng(20261029)
    with LiveSocialCostmap(doc, namespace=f"/socialcost_test/{block}_{int(rolling)}") as live:
        harness.listen(live.raw_topic)
        painted_any = 0
        for scene_idx in range(6):
            peds = _scene(rng, len(params.interaction_weights))
            # Let the layer drop the previous roster, then take a grid painted from this one.
            harness.spin_for(1.5, peds)
            harness.latest = None
            harness.spin_for(1.0, peds)
            msg = harness.latest
            assert msg is not None, f"no costmap_raw on {live.raw_topic}"
            md = msg.metadata
            grid = GridSpec(resolution=md.resolution, width=md.size_x, height=md.size_y,
                            origin_x=md.origin.position.x, origin_y=md.origin.position.y, frame="map")
            live_grid = np.frombuffer(bytes(msg.data), np.uint8).reshape(md.size_y, md.size_x)
            offline = paint(params, peds, grid)
            diff = np.abs(live_grid.astype(int) - offline.astype(int))
            assert diff.max() == 0, (
                f"{block} rolling={rolling} scene {scene_idx}: {int((diff > 0).sum())} cells differ, max {diff.max()}")
            painted_any += int((offline > 0).sum() > 0)
        assert painted_any == 6
