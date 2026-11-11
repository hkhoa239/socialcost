"""Records synchronized RGB-D frames, robot pose and pedestrian states from a running Arena env.

Sampling runs on simulator time: an RGB frame is taken when its stamp reaches the next
slot of the sample grid (1 / sample_hz apart, from episode start + warmup). Depth is paired
by identical stamp (Isaac renders both from one render product), robot and camera poses come
from TF at the image stamp, pedestrian states are interpolated to the image stamp between the
two bracketing arena_peds messages. Labels are not painted here; label_episode() derives them
offline from what was recorded.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from nav2_msgs.msg import Costmap
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from .labels import yaw_from_quat

try:
    from arena_humansim_msgs.msg import AgentMeta, InteractionFrame
except ImportError:  # recorder still works without humansim messages
    AgentMeta = InteractionFrame = None
try:
    from task_generator_msgs.msg import EpisodeRecord
except ImportError:
    EpisodeRecord = None

RUNNING = 1


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _tf_dict(t) -> dict:
    tr, q = t.transform.translation, t.transform.rotation
    return {"x": tr.x, "y": tr.y, "z": tr.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w, "yaw": yaw_from_quat(q.x, q.y, q.z, q.w)}


def _ped_dict(p) -> dict:
    q = p.pose.orientation
    return {
        "name": p.name, "id": int(p.id),
        "x": p.pose.position.x, "y": p.pose.position.y, "z": p.pose.position.z,
        "yaw": yaw_from_quat(q.x, q.y, q.z, q.w),
        "vx": p.twist.linear.x, "vy": p.twist.linear.y, "wz": p.twist.angular.z,
        "animation_state": int(p.animation_state),
        "interaction_id": int(p.interaction_id), "interaction_type": int(p.interaction_type),
        "agent_type": p.agent_type, "model_uri": p.model_uri,
        "gestures": [json.loads(json.dumps(message_to_ordereddict(g))) for g in p.gestures],
        "gait_phase": float(p.gait_phase),
    }


def interpolate_peds(before: tuple[int, list[dict]], after: tuple[int, list[dict]] | None, t_ns: int) -> tuple[list[dict], dict]:
    """Ped states at t_ns, linear in x/y/yaw between bracketing rosters matched by name.

    The roster and the discrete fields (animation_state, interaction_*, gestures) are the ones in
    force at t_ns, i.e. from `before`: a message stamped after the image must not decide whether
    an interaction is active in it. Returns the states and a sync record of the source stamps.
    """
    t0, peds0 = before
    if after is None or after[0] == t0:
        return [dict(p) for p in peds0], {"peds_before_ns": t0, "peds_after_ns": None, "alpha": 0.0}
    t1, peds1 = after
    alpha = min(max((t_ns - t0) / (t1 - t0), 0.0), 1.0)
    by_name0 = {p["name"]: p for p in peds0}
    by_name1 = {p["name"]: p for p in peds1}
    out = []
    for ref in peds0:
        a, b = by_name0.get(ref["name"]), by_name1.get(ref["name"])
        p = dict(ref)
        if a is not None and b is not None:
            for k in ("x", "y", "z", "vx", "vy"):
                p[k] = a[k] + alpha * (b[k] - a[k])
            dyaw = math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"]))
            p["yaw"] = math.atan2(math.sin(a["yaw"] + alpha * dyaw), math.cos(a["yaw"] + alpha * dyaw))
        out.append(p)
    return out, {"peds_before_ns": t0, "peds_after_ns": t1, "alpha": alpha}


@dataclass
class RecorderConfig:
    out_dir: str
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    odom_topic: str
    peds_topic: str
    episode_topic: str
    interactions_topic: str
    agent_meta_topic: str
    map_frame: str
    base_frame: str
    camera_frame: str
    sample_hz: float = 5.0
    duration_s: float = 20.0
    warmup_s: float = 1.0
    max_peds_gap_s: float = 0.25
    live_costmap_topic: str | None = None
    episode_wait_s: float = 600.0
    # An episode whose first frame reaches the recorder later than its window start + this is skipped.
    fresh_margin_s: float = 0.5
    # Episodes with a smaller id are never recorded (1: everything; 2: skip the warm-up episode,
    # whose robot start depends on Nav2 / ros2_control bring-up, see docs/REPRODUCIBILITY.md).
    min_episode_id: int = 1
    # A recorded episode whose robot starts moving later than this after the episode start is flagged.
    max_start_delay_s: float = 0.5
    metadata: dict = field(default_factory=dict)


class EpisodeRecorder(Node):
    """Records one episode of a running env into cfg.out_dir."""

    def __init__(self, cfg: RecorderConfig) -> None:
        super().__init__("socialcost_recorder", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.cfg = cfg
        self.tf = Buffer(cache_time=Duration(seconds=60))
        self._tf_listener = TransformListener(self.tf, self)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.camera_info: CameraInfo | None = None
        self.depth_by_stamp: dict[int, Image] = {}
        self.depth_order: deque[int] = deque()
        self.rgb_pending: deque[Image] = deque(maxlen=10)
        self.peds_stamps: list[int] = []
        self.peds_rosters: list[list[dict]] = []
        self.odom: deque[tuple[int, Odometry]] = deque(maxlen=400)
        self.episode = None
        self.interactions: list[dict] = []
        self.agent_meta: dict | None = None
        self.live_costmaps: deque[Costmap] = deque(maxlen=50)

        self.create_subscription(Image, cfg.rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, cfg.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, cfg.camera_info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Odometry, cfg.odom_topic, self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Pedestrians, cfg.peds_topic, self._on_peds, qos_profile_sensor_data)
        if EpisodeRecord is not None:
            self.create_subscription(EpisodeRecord, cfg.episode_topic, self._on_episode, latched)
        if InteractionFrame is not None:
            self.create_subscription(InteractionFrame, cfg.interactions_topic, self._on_interactions, latched)
            self.create_subscription(AgentMeta, cfg.agent_meta_topic, self._on_agent_meta, latched)
        if cfg.live_costmap_topic:
            self.create_subscription(Costmap, cfg.live_costmap_topic, lambda m: self.live_costmaps.append(m), QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))

        self.state = "waiting"  # waiting -> recording -> done
        self.episode_id: int | None = None
        self.t_start_ns: int | None = None
        self.next_slot = 0
        self.samples = 0
        self.stats = {"rgb_frames": 0, "rgb_without_depth": 0, "tf_failures": 0, "peds_gap_violations": 0,
                      "slots_skipped": 0, "depth_all_invalid": 0, "rgb_depth_dt_ns_max": 0,
                      "peds_bracket_dt_ns_max": 0, "time_jumps": 0, "end_reason": None,
                      "wall_start": time.time()}
        self._files: dict[str, object] = {}
        self.last_rgb_ns = -1

    # ---------------- callbacks ----------------
    def _on_info(self, msg: CameraInfo) -> None:
        if self.camera_info is None:
            self.camera_info = msg

    def _on_depth(self, msg: Image) -> None:
        t = stamp_ns(msg.header.stamp)
        self.depth_by_stamp[t] = msg
        self.depth_order.append(t)
        while len(self.depth_order) > 30:
            self.depth_by_stamp.pop(self.depth_order.popleft(), None)
        self._drain()

    def _on_rgb(self, msg: Image) -> None:
        self.rgb_pending.append(msg)
        self._drain()

    def _on_peds(self, msg: Pedestrians) -> None:
        t = stamp_ns(msg.header.stamp)
        roster = [_ped_dict(p) for p in msg.pedestrians]
        if self.peds_stamps and t < self.peds_stamps[-1]:
            # Simulator time went backwards: a reset. Drop the old stream.
            self.stats["time_jumps"] += 1
            if self.state == "recording":
                self._finish("time_jump")
            self.peds_stamps.clear()
            self.peds_rosters.clear()
        self.peds_stamps.append(t)
        self.peds_rosters.append(roster)
        if self.state == "recording":
            self._write("peds_stream.jsonl", {"stamp_ns": t, "frame_id": msg.header.frame_id, "pedestrians": roster})
        if len(self.peds_stamps) > 4000:
            del self.peds_stamps[:1000]
            del self.peds_rosters[:1000]

    def _on_episode(self, msg) -> None:
        self.episode = msg
        if (self.state == "waiting" and msg.outcome_state == RUNNING and int(msg.episode_id) != getattr(self, "skipped_episode", None)
                and int(msg.episode_id) >= self.cfg.min_episode_id):
            self.episode_id = int(msg.episode_id)
            self.t_start_ns = stamp_ns(msg.start_time)
            self.get_logger().info(f"episode {self.episode_id} running since t={self.t_start_ns / 1e9:.2f}s, seed {msg.seed}")
        elif self.state == "recording" and (int(msg.episode_id) != self.episode_id or msg.outcome_state != RUNNING):
            self._finish(f"episode_outcome_{msg.outcome_state}")

    def _on_interactions(self, msg) -> None:
        offsets = list(msg.member_offset) + [len(msg.member_id)]
        frame = {"stamp_ns": stamp_ns(msg.header.stamp), "interactions": []}
        for k, iid in enumerate(msg.interaction_id):
            a, b = offsets[k], offsets[k + 1] if k + 1 < len(offsets) else len(msg.member_id)
            frame["interactions"].append({
                "interaction_id": int(iid), "interaction_type": int(msg.interaction_type[k]),
                "outcome": int(msg.outcome[k]), "object_id": msg.object_id[k],
                "member_ids": [int(m) for m in msg.member_id[a:b]],
                "member_roles": [int(r) for r in msg.member_role[a:b]],
            })
        self.interactions.append(frame)
        if self.state == "recording":
            self._write("interactions.jsonl", frame)

    def _on_agent_meta(self, msg) -> None:
        self.agent_meta = {"agent_id": [int(a) for a in msg.agent_id], "name": list(msg.name), "agent_type": list(msg.agent_type), "policies": list(msg.policies)}

    # ---------------- sampling ----------------
    def _drain(self) -> None:
        while self.rgb_pending:
            rgb = self.rgb_pending[0]
            t = stamp_ns(rgb.header.stamp)
            depth = self.depth_by_stamp.get(t)
            if depth is None:
                newest_depth = self.depth_order[-1] if self.depth_order else -1
                if newest_depth > t or len(self.rgb_pending) > 5:
                    self.rgb_pending.popleft()
                    self.stats["rgb_without_depth"] += 1
                    continue
                return  # depth for this stamp may still arrive
            self.rgb_pending.popleft()
            self._consider(rgb, depth, t)

    def _consider(self, rgb: Image, depth: Image, t: int) -> None:
        if t <= self.last_rgb_ns:
            if t < self.last_rgb_ns:
                self.stats["time_jumps"] += 1
                if self.state == "recording":
                    self._finish("time_jump")
            return
        self.last_rgb_ns = t
        if self.state == "waiting":
            if self.episode_id is None or self.camera_info is None or self.t_start_ns is None:
                return
            t0 = self.t_start_ns + int(self.cfg.warmup_s * 1e9)
            if not getattr(self, "saw_start", False) and t > t0 + int(self.cfg.fresh_margin_s * 1e9):
                # Attached after this episode's recording window opened: take the next episode,
                # so every window starts at the same offset from its (seeded) episode start.
                self.get_logger().info(f"episode {self.episode_id} window opened before the recorder attached, waiting for the next one")
                self.skipped_episode = self.episode_id
                self.episode_id = None
                return
            self.saw_start = True
            if t < t0 or not self.peds_stamps:
                return
            self._begin(t0)
        if self.state != "recording":
            return
        self.stats["rgb_frames"] += 1
        slot_t = self.t0_ns + int(round(self.next_slot * 1e9 / self.cfg.sample_hz))
        if t < slot_t:
            return
        # The frame belongs to the latest slot it reached; slots it jumped over have no frame.
        slot = int(math.floor((t - self.t0_ns) * self.cfg.sample_hz / 1e9 + 1e-9))
        self.stats["slots_skipped"] += max(0, slot - self.next_slot)
        self.next_slot = slot + 1
        if (t - self.t0_ns) / 1e9 >= self.cfg.duration_s:
            self._finish("duration")
            return
        self._sample(rgb, depth, t, slot)

    def _begin(self, t: int) -> None:
        self.state = "recording"
        self.t0_ns = t
        self.next_slot = 0
        os.makedirs(os.path.join(self.cfg.out_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, "depth"), exist_ok=True)
        if self.cfg.live_costmap_topic:
            os.makedirs(os.path.join(self.cfg.out_dir, "live_social_cost"), exist_ok=True)
        ci = self.camera_info
        cam = {"frame_id": ci.header.frame_id, "width": ci.width, "height": ci.height,
               "K": list(ci.k), "D": list(ci.d), "distortion_model": ci.distortion_model,
               "R": list(ci.r), "P": list(ci.p), "rgb_topic": self.cfg.rgb_topic, "depth_topic": self.cfg.depth_topic,
               "depth_units": "meters, planar distance to the image plane (Isaac DistanceToImagePlane), NaN where invalid",
               "optical_frame_convention": "REP-103 optical: x right, y down, z forward"}
        try:
            t_bc = self.tf.lookup_transform(self.cfg.base_frame, self.cfg.camera_frame, Time(), Duration(seconds=1.0))
            cam["T_base_camera"] = _tf_dict(t_bc)
        except Exception as e:  # noqa: BLE001
            cam["T_base_camera_error"] = str(e)
        with open(os.path.join(self.cfg.out_dir, "camera_info.json"), "w") as f:
            json.dump(cam, f, indent=2)
        # scenario files are authored in humansim's world frame; keep world <- map to map recordings back
        try:
            self.stats["tf_world_map"] = _tf_dict(self.tf.lookup_transform("world", self.cfg.map_frame, Time(), Duration(seconds=1.0)))
        except Exception as e:  # noqa: BLE001
            self.stats["tf_world_map_error"] = str(e)
        for frame in self.interactions[-1:]:
            self._write("interactions.jsonl", frame)
        # the messages bracketing the first samples, so every sample can be rebuilt from the stream
        for ts, roster in list(zip(self.peds_stamps, self.peds_rosters))[-5:]:
            self._write("peds_stream.jsonl", {"stamp_ns": ts, "frame_id": "map", "pedestrians": roster})
        self.get_logger().info(f"recording episode {self.episode_id} on the grid from t={t / 1e9:.3f}s into {self.cfg.out_dir}")

    def _sample(self, rgb: Image, depth: Image, t: int, slot: int) -> None:
        stamp = Time(nanoseconds=t)
        try:
            t_mb = self.tf.lookup_transform(self.cfg.map_frame, self.cfg.base_frame, stamp, Duration(seconds=0.5))
            t_mc = self.tf.lookup_transform(self.cfg.map_frame, self.cfg.camera_frame, stamp, Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            self.stats["tf_failures"] += 1
            self.get_logger().warn(f"slot {slot}: no TF at {t / 1e9:.3f}s ({e}), frame dropped")
            return
        k = bisect.bisect_right(self.peds_stamps, t) - 1
        if k < 0:
            self.stats["peds_gap_violations"] += 1
            return
        before = (self.peds_stamps[k], self.peds_rosters[k])
        after = (self.peds_stamps[k + 1], self.peds_rosters[k + 1]) if k + 1 < len(self.peds_stamps) else None
        gap = (after[0] - before[0]) if after else (t - before[0])
        if gap > self.cfg.max_peds_gap_s * 1e9:
            self.stats["peds_gap_violations"] += 1
            self.get_logger().warn(f"slot {slot}: pedestrian stream gap {gap / 1e9:.3f}s, frame dropped")
            return
        self.stats["peds_bracket_dt_ns_max"] = max(self.stats["peds_bracket_dt_ns_max"], int(gap))
        peds, sync = interpolate_peds(before, after, t)

        idx = self.samples
        name = f"{idx:06d}"
        img = np.frombuffer(bytes(rgb.data), np.uint8).reshape(rgb.height, rgb.step)[:, : rgb.width * 3].reshape(rgb.height, rgb.width, 3)
        if rgb.encoding == "rgb8":
            img = img[:, :, ::-1]
        cv2.imwrite(os.path.join(self.cfg.out_dir, "rgb", name + ".png"), img)
        d = np.frombuffer(bytes(depth.data), np.float32).reshape(depth.height, depth.width).copy()
        d[~np.isfinite(d)] = np.nan
        if np.isnan(d).all():
            self.stats["depth_all_invalid"] += 1
        np.save(os.path.join(self.cfg.out_dir, "depth", name + ".npy"), d)

        odom = self._odom_at(t)
        self._track_motion()
        robot = {"sample": idx, "slot": slot, "stamp_ns": t, "pose_map_base": _tf_dict(t_mb), "pose_map_camera": _tf_dict(t_mc),
                 "odom_twist": odom}
        self._write("robot_states.jsonl", robot)
        self._write("pedestrian_states.jsonl", {"sample": idx, "stamp_ns": t, **sync, "pedestrians": peds})
        sync_rec = {"sample": idx, "stamp_ns": t, "rgb_stamp_ns": t, "depth_stamp_ns": stamp_ns(depth.header.stamp),
                    "rgb_encoding": rgb.encoding, "depth_encoding": depth.encoding, "rgb_frame": rgb.header.frame_id,
                    "depth_frame": depth.header.frame_id}
        if self.live_costmaps:
            cm = self.live_costmaps[-1]
            md = cm.metadata
            grid = np.frombuffer(bytes(cm.data), np.uint8).reshape(md.size_y, md.size_x)
            np.save(os.path.join(self.cfg.out_dir, "live_social_cost", name + ".npy"), grid)
            sync_rec["live_costmap"] = {"stamp_ns": stamp_ns(cm.header.stamp), "frame_id": cm.header.frame_id,
                                        "resolution": md.resolution, "size_x": md.size_x, "size_y": md.size_y,
                                        "origin_x": md.origin.position.x, "origin_y": md.origin.position.y}
        self._write("sync.jsonl", sync_rec)
        self.stats["rgb_depth_dt_ns_max"] = max(self.stats["rgb_depth_dt_ns_max"], abs(t - stamp_ns(depth.header.stamp)))
        self.samples += 1

    def _on_odom(self, m: Odometry) -> None:
        t = stamp_ns(m.header.stamp)
        self.odom.append((t, m))
        self._track_motion()

    def _track_motion(self) -> None:
        """First odometry twist of real motion (|v| > 2 cm/s or |w| > 0.05 rad/s) since the episode start.

        Twist, not displacement: the reset teleports the robot after the episode start, and the
        odometry pose follows the teleport, which a displacement test mistakes for motion.
        """
        if self.stats.get("first_motion_rel_s") is not None or self.t_start_ns is None or not self.odom:
            return
        t, m = self.odom[-1]
        if t < self.t_start_ns:
            return
        tw = m.twist.twist
        if abs(tw.linear.x) > 0.02 or abs(tw.angular.z) > 0.05:
            self.stats["first_motion_rel_s"] = (t - self.t_start_ns) / 1e9

    def _odom_at(self, t: int) -> dict | None:
        if not self.odom:
            return None
        ts, m = min(self.odom, key=lambda e: abs(e[0] - t))
        tw = m.twist.twist
        return {"stamp_ns": ts, "vx": tw.linear.x, "vy": tw.linear.y, "wz": tw.angular.z}

    # ---------------- output ----------------
    def _write(self, name: str, record: dict) -> None:
        f = self._files.get(name)
        if f is None:
            f = self._files[name] = open(os.path.join(self.cfg.out_dir, name), "a")
        f.write(json.dumps(record) + "\n")

    def _finish(self, reason: str) -> None:
        if self.state == "done":
            return
        self.state = "done"
        self.stats["end_reason"] = reason
        self.stats["wall_end"] = time.time()
        for f in self._files.values():
            f.close()
        self._files.clear()
        ep = self.episode
        self._track_motion()
        fm = self.stats.get("first_motion_rel_s")
        self.stats["start_ok"] = fm is not None and fm <= self.cfg.max_start_delay_s
        meta = {
            **self.cfg.metadata,
            "episode_id": self.episode_id,
            "episode_seed": int(ep.seed) if ep is not None else None,
            "world": ep.world if ep is not None else None,
            "tm_robots": ep.tm_robots if ep is not None else None,
            "tm_obstacles": ep.tm_obstacles if ep is not None else None,
            "episode_start_ns": self.t_start_ns,
            "record_start_ns": getattr(self, "t0_ns", None),
            "last_frame_ns": self.last_rgb_ns,
            "sample_hz": self.cfg.sample_hz, "duration_s": self.cfg.duration_s, "warmup_s": self.cfg.warmup_s, "min_episode_id": self.cfg.min_episode_id,
            "samples": self.samples,
            "expected_samples": int(math.floor(self.cfg.duration_s * self.cfg.sample_hz)),
            "stats": self.stats,
            "agent_meta": self.agent_meta,
            "topics": {k: getattr(self.cfg, k) for k in ("rgb_topic", "depth_topic", "camera_info_topic", "odom_topic", "peds_topic", "live_costmap_topic")},
            "frames": {"map": self.cfg.map_frame, "base": self.cfg.base_frame, "camera": self.cfg.camera_frame},
        }
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        with open(os.path.join(self.cfg.out_dir, "episode_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(f"episode {self.episode_id} done ({reason}): {self.samples} samples")

    def run(self) -> dict:
        deadline = time.monotonic() + self.cfg.episode_wait_s
        while rclpy.ok() and self.state != "done":
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.state == "waiting" and time.monotonic() > deadline:
                self.stats["end_reason"] = "no_episode"
                break
        return self.stats
