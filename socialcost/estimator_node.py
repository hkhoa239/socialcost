"""Closed-loop social cost estimate for Nav2 (overlay socialcost_estimate).

Publishes nav_msgs/OccupancyGrid on <env>/socialcost/estimate: the 10 m label grid centred on
base_link, frame base_link, stamped with the input's time, values round(cost * 100 / 254) <= 99
(nav2_mask_overlay_layer maps v -> v * 254 / 100, so cost is reproduced within 2.54).

methods
  zero       an all-zero grid (bodies-only arm through the same path)
  rule       personal space only, from arena_peds positions (clean positions, no interactions)
  oracle     arena-native label from the full arena_peds state
  oracle_ext extended label from arena_peds + interactions
  pos        learned position model on rasterised arena_peds positions/headings (clean positions)
  rgbd       learned RGB-D model on the camera images
  depth/rgb  learned depth-only / RGB-only model
  typemix    TypeMix + counterfactual stage 1 (socialcost/bench/online.py): tracks + RGB-D person crops -> social
             types with uncertainty -> typed costmap; publishes the mean (camera-triggered, full 10 m grid)
  typemix_ucb  the same, publishing mu + k sigma (--ucb-k; risk-averse use of the predicted uncertainty)
  typemix_gt TypeMix fed the true social types (live InteractionFrame + gestures): same model family without
             stage-1 errors (privileged)
  pos_u      probabilistic position-only U-Net of the benchmark (no social semantics)
  bench options: --bench-threshold T zeroes published costs below T (label floor 12: no labelled cell is below it);
             --ema A blends each published grid with the previous one (shifted to the new lattice origin):
             g = A * new + (1 - A) * previous, reset when the robot jumps > 1 m (scenario switch)
  oracle_ext_replay  the extended oracle label sent through exactly the RGB-D path: triggered by the
             camera frames, painted in the robot-centred label grid at the camera stamp, FOV-masked
             and resampled like a prediction (integration control, privileged)

diagnostics (never change what a method publishes unless --post is given)
  --post spec.json     post-process the robot-frame prediction (socialcost.postprocess; sensor data only)
  --shadow             also compute oracle_ext_replay for learned methods and log the agreement (not published)
  --dump-dir D         save every --dump-every-th tick's intermediate grids to D/tick_<seq>.npz

    python -m socialcost.estimator_node --method rgbd --checkpoint <best.pt> [--log out.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from .labels import GridSpec, LayerParams, Ped, paint, yaw_from_quat

ENV = "/arena/env_0"
BENCH_METHODS = ("typemix", "typemix_ucb", "typemix_gt", "pos_u")
CAMERA_METHODS = ("rgbd", "depth", "rgb", "oracle_ext_replay") + BENCH_METHODS
NS = f"{ENV}/task_generator_node/turtlebot"
BASE = "env_0/turtlebot/base_link"
CAM = "env_0/turtlebot/oakd_rgb_camera_optical_frame"


def to_occupancy(cost: np.ndarray) -> np.ndarray:
    return np.minimum(np.round(cost.astype(np.float32) * 100.0 / 254.0), 99).astype(np.int8)


class Estimator(Node):
    def __init__(self, method: str, checkpoint: str | None, log: str | None, rate: float, frame: str | None = None,
                 post: str | None = None, shadow: bool = False, dump_dir: str | None = None, dump_every: int = 10,
                 bench: dict | None = None) -> None:
        super().__init__("socialcost_estimator", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.method = method
        # cost handed to Nav2 = min(round(cost x publish_scale), publish_cap); identity by default (controller diagnosis)
        self.pub_scale = float((bench or {}).get("publish_scale", 1.0))
        self.pub_cap = int((bench or {}).get("publish_cap", 252))
        self.label_spec = (bench or {}).get("label_spec", "v1")  # oracle_ext: v1 (extended/2.0) or a labels_v2 spec
        self.label_channel = (bench or {}).get("label_channel", "extended")  # v2 oracle: native | membership | extended
        from . import postprocess
        self.pp = postprocess
        self.post = postprocess.load_spec(post)
        self.shadow = shadow and method in ("pos", "rgbd", "depth", "rgb")
        self.dump_dir, self.dump_every, self.seq = dump_dir, max(1, dump_every), 0
        if dump_dir:
            import os
            os.makedirs(dump_dir, exist_ok=True)
        self.params = LayerParams.from_overlay()
        self.grid = GridSpec.centered(10.0, 0.05)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, f"{ENV}/socialcost/estimate", latched)
        self.peds = None
        self.rgb = self.depth = None
        # recent camera frames, paired by stamp in tick() (exact match preferred, else nearest within PAIR_TOL_NS):
        # requiring the two latest messages to share a stamp can fail for a whole launch (render products one
        # frame apart / arrival order vs the timer) and then nothing is ever published
        from collections import deque
        self.rgb_buf, self.depth_buf = deque(maxlen=12), deque(maxlen=12)
        self.last_pair_stamp = None
        self.pair_dt_ns = 0
        self.create_subscription(Pedestrians, f"{ENV}/arena_peds", lambda m: setattr(self, "peds", m), qos_profile_sensor_data)
        self.model = None
        if method in ("pos", "rgbd", "depth", "rgb"):
            import torch

            from .learn.models import build
            # --checkpoint a.pt[,b.pt,...]: several checkpoints (e.g. benchmark seeds) are averaged (seed ensemble);
            # benchmark checkpoints (bench_train.py) name the input in config["model"], Gate-5 ones in config["inputs"]
            self.models = []
            for c in checkpoint.split(","):
                ck = torch.load(c, map_location="cuda")
                self.inputs = ck["config"].get("inputs") or ck["config"]["model"]
                m = build(self.inputs).cuda().eval()
                m.load_state_dict(ck["model"])
                self.models.append(m)
            self.model = self.models[0]
            self.torch = torch
            # published values below this are set to 0 (label floor, as --bench-threshold for the bench methods)
            self.img_threshold = float((bench or {}).get("threshold", 0.0))
        self.online = None
        if method in BENCH_METHODS:
            import os
            for k, v in (("SC_DV1_ROOT", "/opt/arena_ws/data/socialcost/dataset_v1"), ("SC_BENCH_ROOT", "/opt/arena_ws/data/costmap_bench"),
                         ("SC_CASCADE_CACHE", "/opt/arena_ws/data/costmap_bench/cascade_cache"), ("SC_STAGE1_ROOT", "/opt/arena_ws/data/costmap_bench/stage1")):
                if os.path.isdir(v):
                    os.environ.setdefault(k, v)
            from .bench.online import OnlineTypeMix
            b = bench or {}
            kind = {"typemix": "typemix", "typemix_ucb": "typemix", "typemix_gt": "typemix_gt", "pos_u": "pos_u"}[method]
            self.online = OnlineTypeMix(stage1=b.get("stage1", "cffull"), model=b.get("model", "typemix_cffullraw"), label=b.get("label", "extended"),
                                        seeds=tuple(b.get("seeds", (0, 1, 2))), kind=kind, runs=b.get("runs", "main"), memory=b.get("memory"))
            self.ucb_k = float(b.get("ucb_k", 1.0)) if method == "typemix_ucb" else 0.0
            self.bench_threshold = float(b.get("threshold", 0.0))
            self.ema = float(b.get("ema", 1.0))
            self._prev = None
            self.get_logger().info(f"{method}: {kind} label {b.get('label', 'extended')} seeds {b.get('seeds', (0, 1, 2))} k {self.ucb_k}")
        self.sngnn = None
        if method == "sngnn":  # SNGNN2D-v2 via the host service (scripts/sngnn_server.py; needs DGL, not in this container)
            import os
            import socket

            import yaml
            from nav_msgs.msg import Odometry
            b = bench or {}
            doc = json.load(open(b["challenge"]))
            cfgc = next(c for c in doc["configs"] if c["config_id"] == b["config_id"])
            scen = os.path.join("/opt/arena_ws/src/Arena/arena_simulation_setup/worlds", cfgc["world"], "scenarios", f"scv1_nav_{b['config_id']}", "scenario.yaml")
            objects = (yaml.safe_load(open(scen)) or {}).get("objects") or [] if os.path.exists(scen) else []
            self.sngnn = socket.create_connection(("127.0.0.1", int(b.get("sngnn_port", 47811))), timeout=30)
            self.sngnn_f = self.sngnn.makefile("rwb")
            self._sngnn_call({"cmd": "scene", "world": cfgc["world"], "objects": objects, "goal": cfgc["robot"]["goal"][:2]})
            self.twist = {"vx": 0.0, "wz": 0.0}
            self.create_subscription(Odometry, f"{NS}/odom", lambda m: setattr(self, "twist", {"vx": m.twist.twist.linear.x, "wz": m.twist.twist.angular.z}),
                                     qos_profile_sensor_data)
            self.bench_threshold = float(b.get("threshold", 0.0))
            self.get_logger().info(f"sngnn: {cfgc['world']} {b['config_id']}, {len(objects)} objects")
        self.K = None
        self.fov = None
        if method in ("pos", "rgbd", "depth", "rgb", "oracle_ext_replay") + BENCH_METHODS:
            # learned models are trained on the camera FOV only; their output elsewhere is untrained
            from sensor_msgs.msg import CameraInfo
            self.create_subscription(CameraInfo, f"{NS}/oakd/camera_info", self._info, qos_profile_sensor_data)
        if method in CAMERA_METHODS:
            self.create_subscription(Image, f"{NS}/oakd/image", lambda m: self.rgb_buf.append(m), qos_profile_sensor_data)
            self.create_subscription(Image, f"{NS}/oakd/depth_image", lambda m: self.depth_buf.append(m), qos_profile_sensor_data)
        self.frames = None
        if method in ("oracle_ext", "oracle_ext_replay", "typemix_gt") or self.shadow:
            from arena_humansim_msgs.msg import InteractionFrame
            self.create_subscription(InteractionFrame, f"{ENV}/task_generator_node/interactions", self._frames, latched)
        self.log = open(log, "a") if log else None
        self.frame = frame or self._costmap_param("global_frame")
        # the mask must share the costmap's cell grid: nav2_mask_overlay_layer samples the mask at the
        # costmap's cell centres, and a 0.05 m mask on the 0.10 m local costmap is read 2.5 cm off-centre
        self.out_res = float(self._costmap_param("resolution"))
        self.get_logger().info(f"publishing in {self.frame} at {self.out_res} m")
        self.create_timer(1.0 / rate, self.tick)

    @staticmethod
    def _costmap_param(name: str) -> str:
        """A local costmap parameter. The mask is published in its global frame (applied without a TF
        lookup: nav2_mask_overlay_layer transforms other frames with the latest TF, which misplaces a
        robot-centred mask while the robot moves) and at its resolution."""
        import subprocess
        for _ in range(60):
            r = subprocess.run(["ros2", "param", "get", f"{NS}/local_costmap/local_costmap", name], capture_output=True, text=True)
            if r.returncode == 0 and " value is:" in r.stdout:
                return r.stdout.split(" value is:")[1].strip()
            time.sleep(2.0)
        raise RuntimeError(f"local costmap {name} not available")

    def _tf(self, target: str, source: str, stamp) -> np.ndarray | None:
        try:
            t = self.tf.lookup_transform(target, source, stamp, rclpy.duration.Duration(seconds=0.3))
        except Exception:  # noqa: BLE001
            return None
        q = t.transform.rotation
        return np.array([t.transform.translation.x, t.transform.translation.y, yaw_from_quat(q.x, q.y, q.z, q.w)])

    def _sngnn_call(self, msg: dict) -> dict:
        self.sngnn_f.write((json.dumps(msg) + "\n").encode())
        self.sngnn_f.flush()
        return json.loads(self.sngnn_f.readline())

    def _info(self, m) -> None:
        self.K = np.array(m.k).reshape(3, 3)

    def _fov_mask(self) -> np.ndarray | None:
        """Label cells inside the camera FOV (as in training); None until calibration and TF are known."""
        if self.fov is None and self.K is not None:
            from .learn.data import fov_mask
            from .qa import pose_matrix
            try:
                tb = self.tf.lookup_transform(BASE, CAM, Time())
            except Exception:  # noqa: BLE001
                return None
            tr, q = tb.transform.translation, tb.transform.rotation
            self.fov = fov_mask(self.grid, self.K, pose_matrix({"x": tr.x, "y": tr.y, "z": tr.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w}))
        return self.fov

    def _frames(self, msg) -> None:
        offs = list(msg.member_offset) + [len(msg.member_id)]
        self.frames = {"interactions": [{"interaction_id": int(i), "interaction_type": int(msg.interaction_type[k]), "outcome": int(msg.outcome[k]),
                                         "member_ids": [int(x) for x in msg.member_id[offs[k]:offs[k + 1]]]} for k, i in enumerate(msg.interaction_id)]}

    def _pose(self, stamp) -> dict | None:
        try:
            t = self.tf.lookup_transform("map", BASE, stamp, rclpy.duration.Duration(seconds=0.2))
        except Exception:  # noqa: BLE001
            return None
        q = t.transform.rotation
        return {"x": t.transform.translation.x, "y": t.transform.translation.y, "yaw": yaw_from_quat(q.x, q.y, q.z, q.w)}

    def _states(self) -> list[dict]:
        from rosidl_runtime_py.convert import message_to_ordereddict
        out = []
        for p in self.peds.pedestrians:
            q = p.pose.orientation
            out.append({"name": p.name, "id": int(p.id), "x": p.pose.position.x, "y": p.pose.position.y, "yaw": yaw_from_quat(q.x, q.y, q.z, q.w),
                        "vx": p.twist.linear.x, "vy": p.twist.linear.y,
                        "animation_state": int(p.animation_state), "interaction_id": int(p.interaction_id), "interaction_type": int(p.interaction_type),
                        "agent_type": p.agent_type, "model_uri": p.model_uri, "gestures": [json.loads(json.dumps(message_to_ordereddict(g))) for g in p.gestures]})
        return out

    def _target_grid(self, pose_f: np.ndarray) -> GridSpec:
        """Frame-aligned 10 m grid around the robot at the costmap resolution, origin on the costmap's cell lattice."""
        r = self.out_res
        size = self.grid.width * self.grid.resolution
        n = int(round(size / r))
        return GridSpec(resolution=r, width=n, height=n,
                        origin_x=round((pose_f[0] - size / 2) / r) * r, origin_y=round((pose_f[1] - size / 2) / r) * r, frame=self.frame)

    def _resample(self, cost_robot: np.ndarray, pose_f: np.ndarray, target: GridSpec) -> np.ndarray:
        """Robot-frame prediction (label grid) -> frame-aligned grid, nearest cell."""
        i, j = np.meshgrid(np.arange(target.width), np.arange(target.height))
        wx = target.origin_x + (i + 0.5) * target.resolution - pose_f[0]
        wy = target.origin_y + (j + 0.5) * target.resolution - pose_f[1]
        c, s_ = math.cos(pose_f[2]), math.sin(pose_f[2])
        lx, ly = c * wx + s_ * wy, -s_ * wx + c * wy
        g = self.grid
        ii = np.floor((lx - g.origin_x) / g.resolution).astype(int)
        jj = np.floor((ly - g.origin_y) / g.resolution).astype(int)
        ok = (ii >= 0) & (ii < g.width) & (jj >= 0) & (jj < g.height)
        out = np.zeros((target.height, target.width), np.uint8)
        out[ok] = cost_robot[jj[ok], ii[ok]]
        return out

    def _states_in(self, T: np.ndarray) -> list[dict]:
        """Pedestrian states moved from map into the mask frame (positions, yaw, gesture targets)."""
        c, s_ = math.cos(T[2]), math.sin(T[2])
        out = []
        for p in self._states():
            q = dict(p, x=T[0] + c * p["x"] - s_ * p["y"], y=T[1] + s_ * p["x"] + c * p["y"], yaw=p["yaw"] + T[2])
            q["gestures"] = [dict(g, at={**g.get("at", {}), "x": T[0] + c * g.get("at", {}).get("x", 0.0) - s_ * g.get("at", {}).get("y", 0.0),
                                          "y": T[1] + s_ * g.get("at", {}).get("x", 0.0) + c * g.get("at", {}).get("y", 0.0)}) for g in p["gestures"]]
            out.append(q)
        return out

    def _smooth(self, cost: np.ndarray, target: GridSpec, pose_f: np.ndarray) -> np.ndarray:
        """Exponential moving average of the frame-aligned grid; the previous grid is shifted by the lattice offset."""
        cur = cost.astype(np.float32)
        if self._prev is not None:
            prev, (ox, oy), ppose = self._prev
            if math.hypot(pose_f[0] - ppose[0], pose_f[1] - ppose[1]) < 1.0 and prev.shape == cur.shape:
                di = int(round((target.origin_x - ox) / target.resolution))
                dj = int(round((target.origin_y - oy) / target.resolution))
                sh = np.zeros_like(prev)
                H, W = prev.shape
                src = prev[max(dj, 0):H + min(dj, 0), max(di, 0):W + min(di, 0)]
                sh[max(-dj, 0):max(-dj, 0) + src.shape[0], max(-di, 0):max(-di, 0) + src.shape[1]] = src
                cur = self.ema * cur + (1 - self.ema) * sh
        self._prev = (cur, (target.origin_x, target.origin_y), pose_f.copy())
        out = np.round(cur).clip(0, 252).astype(np.uint8)
        if self.bench_threshold > 0:
            out[out < self.bench_threshold] = 0
        return out

    PAIR_TOL_NS = 40_000_000

    def _pair(self) -> bool:
        """Newest (rgb, depth) pair: same stamp if available, else the nearest depth within PAIR_TOL_NS. False if
        no pair, or the newest pair was already used."""
        def ns(m):
            return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        depths = list(self.depth_buf)
        if not depths:
            return False
        dstamps = [ns(d) for d in depths]
        for r in reversed(list(self.rgb_buf)):
            t = ns(r)
            k = min(range(len(depths)), key=lambda i: abs(dstamps[i] - t))
            if abs(dstamps[k] - t) <= self.PAIR_TOL_NS:
                if self.last_pair_stamp == t:
                    return False
                self.rgb, self.depth, self.pair_dt_ns, self.last_pair_stamp = r, depths[k], dstamps[k] - t, t
                return True
        return False

    def _oracle_robot(self, pose: dict) -> np.ndarray:
        """Extended oracle label in the robot-centred label grid at `pose` (map frame), as the dataset labels are painted."""
        from .dataset import EXTENDED_CONFIG
        from .extensions import ExtParams, paint_extended
        if not hasattr(self, "_ext"):
            self._ext = ExtParams.load(EXTENDED_CONFIG)
        cost, _ = paint_extended(self.params, self._ext, self._states(), self.frames, self.grid, (pose["x"], pose["y"], pose["yaw"]))
        return cost.astype(np.uint8)

    def tick(self) -> None:
        t0 = time.monotonic()
        if self.method in CAMERA_METHODS:
            if self.K is None or not self._pair() or (self.method in BENCH_METHODS and self.peds is None):
                return
            stamp = self.rgb.header.stamp
        elif self.method == "zero":
            stamp = self.get_clock().now().to_msg()
        else:
            if self.peds is None:
                return
            stamp = self.peds.header.stamp
        # robot pose in the mask frame at the input's time: the mask is placed where the input was taken
        pose_f = self._tf(self.frame, BASE, Time.from_msg(stamp))
        if pose_f is None:
            return
        target = self._target_grid(pose_f)
        dbg: dict = {}
        if self.method == "zero":
            cost = np.zeros((target.height, target.width), np.uint8)
        elif self.method in ("rule", "oracle", "oracle_ext"):
            T_fm = self._tf(self.frame, "map", Time())
            if T_fm is None:
                return
            states = self._states_in(T_fm)
            if self.method == "rule":
                cost = paint(self.params, [Ped(p["x"], p["y"], p["animation_state"]) for p in states], target)
            elif self.method == "oracle":
                cost = paint(self.params, [Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in states], target)
            else:
                if self.label_spec and self.label_spec != "v1":  # socialcost_v2 ground truth (labels_v2.py, extended composition)
                    from .labels_v2 import SPECS, labels_and_components
                    cost = labels_and_components(SPECS[self.label_spec], self.params, target, states, self.frames, (0.0, 0.0, 0.0))[self.label_channel]
                else:
                    from .dataset import EXTENDED_CONFIG
                    from .extensions import ExtParams, paint_extended
                    cost, _ = paint_extended(self.params, ExtParams.load(EXTENDED_CONFIG), states, self.frames, target, (0.0, 0.0, 0.0))
        elif self.method == "pos":
            from .learn.data import raster_peds
            pose = self._pose(Time.from_msg(stamp))
            if pose is None:
                return
            x = {"raster": self.torch.from_numpy(raster_peds(self._states(), pose, self.grid)).unsqueeze(0).cuda()}
            with self.torch.no_grad():
                robot = (self.model(x)[0].cpu().numpy() * self.params.max_cost_value).round().clip(0, 252).astype(np.uint8)
            fov = self._fov_mask()
            if fov is None:
                return
            dbg["robot_raw"] = robot
            if self.post:
                robot = self.pp.apply(robot, self.post)
                dbg["robot_post"] = robot
            cost = self._resample(np.where(fov, robot, 0).astype(np.uint8), pose_f, target)
        elif self.method in BENCH_METHODS:
            pose = self._pose(Time.from_msg(stamp))
            try:
                tc = self.tf.lookup_transform("map", CAM, Time.from_msg(stamp), rclpy.duration.Duration(seconds=0.2))
            except Exception:  # noqa: BLE001
                return
            if pose is None:
                return
            tr, q = tc.transform.translation, tc.transform.rotation
            cam_pose = {"x": tr.x, "y": tr.y, "z": tr.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w}
            rgb = np.frombuffer(bytes(self.rgb.data), np.uint8).reshape(self.rgb.height, self.rgb.width, 3).copy()
            depth = np.frombuffer(bytes(self.depth.data), np.float32).reshape(self.depth.height, self.depth.width).copy()
            depth[~np.isfinite(depth)] = np.nan
            mu, sigma, s1 = self.online.step(self._states(), pose, cam_pose, self.K, rgb, depth, self.frames)
            robot = ((mu + self.ucb_k * sigma) * self.params.max_cost_value).round().clip(0, 252).astype(np.uint8)
            dbg["robot_raw"] = robot
            dbg["sigma_mean"] = float(sigma.mean())
            if s1:
                dbg["stage1"] = {"names": s1["names"], "p_inter": (1 - np.asarray(s1["edge_p"])[..., 0]).round(3).tolist(),
                                 "p_halt": np.asarray(s1["node_p"])[:, 0].round(3).tolist(), "p_wheel": np.asarray(s1["node_p"])[:, 1].round(3).tolist()}
            self._fov_mask()  # logging only: the bench models predict the full 10 m grid from tracks (+ crops in view)
            if self.bench_threshold > 0:
                robot = np.where(robot >= self.bench_threshold, robot, 0).astype(np.uint8)
            cost = self._resample(robot, pose_f, target)
            if self.ema < 1.0:
                cost = self._smooth(cost, target, pose_f)
        elif self.method == "sngnn":
            import base64
            pose = self._pose(Time.from_msg(stamp))
            if pose is None:
                return
            rep = self._sngnn_call({"cmd": "step", "stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec, "robot": pose, "twist": self.twist,
                                    "peds": self._states()})
            if "cost" not in rep:
                self.get_logger().warn(f"sngnn service: {rep.get('error')}")
                return
            robot = np.frombuffer(base64.b64decode(rep["cost"]), np.uint8).reshape(rep["shape"]).copy()
            dbg["robot_raw"] = robot
            dbg["service_ms"] = rep.get("ms")
            if self.bench_threshold > 0:
                robot = np.where(robot >= self.bench_threshold, robot, 0).astype(np.uint8)
            cost = self._resample(robot, pose_f, target)
        elif self.method == "oracle_ext_replay":
            pose = self._pose(Time.from_msg(stamp))
            fov = self._fov_mask()
            if pose is None or fov is None or self.peds is None:
                return
            robot = self._oracle_robot(pose)
            dbg["robot_raw"] = robot
            cost = self._resample(np.where(fov, robot, 0).astype(np.uint8), pose_f, target)
        else:
            from .learn.data import lift_indices
            from .qa import pose_matrix
            try:
                tb = self.tf.lookup_transform(BASE, CAM, Time())
            except Exception:  # noqa: BLE001
                return
            tr, q = tb.transform.translation, tb.transform.rotation
            T_bc = pose_matrix({"x": tr.x, "y": tr.y, "z": tr.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w})
            rgb = np.frombuffer(bytes(self.rgb.data), np.uint8).reshape(self.rgb.height, self.rgb.width, 3).astype(np.float32) / 255.0
            depth = np.frombuffer(bytes(self.depth.data), np.float32).reshape(self.depth.height, self.depth.width).copy()
            depth[~np.isfinite(depth)] = np.nan
            valid = np.isfinite(depth).astype(np.float32)
            d = np.nan_to_num(depth, nan=0.0) / 10.0
            chans = []
            if self.inputs in ("rgbd", "rgb"):
                chans.append(rgb.transpose(2, 0, 1) - 0.5)
            if self.inputs in ("rgbd", "depth"):
                chans += [d[None], valid[None]]
            x = {"image": self.torch.from_numpy(np.concatenate(chans, 0)).unsqueeze(0).cuda(),
                 "lift": self.torch.from_numpy(lift_indices(depth, self.K, T_bc, mode="ground" if self.inputs == "rgb" else "depth")).unsqueeze(0).cuda()}
            with self.torch.no_grad():
                out = self.torch.stack([m(x)[0] for m in self.models]).mean(0)
                robot = (out.cpu().numpy() * self.params.max_cost_value).round().clip(0, 252).astype(np.uint8)
            if self.img_threshold > 0:
                robot = np.where(robot >= self.img_threshold, robot, 0).astype(np.uint8)
            fov = self._fov_mask()
            if fov is None:
                return
            dbg["robot_raw"] = robot
            if self.post:
                vis = self.pp.visibility_mask(depth, self.K, T_bc, self.grid, self.post.get("margin_m", 0.6)) if self.post.get("visibility") else None
                robot = self.pp.apply(robot, self.post, vis)
                dbg["robot_post"] = robot
            cost = self._resample(np.where(fov, robot, 0).astype(np.uint8), pose_f, target)
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        msg.info.resolution = target.resolution
        msg.info.width, msg.info.height = target.width, target.height
        msg.info.origin.position.x, msg.info.origin.position.y = target.origin_x, target.origin_y
        msg.info.origin.orientation.w = 1.0
        if self.pub_scale != 1.0 or self.pub_cap < 252:
            cost = np.minimum(np.round(cost.astype(np.float32) * self.pub_scale), self.pub_cap).astype(np.uint8)
        occ = to_occupancy(cost)
        msg.data = occ.ravel().tolist()
        self.pub.publish(msg)
        compute_ms = (time.monotonic() - t0) * 1000
        self.seq += 1
        if self.log:
            now = self.get_clock().now().nanoseconds
            rec = {"t_ns": now, "input_stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec, "compute_ms": compute_ms,
                   "nonzero": int((cost > 0).sum()), "seq": self.seq, "frame": self.frame, "res": target.resolution, "w": target.width, "h": target.height,
                   "origin": [target.origin_x, target.origin_y], "pose_f": [float(v) for v in pose_f], "cost_max": int(cost.max()),
                   "occ_max": int(occ.max()), "occ_clipped": int((occ >= 99).sum()),
                   "gt10": int((cost > 10).sum()), "gt50": int((cost > 50).sum()), "gt100": int((cost > 100).sum())}
            if self.method in CAMERA_METHODS:
                rec["pair_dt_ms"] = self.pair_dt_ns / 1e6
            if "stage1" in dbg:
                rec["stage1"] = dbg["stage1"]
                rec["sigma_mean"] = dbg["sigma_mean"]
            if "robot_raw" in dbg and self.fov is not None:
                rr = dbg["robot_raw"]
                rec["raw_fov_nonzero"] = int(((rr > 0) & self.fov).sum())
                rec["raw_fov_mean"] = float(rr[self.fov].mean())
            if self.shadow and "robot_raw" in dbg and self.peds is not None:
                pose = self._pose(Time.from_msg(stamp))
                if pose is not None:
                    sh = self._oracle_robot(pose)
                    dbg["shadow_robot"] = sh
                    pr = dbg.get("robot_post", dbg["robot_raw"]).astype(np.int16)
                    f, t_ = self.fov, sh.astype(np.int16)
                    a, b = pr[f], t_[f]
                    rec["shadow"] = {"mae_fov": float(np.abs(a - b).mean()), "label_nonzero_fov": int((b > 0).sum()), "pred_nonzero_fov": int((a > 0).sum()),
                                     **{f"tp{th}": int(((a >= th) & (b >= th)).sum()) for th in (10, 50, 100)},
                                     **{f"fp{th}": int(((a >= th) & (b < th)).sum()) for th in (10, 50, 100)},
                                     **{f"fn{th}": int(((a < th) & (b >= th)).sum()) for th in (10, 50, 100)}}
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        if self.dump_dir and self.seq % self.dump_every == 0:
            import os
            np.savez_compressed(os.path.join(self.dump_dir, f"tick_{self.seq:06d}.npz"), cost=cost, occ=occ, fov=self.fov if self.fov is not None else np.zeros(1),
                                stamp_ns=stamp.sec * 1_000_000_000 + stamp.nanosec, pose_f=pose_f, origin=np.array([target.origin_x, target.origin_y]),
                                res=target.resolution, frame=self.frame, **{k: v for k, v in dbg.items() if isinstance(v, np.ndarray)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["zero", "rule", "oracle", "oracle_ext", "pos", "rgbd", "depth", "rgb", "oracle_ext_replay", "sngnn", *BENCH_METHODS])
    ap.add_argument("--challenge", default="/opt/arena_ws/data/socialcost/navigation_challenge_v1/navigation_challenge_v1.json")
    ap.add_argument("--config-id", default=None, help="sngnn: navigation challenge configuration (scene objects, goal)")
    ap.add_argument("--sngnn-port", type=int, default=47811)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--frame", default=None, help="mask frame; default: the local costmap global frame")
    ap.add_argument("--post", default=None, help="post-processing spec (json) for learned methods")
    ap.add_argument("--shadow", action="store_true", help="log agreement with oracle_ext_replay (not published)")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-every", type=int, default=10)
    ap.add_argument("--bench-label", default="extended", help="bench methods: label the stage-2 models were trained on")
    ap.add_argument("--bench-stage1", default="cffull")
    ap.add_argument("--bench-model", default="typemix_cffullraw")
    ap.add_argument("--bench-seeds", default="0,1,2")
    ap.add_argument("--ucb-k", type=float, default=1.0)
    ap.add_argument("--bench-threshold", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=1.0)
    ap.add_argument("--publish-scale", type=float, default=1.0, help="multiply the published cost (every method)")
    ap.add_argument("--publish-cap", type=int, default=252, help="cap the published cost (every method)")
    ap.add_argument("--bench-runs", default="main", help="bench checkpoints folder under runs/ (main = v1, v2 = socialcost_v2)")
    ap.add_argument("--bench-memory", default=None, choices=[None, "hold", "decay", "decay_geo"], help="typemix: temporal interaction memory")
    ap.add_argument("--label-spec", default="v1", help="oracle_ext ground truth: v1 or a socialcost_v2 spec (v2, ...)")
    ap.add_argument("--label-channel", default="extended", choices=["native", "membership", "extended"], help="v2 oracle label channel")
    args = ap.parse_args()
    rclpy.init()
    bench = {"label": args.bench_label, "stage1": args.bench_stage1, "model": args.bench_model, "seeds": [int(x) for x in args.bench_seeds.split(",")],
             "ucb_k": args.ucb_k, "threshold": args.bench_threshold, "ema": args.ema,
             "publish_scale": args.publish_scale, "publish_cap": args.publish_cap, "runs": args.bench_runs, "memory": args.bench_memory,
             "label_spec": args.label_spec, "label_channel": args.label_channel, "challenge": args.challenge, "config_id": args.config_id, "sngnn_port": args.sngnn_port}
    n = Estimator(args.method, args.checkpoint, args.log, args.rate, args.frame, args.post, args.shadow, args.dump_dir, args.dump_every, bench)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
