"""Online TypeMix + counterfactual stage 1 (the best costmap model of the benchmark) for closed-loop navigation.

Same inputs and code paths as training (group_net.sample_tensors2 on a rolling 2 s buffer, dataset visibility,
crops2 boxes, Dv1Samples typemix inputs), fed from live messages at 5 Hz (the dataset sample rate):

  stage 1   GroupNet ensemble (stage1/<s1>/full_m*.pt, temperatures from its metrics.json), raw factorised export:
            p(type) = sigmoid(same / T_s) softmax(type / T_t), p(none) = 1 - sigmoid(same / T_s); node p = sigmoid(z / T_n);
            averaged over members
  stage 2   TypeMixCost seeds (runs/main/<model>_<label>_s*/best.pt): mu = sigmoid(o0), sigma^2 = exp(o1);
            seed ensemble mu = mean mu_s, sigma^2 = mean sigma_s^2 + var mu_s
  output    robot-centred 200 x 200 grid (0.05 m), cost units 0..1 (x 252 = layer cost): mu and sigma

    est = OnlineTypeMix(); mu, sigma = est.step(states, robot_pose, cam_pose, K, rgb, depth)

Navigation arms (OnlineTypeMix kind): "typemix" (stage 1 -> TypeMix, ours), "typemix_gt" (true types from the live
InteractionFrame and gestures -> TypeMix trained on true types: the stage-1-error-free reference of the same model
family), "pos_u" (probabilistic position-only U-Net: no social semantics). Log-variance clamped to [-12, 0] as in
training (bench_train.mean_logvar).
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import deque

import cv2
import numpy as np
import torch

from ..dv1 import assets
from ..dv1.post import _body_id, _project, body_points, classify, point_states
from ..learn.data import raster_peds
from ..qa import pose_matrix
from .cascade_data import CROP2_H, CROP2_W, person_box
from .data import P_MAX_PEOPLE, ROOT, SCALE
from .group_net import GroupNet, sample_tensors2
from ..learn.models import PositionToGrid
from .data import gt_soft_inputs
from .typemix import TypeMixCost

LOGVAR_MIN, LOGVAR_MAX = -12.0, 0.0

BENCH = os.environ.get("SC_BENCH_ROOT", "/opt/arena_ws/data/costmap_bench")
HIST = 10


class _Buffer:
    """EpisodeCache2-compatible rolling window (labels are dummies; only inputs are read)."""

    def __init__(self, names, track, robot, vis, crop):
        self.names, self.track, self.robot, self.vis, self.crop = names, track, robot, vis, crop
        self.T, self.P = track.shape[:2]
        self.present = np.abs(track).sum(-1) > 0
        self.inter = -np.ones((self.T, self.P), np.int32)
        self.itype = -np.ones((self.T, self.P), np.int32)
        self.halt = np.zeros((self.T, self.P), np.uint8)
        self.wheel = np.zeros(self.P, np.uint8)


class OnlineTypeMix:
    def __init__(self, stage1: str = "cffull", model: str = "typemix_cffullraw", label: str = "membership", seeds=(0, 1, 2),
                 device: str = "cuda", kind: str = "typemix", runs: str = "main", memory: str | None = None) -> None:
        """runs: checkpoint folder under <bench>/runs ("main" = v1, "v2" = socialcost_v2 retraining).
        memory: temporal-memory mode (hold | decay | decay_geo): stage-1 outputs pass through a live BeliefCache with the
        validation-selected configuration of stage1/cffullrawmem<mode>/memory_config.json, and TypeMix gets its staleness
        uncertainty (memory.py). The stage-2 checkpoints must be the matching typemix_cffullrawmem<mode> models."""
        self.dev = torch.device(device)
        self.kind = kind
        self.mem = None
        self.t_frames = 0
        if memory and kind == "typemix":
            from .memory import BeliefCache, MemoryConfig
            cfg = json.load(open(os.path.join(BENCH, "stage1", f"cffullrawmem{memory}", "memory_config.json")))["memory"]
            self.mem = BeliefCache(MemoryConfig(**cfg))
            model = f"typemix_cffullrawmem{memory}"
        if kind == "typemix_gt":
            model = "typemix_gt"
        elif kind == "pos_u":
            model = "pos_u"
        self.s1 = []
        self.s2 = []
        for s in seeds:
            ck = torch.load(os.path.join(BENCH, "runs", runs, f"{model}_{label}_s{s}", "best.pt"), map_location="cpu")
            m = PositionToGrid(cin=3, out_ch=2) if kind == "pos_u" else TypeMixCost(use_semantic_variance=not model.endswith("_nosem"))
            m.load_state_dict(ck["model"])
            self.s2.append(m.to(self.dev).eval())
        from ..labels import GridSpec
        self.grid = GridSpec.centered(10.0, 0.05)
        self.names: list[str] = []
        self.hist = HIST
        self.buf: deque = deque(maxlen=self.hist)
        if kind != "typemix":
            assert self.s2, "missing stage-2 checkpoints"
            return
        s1 = os.path.join(BENCH, "stage1", stage1)
        info = json.load(open(os.path.join(s1, "metrics.json")))["info"]
        a = info["args"]
        self.T = info["temperatures"]
        self.hist = int(a.get("hist", HIST))
        self.buf = deque(maxlen=self.hist)
        for f in sorted(glob.glob(os.path.join(s1, "full_m*.pt"))):
            m = GroupNet(True, d=int(a.get("d", 96)), layers=int(a.get("layers", 3)), dropout=float(a.get("dropout", 0.0)))
            m.load_state_dict(torch.load(f, map_location="cpu"))
            self.s1.append(m.to(self.dev).eval())
        assert self.s1 and self.s2, "missing stage-1 or stage-2 checkpoints"
        _, self.heights = assets.load(ROOT)

    # ------------------------------------------------------------------ per-frame observations
    def _visibility(self, p: dict, Tcm: np.ndarray, K: np.ndarray, depth: np.ndarray, cam_xy) -> float:
        hgt = self.heights.get(_body_id(p.get("model_uri", "")), 1.7)
        uv, z = _project(Tcm, K, body_points(p, cam_xy, hgt))
        _, frac, _ = classify(point_states(depth, uv, z))
        return float(frac)

    def _crop(self, p: dict, Tcm, K, rgb, dep8) -> np.ndarray:
        out = np.zeros((4, CROP2_H, CROP2_W), np.uint8)
        b = person_box(K, Tcm, p["x"], p["y"], height=2.3, r=0.55, W=rgb.shape[1], H=rgb.shape[0])
        if b and b[2] - b[0] >= 2 and b[3] - b[1] >= 4:
            u0, v0, u1, v1 = b
            out[:3] = cv2.resize(rgb[v0:v1, u0:u1], (CROP2_W, CROP2_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            out[3] = cv2.resize(dep8[v0:v1, u0:u1], (CROP2_W, CROP2_H), interpolation=cv2.INTER_NEAREST)
        return out

    def observe(self, states: list[dict], robot_pose: dict, cam_pose: dict, K: np.ndarray, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Append one 5 Hz sample. states: map frame, keys name, x, y, yaw, vx, vy, model_uri. rgb uint8 HxWx3 (RGB);
        depth float32 metres (nan = invalid)."""
        if self.buf:
            rx, ry, _ = self.buf[-1]["robot"]
            if math.hypot(robot_pose["x"] - rx, robot_pose["y"] - ry) > 1.0:  # new episode (scenario switch): drop the old history
                self.buf.clear()
                if self.mem is not None:
                    self.mem.reset()
                    self.t_frames = 0
        for p in states:
            n = p["name"].split("/")[-1]
            if n not in self.names:
                self.names.append(n)
        Tcm = np.linalg.inv(pose_matrix(cam_pose))
        cam_xy = (cam_pose["x"], cam_pose["y"])
        dep8 = (np.clip(np.nan_to_num(depth, nan=0.0), 0, 8) / 8 * 255).astype(np.uint8)
        rec = {"robot": (robot_pose["x"], robot_pose["y"], robot_pose["yaw"]), "people": {}}
        for p in states:
            n = p["name"].split("/")[-1]
            v = self._visibility(p, Tcm, K, depth, cam_xy)
            crop = self._crop(p, Tcm, K, rgb, dep8) if v > 0.05 else np.zeros((4, CROP2_H, CROP2_W), np.uint8)
            rec["people"][n] = ((p["x"], p["y"], p["yaw"], p["vx"], p["vy"]), v, crop)
        self.buf.append(rec)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, states: list[dict], robot_pose: dict, frame: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """(mu, sigma) on the robot-centred label grid in cost units 0..1, plus stage-1 probabilities for logging.
        frame: live InteractionFrame dict (typemix_gt only)."""
        if self.kind == "pos_u":
            return self._stage2({"raster": torch.from_numpy(raster_peds(states, robot_pose, self.grid)).unsqueeze(0)}) + ({},)
        if self.kind == "typemix_gt":
            ep_, _, np_, _ = gt_soft_inputs(states, frame)
            names = [p["name"].split("/")[-1] for p in states][:P_MAX_PEOPLE]
            return self._typemix(states, robot_pose, {"edge_p": ep_[:len(names), :len(names)], "node_p": np_[:len(names)], "names": names})
        rows = list(self.buf)
        T = len(rows)
        names = [p["name"].split("/")[-1] for p in states][:P_MAX_PEOPLE]
        P = len(names)
        dbg: dict = {}
        if P:
            track = np.zeros((T, P, 5), np.float32)
            vis = np.zeros((T, P), np.float32)
            crop = np.zeros((T, P, 4, CROP2_H, CROP2_W), np.uint8)
            robot = np.zeros((T, 3), np.float32)
            for t, r in enumerate(rows):
                robot[t] = r["robot"]
                for j, n in enumerate(names):
                    if n in r["people"]:
                        tr, v, c = r["people"][n]
                        track[t, j], vis[t, j], crop[t, j] = tr, v, c
            b = sample_tensors2(_Buffer(names, track, robot, vis, crop), T - 1, self.hist, True)
            b = {k: v.unsqueeze(0).to(self.dev) for k, v in b.items()}
            Ts, Tt, Tn = self.T["same"], self.T["type"], self.T["node"]
            edge = np.zeros((P, P, 5))
            node = np.zeros((P, 2))
            for m in self.s1:
                s, ty, nd = (x.float()[0].cpu().numpy() for x in m(b))
                ps = 1 / (1 + np.exp(-s[:P, :P] / Ts))
                z = ty[:P, :P] / Tt
                q = np.exp(z - z.max(-1, keepdims=True))
                q /= q.sum(-1, keepdims=True)
                edge += np.concatenate([(1 - ps)[..., None], ps[..., None] * q], -1)
                node += 1 / (1 + np.exp(-nd[:P] / Tn))
            edge /= len(self.s1)
            node /= len(self.s1)
            edge[np.arange(P), np.arange(P)] = (1, 0, 0, 0, 0)
            dbg = {"edge_p": edge, "node_p": node, "names": names}
            if self.mem is not None:  # causal memory over the stage-1 outputs, visibility from the current frame
                last = rows[-1]["people"]
                xy = np.array([last[n][0][:2] if n in last else (np.nan, np.nan) for n in names], np.float64)
                vis_now = np.array([last[n][1] if n in last else 0.0 for n in names], np.float64)
                r = self.mem.step(self.t_frames * 0.2, names, xy, vis_now, edge, node)
                self.t_frames += 1
                dbg.update({"edge_p": r["edge_p"], "node_p": r["node_p"], "edge_u": r["u_edge"], "node_u": r["u_node"],
                            "stale_edge": r["stale_edge"], "stage1_edge_p": edge, "stage1_node_p": node})
        return self._typemix(states, robot_pose, dbg)

    def _typemix(self, states, robot_pose, dbg):
        # stage 2 inputs exactly as Dv1Samples.inputs (typemix)
        pose = robot_pose
        P = min(len(states), P_MAX_PEOPLE)
        c, s_ = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        ppl = np.zeros((P_MAX_PEOPLE, 6), np.float32)
        for a_, p in enumerate(states[:P_MAX_PEOPLE]):
            dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
            yaw = p["yaw"] - pose["yaw"]
            ppl[a_] = (c * dx + s_ * dy, -s_ * dx + c * dy, math.cos(yaw), math.sin(yaw), c * p["vx"] + s_ * p["vy"], -s_ * p["vx"] + c * p["vy"])
        epp = np.zeros((P_MAX_PEOPLE, P_MAX_PEOPLE, 5), np.float32)
        epp[..., 0] = 1.0
        npp = np.zeros((P_MAX_PEOPLE, 2), np.float32)
        msk = np.zeros(P_MAX_PEOPLE, bool)
        if P:
            epp[:P, :P] = dbg["edge_p"]
            npp[:P] = dbg["node_p"]
            msk[:P] = True
        x = {"raster": torch.from_numpy(raster_peds(states, pose, self.grid)).unsqueeze(0), "people": torch.from_numpy(ppl).unsqueeze(0),
             "pmask": torch.from_numpy(msk).unsqueeze(0), "edge_p": torch.from_numpy(epp).unsqueeze(0), "node_p": torch.from_numpy(npp).unsqueeze(0)}
        if self.mem is not None:
            eup = np.zeros((P_MAX_PEOPLE, P_MAX_PEOPLE), np.float32)
            nup = np.zeros((P_MAX_PEOPLE, 2), np.float32)
            if P and "edge_u" in dbg:
                eup[:P, :P] = dbg["edge_u"][:P, :P]
                nup[:P] = dbg["node_u"][:P]
            x.update({"edge_u": torch.from_numpy(eup).unsqueeze(0), "node_u": torch.from_numpy(nup).unsqueeze(0)})
        return self._stage2(x) + (dbg,)

    def _stage2(self, x: dict) -> tuple[np.ndarray, np.ndarray]:
        x = {k: v.to(self.dev) for k, v in x.items()}
        mus, vars_ = [], []
        for m in self.s2:
            with torch.autocast("cuda", dtype=torch.float16):
                o = m(x)
            o = o.float()[0]
            mus.append(torch.sigmoid(o[0]))
            vars_.append(o[1].clamp(LOGVAR_MIN, LOGVAR_MAX).exp())
        M = torch.stack(mus)
        mu = M.mean(0)
        var = torch.stack(vars_).mean(0) + (M.var(0, unbiased=False) if len(mus) > 1 else 0)
        return mu.cpu().numpy(), var.sqrt().cpu().numpy()

    def step(self, states, robot_pose, cam_pose, K, rgb, depth, frame=None):
        if self.kind == "typemix":
            self.observe(states, robot_pose, cam_pose, K, rgb, depth)
        return self.predict(states, robot_pose, frame)


__all__ = ["OnlineTypeMix", "SCALE"]
