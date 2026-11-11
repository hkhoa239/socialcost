"""dataset_v1 samples for the costmap prediction benchmark.

Splits come from dataset_v1/splits (train / val / test_a / test_b / test_c) and Test D from
splits/test_d_counterfactual.json. Targets are `social_cost` (arena-native/1.1) or `social_cost_membership`
(extended-membership/1.1), normalised by 252. Model inputs per variant:

  rgb      RGB image + calibration (ground-plane lifting)       sensor-observable
  depth    depth image + calibration                            sensor-observable
  rgbd     RGB + depth + calibration                            sensor-observable
  pos      rasterised positions and headings of every person    privileged (simulator tracks, no interaction)
  pos_int  pos + privileged interaction metadata (oracle-assisted diagnostic):
           membership-completed interaction segments per class, halt gesture, mobility aid
Evaluation-only maps (never inputs): personal-space component (the "rule" map), interaction region, FOV.
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..dataset import interaction_frame_at, read_jsonl, with_membership
from ..labels import GridSpec
from ..learn.data import fov_mask, lift_indices, raster_peds
from ..qa import pose_matrix

ROOT = os.environ.get("SC_DV1_ROOT", "/home/robotics/arena_ws/data/socialcost/dataset_v1")
LABELS = {"native": "social_cost", "membership": "social_cost_membership", "extended": "social_cost_extended"}  # extended/2.0: + halt zone, mobility aid
# socialcost_v2 labels (socialcost/labels_v2.py, scripts/relabel_v2.py): same directory names in a separate tree
LABELS.update({f"{k}_v2": v for k, v in list(LABELS.items())})
V2_ROOT = os.environ.get("SC_LABELS_V2_ROOT", "/home/robotics/arena_ws/data/socialcost/dataset_v1_labels_v2")


def label_path(eid: str, label: str, k: int, root: str = ROOT) -> str:
    """Target file of sample k for a label name (v1 in dataset_v1, *_v2 in the v2 tree)."""
    base = V2_ROOT if label.endswith("_v2") else root
    return os.path.join(base, "episodes", eid, LABELS[label], f"{k:06d}.npy")


def components_path(eid: str, label: str, k: int, root: str = ROOT) -> str:
    base = V2_ROOT if label.endswith("_v2") else root
    return os.path.join(base, "episodes", eid, "components", f"{k:06d}.npz")


def label_base(label: str) -> str:
    return label[:-3] if label.endswith("_v2") else label
SCALE = 252.0
SPLITS = ("train", "val", "test_a", "test_b", "test_c")
# pos_int interaction channels (membership-completed InteractionFrame types)
INT_CLASSES = {"talk": (0, 1), "queue": (6,), "group_walk": (12,), "other": (3, 4, 5, 7, 8, 9, 10, 11)}
POS_INT_CHANNELS = 3 + len(INT_CLASSES) + 2
# cascade / probabilistic variants: 3 position + 4 class-probability segments + halt p + wheelchair p + edge MI + node MI
CASCADE_CHANNELS = 3 + 4 + 2 + 2
P_MAX_PEOPLE = 10
STAGE1_ROOT = os.environ.get("SC_STAGE1_ROOT", "/home/robotics/arena_ws/data/costmap_bench/stage1")


def split_ids(split: str, root: str = ROOT) -> list[str]:
    return [line.strip() for line in open(os.path.join(root, "splits", f"{split}.txt")) if line.strip()]


def test_d_pairs(root: str = ROOT) -> list[dict]:
    return json.load(open(os.path.join(root, "splits", "test_d_counterfactual.json")))["pairs"]


class Episode:
    """Static per-episode data, loaded once."""

    def __init__(self, eid: str, root: str = ROOT) -> None:
        self.id = eid
        self.dir = os.path.join(root, "episodes", eid)
        self.meta = json.load(open(os.path.join(self.dir, "episode_metadata.json")))
        self.cam = json.load(open(os.path.join(self.dir, "camera_info.json")))
        self.robots = read_jsonl(os.path.join(self.dir, "robot_states.jsonl"))
        self.peds = read_jsonl(os.path.join(self.dir, "pedestrian_states.jsonl"))
        self.frames = read_jsonl(os.path.join(self.dir, "interactions.jsonl"))
        self.K = np.array(self.cam["K"], dtype=np.float64).reshape(3, 3)
        self.T_bc = pose_matrix(self.cam["T_base_camera"])
        self.grid = GridSpec(**self.meta["label"]["grid"])
        dv = self.meta.get("dv1", {})
        self.family, self.condition, self.pair_id, self.split = dv.get("family"), dv.get("condition"), dv.get("pair_id"), dv.get("split")
        self.n = len(self.robots)


def interaction_raster(states: list[dict], frame: dict | None, pose: dict, grid: GridSpec, width: float = 0.25) -> np.ndarray:
    """Privileged interaction metadata in the label grid: one channel per interaction class with the segments
    between every pair of members of an active interaction (membership-completed), a halt-gesture channel
    and a mobility-aid channel (heat at the person)."""
    full = with_membership(states, frame)
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    x = grid.origin_x + (i + 0.5) * grid.resolution
    y = grid.origin_y + (j + 0.5) * grid.resolution
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])

    def local(p):
        dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
        return c * dx + s * dy, -s * dx + c * dy

    out = np.zeros((len(INT_CLASSES) + 2, grid.height, grid.width), np.float32)
    groups: dict[int, list] = {}
    for p in full:
        if p["interaction_id"] >= 0:
            groups.setdefault(p["interaction_id"], []).append(p)
    for members in groups.values():
        t = members[0]["interaction_type"]
        ch = next((k for k, (name, types) in enumerate(INT_CLASSES.items()) if t in types), None)
        if ch is None or len(members) < 2:
            continue
        pts = [local(m) for m in members]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                (ax, ay), (bx, by) = pts[a], pts[b]
                vx, vy = bx - ax, by - ay
                L2 = vx * vx + vy * vy or 1e-9
                t_ = np.clip(((x - ax) * vx + (y - ay) * vy) / L2, 0, 1)
                d2 = (x - ax - t_ * vx) ** 2 + (y - ay - t_ * vy) ** 2
                out[ch] = np.maximum(out[ch], np.exp(-d2 / (2 * width ** 2)))
    for p in full:
        lx, ly = local(p)
        h = np.exp(-((x - lx) ** 2 + (y - ly) ** 2) / (2 * 0.2 ** 2)).astype(np.float32)
        if any(isinstance(g, dict) and str(g.get("slot", "")).startswith("halt") for g in p.get("gestures") or []):
            out[-2] = np.maximum(out[-2], h)
        if "wheelchair" in os.path.basename(str(p.get("model_uri", ""))).lower():
            out[-1] = np.maximum(out[-1], h)
    return out


def soft_interaction_raster(states: list[dict], pose: dict, grid: GridSpec, edge_p: np.ndarray, edge_mi: np.ndarray, node_p: np.ndarray,
                            node_mi: np.ndarray, width: float = 0.25) -> np.ndarray:
    """Probabilistic interaction channels (8 x H x W) in the label grid, shared by the cascades and the ground-truth bound:
    per class c in {talk, queue, group_walk, other}: max over pairs of p_ij(c) * segment(i, j); halt p and mobility-aid p
    as person heat; epistemic channels: max over pairs of MI_ij * segment(i, j) and max over people of max(MI_halt, MI_wheel) * heat.
    edge_p [P, P, 5] (class 0 = none), edge_mi [P, P], node_p [P, 2], node_mi [P, 2], people in `states` order."""
    i, j = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    x = grid.origin_x + (i + 0.5) * grid.resolution
    y = grid.origin_y + (j + 0.5) * grid.resolution
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    pts = []
    for p in states:
        dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
        pts.append((c * dx + s * dy, -s * dx + c * dy))
    out = np.zeros((8, grid.height, grid.width), np.float32)
    P = len(pts)
    for a in range(P):
        for b in range(a + 1, P):
            pc = edge_p[a, b, 1:]
            if pc.max() < 0.02 and edge_mi[a, b] < 0.02:
                continue
            (ax, ay), (bx, by) = pts[a], pts[b]
            if max(abs(ax), abs(ay), abs(bx), abs(by)) > 9:
                continue
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy or 1e-9
            t_ = np.clip(((x - ax) * vx + (y - ay) * vy) / L2, 0, 1)
            seg = np.exp(-((x - ax - t_ * vx) ** 2 + (y - ay - t_ * vy) ** 2) / (2 * width ** 2)).astype(np.float32)
            for k in range(4):
                if pc[k] > 0.02:
                    out[k] = np.maximum(out[k], pc[k] * seg)
            out[6] = np.maximum(out[6], float(edge_mi[a, b]) * seg)
    for a, (lx, ly) in enumerate(pts):
        if max(abs(lx), abs(ly)) > 6:
            continue
        h = np.exp(-((x - lx) ** 2 + (y - ly) ** 2) / (2 * 0.2 ** 2)).astype(np.float32)
        out[4] = np.maximum(out[4], float(node_p[a, 0]) * h)
        out[5] = np.maximum(out[5], float(node_p[a, 1]) * h)
        out[7] = np.maximum(out[7], float(max(node_mi[a, 0], node_mi[a, 1])) * h)
    return out


def gt_soft_inputs(states: list[dict], frame: dict | None) -> tuple:
    """Ground-truth (oracle) interaction state in the soft format: one-hot classes, zero uncertainty."""
    full = with_membership(states, frame)
    P = len(full)
    ep = np.zeros((P, P, 5), np.float32)
    ep[..., 0] = 1.0
    cls = {0: 1, 1: 1, 6: 2, 12: 3}
    for a in range(P):
        for b in range(P):
            if a != b and full[a]["interaction_id"] >= 0 and full[a]["interaction_id"] == full[b]["interaction_id"]:
                ep[a, b] = 0
                ep[a, b, cls.get(full[a]["interaction_type"], 4)] = 1.0
    npp = np.zeros((P, 2), np.float32)
    for a, p in enumerate(full):
        npp[a, 0] = any(isinstance(g, dict) and str(g.get("slot", "")).startswith("halt") for g in p.get("gestures") or [])
        npp[a, 1] = "wheelchair" in os.path.basename(str(p.get("model_uri", ""))).lower()
    return ep, np.zeros((P, P), np.float32), npp, np.zeros((P, 2), np.float32)


class Dv1Samples(Dataset):
    """(episode, sample) items. variant: rgb | depth | rgbd | pos | pos_int | graph (SNGNN2D adapter) | none."""

    def __init__(self, ids: list[str], label: str, variant: str, stride: int = 1, root: str = ROOT, eval_maps: bool = False,
                 graph_builder=None) -> None:
        self.eps = [Episode(e, root) for e in ids if os.path.exists(os.path.join(root, "episodes", e, "DONE"))]
        self.label = LABELS[label]
        self.label_name = label
        self.variant = variant
        self.eval_maps = eval_maps
        self.graph_builder = graph_builder
        self.items = [(ei, k) for ei, e in enumerate(self.eps) for k in range(0, e.n, stride)]
        e0 = self.eps[0]
        self.fov = torch.from_numpy(fov_mask(e0.grid, e0.K, e0.T_bc))

    def __len__(self) -> int:
        return len(self.items)

    def stage1(self, eid: str, var: str | None = None) -> dict:
        """Stage-1 predictions of an episode (out-of-fold for train episodes), loaded once per worker."""
        if not hasattr(self, "_s1"):
            self._s1 = {}
        if eid not in self._s1:
            var = var or self.variant.split("_", 1)[1]
            f = os.path.join(STAGE1_ROOT, var, "preds", f"{eid}.npz")
            if not os.path.exists(f):  # episodes without people (empty controls) have no stage-1 output
                T = len(self.eps[[e.id for e in self.eps].index(eid)].robots) if eid in [e.id for e in self.eps] else 200
                self._s1[eid] = {"edge_p": np.zeros((T, 0, 0, 5), np.float16), "edge_mi": np.zeros((T, 0, 0), np.float16),
                                 "edge_entropy": np.zeros((T, 0, 0), np.float16), "node_p": np.zeros((T, 0, 2), np.float16),
                                 "node_mi": np.zeros((T, 0, 2), np.float16), "index": {}}
                return self._s1[eid]
            z = np.load(f)
            names = list(np.load(os.path.join(os.path.dirname(STAGE1_ROOT), "cascade_cache", f"{eid}.npz"))["names"])
            self._s1[eid] = {k: z[k] for k in z.files} | {"index": {n: i for i, n in enumerate(names)}}
        return self._s1[eid]

    def inputs(self, e: Episode, k: int) -> dict:
        name = f"{k:06d}"
        pose = e.robots[k]["pose_map_base"]
        states = e.peds[k]["pedestrians"]
        x: dict = {}
        if self.variant in ("rgbd", "depth", "rgb"):
            rgb = cv2.imread(os.path.join(e.dir, "rgb", name + ".png"))[:, :, ::-1].astype(np.float32) / 255.0
            depth = np.load(os.path.join(e.dir, "depth", name + ".npy"))
            valid = np.isfinite(depth).astype(np.float32)
            d = np.nan_to_num(depth, nan=0.0) / 10.0
            chans = []
            if self.variant in ("rgbd", "rgb"):
                chans.append(rgb.transpose(2, 0, 1) - 0.5)
            if self.variant in ("rgbd", "depth"):
                chans += [d[None], valid[None]]
            x["image"] = torch.from_numpy(np.concatenate(chans, 0).astype(np.float32))
            x["lift"] = torch.from_numpy(lift_indices(depth, e.K, e.T_bc, mode="ground" if self.variant == "rgb" else "depth"))
        elif self.variant in ("pos", "pos_int"):
            r = raster_peds(states, pose, e.grid)
            if self.variant == "pos_int":
                fr = interaction_frame_at(e.frames, e.robots[k]["stamp_ns"])
                r = np.concatenate([r, interaction_raster(states, fr, pose, e.grid)], 0)
            x["raster"] = torch.from_numpy(r)
        elif self.variant in ("pos_int_u", "cascade_tracks", "cascade_vis"):
            r = raster_peds(states, pose, e.grid)
            if self.variant == "pos_int_u":
                soft = gt_soft_inputs(states, interaction_frame_at(e.frames, e.robots[k]["stamp_ns"]))
            else:
                pr = self.stage1(e.id)
                order = [pr["index"][p["name"].split("/")[-1]] for p in states]
                ix = np.ix_(order, order)
                soft = (pr["edge_p"][k][ix].astype(np.float32), pr["edge_mi"][k][ix].astype(np.float32),
                        pr["node_p"][k][order].astype(np.float32), pr["node_mi"][k][order].astype(np.float32))
            x["raster"] = torch.from_numpy(np.concatenate([r, soft_interaction_raster(states, pose, e.grid, *soft)], 0))
        elif self.variant.startswith("typemix"):
            # TypeMix: positions raster + padded per-person kinematics and type probabilities (stage 1 or ground truth)
            x["raster"] = torch.from_numpy(raster_peds(states, pose, e.grid))
            P = len(states)
            src = self.variant.split("_")[1]  # tracks | vis | gt
            if src == "gt":
                ep_, _, np_, _ = gt_soft_inputs(states, interaction_frame_at(e.frames, e.robots[k]["stamp_ns"]))
            else:
                pr = self.stage1(e.id, src)
                order = [pr["index"][p["name"].split("/")[-1]] for p in states]
                ep_ = pr["edge_p"][k][np.ix_(order, order)].astype(np.float32)
                np_ = pr["node_p"][k][order].astype(np.float32)
                if "edge_u" in pr:  # temporal memory sources (memory.py): staleness uncertainty
                    eu_ = pr["edge_u"][k][np.ix_(order, order)].astype(np.float32)
                    nu_ = pr["node_u"][k][order].astype(np.float32)
            c, s_ = math.cos(pose["yaw"]), math.sin(pose["yaw"])
            ppl = np.zeros((P_MAX_PEOPLE, 6), np.float32)
            for a, p in enumerate(states[:P_MAX_PEOPLE]):
                dx, dy = p["x"] - pose["x"], p["y"] - pose["y"]
                yaw = p["yaw"] - pose["yaw"]
                ppl[a] = (c * dx + s_ * dy, -s_ * dx + c * dy, math.cos(yaw), math.sin(yaw), c * p["vx"] + s_ * p["vy"], -s_ * p["vx"] + c * p["vy"])
            n = min(P, P_MAX_PEOPLE)
            epp = np.zeros((P_MAX_PEOPLE, P_MAX_PEOPLE, 5), np.float32)
            epp[..., 0] = 1.0
            epp[:n, :n] = ep_[:n, :n]
            npp = np.zeros((P_MAX_PEOPLE, 2), np.float32)
            npp[:n] = np_[:n]
            msk = np.zeros(P_MAX_PEOPLE, bool)
            msk[:n] = True
            x.update({"people": torch.from_numpy(ppl), "pmask": torch.from_numpy(msk), "edge_p": torch.from_numpy(epp), "node_p": torch.from_numpy(npp)})
            if "mem" in src:  # every sample of a memory source carries u (zeros for episodes without stage-1 output)
                eup = np.zeros((P_MAX_PEOPLE, P_MAX_PEOPLE), np.float32)
                nup = np.zeros((P_MAX_PEOPLE, 2), np.float32)
                if "edge_u" in self.stage1(e.id, src):
                    eup[:n, :n] = eu_[:n, :n]
                    nup[:n] = nu_[:n]
                x.update({"edge_u": torch.from_numpy(eup), "node_u": torch.from_numpy(nup)})
        elif self.variant == "graph":
            x["graph"] = self.graph_builder(e, k)
        return x

    def __getitem__(self, i: int) -> dict:
        ei, k = self.items[i]
        e = self.eps[ei]
        name = f"{k:06d}"
        target = np.load(label_path(e.id, self.label_name, k, os.path.dirname(os.path.dirname(e.dir)))).astype(np.float32) / SCALE
        comp = np.load(components_path(e.id, self.label_name, k, os.path.dirname(os.path.dirname(e.dir))))
        rule = comp["personal_space"].astype(np.float32) / SCALE
        out = {"x": self.inputs(e, k), "target": torch.from_numpy(target), "rule": torch.from_numpy(rule), "fov": self.fov,
               "meta": {"episode": e.id, "sample": k, "slot": e.robots[k]["slot"], "family": e.family, "condition": e.condition,
                        "pair_id": e.pair_id or ""}}
        if self.eval_maps:
            inter = comp["conversation"] if label_base(self.label_name) == "native" else np.maximum(comp["conversation"], comp["queue"])
            if label_base(self.label_name) == "extended":
                inter = np.maximum(inter, comp["gesture"])
            out["inter"] = torch.from_numpy(inter > 0)
        return out


def collate(batch: list[dict]) -> dict:
    keys = batch[0]["x"].keys()
    x = {}
    for k in keys:
        if k == "graph":
            x[k] = [b["x"][k] for b in batch]
        else:
            x[k] = torch.stack([b["x"][k] for b in batch])
    out = {"x": x, "target": torch.stack([b["target"] for b in batch]), "rule": torch.stack([b["rule"] for b in batch]),
           "fov": torch.stack([b["fov"] for b in batch]), "meta": [b["meta"] for b in batch]}
    if "inter" in batch[0]:
        out["inter"] = torch.stack([b["inter"] for b in batch])
    return out
