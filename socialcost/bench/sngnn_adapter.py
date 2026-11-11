"""Adapter from SOCIALCOST dataset_v1 samples to the official SNGNN2D-v2 implementation (external/SNGNN2D-v2, untouched).

Input side: each sample becomes SNGNN2D's raw JSON sequence (3 frames, 1 s apart: t, t-1 s, t-2 s), each frame in the
robot frame of its own instant, exactly as SocNavGym logs it, then the official graph builder
(`SocNavDataset.generate_final_graph`, alternative 8 = robot/room, people, objects, walls, goal, interaction links,
plus the 19 x 19 grid graph) produces the DGL graph.

  people   pedestrian_states (position, heading, velocity, angular velocity) - privileged simulator tracks
  objects  scenario objects (service desks, occluders) and world furniture within 7 m (asset bounding boxes)
  walls    wall polylines of the world map (contours of map/0.png), split as the original into <= 2.5 m pieces
  goal     the robot's navigation goal; command: the robot's executed [v, 0, w]
  interaction (privileged mode only): p_p links between members of every active interaction (membership-completed
           InteractionFrame), p_o links from queue members to their desk. Sensor mode: no links.

Modifications to the original (all outside external/):
  * grid spacing `area_width` 700 mm -> 10000/19 mm, so the 19 x 19 grid spans exactly the 10 m SOCIALCOST label;
  * the 256 x 256 tanh output is resized (bilinear) to the 200 x 200 label grid; targets use the original
    [-1, 1] scaling (cost / 252 * 2 - 1);
  * nothing else: GAT sizes, ResNet generator, MSE loss and Adam settings come from config/parameters.yml.
"""

from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

EXT = os.environ.get("SNGNN2D_DIR", "/home/robotics/arena_ws/external/SNGNN2D-v2")
for sub in ("dataset", "nets", "utils"):
    p = os.path.join(EXT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
import alternatives_utils as _au  # noqa: E402

AREA_WIDTH_MM = 10000.0 / 19.0
_au.area_width = AREA_WIDTH_MM  # before `alternatives` / `socnav2d_dataset` star-import it
import alternatives as _alt  # noqa: E402
import dgl  # noqa: E402
import socnav2d_dataset as _sd  # noqa: E402

from ..dataset import interaction_frame_at, with_membership  # noqa: E402
from ..dv1.geometry import WORLDS_DIR, World, map_offset, object_footprint  # noqa: E402
from .data import ROOT, Episode  # noqa: E402

assert _sd.area_width == AREA_WIDTH_MM and _alt.area_width == AREA_WIDTH_MM, "grid spacing patch did not apply"
FRAMES_BACK = (0, 5, 10)  # samples at 5 Hz: t, t-1 s, t-2 s
OBJ_RANGE = 7.0
WALL_RANGE = 7.0


def official_params() -> dict:
    return yaml.safe_load(open(os.path.join(EXT, "config", "parameters.yml")))["hyperparameters"]


class _Builder(_sd.SocNavDataset):
    """Calls the official generate_final_graph without the file-based DGLDataset machinery."""

    def __init__(self, edge_features: bool = False) -> None:  # noqa: D401 - deliberately not calling super().__init__
        self.alt = "8"
        self.net = "gat"
        self.edge_features = edge_features
        self.dataloader = _alt.initializeAlt8
        self.device = "cpu"
        self.data = {"typemaps": [], "coordinates": [], "identifiers": [], "wall_segments": []}

    def graph(self, raw: list[dict]):
        g = self.generate_final_graph(raw)
        for k in self.data:
            self.data[k].clear()
        return g


def world_walls(world: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Wall segments (world frame) from the map image: polygonal contours of occupied pixels."""
    m = yaml.safe_load(open(os.path.join(WORLDS_DIR, world, "map", "0.yaml")))
    img = cv2.imread(os.path.join(WORLDS_DIR, world, "map", m["image"]), cv2.IMREAD_GRAYSCALE)
    occ = (img < 128).astype(np.uint8)
    cnts, _ = cv2.findContours(occ, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H = img.shape[0]
    res, ox, oy = m["resolution"], m["origin"][0], m["origin"][1]
    segs = []
    for c in cnts:
        c = cv2.approxPolyDP(c, 2.0, True)[:, 0, :]
        pts = [(ox + (u + 0.5) * res, oy + (H - 1 - v + 0.5) * res) for u, v in c]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            if math.dist(a, b) > 0.15:
                segs.append((a, b))
    return segs


class GraphBuilder:
    """Callable (episode, sample) -> DGL graph, for one input mode: 'privileged' or 'sensor'."""

    def __init__(self, mode: str, root: str = ROOT) -> None:
        assert mode in ("privileged", "sensor")
        self.mode = mode
        self.b = _Builder()
        man = json.load(open(os.path.join(root, "manifests", "manifest_v1.json")))["episodes"]
        self.man = {e["episode_id"]: e for e in man}
        self._walls: dict[str, list] = {}
        self._worlds: dict[str, World] = {}

    def walls(self, world: str):
        if world not in self._walls:
            self._walls[world] = world_walls(world)
        return self._walls[world]

    def furniture(self, world: str) -> list[dict]:
        if world not in self._worlds:
            self._worlds[world] = World.load(world)
        return [o for o in self._worlds[world].objects if o["footprint"]]

    def __call__(self, e: Episode, k: int):
        entry = self.man[e.id]
        world = entry["world"]
        ox, oy = map_offset(world)
        # world-frame objects: scenario objects + furniture, with footprint sizes
        objs = []
        for o in entry["objects"]:
            fp = object_footprint(o["model"])
            if fp:
                objs.append({"x": o["pose"][0] + ox, "y": o["pose"][1] + oy, "a": o["pose"][2] if len(o["pose"]) > 2 else 0.0,
                             "sx": fp[1] - fp[0], "sy": fp[3] - fp[2], "desk": o.get("humansim", {}).get("type") == "desk"})
        for o in self.furniture(world):
            fp = o["footprint"]
            objs.append({"x": o["pose"][0] + ox, "y": o["pose"][1] + oy, "a": o["pose"][2], "sx": fp[1] - fp[0], "sy": fp[3] - fp[2], "desk": False})
        walls_w = [((a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy)) for a, b in self.walls(world)]
        goal_w = (entry["robot"]["goal"][0] + ox, entry["robot"]["goal"][1] + oy)
        r0 = e.robots[k]["pose_map_base"]
        near_objs = [o for o in objs if math.dist((o["x"], o["y"]), (r0["x"], r0["y"])) <= OBJ_RANGE]
        near_walls = [s for s in walls_w if min(math.dist(s[0], (r0["x"], r0["y"])), math.dist(s[1], (r0["x"], r0["y"])),
                                                 math.dist(((s[0][0] + s[1][0]) / 2, (s[0][1] + s[1][1]) / 2), (r0["x"], r0["y"]))) <= WALL_RANGE]
        names = [p["name"] for p in e.peds[k]["pedestrians"]]
        n_obj = len(near_objs)
        pid = {n: 1 + n_obj + i for i, n in enumerate(names)}
        frames = []
        for back in FRAMES_BACK:
            kk = max(0, k - back)
            rp = e.robots[kk]["pose_map_base"]
            c, s = math.cos(rp["yaw"]), math.sin(rp["yaw"])

            def loc(x, y):
                dx, dy = x - rp["x"], y - rp["y"]
                return c * dx + s * dy, -s * dx + c * dy

            people = []
            for p in e.peds[kk]["pedestrians"]:
                if p["name"] not in pid:
                    continue
                lx, ly = loc(p["x"], p["y"])
                vx, vy = c * p["vx"] + s * p["vy"], -s * p["vx"] + c * p["vy"]
                people.append({"id": pid[p["name"]], "x": lx, "y": ly, "a": p["yaw"] - rp["yaw"], "vx": vx, "vy": vy, "va": p.get("wz", 0.0)})
            objects = []
            for i, o in enumerate(near_objs):
                lx, ly = loc(o["x"], o["y"])
                objects.append({"id": 1 + i, "x": lx, "y": ly, "a": o["a"] - rp["yaw"], "vx": 0.0, "vy": 0.0, "va": 0.0, "size_x": o["sx"], "size_y": o["sy"]})
            walls = []
            for a, b in near_walls:
                (x1, y1), (x2, y2) = loc(*a), loc(*b)
                walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            gx, gy = loc(*goal_w)
            tw = e.robots[kk]["odom_twist"]
            frames.append({"ID": f"{e.id}_{k}", "timestamp": e.robots[kk]["stamp_ns"] / 1e9 - (0.0 if kk != 0 or back == 0 else back * 0.2),
                           "command": [tw.get("vx", 0.0), 0.0, tw.get("wz", 0.0)], "people": people, "objects": objects, "walls": walls,
                           "goal": [{"x": gx, "y": gy}], "interaction": []})
        if self.mode == "privileged":
            st = e.peds[k]["pedestrians"]
            full = with_membership(st, interaction_frame_at(e.frames, e.robots[k]["stamp_ns"]))
            groups: dict[int, list] = {}
            for p in full:
                if p["interaction_id"] >= 0 and p["name"] in pid:
                    groups.setdefault(p["interaction_id"], []).append(p)
            links = []
            desks = [1 + i for i, o in enumerate(near_objs) if o["desk"]]
            for members in groups.values():
                ids = [pid[m["name"]] for m in members]
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        links.append({"src": ids[a], "dst": ids[b], "relation": "interacting"})
                if members[0]["interaction_type"] == 6 and desks:
                    links += [{"src": i, "dst": desks[0], "relation": "interacting"} for i in ids]
            frames[0]["interaction"] = links
            for f in frames[1:]:
                f["interaction"] = links  # the builder reads the links of the last frame it visits
        # timestamps must decrease by >= 1 s between the three frames (the builder skips closer frames)
        for i in range(1, len(frames)):
            if frames[i - 1]["timestamp"] - frames[i]["timestamp"] < 1.0:
                frames[i]["timestamp"] = frames[i - 1]["timestamp"] - 1.0
        return self.b.graph(frames)


class SNGNNCostmap(torch.nn.Module):
    """Official SELECT_GNN (GAT + ResnetGenerator) with the output resized to the 200 x 200 label grid."""

    def __init__(self, num_features: int, num_rels: int, params: dict | None = None) -> None:
        super().__init__()
        from select_gnn import SELECT_GNN

        p = params or official_params()
        g0 = dgl.graph(([0], [0]))
        self.net = SELECT_GNN(num_features=num_features, num_edge_feats=None, n_classes=1, num_hidden=p["num_hidden"],
                              gnn_layers=len(p["num_hidden"]) + 1, cnn_layers=3, dropout=p["in_drop"], activation=p["nonlinearity"],
                              final_activation=p["final_activation"], activation_cnn=p["nonlinearity_cnn"],
                              final_activation_cnn=p["final_activation_cnn"], num_channels=p["num_channels"], gnn_type=p["gnn_net"],
                              num_heads=p["heads"], num_rels=num_rels, num_bases=num_rels, g=g0, residual=p["residual"],
                              aggregator_type="mean", attn_drop=p["attn_drop"], alpha=p["alpha"])

    def forward(self, x: dict) -> torch.Tensor:
        g = x["graph"]
        self.net.gnn_object.g = g
        self.net.g = g
        for layer in self.net.gnn_object.layers:
            layer.g = g
        out = self.net(g.ndata["h"].float(), g, None)  # [B, 1, 256, 256] in [-1, 1]
        if out.dim() == 3:
            out = out.unsqueeze(0)
        out = F.interpolate(out, size=(200, 200), mode="bilinear", align_corners=False)
        return out.squeeze(1)  # [-1, 1]; cost = (out + 1) / 2


def batch_graphs(graphs: list, device) -> "dgl.DGLGraph":
    g = dgl.batch(graphs).to(device)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    return g


def num_rels() -> int:
    _, n = _sd.get_relations("8")
    return n + (_sd.N_INTERVALS - 1) * 2
