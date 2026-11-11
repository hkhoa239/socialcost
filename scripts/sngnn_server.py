#!/usr/bin/env python3
"""SNGNN2D-v2 closed-loop costmap service (host side, costmap_bench venv: torch 2.1 + DGL 2.0, which the Arena container's
torch 2.14 cannot load). The estimator (method sngnn) streams JSON lines over TCP 127.0.0.1:<port>:

  {"cmd": "scene", "world": str, "objects": [...scenario objects...], "goal": [x, y] (scenario/world frame)}  -> {"ok": true}
  {"cmd": "step", "stamp_ns": int, "robot": {"x","y","yaw"} (map), "twist": {"vx","wz"}, "peds": [pedestrian states]}
        -> {"cost": base64 uint8 200 x 200 (robot-centred label grid, row j = y), "ms": float}

Graphs are built by the benchmark's own GraphBuilder (sensor mode, socialcost/bench/sngnn_adapter.py) on a live episode
object with the same fields as dataset_v1 (robots[k].pose_map_base / odom_twist / stamp_ns, peds[k].pedestrians), so
the closed-loop input is exactly the training input: frames t, t-1 s, t-2 s at 5 Hz. Output = mean over the seeds of
(tanh + 1) / 2, x 252.

    $PY sngnn_server.py --label extended_v2 --runs v2 --port 47811
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socketserver
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench import sngnn_adapter as sa  # noqa: E402

BENCH = os.environ.get("SC_BENCH_ROOT", "/home/robotics/arena_ws/data/costmap_bench")
HIST = 12  # t .. t-2 s at 5 Hz, +1 so the oldest kept frame is never index 0 once full (GraphBuilder treats index 0 as the episode start)


class LiveEpisode:
    id = "live"

    def __init__(self):
        self.robots: list[dict] = []
        self.peds: list[dict] = []
        self.frames: list[dict] = []

    def add(self, msg: dict) -> int:
        r = msg["robot"]
        self.robots.append({"pose_map_base": {"x": r["x"], "y": r["y"], "yaw": r["yaw"]}, "odom_twist": msg.get("twist", {}), "stamp_ns": int(msg["stamp_ns"])})
        self.peds.append({"pedestrians": msg["peds"]})
        if len(self.robots) > HIST:
            self.robots.pop(0)
            self.peds.pop(0)
        return len(self.robots) - 1


class Service:
    def __init__(self, label: str, runs: str, seeds, device: str = "cuda"):
        self.gb = sa.GraphBuilder("sensor")
        self.dev = torch.device(device)
        self.ckpts = [os.path.join(BENCH, "runs", runs, f"sngnn_sensor_{label}_s{s}", "best.pt") for s in seeds]
        for c in self.ckpts:
            assert os.path.exists(c), c
        self.models = None
        self.ep = LiveEpisode()

    def scene(self, msg: dict) -> dict:
        self.gb.man["live"] = {"world": msg["world"], "objects": msg.get("objects", []), "robot": {"goal": msg["goal"]}}
        self.ep = LiveEpisode()
        return {"ok": True}

    def step(self, msg: dict) -> dict:
        t0 = time.time()
        if self.ep.robots:
            rp = self.ep.robots[-1]["pose_map_base"]
            if np.hypot(msg["robot"]["x"] - rp["x"], msg["robot"]["y"] - rp["y"]) > 1.0:  # scenario switch
                self.ep = LiveEpisode()
        k = self.ep.add(msg)
        g = self.gb(self.ep, k)
        if self.models is None:
            self.models = []
            for c in self.ckpts:
                m = sa.SNGNNCostmap(g.ndata["h"].shape[1], sa.num_rels())
                m.load_state_dict(torch.load(c, map_location="cpu")["model"])
                self.models.append(m.to(self.dev).eval())
        with torch.no_grad():
            gg = sa.batch_graphs([g], self.dev)
            out = torch.stack([((m({"graph": gg}) + 1) / 2).clamp(0, 1)[0] for m in self.models]).mean(0)
        cost = (out.cpu().numpy() * 252).round().clip(0, 252).astype(np.uint8)
        return {"cost": base64.b64encode(cost.tobytes()).decode(), "shape": list(cost.shape), "ms": (time.time() - t0) * 1000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="extended_v2")
    ap.add_argument("--runs", default="v2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--port", type=int, default=47811)
    a = ap.parse_args()
    svc = Service(a.label, a.runs, [int(s) for s in a.seeds.split(",")])

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                try:
                    msg = json.loads(line)
                    rep = svc.scene(msg) if msg.get("cmd") == "scene" else svc.step(msg)
                except Exception as e:  # noqa: BLE001
                    rep = {"error": repr(e)}
                self.wfile.write((json.dumps(rep) + "\n").encode())
                self.wfile.flush()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", a.port), Handler) as srv:
        print(f"sngnn_server on 127.0.0.1:{a.port} ({a.label}, runs/{a.runs})", flush=True)
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
