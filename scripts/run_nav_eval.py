#!/usr/bin/env python3
"""Closed-loop navigation runs: scenario x method x controller x seed, one recorded episode each.

Every arm uses the same Nav2 configuration (overlay socialcost_estimate: obstacle + voxel +
mask overlay of the estimate + inflation) and differs only in what estimator_node publishes.
The bring-up episode is discarded (warm-up reset), the next episode is logged by repro_probe
(robot and pedestrian trajectories, outcome) together with the estimator's own log.

    python run_nav_eval.py --config configs/navigation/nav_v1.yaml

Config `method_specs` (optional) maps an arm name to estimator arguments, e.g.
  rgbd_t13: {method: rgbd, checkpoint: <best.pt>, post: <spec.json>}
Arms without a spec keep the v1 behaviour (bodies -> zero, checkpoint from `checkpoints`).
A warm-up failure (robot never moved: bring-up, not navigation) or a silent estimator (< 20 grids published,
the robot would run on bodies-only costs) is an infrastructure failure, retried once on a fresh launch.
A probe timeout is kept as status "timeout" and scored as a navigation failure by nav_metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
SCRIPTS = "/opt/arena_ws/data/socialcost/scripts"


def log(msg: str) -> None:
    print(f"[nav {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def py(args: list[str], **kw) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": PKG + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.Popen([sys.executable, *args], cwd=PKG, env=env, **kw)


def run_one(cfg: dict, job: dict, out_dir: str) -> dict:
    tag = f"{job['scenario']}__{job['method']}__{job['controller']}__s{job['seed']}"
    run_dir = os.path.join(out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    overlay = "socialcost_estimate" if job["method"] != "bodies_native" else ""
    args = ["task.robots:=scenario", "task.obstacles:=scenario", f"task.scenario.file:={job['scenario']}", f"run_seed:={job['seed']}",
            f"robot.mobile.local_planner:={job['controller']}", *([f"robot.mobile.params_overlay:={overlay}"] if overlay else [])]
    log(f"{tag}: launch")
    launch = subprocess.Popen(["bash", "-l", f"{SCRIPTS}/launch_isaac.sh", f"nav_{tag}", *args], start_new_session=True)
    est = None
    try:
        spec = cfg.get("method_specs", {}).get(job["method"], {})
        ck = spec.get("checkpoint", cfg.get("checkpoints", {}).get(job["method"]))
        if job["method"] != "bodies_native":
            method = spec.get("method", {"bodies": "zero"}.get(job["method"], job["method"]))
            extra = (["--post", spec["post"]] if spec.get("post") else []) + (["--shadow"] if spec.get("shadow") else [])
            for f in ("estimator.jsonl",):
                if os.path.exists(os.path.join(run_dir, f)):
                    os.replace(os.path.join(run_dir, f), os.path.join(run_dir, f + ".prev"))
            est = py(["-m", "socialcost.estimator_node", "--method", method, *(["--checkpoint", ck] if ck else []), *extra,
                      "--log", os.path.join(run_dir, "estimator.jsonl")], stdout=open(os.path.join(run_dir, "estimator.log"), "w"), stderr=subprocess.STDOUT,
                     start_new_session=True)
        w = py(["-m", "socialcost.warmup", "--timeout", "500"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = w.communicate(timeout=600)
        if w.returncode != 0:
            raise RuntimeError(f"warm-up failed: {err.strip()[-300:]}")
        ep = int(out.strip().splitlines()[-1])
        probe = py([os.path.join(HERE, "repro_probe.py"), "--out", os.path.join(run_dir, "episode.jsonl"), "--episodes", "1",
                    "--min-episode-id", str(ep), "--timeout", str(cfg.get("episode_timeout_wall_s", 1500))])
        probe.wait(timeout=cfg.get("episode_timeout_wall_s", 1500) + 60)
        status = "ok" if probe.returncode == 0 else ("timeout" if not os.path.exists(os.path.join(run_dir, "episode.jsonl")) else f"probe exit {probe.returncode}")
        # an estimator that published (almost) nothing leaves the robot on bodies-only costs: not a valid run of the arm
        el = os.path.join(run_dir, "estimator.jsonl")
        n_pub = sum(1 for _ in open(el)) if os.path.exists(el) else 0
        if est is not None and n_pub < 20:
            status = f"failed: estimator_silent ({n_pub} grids)"
    except Exception as e:  # noqa: BLE001
        status = f"failed: {e}"
    finally:
        if est is not None and est.poll() is None:
            os.killpg(est.pid, 2)
        subprocess.run(["bash", f"{SCRIPTS}/stop_isaac.sh"])
        try:
            launch.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, 9)
    rec = {**job, "tag": tag, "status": status, "wall_end": time.time()}
    json.dump(rec, open(os.path.join(run_dir, "run.json"), "w"), indent=2)
    log(f"{tag}: {status}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["output"]
    os.makedirs(out_dir, exist_ok=True)
    jobs = [{"scenario": s, "method": m, "controller": c, "seed": seed}
            for s in cfg["scenarios"] for c in cfg["controllers"] for seed in cfg["seeds"] for m in cfg["methods"]]
    jobs += [{"scenario": s, "method": "bodies", "controller": c, "seed": seed} for s in cfg.get("reference_scenarios", []) for c in cfg["controllers"] for seed in cfg["seeds"]]
    for job in jobs:
        tag = f"{job['scenario']}__{job['method']}__{job['controller']}__s{job['seed']}"
        done = os.path.join(out_dir, tag, "run.json")
        if os.path.exists(done) and json.load(open(done)).get("status") in ("ok", "timeout"):
            continue
        rec = run_one(cfg, job, out_dir)
        if rec["status"].startswith("failed: warm-up") or rec["status"].startswith("failed: estimator_silent"):
            log(f"{tag}: retrying after warm-up failure")
            rec = run_one(cfg, job, out_dir)
            rec["retried"] = True
            json.dump(rec, open(os.path.join(out_dir, tag, "run.json"), "w"), indent=2)
        with open(os.path.join(out_dir, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
