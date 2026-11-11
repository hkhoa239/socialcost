#!/usr/bin/env python3
"""Headless batch collection: for every scenario x seed, launch Arena + Isaac, record, label.

Runs inside the socialcost-arena container (ROS domain 7). Resumable: an episode whose
episode_metadata.json says valid is not recorded again.

    python collect_dataset.py configs/dataset/smoke.yaml [--only socialcost_queue:1]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from socialcost.dataset import FORMAT_VERSION, label_definitions, label_episode, live_agreement, make_splits, scenario_bodies  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402

ARENA_DIR = os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena")
SCRIPTS = "/opt/arena_ws/data/socialcost/scripts"


def log(msg: str) -> None:
    print(f"[collect {datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def git_rev(path: str) -> str:
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def resolve(cfg_path: str, name: str, sub: str) -> str:
    return name if os.path.isabs(name) else os.path.join(PKG, "configs", sub, name + ".yaml")


def wait_for_tf(map_frame: str, base_frame: str, timeout_s: float) -> bool:
    code = (
        "import rclpy,time,sys\n"
        "from rclpy.node import Node\nfrom tf2_ros import Buffer,TransformListener\n"
        "rclpy.init(); n=Node('socialcost_tfwait',parameter_overrides=[rclpy.parameter.Parameter('use_sim_time',value=True)])\n"
        "b=Buffer(); TransformListener(b,n); t0=time.monotonic()\n"
        f"while time.monotonic()-t0<{timeout_s}:\n"
        "    rclpy.spin_once(n,timeout_sec=0.2)\n"
        f"    if b.can_transform('{map_frame}','{base_frame}',rclpy.time.Time()): sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    return subprocess.run([sys.executable, "-c", code]).returncode == 0


def record_one(cfg: dict, sensors: dict, scenario: str, seed: int, ep_dir: str, log_dir: str, extra: dict | None = None) -> dict:
    from socialcost.live import LiveSocialCostmap, costmap_params

    lp = cfg["label"]
    params = LayerParams.from_overlay(os.path.join(ARENA_DIR, lp["overlay"]), lp["costmap"])
    tag = f"{scenario}_seed{seed}"
    launch_args = [f"task.robots:=scenario", "task.obstacles:=scenario", f"task.scenario.file:={scenario}",
                   f"world:={(extra or {}).get('world', cfg['world'])}", f"run_seed:={seed}", *cfg.get("launch_args", []), *(extra or {}).get("launch_args", [])]
    log(f"launch {tag}: {' '.join(launch_args)}")
    launch = subprocess.Popen(["bash", "-l", f"{SCRIPTS}/launch_isaac.sh", f"collect_{tag}", *launch_args], start_new_session=True)
    t_launch = time.time()
    try:
        if not wait_for_tf(sensors["map_frame"], sensors["base_frame"], 600):
            raise RuntimeError("robot TF never appeared")
        log(f"{tag}: robot TF up after {time.time() - t_launch:.0f}s")
        live = None
        if cfg.get("live_costmap", False):
            doc = costmap_params(params, sensors["peds_topic"], sensors["map_frame"], sensors["base_frame"],
                                 lp["size_m"], lp["resolution"], rolling_window=True, use_sim_time=True)
            live = LiveSocialCostmap(doc, namespace=f"/socialcost/live_{tag}", log_path=os.path.join(log_dir, f"live_costmap_{tag}.log"))
            live.__enter__()
        min_episode = 1
        if cfg.get("warmup_episode", False):
            # discard the bring-up episode: wait for real robot motion, reset, record from the next episode
            w = subprocess.run([sys.executable, "-m", "socialcost.warmup", "--timeout", "400"], capture_output=True, text=True,
                               cwd=PKG, env={**os.environ, "PYTHONPATH": PKG + os.pathsep + os.environ.get("PYTHONPATH", "")})
            if w.returncode != 0:
                raise RuntimeError(f"warm-up failed: {w.stderr.strip()[-300:]}")
            min_episode = int(w.stdout.strip().splitlines()[-1])
            log(f"{tag}: warm-up done, recording from episode {min_episode}")
        try:
            rec = subprocess.run([
                sys.executable, os.path.join(HERE, "record_episode.py"), "--out", ep_dir,
                "--sensors", resolve("", cfg["sensors"], "sensors"),
                "--sample-hz", str(cfg["sample_hz"]), "--duration", str(cfg["duration_s"]),
                "--warmup", str(cfg["warmup_s"]), "--fresh-margin", str(cfg.get("fresh_margin_s", 2.0)),
                "--metadata", json.dumps({"scenario": scenario, "run_seed": str(seed), "dataset": cfg["name"], "world": (extra or {}).get("world", cfg["world"]),
                                          **{k: v for k, v in (extra or {}).items() if k not in ("launch_args",)}}),
                *(["--live-costmap-topic", live.raw_topic] if live else []), "--min-episode-id", str(min_episode),
            ], timeout=1800)
            if rec.returncode != 0:
                raise RuntimeError(f"recorder exited {rec.returncode}")
        finally:
            if live is not None:
                live.__exit__(None, None, None)
    finally:
        log(f"{tag}: stopping launch")
        subprocess.run(["bash", f"{SCRIPTS}/stop_isaac.sh"])
        try:
            launch.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, 9)

    grid = GridSpec.centered(lp["size_m"], lp["resolution"], frame="base_link (robot-centred, yaw-aligned)")
    stats = label_episode(ep_dir, params, grid)
    agreement = live_agreement(ep_dir, params) if cfg.get("live_costmap", False) else []
    meta_path = os.path.join(ep_dir, "episode_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["label_definitions"] = {k: v["name"] + "/" + v["version"] for k, v in label_definitions(params, grid).items() if k != "sources"}
    meta["bodies"] = scenario_bodies(os.path.join(ARENA_DIR, "arena_simulation_setup", "worlds", (extra or {}).get("world", cfg["world"])), scenario)
    meta["live_agreement"] = agreement
    meta["wall_seconds_total"] = time.time() - t_launch
    exp = meta["expected_samples"]
    need_cost = cfg.get("require_cost", True)
    meta["valid"] = bool(meta["samples"] >= 0.8 * exp and meta["stats"]["tf_failures"] == 0 and (stats["labels_with_cost"] > 0 or not need_cost))
    # start_ok stays a diagnostic only: odometry around the reset teleport makes the onset estimate unreliable
    # (robot TF trajectories agree to a few cm across runs, docs/REPRODUCIBILITY.md)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"{tag}: {meta['samples']}/{exp} samples, labels with cost {stats['labels_with_cost']}, valid={meta['valid']}")
    return meta


def derive_off(cfg: dict, job: dict, manifest: dict, root: str) -> None:
    """Spawn an OFF condition where its ON twin's people stood when the ON window opened.

    For conditions whose ON people move into place (QUEUE_USE re-slots its members), a static OFF
    can only match the ON geometry if it starts from the recorded ON positions.
    """
    import math

    import yaml

    on_key = f"{job['derive_from']}:{job['seed']}:0"
    ep = os.path.join(root, "episodes", manifest[on_key])
    meta = json.load(open(os.path.join(ep, "episode_metadata.json")))
    # Arena shifts each env by an offset inside the humansim bridge (not in TF). Measured from standing
    # people (recorded map position minus scenario pose): map_empty +5.30, reception +5.25 = 5 - map origin.
    world_name = job.get("world") or cfg["world"]
    off = {"map_empty": 5.3, "reception": 5.25}.get(world_name)
    if off is None:
        origin = yaml.safe_load(open(os.path.join(ARENA_DIR, "arena_simulation_setup", "worlds", world_name, "map", "0.yaml")))["origin"]
        off = 5.0 - origin[0]
    tf = {"x": -off, "y": -off, "yaw": 0.0}
    with open(os.path.join(ep, "pedestrian_states.jsonl")) as f:
        first = json.loads(f.readline())["pedestrians"]
    c, s_ = math.cos(tf["yaw"]), math.sin(tf["yaw"])
    world = (job.get("world") or cfg["world"])
    scen = os.path.join(ARENA_DIR, "arena_simulation_setup", "worlds", world, "scenarios", job["scenario"], "scenario.yaml")
    with open(scen) as f:
        header = f.readline()
        doc = yaml.safe_load(f)
    by_name = {p["name"].split("/")[-1]: p for p in first}
    for h in doc["humans"]:
        p = by_name[h["name"]]
        h["pose"] = [round(tf["x"] + c * p["x"] - s_ * p["y"], 4), round(tf["y"] + s_ * p["x"] + c * p["y"], 4), round(p["yaw"] + tf["yaw"], 4)]
    with open(scen, "w") as f:
        f.write(header.rstrip("\n") + f" Spawn poses derived from {on_key} ({manifest[on_key]}) at its first sample.\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    subprocess.run(["bash", "-lc", "cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1 && colcon build --symlink-install "
                    "--cmake-args -DPython3_EXECUTABLE=/opt/venv/bin/python --base-paths /opt/arena_ws/src --build-base /opt/arena_ws/build "
                    "--install-base /opt/arena_ws/install --packages-select arena_simulation_setup >/dev/null 2>&1"], check=True)
    log(f"{job['scenario']}: spawn poses derived from {on_key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", action="append", default=[], help="scenario:seed to run, repeatable")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(resolve("", cfg["sensors"], "sensors")) as f:
        sensors = yaml.safe_load(f)
    root = cfg["output"]
    log_dir = os.path.join(root, "logs")
    os.makedirs(os.path.join(root, "episodes"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    manifest_path = os.path.join(root, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    # Live costmaps left by an interrupted run would answer lifecycle calls for a new one.
    subprocess.run(["pkill", "-f", "nav2_costmap_2d --ros-args -r __node:=social_costmap"])
    lp = cfg["label"]
    params = LayerParams.from_overlay(os.path.join(ARENA_DIR, lp["overlay"]), lp["costmap"])
    # explicit jobs (scenario, seed, metadata such as pair_id / condition) or the scenarios x seeds grid
    if cfg.get("jobs"):
        jobs = [(j["scenario"], j["seed"], j.get("rep", 0), j) for j in cfg["jobs"]]
    else:
        jobs = [(s, seed, rep, None) for s in cfg["scenarios"] for seed in cfg["seeds"] for rep in range(cfg.get("episodes_per_seed", 1))]
    for scenario, seed, rep, extra in jobs:
        key = f"{scenario}:{seed}:{rep}"
        if args.only and f"{scenario}:{seed}" not in args.only and scenario not in args.only:
            continue
        ep_name = manifest.get(key) or f"episode_{len(manifest) + 1:06d}"
        ep_dir = os.path.join(root, "episodes", ep_name)
        meta_file = os.path.join(ep_dir, "episode_metadata.json")
        if os.path.exists(meta_file) and json.load(open(meta_file)).get("valid"):
            log(f"{key}: {ep_name} already valid, skipping")
            continue
        if os.path.isdir(ep_dir):
            subprocess.run(["rm", "-rf", ep_dir])
        manifest[key] = ep_name
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        try:
            if extra and extra.get("derive_from"):
                derive_off(cfg, extra, manifest, root)
            record_one(cfg, sensors, scenario, seed, ep_dir, log_dir, {k: v for k, v in (extra or {}).items() if k not in ("scenario", "seed", "rep")})
        except Exception as e:  # noqa: BLE001 - keep going, the next job may succeed
            log(f"{key}: FAILED {e}")

    dataset_meta = {
        "format": FORMAT_VERSION, "name": cfg["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "sensors": sensors, "label_params": params.to_dict(),
        "label_grid": GridSpec.centered(lp["size_m"], lp["resolution"]).to_dict(),
        "label_semantics": "uint8 cost in [0, max_cost_value] of arena_social_cost_layer alone (no obstacle/inflation), "
                           "cell (i, j) = array[j, i], centre at origin + (i+0.5, j+0.5) * resolution in the robot base frame "
                           "(x forward, y left), painted from pedestrian states at the image stamp",
        "privileged_fields": ["pedestrian_states.jsonl", "peds_stream.jsonl", "interactions.jsonl"],
        "software": {"arena_commit": git_rev(ARENA_DIR), "arena_isaac_commit": git_rev(os.path.join(ARENA_DIR, "arena_isaac")),
                     "isaac_sim_image": "nvcr.io/nvidia/isaac-sim:6.1.0", "ros_distro": os.environ.get("ROS_DISTRO")},
        "manifest": manifest,
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)
    with open(os.path.join(root, "labels.json"), "w") as f:
        json.dump(label_definitions(params, GridSpec.centered(lp["size_m"], lp["resolution"]), sources=dataset_meta["software"]), f, indent=2)
    splits = make_splits(root, cfg.get("splits", {}))
    log(f"splits: { {k: len(v) for k, v in splits.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
