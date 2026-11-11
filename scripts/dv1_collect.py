#!/usr/bin/env python3
"""dataset_v1 checkpointed collection (Stages A-C): record, label, post-process manifest episodes.

Runs inside socialcost-arena (ROS domain 7). One Arena + Isaac launch per episode, reusing
collect_dataset.record_one. Resumable and idempotent:

  * an episode with episodes/<id>/DONE (valid recording + post-processing) is skipped;
  * a failed or invalid episode is deleted and retried, at most --retries times per run, and every
    attempt is appended to metadata/collection_log.jsonl;
  * an OFF episode whose spawn is derived from its ON twin waits for the ON episode; the derived poses are
    stored in manifests/derived_spawns.json (the frozen manifest is never edited).

    python dv1_collect.py --root /opt/arena_ws/data/socialcost/dataset_v1 --stage subset|pilot|all [--only 000021,000022]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import collect_dataset as cd  # noqa: E402
from socialcost.dv1 import assets, post, spec  # noqa: E402
from socialcost.dv1.geometry import map_offset  # noqa: E402
from socialcost.dv1.scenario import write_episode  # noqa: E402
from dv1_make_scenarios import apply_derived, build  # noqa: E402

CFG = {
    "name": spec.DATASET, "world": "map_empty", "sensors": "turtlebot_oakd", "sample_hz": spec.SAMPLE_HZ, "duration_s": spec.DURATION_S,
    "warmup_s": spec.WARMUP_S, "fresh_margin_s": 0.5, "warmup_episode": True, "require_cost": False, "live_costmap": False, "launch_args": [],
    "label": {"overlay": "arena_robots/arena_robots/config/nav2/overlays/social_cost_layer.yaml", "costmap": "local_costmap", "size_m": 10.0, "resolution": 0.05},
}


def log(msg: str) -> None:
    print(f"[dv1 {datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def status(root: str, line: str) -> None:
    with open(os.path.join(os.path.dirname(root), "dataset_v1_STATUS.txt"), "a") as f:
        f.write(f"{line} {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")


def select(episodes: list[dict], stage: str, only: set[str]) -> list[dict]:
    if only:
        return [e for e in episodes if e["episode_id"] in only]
    if stage == "subset":
        return [e for e in episodes if e["subset"]]
    if stage == "pilot":
        return [e for e in episodes if e["pilot"]]
    if stage == "remaining":
        return [e for e in episodes if not e["pilot"]]
    return episodes


def derive(root: str, ep: dict, on_ep: dict) -> dict:
    """Spawn poses of the OFF episode from the ON episode's first recorded sample (map -> world frame)."""
    on_dir = os.path.join(root, "episodes", on_ep["episode_id"])
    with open(os.path.join(on_dir, "pedestrian_states.jsonl")) as f:
        first = json.loads(f.readline())["pedestrians"]
    ox, oy = map_offset(ep["world"])
    by_name = {p["name"].split("/")[-1]: p for p in first}
    spawn = {}
    for h in ep["humans"]:
        p = by_name.get(h["name"])
        if p is None:
            raise RuntimeError(f"{h['name']} missing from ON episode {on_ep['episode_id']}")
        spawn[h["name"]] = [round(p["x"] - ox, 4), round(p["y"] - oy, 4), round(p["yaw"], 4)]
    return {"from_episode": on_ep["episode_id"], "from_sample": 0, "map_offset": [ox, oy], "spawn": spawn}


def finalize(root: str, ep: dict, ep_dir: str, bboxes: dict, heights: dict, derived: dict | None) -> dict:
    meta_path = os.path.join(ep_dir, "episode_metadata.json")
    meta = json.load(open(meta_path))
    meta["dv1"] = {k: ep[k] for k in ("episode_id", "unit_id", "pair_id", "condition", "family", "subtype", "split", "world_key", "world", "site",
                                      "template_id", "n_peds", "visibility_category", "observability", "reproducibility", "trajectory_classes",
                                      "interaction_condition", "stage", "label_versions")}
    meta["dv1"]["manifest_entry_sha"] = cd_hash(ep)
    meta["dv1"]["derived_spawn"] = derived
    json.dump(meta, open(meta_path, "w"), indent=2)
    robot = robot_health(ep_dir)
    meta["dv1"]["robot_health"] = robot
    json.dump(meta, open(meta_path, "w"), indent=2)
    summary = post.process(ep_dir, bboxes, heights, ep)
    ok = bool(meta["valid"]) and summary["components"]["combined_native_mismatch_samples"] == 0 and robot["ok"]
    done = {"episode_id": ep["episode_id"], "valid": ok, "samples": meta["samples"], "expected": meta["expected_samples"],
            "wall_seconds": meta.get("wall_seconds_total"), "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    if ok:
        json.dump(done, open(os.path.join(ep_dir, "DONE"), "w"))
    return done


def robot_health(ep_dir: str) -> dict:
    """Robot upright (|roll|, |pitch| < 5 deg) and moving (>= 1 m over the window). A tipped robot (seen once in
    office_1 after the warm-up reset) drives its wheels without moving and must not enter the dataset."""
    import math
    rows = [json.loads(line) for line in open(os.path.join(ep_dir, "robot_states.jsonl"))]
    tilts = []
    for r in rows:
        q = r["pose_map_base"]
        roll = math.degrees(math.atan2(2 * (q["qw"] * q["qx"] + q["qy"] * q["qz"]), 1 - 2 * (q["qx"] ** 2 + q["qy"] ** 2)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q["qw"] * q["qy"] - q["qz"] * q["qx"])))))
        tilts.append(max(abs(roll), abs(pitch)))
    tilt = max(tilts)
    tilt_med = sorted(tilts)[len(tilts) // 2]
    tilt_frac5 = sum(t > 5.0 for t in tilts) / len(tilts)
    a, b = rows[0]["pose_map_base"], rows[-1]["pose_map_base"]
    path = sum(math.dist((rows[i]["pose_map_base"]["x"], rows[i]["pose_map_base"]["y"]), (rows[i + 1]["pose_map_base"]["x"], rows[i + 1]["pose_map_base"]["y"]))
               for i in range(len(rows) - 1))
    disp = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
    meta = json.load(open(os.path.join(ep_dir, "episode_metadata.json")))
    fm = meta["stats"].get("first_motion_rel_s")
    # robot start delay after the seeded reset: 0.3-0.9 s normally; a late start shifts the whole robot path
    # sustained tipping (office_1: 17-20 deg throughout) fails; a one-sample bump over a floor seam does not
    upright = tilt_med < 3.0 and tilt_frac5 <= 0.05 and tilt < 12.0
    return {"max_tilt_deg": round(tilt, 2), "median_tilt_deg": round(tilt_med, 2), "frac_tilt_gt5": round(tilt_frac5, 3), "path_m": round(path, 2),
            "displacement_m": round(disp, 2), "first_motion_s": fm, "ok": upright and path >= 1.0 and fm is not None and fm <= 1.5}


def cd_hash(ep: dict) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(ep, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ persistent mode

class Runner:
    """One Arena + Isaac launch serving consecutive episodes of a world: set task.scenario.file, reset with the
    episode's seed, record the next episode. Relaunched on a world change, after an error, and every MAX_PER_LAUNCH."""

    MAX_PER_LAUNCH = 25

    def __init__(self, sensors: dict, log_dir: str):
        self.sensors, self.log_dir = sensors, log_dir
        self.proc, self.world, self.count = None, None, 0
        self.extra_args: list[str] = []

    def stop(self) -> None:
        if self.proc is not None:
            subprocess.run(["bash", f"{cd.SCRIPTS}/stop_isaac.sh"])
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
        self.proc, self.world, self.count = None, None, 0

    def ensure(self, ep: dict) -> None:
        if self.proc is not None and self.world == ep["world"] and self.count < self.MAX_PER_LAUNCH and self.proc.poll() is None:
            return
        self.stop()
        args = ["task.robots:=scenario", "task.obstacles:=scenario", f"task.scenario.file:={ep['scenario']}", f"world:={ep['world']}", f"run_seed:={ep['run_seed']}", *self.extra_args]
        log(f"launch {ep['world']} (persistent) with {ep['scenario']}")
        self.proc = subprocess.Popen(["bash", "-l", f"{cd.SCRIPTS}/launch_isaac.sh", f"dv1_{ep['world']}_{ep['episode_id']}", *args], start_new_session=True)
        t0 = time.time()
        if not cd.wait_for_tf(self.sensors["map_frame"], self.sensors["base_frame"], 600):
            self.stop()
            raise RuntimeError("robot TF never appeared")
        w = subprocess.run([sys.executable, "-m", "socialcost.warmup", "--timeout", "400"], capture_output=True, text=True, cwd=cd.PKG,
                           env={**os.environ, "PYTHONPATH": cd.PKG + os.pathsep + os.environ.get("PYTHONPATH", "")})
        if w.returncode != 0:
            self.stop()
            raise RuntimeError(f"warm-up failed: {w.stderr.strip()[-300:]}")
        log(f"{ep['world']}: up and warm after {time.time() - t0:.0f}s")
        self.world = ep["world"]

    def record(self, ep: dict, ep_dir: str) -> dict:
        t0 = time.time()
        self.ensure(ep)
        sw = subprocess.run([sys.executable, "-m", "socialcost.switch", "--scenario", ep["scenario"], "--seed", str(ep["run_seed"])], capture_output=True,
                            text=True, cwd=cd.PKG, env={**os.environ, "PYTHONPATH": cd.PKG + os.pathsep + os.environ.get("PYTHONPATH", "")}, timeout=200)
        if sw.returncode != 0:
            self.stop()
            raise RuntimeError(f"switch failed: {sw.stderr.strip()[-300:]}")
        new_id = int(sw.stdout.strip().splitlines()[-1])
        self.count += 1
        meta_in = {"scenario": ep["scenario"], "run_seed": str(ep["run_seed"]), "dataset": CFG["name"], "world": ep["world"], "pair_id": ep["pair_id"],
                   "condition": ep["condition"], "family": ep["family"], "split": ep["split"], "dv1_episode_id": ep["episode_id"],
                   "collection_mode": "persistent", "reset_seed": ep["run_seed"]}
        rec = subprocess.run([sys.executable, os.path.join(HERE, "record_episode.py"), "--out", ep_dir, "--sensors", cd.resolve("", CFG["sensors"], "sensors"),
                              "--sample-hz", str(CFG["sample_hz"]), "--duration", str(CFG["duration_s"]), "--warmup", str(CFG["warmup_s"]),
                              "--fresh-margin", str(CFG["fresh_margin_s"]), "--metadata", json.dumps(meta_in), "--min-episode-id", str(new_id)], timeout=900)
        if rec.returncode != 0:
            self.stop()
            raise RuntimeError(f"recorder exited {rec.returncode}")
        return label_meta(ep, ep_dir, t0)


def label_meta(ep: dict, ep_dir: str, t0: float) -> dict:
    """Labelling and validity, as collect_dataset.record_one does after a recording."""
    lp = CFG["label"]
    params = cd.LayerParams.from_overlay(os.path.join(cd.ARENA_DIR, lp["overlay"]), lp["costmap"])
    grid = cd.GridSpec.centered(lp["size_m"], lp["resolution"], frame="base_link (robot-centred, yaw-aligned)")
    stats = cd.label_episode(ep_dir, params, grid)
    meta_path = os.path.join(ep_dir, "episode_metadata.json")
    meta = json.load(open(meta_path))
    meta["label_definitions"] = {k: v["name"] + "/" + v["version"] for k, v in cd.label_definitions(params, grid).items() if k != "sources"}
    meta["bodies"] = cd.scenario_bodies(os.path.join(cd.ARENA_DIR, "arena_simulation_setup", "worlds", ep["world"]), ep["scenario"])
    meta["live_agreement"] = []
    meta["wall_seconds_total"] = time.time() - t0
    meta["valid"] = bool(meta["samples"] >= 0.8 * meta["expected_samples"] and meta["stats"]["tf_failures"] == 0)
    json.dump(meta, open(meta_path, "w"), indent=2)
    log(f"{ep['episode_id']}: {meta['samples']}/{meta['expected_samples']} samples, labels with cost {stats['labels_with_cost']}, valid={meta['valid']}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", default="subset", choices=["subset", "pilot", "remaining", "all"])
    ap.add_argument("--only", default="")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--mode", default="persistent", choices=["persistent", "relaunch"], help="persistent: one Isaac launch per world group")
    args = ap.parse_args()
    root = args.root
    man = json.load(open(os.path.join(root, "manifests", "manifest_v1.json")))
    by_id = {e["episode_id"]: e for e in man["episodes"]}
    eps = select(man["episodes"], args.stage, {x for x in args.only.split(",") if x})
    # ON before OFF within a pair (manifest order already is), derived OFF episodes after their ON twin
    dpath = os.path.join(root, "manifests", "derived_spawns.json")
    derived_all = json.load(open(dpath)) if os.path.exists(dpath) else {}
    bboxes, heights = assets.load(root)
    sensors = __import__("yaml").safe_load(open(cd.resolve("", CFG["sensors"], "sensors")))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    attempts_log = os.path.join(root, "metadata", "collection_log.jsonl")
    status(root, f"COLLECT_BEGIN stage={args.stage} episodes={len(eps)}")
    # write every selected scenario once and build
    for ep in eps:
        write_episode(apply_derived(ep, derived_all))
    build()
    def is_done(e: dict) -> bool:
        """DONE and recorded from the current manifest entry (a regenerated manifest invalidates stale recordings)."""
        d = os.path.join(root, "episodes", e["episode_id"])
        if not os.path.exists(os.path.join(d, "DONE")):
            return False
        try:
            sha = json.load(open(os.path.join(d, "episode_metadata.json")))["dv1"]["manifest_entry_sha"]
        except Exception:  # noqa: BLE001
            return False
        if sha != cd_hash(e):
            log(f"{e['episode_id']}: recorded from a different manifest entry, re-recording")
            os.remove(os.path.join(d, "DONE"))
            return False
        return True

    todo = [e for e in eps if not is_done(e)]
    log(f"{len(eps)} selected, {len(eps) - len(todo)} already done, {len(todo)} to record")
    tries = {e["episode_id"]: 0 for e in todo}
    if args.mode == "persistent":
        order = {w: i for i, w in enumerate(dict.fromkeys(e["world"] for e in todo))}
        todo = sorted(todo, key=lambda e: (order[e["world"]], int(e["episode_id"])))
    queue = list(todo)
    runner = Runner(sensors, log_dir) if args.mode == "persistent" else None
    n_ok = n_fail = 0
    while queue:
        ep = queue.pop(0)
        eid = ep["episode_id"]
        ep_dir = os.path.join(root, "episodes", eid)
        if os.path.exists(os.path.join(ep_dir, "DONE")):
            continue
        derived = None
        if ep.get("derive_spawn_from") == "on":
            on_ep = next(x for x in man["episodes"] if x["pair_id"] == ep["pair_id"] and x["condition"] == "on")
            if not os.path.exists(os.path.join(root, "episodes", on_ep["episode_id"], "DONE")):
                if any(q["episode_id"] == on_ep["episode_id"] for q in queue):
                    queue.append(ep)  # ON twin still pending (first try or a retry): come back after it
                    continue
                log(f"{eid}: ON twin {on_ep['episode_id']} not valid, skipping (shortfall)")
                _append(attempts_log, {"episode_id": eid, "result": "skipped_on_missing", "on": on_ep["episode_id"]})
                n_fail += 1
                continue
            derived = derive(root, ep, on_ep)
            derived_all[eid] = derived
            json.dump(derived_all, open(dpath, "w"), indent=1)
            write_episode(apply_derived(ep, derived_all))
            if runner is None:
                build()  # persistent mode: the directory exists and install holds symlinks to it, no rebuild needed
        tries[eid] += 1
        if os.path.isdir(ep_dir):
            shutil.rmtree(ep_dir)
        t0 = time.time()
        rec = {"episode_id": eid, "attempt": tries[eid], "family": ep["family"], "world": ep["world"], "started": datetime.datetime.now().isoformat(timespec="seconds")}
        try:
            extra = {"world": ep["world"], "pair_id": ep["pair_id"], "condition": ep["condition"], "family": ep["family"], "split": ep["split"],
                     "dv1_episode_id": eid}
            if runner is not None:
                runner.record(ep, ep_dir)
            else:
                cd.record_one(CFG, sensors, ep["scenario"], ep["run_seed"], ep_dir, log_dir, extra)
            done = finalize(root, ep, ep_dir, bboxes, heights, derived)
            rec.update({"result": "ok" if done["valid"] else "invalid", "samples": done["samples"]})
        except Exception as e:  # noqa: BLE001
            rec.update({"result": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        rec["wall_s"] = round(time.time() - t0, 1)
        _append(attempts_log, rec)
        log(f"{eid} ({ep['family']}/{ep['condition']}/{ep['world']}): {rec['result']} in {rec['wall_s']}s (attempt {tries[eid]})")
        if rec["result"] == "ok":
            n_ok += 1
            status(root, f"EP_OK {eid}")
        elif tries[eid] <= args.retries:
            queue.append(ep)
            status(root, f"EP_RETRY {eid} {rec['result']}")
        else:
            n_fail += 1
            status(root, f"EP_FAILED {eid} {rec['result']}")
    if runner is not None:
        runner.stop()
    status(root, f"COLLECT_DONE stage={args.stage} ok={n_ok} failed={n_fail} mode={args.mode}")
    return 0


def _append(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())
