#!/usr/bin/env python3
"""Closed-loop evaluation of costmap estimators on navigation_challenge_v1 (36 configurations), persistent Isaac.

Every arm uses the same Nav2 stack (overlay socialcost_estimate: obstacle + voxel + mask overlay of the estimate +
inflation, one controller) and differs only in what estimator_node publishes. One Isaac launch per world serves all
its configurations and arms (task.scenario.file + lifecycle/reset_episode with the configuration's run_seed, so the
pedestrians replay identically across arms). Per run:

  1. start estimator_node for the arm and wait until it publishes (no run starts on another arm's stale mask)
  2. start the ground-truth cost meter (socialcost.cost_meter) and the episode probe
  3. switch to the configuration (new episode), record until the episode ends or the wall budget is spent
  4. stop estimator / meter, write run.json

    python run_nav_challenge.py --config configs/navigation/challenge_v1.yaml

Resumable: runs whose run.json says ok / timeout are skipped. Infrastructure failures (launch, warm-up, switch,
silent estimator) relaunch Isaac and retry the run once.
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
ENV = {**os.environ, "PYTHONPATH": PKG + os.pathsep + os.environ.get("PYTHONPATH", "")}
MAX_PER_LAUNCH = 40


def log(msg: str) -> None:
    print(f"[challenge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def py(args: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, *args], cwd=PKG, env=ENV, start_new_session=True, **kw)


def stop(p: subprocess.Popen | None, sig: int = 2) -> None:
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, sig)
            p.wait(20)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(p.pid, 9)
            except Exception:  # noqa: BLE001
                pass


class Sim:
    def __init__(self, controller: str, overlay: str = "socialcost_estimate"):
        self.controller = controller
        self.overlay = overlay  # name in arena_robots/config/nav2/overlays, or an absolute path
        self.proc, self.world, self.count = None, None, 0

    def stop(self) -> None:
        if self.proc is not None:
            subprocess.run(["bash", f"{SCRIPTS}/stop_isaac.sh"])
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
        self.proc, self.world, self.count = None, None, 0

    def ensure(self, cfg: dict) -> None:
        if self.proc is not None and self.proc.poll() is None and self.world == cfg["world"] and self.count < MAX_PER_LAUNCH:
            return
        self.stop()
        args = ["task.robots:=scenario", "task.obstacles:=scenario", f"task.scenario.file:={cfg['scenario']}", f"world:={cfg['world']}",
                f"run_seed:={cfg['run_seed']}", f"robot.mobile.local_planner:={self.controller}", f"robot.mobile.params_overlay:={self.overlay}"]
        log(f"launch {cfg['world']} with {cfg['scenario']}")
        self.proc = subprocess.Popen(["bash", "-l", f"{SCRIPTS}/launch_isaac.sh", f"challenge_{cfg['world']}_{cfg['config_id']}", *args],
                                     start_new_session=True)
        t0 = time.time()
        w = subprocess.run([sys.executable, "-m", "socialcost.warmup", "--timeout", "600"], capture_output=True, text=True, cwd=PKG, env=ENV)
        if w.returncode != 0:
            self.stop()
            raise RuntimeError(f"warm-up failed: {w.stderr.strip()[-300:]}")
        log(f"{cfg['world']}: up and warm after {time.time() - t0:.0f} s")
        self.world = cfg["world"]


def estimator_cmd(method: str, spec: dict, run_dir: str, extra: list | None = None, config_id: str = "") -> list[str]:
    m = spec.get("method", method)
    sub = lambda a: str(a).replace("{run_dir}", run_dir).replace("{config_id}", config_id)  # noqa: E731
    return ["-m", "socialcost.estimator_node", "--method", m, "--log", os.path.join(run_dir, "estimator.jsonl"),
            *[sub(a) for a in spec.get("args", [])], *[sub(a) for a in (extra or [])]]


def wait_lines(path: str, n: int, timeout: float, proc: subprocess.Popen) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(path) and sum(1 for _ in open(path)) >= n:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(1.0)
    return False


def episode_sim_s(path: str) -> float | None:
    """Simulated seconds from episode start to the last robot sample (None if no episode record)."""
    try:
        e = [json.loads(line) for line in open(path) if line.strip()][-1]
        return (e["robot"][-1][0] - e["start_ns"]) / 1e9 if e.get("robot") else 0.0
    except Exception:  # noqa: BLE001
        return None


def wait_for_load(max_load: float) -> None:
    """Block while the machine is overloaded (1-min load average, shared with the host)."""
    said = False
    while os.getloadavg()[0] > max_load:
        if not said:
            log(f"load {os.getloadavg()[0]:.0f} > {max_load:.0f}: waiting for the machine before the next run")
            said = True
        time.sleep(60)
    if said:
        log(f"load {os.getloadavg()[0]:.0f}: resuming")


def run_one(sim: Sim, cfg: dict, method: str, spec: dict, out_dir: str, wall_budget: float, rep: int = 0, extra: list | None = None,
            time_limit: float = 180.0) -> dict:
    tag = f"{cfg['config_id']}__{method}__{sim.controller}" + (f"__r{rep}" if rep else "")
    run_dir = os.path.join(out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("estimator.jsonl", "meter.jsonl", "episode.jsonl"):
        if os.path.exists(os.path.join(run_dir, f)):
            os.replace(os.path.join(run_dir, f), os.path.join(run_dir, f + ".prev"))
    est = meter = probe = None
    t0 = time.time()
    try:
        sim.ensure(cfg)
        est = py(estimator_cmd(method, spec, run_dir, extra, cfg["config_id"]), stdout=open(os.path.join(run_dir, "estimator.log"), "w"), stderr=subprocess.STDOUT)
        if not wait_lines(os.path.join(run_dir, "estimator.jsonl"), 3, 240, est):
            raise RuntimeError("estimator_silent before the run")
        meter = py(["-m", "socialcost.cost_meter", "--log", os.path.join(run_dir, "meter.jsonl")],
                   stdout=open(os.path.join(run_dir, "meter.log"), "w"), stderr=subprocess.STDOUT)
        sw = subprocess.run([sys.executable, "-m", "socialcost.switch", "--scenario", cfg["scenario"], "--seed", str(cfg["run_seed"])],
                            capture_output=True, text=True, cwd=PKG, env=ENV, timeout=240)
        if sw.returncode != 0:
            sim.stop()
            raise RuntimeError(f"switch failed: {sw.stderr.strip()[-300:]}")
        sim.count += 1
        ep = int(sw.stdout.strip().splitlines()[-1])
        probe = py([os.path.join(HERE, "repro_probe.py"), "--out", os.path.join(run_dir, "episode.jsonl"), "--episodes", "1",
                    "--min-episode-id", str(ep), "--timeout", str(wall_budget), "--end-on-abort"], stdout=open(os.path.join(run_dir, "probe.log"), "w"),
                   stderr=subprocess.STDOUT)
        probe.wait(timeout=wall_budget + 120)
        status = "ok" if os.path.exists(os.path.join(run_dir, "episode.jsonl")) else "timeout"
        if status == "ok":
            oc = json.loads(open(os.path.join(run_dir, "episode.jsonl")).readline()).get("outcome")
            status = {-6: "nav2_aborted", -1: "timeout"}.get(oc, "ok")
        el = os.path.join(run_dir, "estimator.jsonl")
        n_pub = sum(1 for _ in open(el)) if os.path.exists(el) else 0
        if n_pub < 20:
            status = f"failed: estimator_silent ({n_pub} grids)"
        if est.poll() is not None:
            status = f"failed: estimator exited {est.returncode}"
        # speed guard: the probe's budget is wall time, so a slow simulator (machine shared with other jobs) cuts runs
        # short; a timeout only counts if the run covered the simulated time limit, otherwise it is redone
        sim_s = episode_sim_s(os.path.join(run_dir, "episode.jsonl"))
        if status == "timeout" and (sim_s is None or sim_s < time_limit):
            status = f"failed: slow_sim ({0 if sim_s is None else sim_s:.0f} s of {time_limit:.0f} s simulated)"
    except Exception as e:  # noqa: BLE001
        status = f"failed: {e}"
    finally:
        stop(probe)
        stop(meter)
        stop(est)
    rec = {"config_id": cfg["config_id"], "kind": cfg["kind"], "world": cfg["world"], "scenario": cfg["scenario"], "run_seed": cfg["run_seed"],
           "method": method, "controller": sim.controller, "repeat": rep, "tag": tag, "status": status, "wall_s": time.time() - t0, "wall_end": time.time()}
    sim_s = episode_sim_s(os.path.join(run_dir, "episode.jsonl"))
    if sim_s:
        rec["sim_s"] = sim_s
        rec["sim_over_wall"] = sim_s / max(1.0, rec["wall_s"])
    json.dump(rec, open(os.path.join(run_dir, "run.json"), "w"), indent=2)
    log(f"{tag}: {status} ({rec['wall_s']:.0f} s)")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default="", help="comma-separated config ids")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    doc = json.load(open(cfg["challenge"]))
    confs = [{"config_id": c["config_id"], "kind": c["kind"], "world": c["world"], "run_seed": c["run_seed"], "scenario": f"scv1_nav_{c['config_id']}"}
             for c in doc["configs"]]
    only = {x for x in args.only.split(",") if x} or set(cfg.get("only", []))
    if only:
        confs = [c for c in confs if c["config_id"] in only]
    confs.sort(key=lambda c: (c["world"], c["config_id"]))  # one launch per world
    out_dir = cfg["output"]
    os.makedirs(out_dir, exist_ok=True)
    if not only and os.path.exists(os.path.join(out_dir, "HOLD")):
        log(f"HOLD file in {out_dir}: full run not started (remove the file to allow it)")
        return 0
    max_load = float(cfg.get("max_load", 24))  # 24 cores; Isaac + Nav2 + estimator alone stay well below
    for controller in cfg["controllers"]:
        sim = Sim(controller, cfg.get("overlay", "socialcost_estimate"))
        extra = cfg.get("estimator_args_all", [])  # appended to every arm's estimator (same transform for all arms)
        for c, rep in [(c, r) for r in range(cfg.get("repeats", 1)) for c in confs]:  # a full pass per repeat
            for method in cfg["methods"]:
                tag = f"{c['config_id']}__{method}__{controller}" + (f"__r{rep}" if rep else "")
                done = os.path.join(out_dir, tag, "run.json")
                if os.path.exists(done) and json.load(open(done)).get("status") in ("ok", "timeout", "nav2_aborted"):
                    continue
                spec = cfg.get("method_specs", {}).get(method, {})
                tl, wb = float(cfg.get("time_limit_s", 180)), cfg.get("wall_budget_s", 700)
                wait_for_load(max_load)
                rec = run_one(sim, c, method, spec, out_dir, wb, rep, extra, tl)
                for _ in range(3):  # slow simulator: wait for the machine, then redo (Isaac stays up)
                    if not rec["status"].startswith("failed: slow_sim"):
                        break
                    log(f"{tag}: {rec['status']}; redoing when the machine is free")
                    time.sleep(300)
                    wait_for_load(max_load)
                    rec = run_one(sim, c, method, spec, out_dir, wb, rep, extra, tl)
                    rec["retried_slow"] = True
                    json.dump(rec, open(os.path.join(out_dir, tag, "run.json"), "w"), indent=2)
                if rec["status"].startswith("failed") and not rec["status"].startswith("failed: slow_sim"):
                    log(f"{tag}: infrastructure failure, relaunching and retrying once")
                    sim.stop()
                    rec = run_one(sim, c, method, spec, out_dir, wb, rep, extra, tl)
                    rec["retried"] = True
                    json.dump(rec, open(os.path.join(out_dir, tag, "run.json"), "w"), indent=2)
                with open(os.path.join(out_dir, "manifest.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
        sim.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
