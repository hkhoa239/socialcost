#!/usr/bin/env python3
"""socialcost_v2 retraining queue (host). Same protocol as v1 (bench_train.py, 48k samples, val every 6k, v1-selected learning
rates, seeds 0-2); outputs in runs/v2/<model>_<label>_s<seed> so v1 runs/main is never touched.

GPU is shared with other users' jobs and Isaac: a job starts only when free GPU memory >= MIN_FREE_MIB and fewer than MAX_PAR
jobs of this queue run. A failed job (GPU contention: OOM / cuDNN init) is retried up to MAX_TRIES times. Jobs already running from an
earlier instance are adopted. Checkpoint selection adds 2 x whole-grid MAE (--score-fp-weight 2). Resumable (jobs with metrics.json are skipped).
Status: STATUS_v2_train.txt.   Detached:  setsid nohup .venv/bin/python run_v2_train.py > logs/v2_train.log 2>&1 &
"""

import json
import os
import subprocess
import time
from datetime import datetime

ROOT = "/home/robotics/arena_ws/data/costmap_bench"
PY = f"{ROOT}/.venv/bin/python"
TRAIN = "/home/robotics/arena_ws/src/deps/socialcost/scripts/bench_train.py"
MAX_PAR, MIN_FREE_MIB = 2, 2600
LR = {"pos": 1e-3, "pos_u": 1e-3, "pos_int": 1e-3, "rgb": 3e-4, "depth": 3e-4, "rgbd": 3e-4, "sngnn_sensor": 5e-5, "sngnn_priv": 2e-4}
CORE = ["typemix_cffullraw", "typemix_gt", "pos", "pos_u", "rgb", "depth", "rgbd", "cascade_tracks", "cascade_vis", "typemix_tracks",
        "typemix_vis", "sngnn_sensor", "sngnn_priv"]
MEM = ["typemix_cffullrawmemhold", "typemix_cffullrawmemdecay", "typemix_cffullrawmemdecay_geo"]
NAV = ["typemix_cffullraw", "typemix_gt", "pos_u", "rgb", "rgbd", "typemix_cffullrawmemdecay_geo", "sngnn_sensor"]


def jobs():
    out = []
    for seed in (0, 1, 2):
        out += [(m, "extended_v2", seed) for m in NAV]
    for seed in (0, 1, 2):
        out += [(m, "membership_v2", seed) for m in CORE + MEM]
    for seed in (0, 1, 2):
        out += [(m, "native_v2", seed) for m in CORE]
    return out


def free_mib() -> int:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    except Exception:  # noqa: BLE001
        return 0


def st(msg: str) -> None:
    with open(f"{ROOT}/STATUS_v2_train.txt", "a") as f:
        f.write(f"{msg} {datetime.now().isoformat(timespec='seconds')}\n")


def launch(m, label, seed):
    out = f"{ROOT}/runs/v2/{m}_{label}_s{seed}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, TRAIN, "--out", out, "--model", m, "--label", label, "--seed", str(seed), "--lr", str(LR.get(m, 1e-3)), "--samples", "48000",
           "--val-every", "6000", "--batch", "16", "--workers", "6", "--score-fp-weight", "2.0"]
    open(f"{out}/cmd.txt", "w").write(" ".join(cmd) + "\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = subprocess.run(["bash", "-c", f"source {ROOT}/env.sh; echo $LD_LIBRARY_PATH"], capture_output=True, text=True).stdout.strip()
    env["PYTHONPATH"] = "/home/robotics/arena_ws/src/deps/socialcost"
    env["DGLBACKEND"] = "pytorch"
    return subprocess.Popen(cmd, stdout=open(f"{out}/stdout.log", "a"), stderr=subprocess.STDOUT, env=env)


class External:
    """A bench_train process started by an earlier instance of this queue (adopted, not restarted)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        if os.path.exists(f"/proc/{self.pid}"):
            return None
        self.returncode = 0
        return 0


def live_jobs() -> dict:
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            a = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(x.endswith(b"bench_train.py") for x in a) and b"--out" in a:
            run = os.path.basename(a[a.index(b"--out") + 1].decode())
            m, rest = run.rsplit("_s", 1)
            for lab in ("extended_v2", "membership_v2", "native_v2"):
                if m.endswith("_" + lab):
                    j = (m[: -len(lab) - 1], lab, int(rest))
                    out.setdefault(j, External(int(d)))
    return out


MAX_TRIES = 4


def main():
    todo = [j for j in jobs() if not os.path.exists(f"{ROOT}/runs/v2/{j[0]}_{j[1]}_s{j[2]}/metrics.json")]
    running = live_jobs()
    todo = [j for j in todo if j not in running]
    st(f"BEGIN {len(todo)} jobs, adopted {len(running)} running")
    tries = {j: 1 for j in running}
    retried = set()
    while todo or running:
        for j, p in list(running.items()):
            if p.poll() is not None:
                ok = os.path.exists(f"{ROOT}/runs/v2/{j[0]}_{j[1]}_s{j[2]}/metrics.json")
                st(f"EXIT {p.returncode} {'ok' if ok else 'FAILED'} {j[0]} {j[1]} s{j[2]}")
                del running[j]
                if not ok and tries.get(j, 1) < MAX_TRIES:
                    retried.add(j)
                    todo.append(j)
        if todo and len(running) < MAX_PAR and free_mib() >= MIN_FREE_MIB:
            j = todo.pop(0)
            tries[j] = tries.get(j, 0) + 1
            running[j] = launch(*j)
            st(f"START {j[0]} {j[1]} s{j[2]} (free {free_mib()} MiB, running {len(running)})")
            time.sleep(90)  # let it allocate before judging free memory again
            continue
        time.sleep(30)
    st("DONE")


if __name__ == "__main__":
    main()
