#!/usr/bin/env python3
"""Stage D: checksum and provenance report, then mark dataset_v1 immutable.

    python dv1_freeze.py --root /opt/arena_ws/data/socialcost/dataset_v1 [--no-lock]

Writes metadata/checksums.jsonl (sha256 of every file under episodes/, manifests/, splits/, metadata/ except itself),
metadata/provenance.json (dataset digest, code and world hashes, software versions, label versions, collection
statistics) and manifests/DATASET_FROZEN, then removes write permission from episodes/, manifests/ and splits/.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from socialcost.dv1 import spec  # noqa: E402
from socialcost.dv1.geometry import WORLDS_DIR  # noqa: E402

ARENA = os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(path: str) -> str | None:
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args()
    root = args.root
    out_ck = os.path.join(root, "metadata", "checksums.jsonl")
    digest = hashlib.sha256()
    n_files, n_bytes = 0, 0
    with open(out_ck, "w") as f:
        for top in ("manifests", "splits", "metadata", "episodes"):
            for base, dirs, files in os.walk(os.path.join(root, top)):
                dirs.sort()
                for name in sorted(files):
                    p = os.path.join(base, name)
                    if p == out_ck or name.endswith("provenance.json"):
                        continue
                    rel = os.path.relpath(p, root)
                    h = sha(p)
                    size = os.path.getsize(p)
                    f.write(json.dumps({"path": rel, "sha256": h, "bytes": size}) + "\n")
                    digest.update(f"{rel}\0{h}\n".encode())
                    n_files += 1
                    n_bytes += size
    code = {}
    for base in (os.path.join(PKG, "socialcost"), os.path.join(PKG, "scripts"), os.path.join(PKG, "configs"), os.path.join(PKG, "src")):
        for b, _d, files in os.walk(base):
            for name in sorted(files):
                if name.endswith((".py", ".yaml", ".cpp", ".json")) and "__pycache__" not in b:
                    p = os.path.join(b, name)
                    code[os.path.relpath(p, PKG)] = sha(p)
    worlds = {v["name"]: {"world.yaml": sha(os.path.join(WORLDS_DIR, v["name"], "world.yaml")),
                          "map": sha(os.path.join(WORLDS_DIR, v["name"], "map", "0.png"))} for v in spec.WORLDS.values()}
    eps = sorted(os.listdir(os.path.join(root, "episodes")))
    done = [e for e in eps if os.path.exists(os.path.join(root, "episodes", e, "DONE"))]
    prov = {
        "dataset": spec.DATASET, "spec_version": spec.SPEC_VERSION, "frozen": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": digest.hexdigest(), "files": n_files, "bytes": n_bytes, "episodes_recorded": len(done),
        "manifest_sha256": open(os.path.join(root, "manifests", "manifest_v1.sha256")).read().split()[0],
        "label_versions": spec.LABELS, "worlds": worlds,
        "software": {"arena_commit": git_rev(ARENA), "arena_isaac_commit": git_rev(os.path.join(ARENA, "arena_isaac")),
                     "humansim_commit": git_rev(os.path.join(ARENA, "humansim")), "isaac_sim_image": "nvcr.io/nvidia/isaac-sim:6.1.0",
                     "ros_distro": os.environ.get("ROS_DISTRO"), "physics": "PhysX, 60 Hz, render every step"},
        "code_sha256": code,
        "collection": "persistent mode (scripts/dv1_collect.py): one Isaac launch per world group, task.scenario.file + seeded reset_episode per episode; "
                      "scenario compiler v1.1 (staggered seek for approach-and-stop conversations)",
    }
    json.dump(prov, open(os.path.join(root, "metadata", "provenance.json"), "w"), indent=1)
    with open(os.path.join(root, "manifests", "DATASET_FROZEN"), "w") as f:
        f.write(f"frozen {prov['frozen']}\ndataset_sha256 {prov['dataset_sha256']}\nfiles {n_files}\nepisodes {len(done)}\n")
    if not args.no_lock:
        for top in ("episodes", "manifests", "splits"):
            for b, dirs, files in os.walk(os.path.join(root, top)):
                for name in files + dirs:
                    p = os.path.join(b, name)
                    os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({k: prov[k] for k in ("dataset_sha256", "files", "bytes", "episodes_recorded", "manifest_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
