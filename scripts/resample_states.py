#!/usr/bin/env python3
"""Rebuild pedestrian_states.jsonl from the recorded peds_stream.jsonl with the current
interpolate_peds rule, then relabel. For datasets recorded before label version 1.1, whose
discrete pedestrian fields came from the nearer (possibly later) arena_peds message.

The old files are kept as pedestrian_states.v1_0.jsonl, social_cost.v1_0/, social_cost_membership.v1_0/
(once). The samples whose states or labels changed are listed in episode_metadata.json["resample_1_1"].

    python resample_states.py <dataset_root> [...]
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.dataset import label_episode, read_jsonl  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402
from socialcost.recorder import interpolate_peds  # noqa: E402

DISCRETE = ("animation_state", "interaction_id", "interaction_type")


def main() -> int:
    for root in sys.argv[1:]:
        for ep in sorted(glob.glob(os.path.join(root, "episodes", "episode_*"))):
            stream = {s["stamp_ns"]: s["pedestrians"] for s in read_jsonl(os.path.join(ep, "peds_stream.jsonl"))}
            old = read_jsonl(os.path.join(ep, "pedestrian_states.jsonl"))
            if not os.path.exists(os.path.join(ep, "pedestrian_states.v1_0.jsonl")):
                shutil.copy(os.path.join(ep, "pedestrian_states.jsonl"), os.path.join(ep, "pedestrian_states.v1_0.jsonl"))
                for ch in ("social_cost", "social_cost_membership"):
                    shutil.copytree(os.path.join(ep, ch), os.path.join(ep, ch + ".v1_0"))
            new, state_changed, missing = [], [], 0
            for rec in old:
                b, a = rec["peds_before_ns"], rec["peds_after_ns"]
                if b not in stream or (a is not None and a not in stream):
                    missing += 1
                    new.append(rec)
                    continue
                peds, sync = interpolate_peds((b, stream[b]), (a, stream[a]) if a is not None else None, rec["stamp_ns"])
                n = {"sample": rec["sample"], "stamp_ns": rec["stamp_ns"], **sync, "pedestrians": peds}
                key = lambda ps: sorted((p["name"], *(p[f] for f in DISCRETE)) for p in ps)  # noqa: E731
                if key(rec["pedestrians"]) != key(peds):
                    state_changed.append(rec["sample"])
                new.append(n)
            with open(os.path.join(ep, "pedestrian_states.jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in new))
            meta = json.load(open(os.path.join(ep, "episode_metadata.json")))
            params = LayerParams.from_mapping(meta["label"]["layer_params"])
            grid = GridSpec(**meta["label"]["grid"])
            label_episode(ep, params, grid)
            label_changed = {ch: [k for k in range(len(new)) if not np.array_equal(np.load(f"{ep}/{ch}/{k:06d}.npy"), np.load(f"{ep}/{ch}.v1_0/{k:06d}.npy"))]
                             for ch in ("social_cost", "social_cost_membership")}
            meta = json.load(open(os.path.join(ep, "episode_metadata.json")))
            meta["resample_1_1"] = {"rule": "discrete pedestrian fields from the arena_peds message in force at the image stamp",
                                    "samples_with_changed_discrete_state": state_changed, "samples_missing_stream": missing,
                                    "labels_changed": label_changed}
            json.dump(meta, open(os.path.join(ep, "episode_metadata.json"), "w"), indent=2)
            print(f"{ep}: states changed {state_changed}, labels changed { {k: v for k, v in label_changed.items()} }, missing {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
