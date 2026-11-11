#!/usr/bin/env python3
"""Write <root>/labels.json and each episode's label_definitions for an existing dataset.

Metadata only: label arrays are not touched. Refuses to proceed if an episode's recorded label
parameters differ from the dataset's.

    python label_manifest.py <dataset_root> [...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.dataset import label_definitions  # noqa: E402
from socialcost.labels import GridSpec, LayerParams  # noqa: E402


def main() -> int:
    for root in sys.argv[1:]:
        meta = json.load(open(os.path.join(root, "metadata.json")))
        params = LayerParams.from_mapping(meta["label_params"])
        grid = GridSpec(**meta["label_grid"])
        defs = label_definitions(params, grid, sources=meta.get("software"))
        tags = {k: v["name"] + "/" + v["version"] for k, v in defs.items() if k != "sources"}
        for ep in sorted(glob.glob(os.path.join(root, "episodes", "episode_*"))):
            mp = os.path.join(ep, "episode_metadata.json")
            m = json.load(open(mp))
            if LayerParams.from_mapping(m["label"]["layer_params"]) != params:
                raise SystemExit(f"{ep}: label parameters differ from the dataset's")
            m["label_definitions"] = tags
            json.dump(m, open(mp, "w"), indent=2)
        json.dump(defs, open(os.path.join(root, "labels.json"), "w"), indent=2)
        print(f"{root}: {tags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
