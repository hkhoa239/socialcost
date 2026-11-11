"""Body bounding boxes and heights from the dataset_v1 asset manifest."""

from __future__ import annotations

import json
import os


def load(root: str) -> tuple[dict, dict]:
    m = json.load(open(os.path.join(root, "metadata", "asset_manifest.json")))
    bboxes = {b["body_id"]: b["bounding_box"][:2] for b in m["bodies"] if b.get("bounding_box")}
    heights = {b["body_id"]: b["size_m"]["height"] for b in m["bodies"] if b.get("size_m")}
    return bboxes, heights
