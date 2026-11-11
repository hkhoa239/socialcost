"""Offline labelling, dataset metadata, splits and the sample loader."""

from __future__ import annotations

import bisect
import glob
import json
import os
from collections.abc import Iterator

import cv2
import numpy as np
import yaml

from .labels import GridSpec, LayerParams, Ped, paint

FORMAT_VERSION = "socialcost-dataset/1"
EXTENDED_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs", "labels", "extended_v2.yaml")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def peds_for_layer(states: list[dict]) -> list[Ped]:
    return [Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in states]


def with_membership(states: list[dict], frame: dict | None) -> list[dict]:
    """Pedestrian states with interaction membership completed from humansim's InteractionFrame.

    humansim publishes Pedestrian.interaction_id only for ACTIVE participants
    (InteractionManager.active_interaction), so people waiting in a QUEUE_USE line carry -1 and
    the layer never pairs them. Here every member of an active interaction (participants and
    queued) takes that interaction's id and type, which lets the layer's own group term paint
    the queue with its QUEUE_USE weight. People already in an active interaction keep theirs.
    """
    if not frame:
        return states
    member_of: dict[int, tuple[int, int]] = {}
    for it in frame["interactions"]:
        if it["outcome"] != 1:  # ACTIVE
            continue
        for m in it["member_ids"]:
            member_of.setdefault(m, (it["interaction_id"], it["interaction_type"]))
    out = []
    for p in states:
        q = dict(p)
        if q["interaction_id"] < 0 and q["id"] in member_of:
            q["interaction_id"], q["interaction_type"] = member_of[q["id"]]
        out.append(q)
    return out


def interaction_frame_at(frames: list[dict], t_ns: int) -> dict | None:
    k = bisect.bisect_right([f["stamp_ns"] for f in frames], t_ns) - 1
    return frames[k] if k >= 0 else None


def label_episode(ep_dir: str, params: LayerParams, grid: GridSpec) -> dict:
    """Writes the labels of every recorded sample, returns label stats.

    social_cost/<sample>.npy is the Arena social layer alone, painted on `grid` placed at the
    robot's base pose (x, y, yaw) in the map frame at the image stamp, from the pedestrian
    states interpolated to that stamp: exactly what the live layer paints for those states.
    social_cost_membership/<sample>.npy is the same layer with interaction membership completed
    from the InteractionFrame in force at the stamp (with_membership), which adds the queue term.
    """
    robots = {r["sample"]: r for r in read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))}
    peds = {p["sample"]: p for p in read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))}
    frames = read_jsonl(os.path.join(ep_dir, "interactions.jsonl"))
    from .extensions import ExtParams, definition, paint_extended
    ext = ExtParams.load(EXTENDED_CONFIG)
    for d in ("social_cost", "social_cost_membership", "social_cost_extended"):
        os.makedirs(os.path.join(ep_dir, d), exist_ok=True)
    fired_total = {"membership": 0, "halt": 0, "mobility": 0}
    nonzero, differs = [], 0
    for k in sorted(robots):
        pose = robots[k]["pose_map_base"]
        gp = (pose["x"], pose["y"], pose["yaw"])
        states = peds[k]["pedestrians"]
        label = paint(params, peds_for_layer(states), grid, gp)
        full = paint(params, peds_for_layer(with_membership(states, interaction_frame_at(frames, robots[k]["stamp_ns"]))), grid, gp)
        np.save(os.path.join(ep_dir, "social_cost", f"{k:06d}.npy"), label)
        np.save(os.path.join(ep_dir, "social_cost_membership", f"{k:06d}.npy"), full)
        extended, fired = paint_extended(params, ext, states, interaction_frame_at(frames, robots[k]["stamp_ns"]), grid, gp)
        np.save(os.path.join(ep_dir, "social_cost_extended", f"{k:06d}.npy"), extended)
        for key, v in fired.items():
            fired_total[key] += int(v > 0)
        nonzero.append(int((label > 0).sum()))
        differs += int((full != label).any())
    stats = {"labels": len(nonzero), "nonzero_cells_mean": float(np.mean(nonzero)) if nonzero else 0.0,
             "labels_with_cost": int(sum(n > 0 for n in nonzero)), "membership_labels_differing": differs}
    meta_path = os.path.join(ep_dir, "episode_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    stats["extended_samples_with_extension"] = fired_total
    meta["label"] = {"layer_params": params.to_dict(), "grid": grid.to_dict(), **stats, "extended": definition(ext)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return stats


def live_agreement(ep_dir: str, params: LayerParams) -> list[dict]:
    """Per sample: the recorded live social costmap against the offline painter on the same grid.

    The offline grid uses pedestrians interpolated at the live costmap's stamp, so what is left
    is the layer's own data latency (it paints the newest arena_peds it has, not the stamp's).
    """
    stream = read_jsonl(os.path.join(ep_dir, "peds_stream.jsonl"))
    stamps = [s["stamp_ns"] for s in stream]
    out = []
    for rec in read_jsonl(os.path.join(ep_dir, "sync.jsonl")):
        lc = rec.get("live_costmap")
        if lc is None or not stamps:
            continue
        live = np.load(os.path.join(ep_dir, "live_social_cost", f"{rec['sample']:06d}.npy"))
        t = lc["stamp_ns"]
        k = bisect.bisect_right(stamps, t) - 1
        if k < 0:
            continue
        before = (stamps[k], stream[k]["pedestrians"])
        after = (stamps[k + 1], stream[k + 1]["pedestrians"]) if k + 1 < len(stamps) else None
        from .recorder import interpolate_peds  # ROS import, only needed for live agreement

        peds, _ = interpolate_peds(before, after, t)
        # nav2_msgs/Costmap carries resolution as float32; the layer worked with the configured double.
        grid = GridSpec(resolution=round(lc["resolution"], 6), width=lc["size_x"], height=lc["size_y"], origin_x=lc["origin_x"], origin_y=lc["origin_y"], frame=lc["frame_id"])
        off = paint(params, peds_for_layer(peds), grid)
        diff = np.abs(live.astype(int) - off.astype(int))
        a, b = live > 0, off > 0
        union = int((a | b).sum())
        out.append({"sample": rec["sample"], "exact_fraction": float((diff == 0).mean()), "mae": float(diff.mean()),
                    "max_abs": int(diff.max()), "iou_nonzero": float((a & b).sum() / union) if union else 1.0,
                    "live_minus_image_stamp_s": (t - rec["stamp_ns"]) / 1e9})
    return out


def load_sample(ep_dir: str, k: int, with_privileged: bool = False) -> dict:
    """One training sample. Pedestrian states are privileged (labels, debugging, baselines) and
    are only returned when asked for; they are not estimator inputs."""
    name = f"{k:06d}"
    with open(os.path.join(ep_dir, "camera_info.json")) as f:
        cam = json.load(f)
    robot = read_jsonl(os.path.join(ep_dir, "robot_states.jsonl"))[k]
    sample = {
        "rgb": cv2.imread(os.path.join(ep_dir, "rgb", name + ".png"))[:, :, ::-1],
        "depth": np.load(os.path.join(ep_dir, "depth", name + ".npy")),
        "social_cost": np.load(os.path.join(ep_dir, "social_cost", name + ".npy")),
        "social_cost_membership": np.load(os.path.join(ep_dir, "social_cost_membership", name + ".npy")),
        **({"social_cost_extended": np.load(os.path.join(ep_dir, "social_cost_extended", name + ".npy"))} if os.path.exists(os.path.join(ep_dir, "social_cost_extended", name + ".npy")) else {}),
        "robot_pose": robot["pose_map_base"],
        "camera_intrinsics": np.array(cam["K"]).reshape(3, 3),
        "camera_extrinsics": {"T_base_camera": cam.get("T_base_camera"), "T_map_camera": robot["pose_map_camera"]},
        "timestamp": robot["stamp_ns"] / 1e9,
    }
    if with_privileged:
        sample["pedestrian_states"] = read_jsonl(os.path.join(ep_dir, "pedestrian_states.jsonl"))[k]["pedestrians"]
    return sample


class SocialCostDataset:
    """Iterates samples of the episodes listed in one split file."""

    def __init__(self, root: str, split: str, with_privileged: bool = False) -> None:
        self.root = root
        with open(os.path.join(root, "splits", f"{split}.txt")) as f:
            episodes = [line.strip() for line in f if line.strip()]
        self.index = []
        for ep in episodes:
            n = len(glob.glob(os.path.join(root, "episodes", ep, "social_cost", "*.npy")))
            self.index += [(ep, k) for k in range(n)]
        self.with_privileged = with_privileged

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ep, k = self.index[i]
        return load_sample(os.path.join(self.root, "episodes", ep), k, self.with_privileged)

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]


def make_splits(root: str, rules: dict) -> dict[str, list[str]]:
    """Splits by whole episodes, grouped so that one (scenario, world, seed) never spans splits.

    rules: {"test": {"worlds": [...], "bodies": [...], "seeds": [...]}, "val": {...}}; an episode
    goes to the first split whose rule it matches (any listed world, any pedestrian body or any
    run seed), else to train. Matching on run seed keeps correlated trajectories together.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ep_dir in sorted(glob.glob(os.path.join(root, "episodes", "episode_*"))):
        with open(os.path.join(ep_dir, "episode_metadata.json")) as f:
            meta = json.load(f)
        if not meta.get("valid", True):
            continue
        bodies = set(meta.get("bodies", []))
        target = "train"
        for split in ("test", "val"):
            r = rules.get(split) or {}
            if (meta.get("world") in set(r.get("worlds", [])) or bodies & set(r.get("bodies", [])) or meta.get("run_seed") in set(map(str, r.get("seeds", [])))
                    or meta.get("body_set") in set(r.get("body_sets", [])) or meta.get("pair_id") in set(r.get("pair_ids", []))):
                target = split
                break
        splits[target].append(os.path.basename(ep_dir))
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    for name, eps in splits.items():
        with open(os.path.join(root, "splits", f"{name}.txt"), "w") as f:
            f.write("".join(e + "\n" for e in eps))
    return splits


def scenario_bodies(world_dir: str, scenario: str) -> list[str]:
    with open(os.path.join(world_dir, "scenarios", scenario, "scenario.yaml")) as f:
        doc = yaml.safe_load(f)
    return sorted({h["model"] for h in doc.get("humans", [])})


def label_definitions(params: LayerParams, grid: GridSpec, sources: dict | None = None) -> dict:
    """The two label channels, versioned. Written to <root>/labels.json and every episode's metadata."""
    common = {
        "dtype": "uint8", "range": [0, params.max_cost_value], "zero": "no social cost",
        "grid": {**grid.to_dict(), "frame": "base_link at the image stamp, x forward, y left", "indexing": "label[j, i], cell centre (origin_x + (i+0.5) res, origin_y + (j+0.5) res)"},
        "time_reference": "pedestrian positions, velocities and yaw interpolated to the RGB-D image stamp; roster and discrete fields (animation_state, interaction_id, interaction_type) from the arena_peds message in force at the stamp (1.1; 1.0 took them from the nearer, possibly later, message); robot pose = TF map->base_link at that stamp",
        "parameters": {"source": "arena_robots/config/nav2/overlays/social_cost_layer.yaml, block local_costmap", **params.to_dict()},
        "terms": {
            "personal_space": "per pedestrian, cells with centre distance d <= personal_space_radius: A * exp(-(d^2 / (2 sigma^2))^falloff_shape); "
                              "sigma, A by animation_state: default (personal_space_sigma, 0.5), SURPRISED/CURIOUS (0.75/0.45 sigma, 0.75), PANIC/THREATENING (1.2/0.45 sigma, 1.0); isotropic, heading unused",
            "interaction": "per pair of pedestrians sharing interaction_id >= 0, cells within group_extra_radius of the segment between them: "
                           "w * exp(-(d_seg^2 / (2 group_sigma^2))^falloff_shape), w = interaction_weights[type of the group's first member]; w = 0 paints nothing",
            "cell_value": "floor(clamp(v * cost_scale + cost_bias, 0, 1) * max_cost_value)",
            "combination": "per-cell maximum over all terms, starting from 0",
        },
        "implementation": "libsocialcost_grid.so over arena_social_cost_layer/cost_math.hpp (socialcost/labels.py)",
    }
    return {
        "social_cost": {"name": "arena-native", "version": "1.1", **common,
                        "inputs": "arena_people_msgs/Pedestrian x, y, animation_state, interaction_id, interaction_type exactly as published "
                                  "(humansim publishes interaction_id only for ACTIVE participants)",
                        "definition": "exact output of arena_social_cost_layer for the published pedestrian states"},
        "social_cost_membership": {"name": "extended-membership", "version": "1.1", **common,
                                   "inputs": "as arena-native, plus humansim InteractionFrame (interactions.jsonl) in force at the image stamp",
                                   "definition": "arena-native cost functions, weights and operator unchanged; every member of an ACTIVE interaction "
                                                 "(participant or QUEUED) takes that interaction's id and type when its published interaction_id is -1",
                                   "extensions": [{"id": "membership-completion", "version": "1.0",
                                                   "effect": "queued members join their interaction, so the interaction term of that kind (QUEUE_USE 1.0; "
                                                             "also USE/SIT_ON/LIE_ON/SERVICE queues if present) is painted between all member pairs"}],
                                   "relation_to_native": "extended >= native cell-wise; identical when no interaction has queued members"},
        "social_cost_extended": {**common, **_extended_definition(),
                                 "inputs": "as extended-membership, plus Pedestrian gestures (halt slots) and agent_type/model_uri (mobility aid)"},
        "sources": sources or {},
    }


def _extended_definition() -> dict:
    from .extensions import ExtParams, definition
    return definition(ExtParams.load(EXTENDED_CONFIG))
