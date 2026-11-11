"""socialcost_v2 labels: revised interaction geometry on top of the unchanged Arena social layer math.

The v1 labels (arena-native/1.1, extended-membership/1.1, extended/2.1) are not touched. v2 is painted by the same
C implementation as v1 and as the Nav2 layer (socialcost_grid.cpp -> arena_social_cost_layer/cost_math.hpp), so every
v2 cell is exactly what SocialCostLayer would paint with the v2 parameters; only two things change:

  1. conversation / interaction capsule: clip radius group_extra_radius 2.0 m -> 1.2 m, and the capsule's super-Gaussian
     sigma either kept (0.85, "fixed") or scaled with the radius (0.85 x 1.2 / 2.0 = 0.51, "proportional")
  2. queues (interaction_type 6, QUEUE_USE): optional order-aware corridor. v1 paints a capsule for every pair of queue
     members (all pairs), which joins the front and the back of a bent or long line across the floor. The corridor
     paints a capsule only between neighbouring members, the edges of the Euclidean minimum spanning tree over the
     members' positions (for a line of people this is the chain front-to-back; humansim's member_ids order is not the
     spatial order: in dataset_v1 it gives consecutive gaps > 1.5 m in 31 % of queue frames). Implemented by handing each
     tree edge to the C painter as its own two-person interaction of the same type and weight: the painter's per-cell
     maximum makes the duplicated personal-space terms a no-op, so the result is the layer's capsule term restricted to
     the tree edges.

Unchanged on purpose: extended/2.0 composition incl. the wheelchair hypothesis (as v1 social_cost_extended), personal space (radius 1.0 m, sigma, amplitudes per animation state), falloff_shape 3, cost_scale
0.8, cost_bias 0.05, max_cost_value 252, interaction weights (so group walking stays at weight 0 and no wheelchair term),
membership completion, halt-gesture/1.0 wedge, per-cell max compositing, uint8 floor((v*scale+bias) clipped x 252).

Groups of 3-4 conversing people stay pairwise (all pairs, as the layer): with dataset_v1 spacing (pairwise 1.5-2.3 m) the
union of pair capsules of radius 1.2 m covers the group's interior (validated in scripts/labels_v2_validate.py), so no
group-level definition is introduced.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np

from .dataset import EXTENDED_CONFIG, peds_for_layer, with_membership
from .extensions import ExtParams, halt_term, is_mobility_aid
from .labels import GridSpec, LayerParams, Ped, paint

QUEUE_TYPE = 6
TALK_TYPES = (0, 1)
V1_GROUP_RADIUS, V1_GROUP_SIGMA = 2.0, 0.85


@dataclasses.dataclass(frozen=True)
class SpecV2:
    name: str
    version: str
    group_extra_radius: float
    group_sigma: float
    queue_mode: str  # "all_pairs" (v1 behaviour) | "mst" (neighbour corridor)
    note: str = ""

    def layer_params(self, base: LayerParams) -> LayerParams:
        return dataclasses.replace(base, group_extra_radius=self.group_extra_radius, group_sigma=self.group_sigma)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


PROP_SIGMA = round(V1_GROUP_SIGMA * 1.2 / V1_GROUP_RADIUS, 4)  # 0.51
SPECS = {
    # final v2 label
    "v2": SpecV2("v2", "socialcost_v2/2.0", 1.2, PROP_SIGMA, "mst",
                 "radius 1.2 m, sigma scaled with the radius, queue neighbour corridor"),
    # geometry variants (documented comparison / ablation)
    "v2_fixedsigma": SpecV2("v2_fixedsigma", "socialcost_v2-fixedsigma/2.0", 1.2, V1_GROUP_SIGMA, "mst",
                            "radius 1.2 m, v1 sigma 0.85: the capsule is cut at 1.2 m while still at ~37 % of its peak"),
    "v2_allpairs": SpecV2("v2_allpairs", "socialcost_v2-allpairsqueue/2.0", 1.2, PROP_SIGMA, "all_pairs",
                          "as v2 with the v1 all-pairs queue (queue ablation)"),
    # v1 reproduced through this module (equivalence check only)
    "v1": SpecV2("v1", "arena-native/1.1", V1_GROUP_RADIUS, V1_GROUP_SIGMA, "all_pairs", "v1 parameters, for equivalence tests"),
}


def mst_edges(xy: np.ndarray) -> list[tuple[int, int]]:
    """Euclidean minimum spanning tree (Prim), edges as index pairs; deterministic (ties by index)."""
    n = len(xy)
    if n < 2:
        return []
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    best = d[0].copy()
    parent = np.zeros(n, int)
    edges = []
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        j = int(np.argmin(cand))
        edges.append((min(parent[j], j), max(parent[j], j)))
        in_tree[j] = True
        closer = d[j] < best
        best = np.where(closer, d[j], best)
        parent = np.where(closer, j, parent)
    return edges


def expand_queue_groups(peds: list[Ped], spec: SpecV2) -> list[Ped]:
    """Queue groups -> one two-person interaction per neighbour edge (mst mode); other groups untouched."""
    if spec.queue_mode == "all_pairs":
        return peds
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(peds):
        if p.interaction_id >= 0:
            groups.setdefault(p.interaction_id, []).append(i)
    out = []
    queue_ids = set()
    for gid, idx in groups.items():
        # the layer takes the group's type from its first member in roster order
        if peds[idx[0]].interaction_type == QUEUE_TYPE and len(idx) > 2:
            queue_ids.add(gid)
    next_id = max([p.interaction_id for p in peds] + [0]) + 1
    for p in peds:
        if p.interaction_id in queue_ids:
            out.append(Ped(p.x, p.y, p.animation_state))  # personal space only; corridor edges below
        else:
            out.append(p)
    for gid in sorted(queue_ids):
        idx = groups[gid]
        xy = np.array([[peds[i].x, peds[i].y] for i in idx])
        for a, b in mst_edges(xy):
            pa, pb = peds[idx[a]], peds[idx[b]]
            out.append(Ped(pa.x, pa.y, pa.animation_state, next_id, QUEUE_TYPE))
            out.append(Ped(pb.x, pb.y, pb.animation_state, next_id, QUEUE_TYPE))
            next_id += 1
    return out


def paint_v2(spec: SpecV2, base: LayerParams, peds: list[Ped], grid: GridSpec, pose) -> np.ndarray:
    return paint(spec.layer_params(base), expand_queue_groups(list(peds), spec), grid, pose)


def _ext20() -> ExtParams:
    """extended/2.0 (configs/labels/extended_v2.yaml): what dataset_v1's social_cost_extended, the navigation models and the
    oracle_ext arm use: membership + halt wedge + mobility-aid hypothesis (wheelchair personal space x1.5)."""
    return ExtParams.load(EXTENDED_CONFIG)


def labels_and_components(spec: SpecV2, base: LayerParams, grid: GridSpec, states: list[dict], frame: dict | None, pose) -> dict[str, np.ndarray]:
    """native / membership / extended labels and the benchmark's component maps, for one sample."""
    P = spec.layer_params(base)
    no_personal = dataclasses.replace(P, personal_space_radius=1e-6)
    full = with_membership(states, frame)
    native = peds_for_layer(states)
    member = peds_for_layer(full)
    gesture = np.zeros((grid.height, grid.width), np.uint8)
    ext = _ext20()
    for p in states:
        gesture = np.maximum(gesture, halt_term(base, ext, p, grid, pose))
    mob = dataclasses.replace(base, personal_space_radius=base.personal_space_radius * ext.mobility_radius_scale,
                              personal_space_sigma=base.personal_space_sigma * ext.mobility_sigma_scale)
    aids = [Ped(p["x"], p["y"], p["animation_state"]) for p in full if is_mobility_aid(p, ext)]
    mobility = paint(mob, aids, grid, pose) if (aids and ext.mobility) else np.zeros_like(gesture)
    talk = [p for p in native if p.interaction_id >= 0 and p.interaction_type != QUEUE_TYPE]
    queue = [p for p in member if p.interaction_id >= 0 and p.interaction_type == QUEUE_TYPE]
    out = {
        "native": paint_v2(spec, base, native, grid, pose),
        "membership": paint_v2(spec, base, member, grid, pose),
        "personal_space": paint(P, [Ped(p.x, p.y, p.animation_state) for p in native], grid, pose),
        "conversation": paint_v2(spec, no_personal, talk, grid, pose) if talk else np.zeros_like(gesture),
        "queue": paint_v2(spec, no_personal, queue, grid, pose) if queue else np.zeros_like(gesture),
        "gesture": gesture,
    }
    out["mobility"] = mobility
    out["extended"] = np.maximum(np.maximum(out["membership"], gesture), mobility)  # extended/2.0 composition
    return out


def manifest(spec: SpecV2, base: LayerParams, grid: GridSpec) -> dict:
    P = spec.layer_params(base)
    return {
        "label_version": spec.version, "spec": spec.to_dict(), "layer_params": P.to_dict(), "grid": grid.to_dict(),
        "equations": {
            "cell_value": "uint8(floor(clip(v * cost_scale + cost_bias, 0, 1) * max_cost_value)), cells outside every term = 0",
            "personal_space": "v = A(state) * exp(-((d^2) / (2 sigma(state)^2))^falloff_shape) for d <= personal_space_radius; "
                              "sigma(base/alert/danger) = (0.45, 0.75, 1.2) * personal_space_sigma / 0.45, A = (0.5, 0.75, 1.0)",
            "interaction_capsule": "for each painted pair (a, b): v = w(type) * exp(-((dseg^2) / (2 group_sigma^2))^falloff_shape) for "
                                   "dseg <= group_extra_radius, dseg = distance to segment ab; w = interaction_weights[type of the group's first member]",
            "pairs": "every pair inside one interaction_id; queues (type 6, > 2 members) in mst mode: only Euclidean minimum-spanning-tree edges",
            "membership": "queued members of an ACTIVE InteractionFrame interaction take its id/type (with_membership), as extended-membership/1.1",
            "extended": "max(membership label, halt-gesture/1.0 wedge, mobility-aid hypothesis: wheelchair personal space radius/sigma x1.5) (extended/2.0 composition, as dataset_v1 social_cost_extended)",
            "compositing": "per-cell maximum over all terms",
        },
        "unchanged_from_v1": ["personal space", "falloff_shape", "cost_scale", "cost_bias", "max_cost_value", "interaction_weights",
                              "halt-gesture/1.0", "membership completion", "grid", "splits", "counterfactual pairs"],
        "implementation": "socialcost/labels_v2.py -> labels.paint (libsocialcost_grid.so, cost_math.hpp shared with the Nav2 SocialCostLayer)",
    }


def dumps_manifest(spec: SpecV2, base: LayerParams, grid: GridSpec) -> str:
    return json.dumps(manifest(spec, base, grid), indent=2)


__all__ = ["SPECS", "SpecV2", "mst_edges", "expand_queue_groups", "paint_v2", "labels_and_components", "manifest", "QUEUE_TYPE",
           "PROP_SIGMA", "math"]
