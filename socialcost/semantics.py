"""Semantic analysis of the social cost labels: decomposition, counterfactuals, queue geometry,
group walking, temporal change attribution and live-layer data age.

All paints go through labels.paint (the layer's own cost functions). The decomposition uses the
layer's combination operator, a per-cell maximum, so native = max(personal-only, interaction-only)
holds exactly and is checked.
"""

from __future__ import annotations

import bisect
import dataclasses
import math

import numpy as np

from .dataset import interaction_frame_at, peds_for_layer, read_jsonl
from .labels import GridSpec, LayerParams, Ped, paint
from .reference import to_cell_cost
from .verify import Ep

NATIVE, EXTENDED = "social_cost", "social_cost_membership"


def strip_interactions(peds: list[Ped]) -> list[Ped]:
    return [Ped(p.x, p.y, p.animation_state) for p in peds]


def interaction_only_params(params: LayerParams) -> LayerParams:
    """Personal-space clip radius 0: the layer paints no personal-space cell, only interaction terms."""
    return dataclasses.replace(params, personal_space_radius=0.0)


def components(params: LayerParams, peds: list[Ped], grid: GridSpec, pose: tuple[float, float, float]) -> dict[str, np.ndarray]:
    personal = paint(params, strip_interactions(peds), grid, pose)
    interaction = paint(interaction_only_params(params), peds, grid, pose)
    combined = paint(params, peds, grid, pose)
    return {"personal": personal, "interaction": interaction, "combined": combined}


# ------------------------------------------------------------------ decomposition over recorded data
def decomposition(ep: Ep) -> dict:
    n = len(ep.robots)
    max_identity_bad = 0
    stored_bad = 0
    stats = {ch: {"samples_with_interaction_cost": 0, "interaction_cells": [], "interaction_only_cells": [], "personal_cells": []} for ch in (NATIVE, EXTENDED)}
    for k in range(n):
        pose = ep.pose(k)
        for ch, membership in ((NATIVE, False), (EXTENDED, True)):
            c = components(ep.params, peds_for_layer(ep.states(k, membership)), ep.grid, pose)
            if not np.array_equal(np.maximum(c["personal"], c["interaction"]), c["combined"]):
                max_identity_bad += 1
            if not np.array_equal(c["combined"], ep.label(k, ch)):
                stored_bad += 1
            raised = c["combined"] > c["personal"]  # cells where the interaction term wins the max
            s = stats[ch]
            s["samples_with_interaction_cost"] += int(raised.any())
            s["interaction_cells"].append(int((c["interaction"] > 0).sum()))
            s["interaction_only_cells"].append(int(raised.sum()))
            s["personal_cells"].append(int((c["personal"] > 0).sum()))
    for s in stats.values():
        for key in ("interaction_cells", "interaction_only_cells", "personal_cells"):
            v = s.pop(key)
            s[key + "_mean"] = float(np.mean(v)) if v else 0.0
            s[key + "_max"] = int(np.max(v)) if v else 0
    diff = [int((ep.label(k, EXTENDED) != ep.label(k, NATIVE)).sum()) for k in range(n)]
    below = sum(int((ep.label(k, EXTENDED) < ep.label(k, NATIVE)).sum()) for k in range(n))
    return {"episode": ep.name, "scenario": ep.scenario, "samples": n, "max_identity_violations": max_identity_bad,
            "stored_label_mismatches": stored_bad, "channels": stats,
            "extended_minus_native": {"samples_differing": int(sum(d > 0 for d in diff)), "cells_differing_mean": float(np.mean(diff)) if diff else 0.0,
                                      "cells_differing_max": int(max(diff, default=0)), "cells_where_extended_below_native": below}}


# ------------------------------------------------------------------ ON/OFF counterfactual
def segment_mask(grid: GridSpec, pose, a: tuple[float, float], b: tuple[float, float], halfwidth: float) -> np.ndarray:
    from .reference import cell_centres
    wx, wy = cell_centres(grid, pose)
    abx, aby = b[0] - a[0], b[1] - a[1]
    ab2 = abx * abx + aby * aby
    t = np.clip(((wx - a[0]) * abx + (wy - a[1]) * aby) / ab2, 0, 1) if ab2 > 1e-9 else 0.0
    return (wx - (a[0] + t * abx)) ** 2 + (wy - (a[1] + t * aby)) ** 2 <= halfwidth ** 2


def conversation_counterfactual(ep: Ep) -> dict:
    """Same states, robot pose, grid and parameters, with and without the TALK_TO interaction."""
    rows = []
    for k in range(len(ep.robots)):
        ps = {p["name"].split("/")[-1]: p for p in ep.states(k)}
        a, b = ps.get("talker_a"), ps.get("talker_b")
        if a is None or b is None:
            continue
        pose = ep.pose(k)
        peds = peds_for_layer(ep.states(k))
        on = paint(ep.params, peds, ep.grid, pose)
        off = paint(ep.params, strip_interactions(peds), ep.grid, pose)
        if not on.any() and not off.any():
            continue
        seg = segment_mask(ep.grid, pose, (a["x"], a["y"]), (b["x"], b["y"]), 0.1)
        d = on.astype(int) - off.astype(int)
        rows.append({"sample": k, "active": a["interaction_id"] >= 0 and a["interaction_id"] == b["interaction_id"],
                     "cells_raised": int((d > 0).sum()), "cells_lowered": int((d < 0).sum()), "added_cost_sum": int(d.sum()), "added_cost_max": int(d.max()),
                     "between_cells": int(seg.sum()),
                     "between_on_mean": float(on[seg].mean()) if seg.any() else None, "between_off_mean": float(off[seg].mean()) if seg.any() else None,
                     "between_off_min": int(off[seg].min()) if seg.any() else None, "between_on_min": int(on[seg].min()) if seg.any() else None,
                     "separation_m": math.dist((a["x"], a["y"]), (b["x"], b["y"]))})
    act = [r for r in rows if r["active"]]
    full = [r for r in act if r["between_cells"] > 0]
    return {"episode": ep.name, "scenario": ep.scenario, "samples_compared": len(rows), "samples_active": len(act),
            "cells_lowered_total": sum(r["cells_lowered"] for r in rows),
            "inactive_samples_with_added_cost": sum(1 for r in rows if not r["active"] and r["cells_raised"]),
            "active": {"cells_raised_mean": float(np.mean([r["cells_raised"] for r in act])) if act else None,
                       "added_cost_max": max((r["added_cost_max"] for r in act), default=None),
                       "between_on_mean": float(np.mean([r["between_on_mean"] for r in full])) if full else None,
                       "between_off_mean": float(np.mean([r["between_off_mean"] for r in full])) if full else None,
                       "between_on_min": min((r["between_on_min"] for r in full), default=None),
                       "between_off_min": min((r["between_off_min"] for r in full), default=None),
                       "separation_m_mean": float(np.mean([r["separation_m"] for r in act])) if act else None},
            "rows": rows[::5]}


# ------------------------------------------------------------------ queue geometry (synthetic lines)
def queue_geometry(params: LayerParams, spacings=(0.6, 0.8, 1.0, 1.4, 2.0, 2.6), orientations_deg=(0, 30, 45, 90, 135), members: int = 4) -> dict:
    # large enough that no line plus its capsule touches the grid edge at any orientation
    grid = GridSpec.centered(20.0, 0.05)
    from .reference import cell_centres
    peak = to_cell_cost(params, params.interaction_weights[6])
    rows = []
    for s in spacings:
        for th in orientations_deg:
            ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
            L = s * (members - 1)
            pts = [(-L / 2 + i * s) for i in range(members)]
            queue = [Ped(t * ux, t * uy, 0, 1, 6) for t in pts]
            native = [Ped(q.x, q.y, 0, 1 if i == 0 else -1, 6) for i, q in enumerate(queue)]  # only the served person is published
            ext = paint(params, queue, grid)
            nat = paint(params, native, grid)
            wx, wy = cell_centres(grid, (0, 0, 0))
            along = wx * ux + wy * uy
            across = -wx * uy + wy * ux
            on_line = (np.abs(across) <= grid.resolution / 2) & (along >= pts[0]) & (along <= pts[-1])
            width_cells = ext[(np.abs(along) < grid.resolution)]
            span = across[(np.abs(along) < grid.resolution) & (ext > 0)]
            rows.append({"spacing_m": s, "orientation_deg": th,
                         "extended_line_min": int(ext[on_line].min()), "native_line_min": int(nat[on_line].min()),
                         "native_gap_cells_on_line": int((nat[on_line] == 0).sum()), "line_cells": int(on_line.sum()),
                         "extended_halfwidth_m": float(np.abs(span).max()) if span.size else 0.0,
                         "extended_area_m2": float((ext > 0).sum() * grid.resolution ** 2), "native_area_m2": float((nat > 0).sum() * grid.resolution ** 2),
                         "extended_sum": int(ext.astype(np.int64).sum())})
    # rotation invariance per spacing
    inv = {}
    for s in spacings:
        r = [x for x in rows if x["spacing_m"] == s]
        areas = np.array([x["extended_area_m2"] for x in r])
        inv[str(s)] = {"area_rel_spread": float((areas.max() - areas.min()) / areas.mean()), "line_min_values": sorted({x["extended_line_min"] for x in r})}
    ok = all(x["extended_line_min"] >= peak - 1 for x in rows) and all(abs(x["extended_halfwidth_m"] - params.group_extra_radius) <= grid.resolution for x in rows) \
        and all(v["area_rel_spread"] < 0.02 for v in inv.values())
    return {"status": "PASS" if ok else "FAIL", "expected_line_cost": peak, "group_extra_radius": params.group_extra_radius,
            "rotation_invariance": inv, "cases": rows,
            "note": "extended: continuous capsule of the QUEUE_USE weight along the whole line at every spacing and orientation, "
                    "half-width = group_extra_radius; native (only the served person carries the id) leaves gaps on the line once spacing exceeds 2 x personal_space_radius"}


# ------------------------------------------------------------------ group walking
def group_walk_coverage(ep: Ep) -> dict:
    pair_d, mid_cost, seg_cov, seg_min = [], [], [], []
    for k in range(len(ep.robots)):
        ps = ep.states(k)
        pose = ep.pose(k)
        lab = ep.label(k)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                seg = segment_mask(ep.grid, pose, (a["x"], a["y"]), (b["x"], b["y"]), 0.05)
                if seg.sum() < 3:
                    continue
                # only pairs fully inside the grid
                from .verify import _local
                if not all(abs(c) < 4.5 for p in (a, b) for c in _local(pose, p["x"], p["y"])):
                    continue
                pair_d.append(math.dist((a["x"], a["y"]), (b["x"], b["y"])))
                vals = lab[seg]
                seg_cov.append(float((vals > 0).mean()))
                seg_min.append(int(vals.min()))
    hyp = to_cell_cost(ep.params, 1.0)
    return {"episode": ep.name, "pairs_checked": len(pair_d), "pair_distance_m": {"min": min(pair_d, default=None), "mean": float(np.mean(pair_d)) if pair_d else None, "max": max(pair_d, default=None)},
            "segment_cells_with_cost_fraction_min": min(seg_cov, default=None), "segment_cost_min": min(seg_min, default=None),
            "personal_space_radius": ep.params.personal_space_radius, "capsule_value_if_weight_1": hyp,
            "note": "the inter-member segment is fully covered by personal space whenever members are closer than 2 x personal_space_radius"}


# ------------------------------------------------------------------ temporal change attribution
def temporal_attribution(ep: Ep, channel: str = NATIVE) -> dict:
    """L(k+1) - L(k) split into grid motion, pedestrian motion and interaction-state change.

    P(pos, int, g) paints positions `pos` with interaction state `int` on the grid at robot pose g.
      grid:        P(pos_k, int_k, g_k+1)   - P(pos_k, int_k, g_k)
      movement:    P(pos_k+1, int_k, g_k+1) - P(pos_k, int_k, g_k+1)
      interaction: P(pos_k+1, int_k+1, g_k+1) - P(pos_k+1, int_k, g_k+1)
    The three terms add up to the label change exactly (checked).
    """
    membership = channel == EXTENDED
    totals = {"grid": [], "movement": [], "interaction": []}
    bad = 0
    events = []
    for k in range(len(ep.robots) - 1):
        s0, s1 = ep.states(k, membership), ep.states(k + 1, membership)
        g0, g1 = ep.pose(k), ep.pose(k + 1)
        by0 = {p["name"]: p for p in s0}
        # positions of k+1 with the interaction state of k (names present in both)
        mixed = [Ped(p["x"], p["y"], by0[p["name"]]["animation_state"], by0[p["name"]]["interaction_id"], by0[p["name"]]["interaction_type"]) if p["name"] in by0 else Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in s1]
        L0 = paint(ep.params, peds_for_layer(s0), ep.grid, g0).astype(int)
        A = paint(ep.params, peds_for_layer(s0), ep.grid, g1).astype(int)
        B = paint(ep.params, mixed, ep.grid, g1).astype(int)
        L1 = paint(ep.params, peds_for_layer(s1), ep.grid, g1).astype(int)
        if not np.array_equal(L1 - L0, (A - L0) + (B - A) + (L1 - B)):
            bad += 1
        for key, d in (("grid", A - L0), ("movement", B - A), ("interaction", L1 - B)):
            totals[key].append(int(np.abs(d).sum()))
        if np.abs(L1 - B).sum():
            events.append({"between_samples": [k, k + 1], "t_s": ep.robots[k + 1]["stamp_ns"] / 1e9, "abs_change": int(np.abs(L1 - B).sum())})
    return {"episode": ep.name, "scenario": ep.scenario, "channel": channel, "transitions": len(totals["grid"]), "decomposition_errors": bad,
            **{f"{k}_abs_change_sum": int(sum(v)) for k, v in totals.items()},
            **{f"{k}_abs_change_mean": float(np.mean(v)) if v else 0.0 for k, v in totals.items()},
            "interaction_change_events": events}


# ------------------------------------------------------------------ live layer data age
def live_age(ep: Ep, window_s: float = 1.0) -> dict:
    """Which arena_peds message did the live layer paint? The newest stream message, at or before the
    live costmap stamp, whose roster reproduces the live grid exactly. Its age is live stamp minus message stamp."""
    stream = ep.peds_stream()
    if not stream:
        return {"episode": ep.name, "status": "NOT_TESTED"}
    stamps = [s["stamp_ns"] for s in stream]
    ages, vs_image, unique, ambiguous, unmatched = [], [], 0, 0, 0
    for k, rec in enumerate(ep.sync):
        lc = rec.get("live_costmap")
        if lc is None:
            continue
        live = np.load(f"{ep.dir}/live_social_cost/{k:06d}.npy")
        g = GridSpec(resolution=round(lc["resolution"], 6), width=lc["size_x"], height=lc["size_y"], origin_x=lc["origin_x"], origin_y=lc["origin_y"])
        hi = bisect.bisect_right(stamps, lc["stamp_ns"])
        lo = bisect.bisect_left(stamps, lc["stamp_ns"] - int(window_s * 1e9))
        matches = []
        for m in range(hi - 1, lo - 1, -1):
            if np.array_equal(paint(ep.params, peds_for_layer(stream[m]["pedestrians"]), g), live):
                matches.append(m)
        if not matches:
            unmatched += 1
            continue
        distinct = len({tuple((round(p["x"], 6), round(p["y"], 6)) for p in stream[m]["pedestrians"]) for m in matches})
        if distinct > 1 or len(matches) == hi - lo:
            ambiguous += 1  # static scene: every candidate paints the same grid
            continue
        unique += 1
        newest = matches[0]
        ages.append((lc["stamp_ns"] - stamps[newest]) / 1e6)
        vs_image.append((stamps[newest] - rec["stamp_ns"]) / 1e6)
    return {"episode": ep.name, "scenario": ep.scenario, "live_grids": sum(1 for r in ep.sync if "live_costmap" in r),
            "identified": unique, "ambiguous_static": ambiguous, "unmatched": unmatched,
            "data_age_ms": {"mean": float(np.mean(ages)), "min": float(np.min(ages)), "max": float(np.max(ages))} if ages else None,
            "painted_roster_minus_image_ms": {"mean": float(np.mean(vs_image)), "min": float(np.min(vs_image)), "max": float(np.max(vs_image))} if vs_image else None,
            "note": "live-layer age only; dataset labels never use the live layer (they are painted from states interpolated to the image stamp)"}
