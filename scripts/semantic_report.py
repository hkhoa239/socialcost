#!/usr/bin/env python3
"""Semantic validation of the social cost labels; writes <output>/*.json and semantic_report.md.

    python semantic_report.py --dataset <root> [--dataset ...] --output <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost import semantics as S  # noqa: E402
from socialcost import verify as V  # noqa: E402
from socialcost.labels import LayerParams  # noqa: E402

ANIM = {0: "IDLE", 1: "WALKING", 2: "RUNNING", 3: "PANIC", 4: "SURPRISED", 5: "CURIOUS", 6: "THREATENING"}


def interaction_audit(eps: list[V.Ep]) -> list[dict]:
    from arena_humansim.core.interaction_kinds import InteractionType
    local = LayerParams.from_overlay(costmap="local_costmap")
    glob_ = LayerParams.from_overlay(costmap="global_costmap")
    seen_frame, seen_pub = Counter(), Counter()
    queued_seen = Counter()
    for e in eps:
        for f in e.frames:
            for i in f["interactions"]:
                seen_frame[i["interaction_type"]] += 1
                queued_seen[i["interaction_type"]] += sum(1 for r in i["member_roles"] if r == 1)
        for ps in e.peds:
            for t in {(p["interaction_id"], p["interaction_type"]) for p in ps["pedestrians"] if p["interaction_id"] >= 0}:
                seen_pub[t[1]] += 1
    rows = []
    for t in InteractionType:
        k = t.kind
        c = k.contract_defaults
        handle = str(getattr(k.handle, "kind", k.handle)).split(".")[-1]
        w = local.interaction_weights[int(t)] if int(t) < len(local.interaction_weights) else 0.0
        pairable = c.max_participants == -1 or c.max_participants >= 2
        if handle == "OBJECT":
            pair_note = "only if the bound object's capacity >= 2 (default 1)"
        else:
            pair_note = "yes" if pairable else "no (single participant)"
        native = "yes" if w > 0 and pairable else ("object capacity >= 2 only" if w > 0 and handle == "OBJECT" else "no")
        rows.append({
            "type": int(t), "name": t.name, "weight_local": w, "weight_global": glob_.interaction_weights[int(t)] if int(t) < len(glob_.interaction_weights) else 0.0,
            "handle": handle, "participants": [c.min_participants, c.max_participants], "queueable": c.queueable,
            "formation": getattr(k.formation_default, "type", None) if k.formation_default else None,
            "required_state": "interaction_id and interaction_type on >= 2 pedestrians (Pedestrian msg)",
            "published": "interaction_id/type for ACTIVE participants only; queued members -1",
            "two_active_participants_possible": pair_note,
            "native_term_generated": native,
            "extended_term_generated": "yes, between participants and queued members" if w > 0 and c.queueable else native,
            "seen_in_interaction_frames": seen_frame.get(int(t), 0), "seen_published_on_pedestrians": seen_pub.get(int(t), 0),
            "queued_member_observations": queued_seen.get(int(t), 0),
        })
    return rows


def animation_audit(eps: list[V.Ep]) -> dict:
    states = Counter()
    for e in eps:
        for ps in e.peds:
            for p in ps["pedestrians"]:
                states[ANIM.get(p["animation_state"], p["animation_state"])] += 1
    return {"observed_animation_states": dict(states),
            "note": "SURPRISED/CURIOUS and PANIC/THREATENING widen the personal space; none occurred in these scenes, so those branches are covered only by tests/test_live_agreement.py and tests/test_labels.py"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    os.makedirs(out, exist_ok=True)
    eps = V.load_episodes(*args.dataset)
    by = {}
    for e in eps:
        by.setdefault(e.scenario, []).append(e)

    audit = {"interactions": interaction_audit(eps), "animation_states": animation_audit(eps)}
    json.dump(audit, open(os.path.join(out, "interaction_audit.json"), "w"), indent=2)

    decomp = [S.decomposition(e) for e in eps]
    json.dump(decomp, open(os.path.join(out, "label_decomposition.json"), "w"), indent=2)

    controls = {
        "A_empty_world": V.test_empty(eps), "B_single_pedestrian": V.test_single(eps),
        "C_conversation_on_vs_off_sim": V.test_conversation(eps),
        "C_conversation_counterfactual_identical_states": [S.conversation_counterfactual(e) for e in by.get("socialcost_conversation", []) + by.get("socialcost_verify_lifecycle", [])],
        "C_control_episode_has_no_interaction_term": [S.conversation_counterfactual(e) for e in by.get("socialcost_verify_conversation_off", [])],
        "F_interaction_lifecycle": V.test_lifecycle(eps),
    }
    json.dump(controls, open(os.path.join(out, "controls.json"), "w"), indent=2, default=str)

    queue = {"recorded": V.test_queue([e for e in eps if e.scenario in ("socialcost_queue",)]),
             "recorded_rotated": V.test_queue([_as(e, "socialcost_queue") for e in by.get("socialcost_verify_queue_rotated", [])]),
             "synthetic_geometry": S.queue_geometry(eps[0].params),
             "membership_extraction": [_membership(e) for e in by.get("socialcost_queue", []) + by.get("socialcost_verify_queue_rotated", [])]}
    json.dump(queue, open(os.path.join(out, "queue.json"), "w"), indent=2, default=str)

    gw = {"tracking": V.test_group_walk(eps), "coverage": [S.group_walk_coverage(e) for e in by.get("socialcost_group_walk", [])]}
    json.dump(gw, open(os.path.join(out, "group_walk.json"), "w"), indent=2)

    temporal = [S.temporal_attribution(e, ch) for e in eps for ch in (S.NATIVE, S.EXTENDED)]
    json.dump(temporal, open(os.path.join(out, "temporal.json"), "w"), indent=2)
    age = [S.live_age(e) for e in eps]
    json.dump(age, open(os.path.join(out, "live_age.json"), "w"), indent=2)

    # ---------------- report
    L = ["# Social cost semantics", "", f"Datasets: {', '.join(args.dataset)}. {len(eps)} episodes, {sum(len(e.robots) for e in eps)} samples.", ""]
    L += ["## Interaction audit", "", "| type | name | w (local) | participants | published members | 2 active participants possible | native term | extended term | seen (frames / published) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in audit["interactions"]:
        L.append(f"| {r['type']} | {r['name']} | {r['weight_local']} | {r['participants']} | active participants | {r['two_active_participants_possible']} | {r['native_term_generated']} | {r['extended_term_generated']} | {r['seen_in_interaction_frames']} / {r['seen_published_on_pedestrians']} |")
    L += ["", f"Animation states observed: `{json.dumps(audit['animation_states']['observed_animation_states'])}`. {audit['animation_states']['note']}", ""]
    L += ["## Controls", "", "| control | status | key numbers |", "|---|---|---|"]
    for name in ("A_empty_world", "B_single_pedestrian", "C_conversation_on_vs_off_sim", "F_interaction_lifecycle"):
        c = controls[name]
        L.append(f"| {name} | **{c['status']}** | `{json.dumps({k: v for k, v in c['metrics'].items() if k != 'trace'}, default=str)[:600]}` |")
    for cf in controls["C_conversation_counterfactual_identical_states"]:
        a = cf["active"]
        # between-talker cells are measurable only while the talkers' segment lies inside the robot's grid
        between = "not measurable (segment outside the grid while active)" if a["between_on_min"] is None else ("raised" if a["between_on_min"] > a["between_off_min"] else "NOT raised")
        ok = cf["samples_active"] and cf["cells_lowered_total"] == 0 and cf["inactive_samples_with_added_cost"] == 0 and (a["cells_raised_mean"] or 0) > 0 and between != "NOT raised"
        L.append(f"| C counterfactual {cf['episode']} | **{'PASS' if ok else 'FAIL'}** | `{json.dumps(a)}` samples active {cf['samples_active']}/{cf['samples_compared']}, inactive with added cost {cf['inactive_samples_with_added_cost']}, between talkers: {between} |")
    for cf in controls["C_control_episode_has_no_interaction_term"]:
        L.append(f"| C control episode {cf['episode']} | **{'PASS' if cf['samples_active'] == 0 and cf['inactive_samples_with_added_cost'] == 0 else 'FAIL'}** | samples {cf['samples_compared']}, active {cf['samples_active']}, with added cost {cf['inactive_samples_with_added_cost']} |")
    L += ["", "## Native vs extended labels", "", "| episode | scenario | max identity violations | stored ≠ regenerated | native samples w/ interaction cost | extended samples w/ interaction cost | samples differing | cells extended < native |", "|---|---|---|---|---|---|---|---|"]
    for d in decomp:
        L.append(f"| {d['episode']} | {d['scenario']} | {d['max_identity_violations']} | {d['stored_label_mismatches']} | {d['channels']['social_cost']['samples_with_interaction_cost']} | "
                 f"{d['channels']['social_cost_membership']['samples_with_interaction_cost']} | {d['extended_minus_native']['samples_differing']} | {d['extended_minus_native']['cells_where_extended_below_native']} |")
    L += ["", "## Queue", "", f"Recorded (0.8 m spacing): **{queue['recorded']['status']}** `{json.dumps(queue['recorded']['metrics'])}`",
          "", f"Recorded, rotated 90°, 1.2 m spacing: **{queue['recorded_rotated']['status']}** `{json.dumps(queue['recorded_rotated']['metrics'])}`",
          "", f"Synthetic geometry: **{queue['synthetic_geometry']['status']}**. {queue['synthetic_geometry']['note']}", "",
          "| spacing m | orientation ° | extended min on line | native min on line | native gap cells | extended half-width m |", "|---|---|---|---|---|---|"]
    for c in queue["synthetic_geometry"]["cases"]:
        if c["orientation_deg"] in (0, 45, 90):
            L.append(f"| {c['spacing_m']} | {c['orientation_deg']} | {c['extended_line_min']} | {c['native_line_min']} | {c['native_gap_cells_on_line']}/{c['line_cells']} | {c['extended_halfwidth_m']:.3f} |")
    L += ["", "Membership extraction:", ""] + [f"- {m['episode']}: `{json.dumps(m)}`" for m in queue["membership_extraction"]]
    L += ["", "## Group walking", "", f"Tracking: **{gw['tracking']['status']}** `{json.dumps(gw['tracking']['metrics'])}`", ""] + [f"- coverage {c['episode']}: `{json.dumps(c)}`" for c in gw["coverage"]]
    L += ["", "## Temporal change attribution (sum over consecutive samples of |ΔL|, cost units)", "", "| episode | scenario | channel | grid motion | pedestrian motion | interaction state | decomposition errors | interaction events (sample pairs) |", "|---|---|---|---|---|---|---|---|"]
    for t in temporal:
        L.append(f"| {t['episode']} | {t['scenario']} | {t['channel']} | {t['grid_abs_change_sum']} | {t['movement_abs_change_sum']} | {t['interaction_abs_change_sum']} | {t['decomposition_errors']} | {[e['between_samples'] for e in t['interaction_change_events']][:6]} |")
    L += ["", "## Live layer data age (not dataset synchronization)", "", "| episode | scenario | live grids | identified | ambiguous (static) | unmatched | age at costmap stamp ms (mean/min/max) | painted roster − image ms (mean/min/max) |", "|---|---|---|---|---|---|---|---|"]
    for a in age:
        if a.get("status") == "NOT_TESTED":
            continue
        da, pr = a["data_age_ms"], a["painted_roster_minus_image_ms"]
        L.append(f"| {a['episode']} | {a['scenario']} | {a['live_grids']} | {a['identified']} | {a['ambiguous_static']} | {a['unmatched']} | "
                 f"{'–' if not da else f'{da['mean']:.1f} / {da['min']:.1f} / {da['max']:.1f}'} | {'–' if not pr else f'{pr['mean']:.1f} / {pr['min']:.1f} / {pr['max']:.1f}'} |")
    open(os.path.join(out, "semantic_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:3]))
    return 0


def _as(e: V.Ep, scenario: str) -> V.Ep:
    e.scenario = scenario
    return e


def _membership(e: V.Ep) -> dict:
    """InteractionFrame membership vs. published ids vs. pedestrians, per sample."""
    complete, mism, n = 0, 0, 0
    sizes = Counter()
    for k in range(len(e.robots)):
        from socialcost.dataset import interaction_frame_at
        f = interaction_frame_at(e.frames, e.robots[k]["stamp_ns"])
        q = [i for i in (f["interactions"] if f else []) if i["interaction_type"] == 6 and i["outcome"] == 1]
        if not q:
            continue
        n += 1
        ids = {p["id"] for p in e.states(k)}
        members = set(q[0]["member_ids"])
        sizes[len(members)] += 1
        complete += int(members <= ids)
        done = {p["id"] for p in e.states(k, membership=True) if p["interaction_id"] == q[0]["interaction_id"]}
        mism += int(done != members)
    return {"episode": e.name, "samples_with_queue": n, "member_count_histogram": dict(sizes), "members_all_present_as_pedestrians": complete,
            "completed_membership_differs_from_frame": mism}


if __name__ == "__main__":
    sys.exit(main())
