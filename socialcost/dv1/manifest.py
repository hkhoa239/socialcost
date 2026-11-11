"""dataset_v1 manifest generation (Stage A): units -> splits -> worlds -> sites -> counts -> scenes -> bodies -> episodes."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict

from . import scenes, spec
from .geometry import Route, World
from .scenario import scenario_name

# subtype mixes per family (weights); paired families also choose ON/OFF semantics
CONV_MODES = {"static": 0.5, "approach_and_stop": 0.25, "lifecycle": 0.25}
STOP_VIEWS = {"frontal": 0.4, "oblique_30": 0.3, "oblique_60": 0.3}
STOP_TIMING = {"active_from_start": 0.3, "activates_on_approach": 0.25, "active_then_deactivates": 0.25, "pulse": 0.2}
WC_TRAJ = {"stationary": 0.25, "slow_forward_same": 0.15, "slow_forward_opposite": 0.15, "crossing": 0.2, "boundary_wait": 0.15, "turning": 0.1}
QUEUE_ORIENT = {"parallel": 0.4, "perpendicular": 0.35, "oblique": 0.25}
GW_TRAJ = {"same_direction": 0.2, "opposite": 0.2, "perpendicular": 0.15, "passing": 0.15, "fov_enter_leave": 0.15, "corridor_narrowing": 0.15}
GW_FORM = {"side_by_side": 0.35, "single_file": 0.2, "v_formation": 0.2, "loose": 0.25}
PO_CATS = {"fov_exit": 20, "fov_reentry": 20, "static_occlusion": 20, "dynamic_occlusion": 20}
CTRL_KINDS = {"empty": 8, "single_static": 8, "single_moving": 8, "two_independent": 8, "neutral_crowd": 8}

OBSERVABILITY = {
    "conversation": "visual_temporal",  # talk-gesture clip and partner gaze; clearer over a short history
    "stop_gesture": "visual",  # raised hand, confirmed per sample from rendered frames (cue_visible)
    "wheelchair": "visual_appearance",  # body asset differs by design
    "queue": "hidden_state",  # same people and poses; membership not visible in a single frame
}


def _weighted(rng: random.Random, d: dict):
    keys = list(d)
    return rng.choices(keys, weights=[d[k] for k in keys])[0]


def _largest_remainder(total: int, weights: dict) -> dict:
    raw = {k: total * w / sum(weights.values()) for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


OPEN_ONLY = {("group_walking", "perpendicular"), ("group_walking", "fov_enter_leave"), ("group_walking", "passing"), ("partial_observability", "fov_reentry")}


def _counts(total: int, weights: dict) -> list:
    return [k for k, n in _largest_remainder(total, weights).items() for _ in range(n)]


def _plans(fam: str, n: int, rng: random.Random) -> list[dict]:
    def shuf(v):
        v = list(v)
        rng.shuffle(v)
        return v
    if fam == "conversation":
        return [{"mode": m} for m in shuf(_counts(n, CONV_MODES))]
    if fam == "stop_gesture":
        return [{"view": a, "timing": b} for a, b in zip(shuf(_counts(n, STOP_VIEWS)), shuf(_counts(n, STOP_TIMING)))]
    if fam == "wheelchair":
        return [{"traj": t} for t in shuf(_counts(n, WC_TRAJ))]
    if fam == "queue":
        return [{"regime": r, "orient": o, "advancing": a} for r, o, a in zip(shuf(_counts(n, {"dense": 1, "normal": 1, "sparse": 1})),
                                                                           shuf(_counts(n, QUEUE_ORIENT)), shuf(_counts(n, {True: 0.4, False: 0.6})))]
    if fam == "group_walking":
        trajs = shuf(_counts(n, GW_TRAJ))
        forms = shuf(_counts(n - trajs.count("corridor_narrowing"), GW_FORM))
        return [{"traj": t, "form": "side_by_side" if t == "corridor_narrowing" else forms.pop()} for t in trajs]
    if fam == "partial_observability":
        return [{"kind": c} for c in shuf([c for c, k in PO_CATS.items() for _ in range(k)])]
    return [{"kind": c} for c in shuf([c for c, k in CTRL_KINDS.items() for _ in range(k)])]


def _needs_open(fam: str, plan: dict) -> bool:
    return (fam, plan.get("traj") or plan.get("kind")) in OPEN_ONLY


def match_plans(units: list[dict]) -> None:
    """Give plans that need open space (lateral crossings, lateral FOV exit) to units in worlds with open sites."""
    for fam in spec.FAMILIES:
        us = [u for u in units if u["family"] == fam]
        plans = [u["plan"] for u in us]
        need = [p for p in plans if _needs_open(fam, p)]
        rest = [p for p in plans if not _needs_open(fam, p)]
        open_units = [u for u in us if u["world_key"] in ("C", "D")]
        other = [u for u in us if u["world_key"] not in ("C", "D")]
        order = open_units + other
        for u, p in zip(order, need + rest):
            u["plan"] = p
            if fam in ("partial_observability", "controls"):
                u["subtype"] = p["kind"]
            if fam == "controls":
                u["count_bin"] = {"empty": (0, 1), "single_static": (0, 1), "single_moving": (0, 1), "two_independent": (2, 2), "neutral_crowd": (3, 4)}[p["kind"]]


def make_units(rng: random.Random) -> list[dict]:
    units = []
    for fam, f in spec.FAMILIES.items():
        size = 2 if f["paired"] else 1
        n_units = f["quota"] // size
        # split quotas in units
        per_split = {s: round(spec.SPLIT_FRACTION[s] * n_units) for s in spec.SPLITS}
        per_split["train"] += n_units - sum(per_split.values())
        split_list = [s for s in spec.SPLITS for _ in range(per_split[s])]
        rng.shuffle(split_list)
        # count bins (in units)
        bins = [b for b, n in spec.PED_COUNT_ALLOCATION[fam].items() for _ in range(n // size)]
        rng.shuffle(bins)
        # subtype plans with exact (largest-remainder) totals, so retries cannot skew the mix toward easy geometry
        sub = _plans(fam, n_units, rng)
        for i in range(n_units):
            units.append({"family": fam, "paired": f["paired"], "split": split_list[i], "count_bin": bins[i], "plan": sub[i],
                          "subtype": sub[i].get("kind") if fam in ("partial_observability", "controls") else None})
    # controls: count bin follows the control kind
    for u in units:
        if u["family"] == "controls":
            u["count_bin"] = {"empty": (0, 1), "single_static": (0, 1), "single_moving": (0, 1), "two_independent": (2, 2), "neutral_crowd": (3, 4)}[u["subtype"]]
    return units


def assign_worlds(units: list[dict], rng: random.Random) -> None:
    for fam in spec.FAMILIES:
        for split in spec.SPLITS:
            us = [u for u in units if u["family"] == fam and u["split"] == split]
            if spec.SPLIT_WORLD_ROLE[split] == "heldout":
                for u in us:
                    u["world_key"] = "D"
                continue
            alloc = _largest_remainder(len(us), spec.FAMILIES[fam]["worlds"])
            wl = [w for w, n in alloc.items() for _ in range(n)]
            rng.shuffle(wl)
            for u, w in zip(us, wl):
                u["world_key"] = w


def sites_for(world_key: str, family: str, subtype) -> list[str]:
    kinds = spec.FAMILY_SITES[family]
    out = [sid for sid, (wk, kind, _) in spec.SITES.items() if wk == world_key and kind in kinds]
    if family == "partial_observability" and subtype == "fov_reentry":
        out = [sid for sid in out if spec.SITES[sid][1] in ("open", "aisle", "junction", "door")]
    return out


def build_unit(u: dict, worlds: dict, rng: random.Random, site_use: Counter) -> None:
    fam = u["family"]
    conds = ("on", "off") if u["paired"] else ("na",)
    lo, hi = u["count_bin"]
    last_err = Counter()
    candidates = sites_for(u["world_key"], fam, u["subtype"])
    if _needs_open(fam, u["plan"]) and any(spec.SITES[c][1] in ("open", "aisle") for c in candidates):
        candidates = [c for c in candidates if spec.SITES[c][1] in ("open", "aisle")]
    for attempt in range(600):
        # least-used site first, random tie-break
        site = min(candidates, key=lambda s: (site_use[s], rng.random())) if attempt < 100 else rng.choice(candidates)
        wk, kind, pts = spec.SITES[site]
        route = Route(pts)
        sc = scenes.Scene(worlds[wk], route, rng, conds)
        n = rng.randint(lo, hi)
        if fam == "controls" and u["subtype"] == "empty":
            n = 0
        elif fam == "controls" and u["subtype"] in ("single_static", "single_moving"):
            n = 1
        elif fam in ("stop_gesture", "wheelchair") and (lo, hi) == (0, 1):
            n = 1
        elif fam == "queue":
            n = min(n, 8)
        plan = dict(u["plan"])
        fallback = attempt >= 300 and fam not in ("partial_observability", "controls")
        if fallback:  # plan infeasible here: resample, recorded on the unit
            plan = rng.choice(_plans(fam, 20, rng))
        try:
            if fam == "conversation":
                scenes.build_conversation(sc, n, plan["mode"])
                u["subtype"] = plan["mode"]
            elif fam == "stop_gesture":
                scenes.build_stop(sc, n, plan["view"], plan["timing"])
                u["subtype"] = f"{plan['view']}/{plan['timing']}"
            elif fam == "wheelchair":
                scenes.build_wheelchair(sc, n, plan["traj"])
                u["subtype"] = plan["traj"]
            elif fam == "queue":
                scenes.build_queue(sc, n, plan["regime"], plan["orient"], plan["advancing"])
                u["subtype"] = f"{plan['regime']}/{plan['orient']}/{'advancing' if plan['advancing'] else 'static'}"
            elif fam == "group_walking":
                traj = plan["traj"]
                if traj == "corridor_narrowing" and kind not in ("corridor", "aisle"):
                    raise scenes.Reject("needs_corridor")
                scenes.build_group_walk(sc, n, traj, plan["form"])
                u["subtype"] = f"{traj}/{plan['form']}"
            elif fam == "partial_observability":
                scenes.build_partial(sc, n, u["subtype"])
            else:
                scenes.build_control(sc, n, u["subtype"])
            u["plan_fallback"] = fallback
        except scenes.Reject as e:
            last_err[str(e)] += 1
            continue
        if len(sc.humans) != n:
            last_err["count_mismatch"] += 1
            continue
        site_use[site] += 1
        u.update({"site": site, "site_kind": kind, "route": [list(p) for p in pts], "humans": sc.humans, "objects": sc.objects, "scene": sc.meta,
                  "n_peds": n, "build_attempts": attempt + 1})
        return
    raise RuntimeError(f"unit {u.get('unit_id')} ({fam}/{u['world_key']}/{u['count_bin']}) failed: {dict(last_err)}")


def assign_bodies(units: list[dict], pools: dict, rng: random.Random) -> None:
    known, held = pools["known_walking"], pools["heldout_walking"]
    wheel = {**pools["wheelchair_known"], **pools["wheelchair_heldout"]}
    use = Counter()
    for u in units:
        pool = held if spec.SPLIT_BODY_ROLE[u["split"]] == "heldout" else known
        wc_ids = [i for i in pool if i in wheel]
        chosen = set()
        for h in u["humans"]:
            if h["body_slot"] == "wheelchair":
                ident = rng.choice(wc_ids)
                h["body_by_condition"] = {"on": wheel[ident], "off": ident}
                h["identity"] = ident
            else:
                free = [i for i in pool if i not in chosen and i not in wheel]
                ident = min(free, key=lambda i: (use[i], rng.random()))
                h["body_by_condition"] = {c: ident for c in ("on", "off", "na")}
                h["identity"] = ident
            chosen.add(h["identity"])
            use[h["identity"]] += 1


def pick_stages(units: list[dict], rng: random.Random) -> None:
    for fam in spec.FAMILIES:
        fam_units = [u for u in units if u["family"] == fam]
        quota = round(spec.PILOT_FRACTION * len(fam_units))
        by_split = defaultdict(list)
        for u in fam_units:
            by_split[u["split"]].append(u)
        alloc = _largest_remainder(quota, {s: len(v) for s, v in by_split.items()})
        for s, v in by_split.items():
            k = max(alloc.get(s, 0), 1)
            rng.shuffle(v)
            for u in v[:k]:
                u["pilot"] = True
        # trim to the quota if the at-least-one rule overshot (drop from train)
        extra = sum(u.get("pilot", False) for u in fam_units) - quota
        for u in [u for u in fam_units if u.get("pilot") and u["split"] == "train"][:max(0, extra)]:
            u["pilot"] = False
    # Stage A subset: from the pilot, spread over all four worlds (episodes per world as even as possible)
    per_world = Counter()
    for fam, k in spec.SUBSET_PER_FAMILY.items():
        pil = [u for u in units if u["family"] == fam and u.get("pilot")]
        rng.shuffle(pil)
        for _ in range(k):
            cand = [u for u in pil if not u.get("subset")]
            if not cand:
                break
            u = min(cand, key=lambda u: per_world[u["world_key"]])
            u["subset"] = True
            per_world[u["world_key"]] += 2 if u["paired"] else 1

def expand(units: list[dict]) -> list[dict]:
    episodes = []
    eid = 0
    for u in units:
        conds = ("on", "off") if u["paired"] else ("na",)
        pair_id = f"P{u['unit_index']:04d}" if u["paired"] else None
        for c in conds:
            eid += 1
            e_id = f"{eid:06d}"
            humans = []
            for h in u["humans"]:
                hh = {k: v for k, v in h.items() if k not in ("body_by_condition",)}
                hh["body"] = h["body_by_condition"][c]
                hh["behavior"] = {c: h["behavior"][c]}
                hh["agent_type"] = {c: h["agent_type"][c]}
                humans.append(hh)
            fam = u["family"]
            obs = OBSERVABILITY.get(fam, "na")
            derive = None
            if c == "off" and (fam == "queue" or (fam == "conversation" and u["subtype"] == "static")):
                derive = "on"  # OFF spawn poses come from the ON episode's first recorded sample
            interaction = {
                "conversation": u["scene"].get("interaction") if c == "on" else None,
                "stop_gesture": "halt_gesture" if c == "on" else None,
                "wheelchair": "mobility_aid" if c == "on" else None,
                "queue": "QUEUE_USE" if c == "on" else None,
                "group_walking": "GROUP_WALK",
            }.get(fam)
            reproducibility = _reproducibility(u, c)
            ep = {
                "episode_id": e_id, "scenario": scenario_name(e_id), "unit_id": u["unit_id"], "pair_id": pair_id, "condition": c,
                "family": fam, "subtype": u["subtype"], "split": u["split"], "world_key": u["world_key"], "world": spec.WORLDS[u["world_key"]]["name"],
                "site": u["site"], "site_kind": u["site_kind"], "robot": {"start": [*u["route"][0], round(math.atan2(u["route"][1][1] - u["route"][0][1], u["route"][1][0] - u["route"][0][0]), 4)],
                                                                          "goal": [*u["route"][-1], round(math.atan2(u["route"][-1][1] - u["route"][-2][1], u["route"][-1][0] - u["route"][-2][0]), 4)],
                                                                          "route": u["route"]},
                "run_seed": u["run_seed"], "template_id": u["template_id"], "n_peds": u["n_peds"], "count_bin": list(u["count_bin"]),
                "humans": humans, "objects": u["objects"], "scene": u["scene"], "interaction_condition": interaction,
                "visibility_category": u["scene"].get("visibility_category", "natural"), "observability": obs if u["paired"] else "na",
                "trajectory_classes": sorted({h.get("trajectory_class", _traj_of(h, c)) for h in humans}),
                "reproducibility": reproducibility, "derive_spawn_from": derive,
                "bodies": sorted({h["body"] for h in humans}), "identities": sorted({h["identity"] for h in humans}),
                "label_versions": spec.LABELS, "stage": "subset" if u.get("subset") else ("pilot" if u.get("pilot") else "full"),
                "pilot": bool(u.get("pilot")), "subset": bool(u.get("subset")),
                "plan": u["plan"], "plan_fallback": bool(u.get("plan_fallback")), "build_attempts": u["build_attempts"],
            }
            episodes.append(ep)
    return episodes


def _traj_of(h: dict, c: str) -> str:
    steps = h["behavior"][c]["steps"]
    if any("group_form" in s or "group_join" in s for s in steps):
        return "group_following"
    if any("queue" in s for s in steps):
        return "social_transition" if h["behavior"][c].get("loop") and any(s.get("service", 1000) < 1000 for s in steps if "queue" in s) else "static"
    gotos = [s for s in steps if "goto" in s]
    if not gotos:
        return "static"
    if any("interact" in s for s in steps):
        return "social_transition"
    if len(gotos) > 1:
        return "waypoint"
    if any("wait" in s and s["wait"] > 1.5 for s in steps[1:]):
        return "stop_and_go"
    return "linear"


def _reproducibility(u: dict, c: str) -> str:
    fam, sub = u["family"], u["subtype"]
    moving = any(any("goto" in s or "group_form" in s or "group_join" in s for s in h["behavior"][c]["steps"]) for h in u["humans"])
    if not moving and fam != "queue":
        return "fully_reproducible"
    if fam == "queue":
        return "fully_reproducible" if "static" in sub or c == "off" else "approximately_reproducible"
    return "reactive" if fam in ("group_walking",) or "crossing" in str(sub) else "approximately_reproducible"


def generate(pools: dict, seed: int = spec.MASTER_SEED) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    units = make_units(rng)
    assign_worlds(units, rng)
    match_plans(units)
    worlds = {k: World.load(v["name"]) for k, v in spec.WORLDS.items()}
    site_use = Counter()
    for i, u in enumerate(units):
        u["unit_index"] = i + 1
        u["unit_id"] = f"U{i + 1:04d}"
        u["run_seed"] = 1000 + i + 1
        urng = random.Random(seed * 1000 + i)
        build_unit(u, worlds, urng, site_use)
        u["template_id"] = f"T{_sha([u['family'], u['site'], u['humans'], u['objects']])}"
    assign_bodies(units, pools, rng)
    pick_stages(units, rng)
    episodes = expand(units)
    info = {"site_use": dict(site_use), "units": len(units), "episodes": len(episodes)}
    return episodes, info
