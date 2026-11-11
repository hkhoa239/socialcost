"""Scene builders for dataset_v1: one sampled, geometry-checked scene per manifest unit.

A unit is one counterfactual pair (conditions "on"/"off") or one unpaired episode (condition "na").
Builders work in the robot route's local frame (s along the route from the start, v to the left) and
return world-frame poses. Every person has per-condition behaviour as a list of abstract steps that
`scenario.py` compiles into humansim behaviour trees:

    {"wait": t}                       stand still t seconds
    {"goto": [x, y, theta]}           walk to a scripted waypoint (world frame)
    {"interact": TYPE, "duration": d} symmetric interaction (TALK_TO / GROUP_CONVERSATION), gaze at partner
    {"halt": {"azimuth": a, "elevation": e}, "duration": d}   raised stop hand, relative to the body heading
    {"queue": desk, "service": d}     QUEUE_USE at a scenario desk (loops)
    {"group_form": true} / {"group_join": true}  GROUP_WALK leader / follower
A behaviour ends in an indefinite hold unless it loops (`"loop": true`).

Times are seconds from the seeded episode start; the recording window is [WARMUP_S, WARMUP_S + DURATION_S].
"""

from __future__ import annotations

import math
import random

import numpy as np

from . import spec
from .geometry import Route, World, wrap

W0, W1 = spec.WARMUP_S, spec.WARMUP_S + spec.DURATION_S
CONTROLLED = ("primary", "participant", "member", "target")
LAT_GROUP, LAT_QUEUE = 2.5, 3.0
FOREVER = 1000.0


class Reject(Exception):
    pass


def _deg(a: float) -> float:
    return math.radians(a)


class Scene:
    """Collects people and objects for one unit and checks the geometry as it goes."""

    def __init__(self, world: World, route: Route, rng: random.Random, conditions: tuple[str, ...]):
        self.world, self.route, self.rng, self.conditions = world, route, rng, conditions
        self.humans: list[dict] = []
        self.objects: list[dict] = []
        self.meta: dict = {}
        self._occ = world

    # --- frames ---
    def w(self, s: float, v: float) -> tuple[float, float]:
        return self.route.local_to_world(s, v)

    def h(self, s: float, rel: float) -> float:
        """World heading at route position s, rel radians from the route direction (0 = along the route)."""
        return wrap(self.route.heading(s) + rel)

    def toward_robot(self, s: float) -> float:
        return self.h(s, math.pi)

    def lat(self, s: float, margin: float = spec.PED_WALL_CLEARANCE + 0.05, cap: float | None = None) -> tuple[float, float]:
        """Lateral interval free of walls at route position s; `cap` keeps controlled people near the camera's view
        (open halls give 7-8 m of free space, beyond the 72 deg FOV from most robot positions)."""
        left, right = self.route.lateral_free(self._occ, s)
        lo, hi = -(right - margin), left - margin
        if cap is not None:
            lo, hi = max(lo, -cap), min(hi, cap)
        return lo, hi

    # --- checks ---
    def free(self, x: float, y: float, margin: float = spec.PED_WALL_CLEARANCE) -> bool:
        return self._occ.clearance(x, y) >= margin

    def check_point(self, x: float, y: float, others: list[tuple[float, float]] | None = None, min_d: float = spec.PED_PED_MIN) -> None:
        if not self.free(x, y):
            raise Reject("wall")
        for o in others if others is not None else [tuple(hh["spawn"][:2]) for hh in self.humans]:
            if math.dist((x, y), o) < min_d:
                raise Reject("overlap")
        rs = self.route.pts[0]
        if math.dist((x, y), rs) < spec.ROBOT_START_PED_MIN:
            raise Reject("robot_start")

    def check_path(self, pts: list[tuple[float, float]]) -> None:
        for a, b in zip(pts, pts[1:]):
            if not self._occ.segment_clear(a, b, 0.3):
                if self._occ.path_length(a, b, 0.3) is None:
                    raise Reject("unreachable")

    def visible_from_route(self, x: float, y: float) -> bool:
        """Line of sight (walls and world furniture, not scenario objects) from some robot position on the route."""
        for s in np.arange(0.0, min(self.route.length, 10.0), 0.5):
            if self.world.line_of_sight(self.route.at(float(s))[:2], (x, y)):
                return True
        return False

    # --- people / objects ---
    def add(self, role: str, spawn: tuple[float, float, float], behavior: dict, *, agent: dict | str = "adult", speed: float | None = None,
            body_slot: str | None = None, waypoints: list | None = None, check: bool = True, min_d: float = spec.PED_PED_MIN) -> dict:
        x, y, yaw = spawn
        if check:
            self.check_point(x, y, min_d=min_d)
        if role in CONTROLLED and not self.visible_from_route(x, y):
            raise Reject("no_line_of_sight")
        if waypoints:
            self.check_path([(x, y)] + [tuple(p[:2]) for p in waypoints])
            for p in waypoints:
                if not self.free(p[0], p[1], 0.3):
                    raise Reject("waypoint_wall")
        per_cond = set(behavior) <= {"on", "off", "na"}
        beh = {c: behavior[c] for c in self.conditions} if per_cond else {c: behavior for c in self.conditions}
        ag = {c: agent[c] for c in self.conditions} if isinstance(agent, dict) else {c: agent for c in self.conditions}
        hmn = {"name": f"{role}_{len(self.humans)}", "role": role, "spawn": [round(x, 3), round(y, 3), round(wrap(yaw), 4)],
               "behavior": beh, "agent_type": ag, "speed": round(speed, 3) if speed else None, "body_slot": body_slot or role,
               "intended_waypoints": [[round(p[0], 3), round(p[1], 3)] for p in (waypoints or [])]}
        self.humans.append(hmn)
        return hmn

    def add_object(self, name: str, model: str, x: float, y: float, yaw: float, humansim: dict | None = None, margin: float = 0.15) -> dict:
        if not self.free(x, y, margin + 0.3):
            raise Reject("object_wall")
        for hh in self.humans:
            if math.dist((x, y), hh["spawn"][:2]) < 0.9:
                raise Reject("object_overlap")
        o = {"name": name, "model": model, "pose": [round(x, 3), round(y, 3), round(wrap(yaw), 4)]}
        if humansim:
            o["humansim"] = humansim
        self.objects.append(o)
        self._occ = self._occ.add_obstacles([o])
        return o

    # --- common helpers ---
    def bystanders(self, n: int, s_range: tuple[float, float] = (4.0, 13.0), moving_p: float = 0.5, avoid: list | None = None) -> None:
        for _ in range(n):
            for _try in range(60):
                try:
                    self._one_bystander(s_range, moving_p, avoid or [])
                    break
                except Reject:
                    continue
            else:
                raise Reject("bystander")

    def _one_bystander(self, s_range, moving_p, avoid) -> None:
        rng = self.rng
        s = rng.uniform(*s_range)
        lo, hi = self.lat(s)
        if hi - lo < 0.2:
            raise Reject("narrow")
        v = rng.uniform(lo, hi)
        x, y = self.w(s, v)
        for a in avoid:
            if math.dist((x, y), a) < 1.6:
                raise Reject("near_primary")
        if rng.random() >= moving_p:
            self.add("bystander", (x, y, rng.uniform(-math.pi, math.pi)), {"steps": [{"wait": FOREVER}]})
            self.humans[-1]["trajectory_class"] = "static"
            return
        kind = rng.choices(list(BYSTANDER_MOTION), weights=list(BYSTANDER_MOTION.values()))[0]
        speed = rng.uniform(0.5, 1.2)
        if kind in ("linear", "parallel", "crossing"):
            if kind == "crossing":
                lo2, hi2 = self.lat(s)
                target = self.w(s + rng.uniform(-1, 1), hi2 if v < 0 else lo2)
            elif kind == "parallel":
                target = self.w(min(self.route.length + 2, s + rng.choice([-1, 1]) * rng.uniform(4, 8)), v)
            else:
                ang = rng.uniform(-math.pi, math.pi)
                target = (x + 5 * math.cos(ang), y + 5 * math.sin(ang))
            if not self.free(*target, 0.4):
                raise Reject("target_wall")
            yaw = math.atan2(target[1] - y, target[0] - x)
            self.add("bystander", (x, y, yaw), {"steps": [{"wait": rng.uniform(0, 6)}, {"goto": [target[0], target[1], yaw]}]}, speed=speed,
                     waypoints=[target])
            self.humans[-1].update({"trajectory_class": kind, "lisn_pattern": "single_pass"})
            return
        if kind == "pacing":  # LISN-style hovering: back and forth between nearby points, looping
            pts = []
            for _ in range(rng.choice([2, 2, 3])):
                for _try in range(20):
                    a = rng.uniform(-math.pi, math.pi)
                    d = rng.uniform(0.8, 2.0)
                    p = (x + d * math.cos(a), y + d * math.sin(a))
                    if self.free(*p, 0.4) and self._occ.segment_clear((x, y), p, 0.3):
                        pts.append(p)
                        break
            if len(pts) < 2:
                raise Reject("pacing_space")
            steps, prev = [], (x, y)
            for p in pts + [(x, y)]:
                yaw = math.atan2(p[1] - prev[1], p[0] - prev[0])
                steps += [{"goto": [p[0], p[1], yaw]}, {"wait": round(rng.uniform(0.5, 2.5), 2)}]
                prev = p
            self.add("bystander", (x, y, rng.uniform(-math.pi, math.pi)), {"steps": steps, "loop": True}, speed=rng.uniform(0.5, 0.9),
                     waypoints=pts + [(x, y)])
            self.humans[-1].update({"trajectory_class": "pacing", "lisn_pattern": "patrol_loop (waypoint_mode 2), short legs"})
            return
        # traverse (single pass) or patrol (loop): multi-waypoint paths routed on the map through corridors and doorways
        n_anchor = 1 if kind == "traverse" else rng.choice([2, 3])
        anchors, prev = [], (x, y)
        for _ in range(n_anchor):
            for _try in range(25):
                if kind == "traverse":
                    a = rng.uniform(-math.pi, math.pi)
                    d = rng.uniform(8.0, 15.0)
                    p = (prev[0] + d * math.cos(a), prev[1] + d * math.sin(a))
                else:
                    sp = rng.uniform(-2.0, self.route.length + 2.0)
                    l2, h2 = self.lat(min(max(sp, 0.0), self.route.length))
                    p = self.w(sp, rng.uniform(l2 - 1.5, h2 + 1.5))
                if self.free(*p, 0.5) and math.dist(p, prev) > 3.0:
                    anchors.append(p)
                    prev = p
                    break
        if len(anchors) < n_anchor:
            raise Reject("anchor_space")
        legs = [(x, y)] + anchors + ([(x, y)] if kind == "patrol" else [])
        wps = []
        for a_, b_ in zip(legs, legs[1:]):
            r = self._occ.path_length(a_, b_, 0.35, box=4.0)
            if r is None:
                raise Reject("unreachable")
            wps += simplify(r[1])[1:]
        steps, prev = [{"wait": round(rng.uniform(0.0, 8.0), 2)}], (x, y)
        for p in wps:
            yaw = math.atan2(p[1] - prev[1], p[0] - prev[0])
            steps.append({"goto": [p[0], p[1], yaw]})
            if kind == "patrol" and any(math.dist(p, q) < 0.05 for q in anchors):
                steps.append({"wait": round(rng.uniform(0.0, 3.0), 2)})
            prev = p
        self.add("bystander", (x, y, math.atan2(wps[0][1] - y, wps[0][0] - x)), {"steps": steps, "loop": kind == "patrol"}, speed=speed, waypoints=wps)
        self.humans[-1].update({"trajectory_class": kind, "lisn_pattern": "single_pass (waypoint_mode 1)" if kind == "traverse" else "patrol_loop (waypoint_mode 2)"})


# moving-bystander mix: the three single-leg classes plus LISN-style pacing, routed traversals and patrol loops
BYSTANDER_MOTION = {"linear": 0.12, "parallel": 0.14, "crossing": 0.14, "pacing": 0.2, "traverse": 0.2, "patrol": 0.2}


def simplify(path: list[tuple[float, float]], tol: float = 0.25, max_leg: float = 4.0) -> list[tuple[float, float]]:
    """Douglas-Peucker on an A* cell path, then split legs longer than max_leg (humansim go_to waypoints)."""
    def dp(pts):
        if len(pts) < 3:
            return pts
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        n = math.hypot(x1 - x0, y1 - y0) or 1e-9
        d = [abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0) / n for px, py in pts[1:-1]]
        i = int(np.argmax(d)) + 1
        if d[i - 1] > tol:
            return dp(pts[:i + 1])[:-1] + dp(pts[i:])
        return [pts[0], pts[-1]]
    out = dp(list(path))
    res = [out[0]]
    for p in out[1:]:
        q = res[-1]
        k = int(math.dist(p, q) // max_leg)
        for j in range(1, k + 1):
            f = j / (k + 1)
            res.append((q[0] + (p[0] - q[0]) * f, q[1] + (p[1] - q[1]) * f))
        res.append(p)
    return [(round(a, 3), round(b, 3)) for a, b in res]

def hold() -> dict:
    return {"steps": [{"wait": FOREVER}]}


# =====================================================================================================
# Conversation
# =====================================================================================================

def build_conversation(sc: Scene, n: int, mode: str) -> None:
    rng = sc.rng
    core = n if n == 2 else rng.choice([n, n] + ([n - 1] if n - 1 >= 2 else []))
    if n == 4 and core == 3 and rng.random() < 0.3:
        core = 2
    inter = "TALK_TO" if core == 2 else "GROUP_CONVERSATION"
    s_c = rng.uniform(6.0, 10.0) if mode != "approach_and_stop" else rng.uniform(7.5, 10.5)
    lo, hi = sc.lat(s_c, cap=LAT_GROUP)
    gap = core == 2 and rng.random() < 0.3
    if core == 2:
        sep = rng.uniform(0.9, 2.0)
        axis = rng.uniform(_deg(70), _deg(110)) if gap else rng.uniform(0, math.pi)
        arrangement = "face_to_face" if rng.random() < 0.7 else "L_shape"
        vm = 0.0 if gap else rng.uniform(lo + 0.5 * sep, hi - 0.5 * sep) if hi - lo > sep else None
        if vm is None:
            raise Reject("narrow")
        cx, cy = sc.w(s_c, vm)
        a = sc.h(s_c, axis)
        pts = [(cx - 0.5 * sep * math.cos(a), cy - 0.5 * sep * math.sin(a)), (cx + 0.5 * sep * math.cos(a), cy + 0.5 * sep * math.sin(a))]
        yaws = [math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]), math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0])]
        jit = 25 if arrangement == "face_to_face" else 45
        yaws = [y + _deg(rng.uniform(-jit, jit)) for y in yaws]
        formation = {"type": "dyad", "separation_m": round(sep, 3), "arrangement": arrangement, "axis_rel_route_deg": round(math.degrees(axis), 1)}
    else:
        r = rng.uniform(0.7, 1.0) if core == 3 else rng.uniform(0.8, 1.1)
        vm = rng.uniform(lo + r, hi - r) if hi - lo > 2 * r else None
        if vm is None:
            raise Reject("narrow")
        cx, cy = sc.w(s_c, vm)
        rot = rng.uniform(0, 2 * math.pi)
        angs = [rot + 2 * math.pi * i / core + _deg(rng.uniform(-12, 12)) for i in range(core)]
        pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in angs]
        yaws = [math.atan2(cy - p[1], cx - p[0]) + _deg(rng.uniform(-10, 10)) for p in pts]
        formation = {"type": "triangle" if core == 3 else "circle", "radius_m": round(r, 3)}
    group_rel = wrap(math.atan2(cy - sc.route.pts[0][1], cx - sc.route.pts[0][0]) - sc.route.heading(0))
    on_off = sc.conditions
    for i, (p, yaw) in enumerate(zip(pts, yaws)):
        if mode == "static":
            beh = {"on": {"steps": [{"interact": inter, "duration": FOREVER}]}, "off": hold()}
            sc.add("participant", (p[0], p[1], yaw), beh, min_d=0.6)
        elif mode == "lifecycle":
            t_on = sc.meta.setdefault("t_on", round(rng.uniform(W0 + 1.0, W0 + 6.0), 2))
            dur = sc.meta.setdefault("talk_duration", round(rng.uniform(6.0, 10.0), 2))
            beh = {"on": {"steps": [{"wait": t_on}, {"interact": inter, "duration": dur}]}, "off": hold()}
            sc.add("participant", (p[0], p[1], yaw), beh, min_d=0.6)
        else:  # approach_and_stop: walk into formation, then interact (ON) or stand (OFF)
            for _ in range(30):
                ang = rng.uniform(-math.pi, math.pi)
                d = rng.uniform(3.0, 5.0)
                sx, sy = p[0] + d * math.cos(ang), p[1] + d * math.sin(ang)
                if sc.free(sx, sy) and math.dist((sx, sy), sc.route.pts[0]) > spec.ROBOT_START_PED_MIN:
                    break
            else:
                raise Reject("approach_start")
            delay = round(rng.uniform(0.0, 2.0), 2)
            go = {"goto": [p[0], p[1], yaw]}
            beh = {"on": {"steps": [{"wait": delay}, go, {"interact": inter, "duration": FOREVER}]}, "off": {"steps": [{"wait": delay}, go]}}
            sc.add("participant", (sx, sy, math.atan2(p[1] - sy, p[0] - sx)), beh, speed=rng.uniform(0.7, 1.0), waypoints=[(p[0], p[1], yaw)], min_d=0.6)
            sc.humans[-1]["formation_pose"] = [round(p[0], 3), round(p[1], 3), round(wrap(yaw), 4)]
    sc.meta.update({"mode": mode, "interaction": inter, "core_participants": core, "formation": formation, "gap_on_route": gap,
                    "group_centre": [round(cx, 3), round(cy, 3)], "group_s": round(s_c, 2), "group_bearing_from_start_deg": round(math.degrees(group_rel), 1)})
    sc.bystanders(n - core, avoid=[(cx, cy)])


# =====================================================================================================
# Stop gesture
# =====================================================================================================

def build_stop(sc: Scene, n: int, view: str, timing: str) -> None:
    rng = sc.rng
    s_p = rng.uniform(7.0, 11.0)
    lo, hi = sc.lat(s_p, cap=LAT_GROUP)
    if view == "frontal":
        v = rng.uniform(max(lo, -0.3), min(hi, 0.3))
        rel = 0.0
    else:
        deg = 30 if view == "oblique_30" else 60
        side = rng.choice([-1, 1])
        v = side * rng.uniform(0.4, 1.4)
        v = min(max(v, lo), hi)
        rel = side * _deg(deg + rng.uniform(-5, 5))  # body turned away from the robot by ~deg
    x, y = sc.w(s_p, v)
    yaw = wrap(sc.toward_robot(s_p) + rel)
    halt = {"halt": {"azimuth": 0, "elevation": 10}}
    if timing == "active_from_start":
        on = {"steps": [{**halt, "duration": FOREVER}]}
        t_on, t_off = 0.0, None
    elif timing == "activates_on_approach":
        t_on = round(rng.uniform(W0 + 4.0, W0 + 12.0), 2)
        on = {"steps": [{"wait": t_on}, {**halt, "duration": FOREVER}]}
        t_off = None
    elif timing == "active_then_deactivates":
        t_on = round(rng.uniform(W0 - 2.0, W0 + 3.0), 2)
        d = round(rng.uniform(5.0, 9.0), 2)
        on = {"steps": [{"wait": t_on}, {**halt, "duration": d}]}
        t_off = round(t_on + d, 2)
    else:  # pulse: active for several seconds mid-window
        t_on = round(rng.uniform(W0 + 3.0, W0 + 9.0), 2)
        d = round(rng.uniform(4.0, 8.0), 2)
        on = {"steps": [{"wait": t_on}, {**halt, "duration": d}]}
        t_off = round(t_on + d, 2)
    sc.add("primary", (x, y, yaw), {"on": on, "off": hold()})
    occl = n >= 2 and rng.random() < 0.2
    if occl:  # a bystander between robot and primary, slightly offset: partial arm / body occlusion
        so = s_p - rng.uniform(1.5, 3.0)
        vo = v + rng.choice([-1, 1]) * rng.uniform(0.25, 0.5)
        lo2, hi2 = sc.lat(so)
        vo = min(max(vo, lo2), hi2)
        ox, oy = sc.w(so, vo)
        sc.add("bystander", (ox, oy, rng.uniform(-math.pi, math.pi)), hold())
        sc.humans[-1]["trajectory_class"] = "static_occluder"
    sc.meta.update({"view": view, "timing": timing, "gesture_t_on_s": t_on, "gesture_t_off_s": t_off, "primary_s": round(s_p, 2),
                    "body_rel_robot_deg": round(math.degrees(rel), 1), "occluding_bystander": occl})
    sc.bystanders(n - 1 - int(occl), avoid=[(x, y)])


# =====================================================================================================
# Wheelchair
# =====================================================================================================

def build_wheelchair(sc: Scene, n: int, traj: str) -> None:
    rng = sc.rng
    s_p = rng.uniform(6.5, 10.5)
    lo, hi = sc.lat(s_p, cap=LAT_GROUP)
    speed = round(rng.uniform(0.45, 0.7), 2)
    agent = {"on": "wheelchair_manual", "off": "adult"}
    wps = None
    if traj == "stationary":
        v = rng.uniform(lo, hi)
        x, y = sc.w(s_p, v)
        yaw = rng.uniform(-math.pi, math.pi)
        beh = hold()
    elif traj == "boundary_wait":
        side = rng.choice([-1, 1])
        v = (hi if side > 0 else lo)
        x, y = sc.w(s_p, v)
        yaw = sc.h(s_p, rng.choice([0, math.pi]))
        beh = hold()
    elif traj in ("slow_forward_same", "slow_forward_opposite"):
        v = rng.uniform(max(lo, 0.7), hi) * rng.choice([1, -1]) if hi > 0.7 else rng.uniform(lo, hi)
        v = min(max(v, lo), hi)
        if traj == "slow_forward_same":
            s0, s1 = s_p - 2.0, min(sc.route.length + 3, s_p + 10)
        else:
            s0, s1 = s_p + 4.0, max(-2.0, s_p - 8.0)
        x, y = sc.w(s0, v)
        tx, ty = sc.w(s1, v)
        yaw = math.atan2(ty - y, tx - x)
        wps = [(tx, ty, yaw)]
        beh = {"steps": [{"wait": round(rng.uniform(W0 - 4, W0 + 2), 2)}, {"goto": [tx, ty, yaw]}]}
    elif traj == "crossing":
        side = rng.choice([-1, 1])
        far = 3.0
        lo2, hi2 = sc.lat(s_p, 0.5)
        v0 = (hi2 if side > 0 else lo2)
        v1 = (lo2 if side > 0 else hi2)
        if abs(v0 - v1) < 2.0:
            raise Reject("narrow_cross")
        x, y = sc.w(s_p, max(min(v0, far), -far))
        tx, ty = sc.w(s_p + rng.uniform(-1, 1), max(min(v1, far), -far))
        yaw = math.atan2(ty - y, tx - x)
        wps = [(tx, ty, yaw)]
        beh = {"steps": [{"wait": round(rng.uniform(W0, W0 + 8), 2)}, {"goto": [tx, ty, yaw]}]}
    else:  # turning: two legs with a ~90 deg turn
        v = rng.uniform(lo, hi)
        x, y = sc.w(s_p, v)
        a0 = rng.uniform(-math.pi, math.pi)
        p1 = (x + 2.5 * math.cos(a0), y + 2.5 * math.sin(a0))
        a1 = a0 + rng.choice([-1, 1]) * math.pi / 2
        p2 = (p1[0] + 2.5 * math.cos(a1), p1[1] + 2.5 * math.sin(a1))
        yaw = a0
        wps = [(p1[0], p1[1], a0), (p2[0], p2[1], a1)]
        beh = {"steps": [{"wait": round(rng.uniform(W0, W0 + 6), 2)}, {"goto": list(wps[0])}, {"goto": list(wps[1])}]}
    sc.add("primary", (x, y, yaw), beh, agent=agent, speed=speed, waypoints=wps, body_slot="wheelchair")
    sc.humans[-1]["trajectory_class"] = traj
    sc.meta.update({"trajectory": traj, "primary_s": round(s_p, 2), "speed": speed,
                    "geometry_note": "ON wheelchair body (capsule footprint 1.1 m, along_heading kinematics) vs OFF walking twin; "
                                     "positions and headings are controlled, visual geometry differs by design"})
    sc.bystanders(n - 1, avoid=[(x, y)])


# =====================================================================================================
# Queue
# =====================================================================================================

def build_queue(sc: Scene, n: int, spacing_regime: str, orientation: str, advancing: bool) -> None:
    rng = sc.rng
    lo_s, hi_s = {"dense": (0.6, 0.9), "normal": (0.9, 1.3), "sparse": (1.3, 2.0)}[spacing_regime]
    step = round(rng.uniform(lo_s, hi_s), 3)
    front = 0.6
    length = front + step * (n - 1)
    s_d = rng.uniform(7.0, 11.0)
    lo, hi = sc.lat(s_d, 0.6, cap=LAT_QUEUE)
    side = rng.choice([-1, 1])
    if orientation == "parallel":
        # desk beside the route, queue extending back along the route on the same side
        vd = min(max(side * rng.uniform(1.0, 1.7), lo), hi)
        if abs(vd) < 0.9:
            raise Reject("narrow")
        dx, dy = sc.w(s_d, vd)
        qdir = sc.h(s_d, rng.choice([0, math.pi]))  # queue axis direction from desk toward tail
    elif orientation == "perpendicular":
        vd = (hi if side > 0 else lo)
        dx, dy = sc.w(s_d, vd)
        qdir = sc.h(s_d, -side * math.pi / 2)  # tail toward the route centre line
    else:  # oblique
        vd = min(max(side * rng.uniform(1.0, 2.5), lo), hi)
        dx, dy = sc.w(s_d, vd)
        qdir = sc.h(s_d, rng.choice([-1, 1]) * _deg(rng.uniform(35, 55)) + (math.pi if rng.random() < 0.5 else 0))
    desk_yaw = wrap(qdir + math.pi)  # desk faces its queue
    # people face the desk; the first stands ~1.25 m from the desk centre (where humansim's line formation with
    # front_offset 0.6 put the served person in pilot_v1), the rest `step` apart along qdir
    first = 1.25
    slots = [(dx + (first + i * step) * math.cos(qdir), dy + (first + i * step) * math.sin(qdir)) for i in range(n)]
    tail_dist = min(math.dist(slots[-1], sc.route.local_to_world(s, 0.0)) for s in [x * 0.25 for x in range(int(sc.route.length * 4))])
    if tail_dist < 0.6:
        raise Reject("queue_on_route")
    if not sc.free(dx, dy, 0.6):
        raise Reject("object_wall")
    service = round(rng.uniform(5.0, 9.0), 2) if advancing else FOREVER
    on = {"steps": [{"queue": "service_desk", "service": service}], "loop": True}
    for i, (x, y) in enumerate(slots):
        sc.add("member", (x, y, wrap(qdir + math.pi)), {"on": on, "off": hold()}, min_d=0.55)
    desk = sc.add_object("service_desk", spec.QUEUE_DESK_MODEL, dx, dy, desk_yaw,
                         humansim={"type": "desk", "capacity": 1, "interaction_radius": 1.0,
                                   "formation": {"type": "line", "params": {"base_step": step, "front_offset": front}}})
    sc.meta.update({"spacing_regime": spacing_regime, "spacing_m": step, "front_offset_m": front, "orientation": orientation,
                    "advancing": advancing, "service_s": service if advancing else None, "desk": desk["pose"], "queue_length_m": round(first + step * (n - 1), 2),
                    "bent": False, "bent_note": "humansim LineFormation places members on a straight line; bent queues are not producible"})


# =====================================================================================================
# Group walking (unpaired)
# =====================================================================================================

def build_group_walk(sc: Scene, n: int, traj: str, formation: str, safe_goal: bool = True) -> None:
    rng = sc.rng
    speed = round(rng.uniform(0.4, 1.4), 2)
    L = sc.route.length
    if traj == "same_direction":
        s0, s1, v = rng.uniform(4.0, 7.0), L + 4.0, 0.0
    elif traj == "opposite":
        s0, s1, v = min(L + 2, rng.uniform(13.0, 16.0) + speed * 4), -3.0, 0.0
    elif traj == "passing":  # starts behind the robot start, beside its path, walks past it
        s0, s1, v = -rng.uniform(2.3, 4.0), L + 6.0, 0.0
        speed = round(rng.uniform(0.8, 1.4), 2)
    elif traj in ("perpendicular", "fov_enter_leave"):
        s_c = rng.uniform(6.0, 9.0) if traj == "perpendicular" else rng.uniform(10.0, 13.0)
        lo, hi = sc.lat(s_c, 0.6)
        side = rng.choice([-1, 1])
        a, b = (hi, lo) if side > 0 else (lo, hi)
        if abs(a - b) < 3.0:
            raise Reject("narrow_cross")
        a, b = max(min(a, 6.0), -6.0), max(min(b, 6.0), -6.0)
        sx, sy = sc.w(s_c, a)
        tx, ty = sc.w(s_c, b)
        return _group(sc, n, (sx, sy), (tx, ty), speed, traj, formation, rng.uniform(W0 - 3, W0 + 6))
    else:  # corridor_narrowing: abreast group head-on in a corridor
        s0, s1, v = min(L + 2, rng.uniform(13.0, 16.0)), -3.0, 0.0
    lo, hi = sc.lat(min(max(s0, 0), L))
    if traj == "passing":
        side = rng.choice([-1, 1])
        v = side * rng.uniform(0.8, 1.3)
        if not lo <= v <= hi:
            raise Reject("narrow_passing")
    else:
        v = rng.uniform(max(lo, -0.6), min(hi, 0.6))
    start = sc.w(s0, v)
    goal = sc.w(s1, v)
    if safe_goal and s1 < 0:  # pull a goal behind the robot start back while it is inside walls / furniture
        for s_try in (s1, -2.0, -1.0, 0.0):
            if sc.free(*sc.w(s_try, v), 0.5):
                goal = sc.w(s_try, v)
                break
    delay = rng.uniform(0.0, 3.0) if traj in ("same_direction", "passing") else rng.uniform(W0 - 6, W0)
    return _group(sc, n, start, goal, speed, traj, formation, max(0.0, delay))


def _group(sc: Scene, n: int, start, goal, speed, traj, formation, delay) -> None:
    rng = sc.rng
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    sp = rng.uniform(0.7, 0.95)
    offs = []
    for i in range(n):
        if formation == "side_by_side":
            offs.append((0.0, (i - (n - 1) / 2) * sp))
        elif formation == "single_file":
            offs.append((-i * sp * 1.2, 0.0))
        elif formation == "v_formation":
            k = (i + 1) // 2
            offs.append((-k * 0.7, (1 if i % 2 else -1) * k * sp if i else 0.0))
        else:  # loose
            offs.append((rng.uniform(-1.2, 0.3), rng.uniform(-1.0, 1.0)))
    c, s_ = math.cos(heading), math.sin(heading)
    pts = [(start[0] + c * a - s_ * b, start[1] + s_ * a + c * b) for a, b in offs]
    for i, p in enumerate(pts):
        if i == 0:
            beh = {"steps": [{"wait": round(delay, 2)}, {"group_form": True}, {"goto": [goal[0], goal[1], heading]}]}
            sc.add("leader", (p[0], p[1], heading), beh, speed=speed, waypoints=[(goal[0], goal[1], heading)], min_d=0.55)
        else:
            beh = {"steps": [{"wait": 1.0}, {"group_join": True}]}
            sc.add("follower", (p[0], p[1], heading), beh, speed=min(1.6, speed + rng.uniform(0.1, 0.25)), min_d=0.55)
        sc.humans[-1]["trajectory_class"] = "group_following"
    sc.meta.update({"trajectory": traj, "requested_formation": formation, "leader_speed": speed, "group_size": n,
                    "path": [[round(start[0], 3), round(start[1], 3)], [round(goal[0], 3), round(goal[1], 3)]], "start_delay_s": round(delay, 2),
                    "formation_note": "humansim WalkFormation reshapes the group (abreast / V / single file / circle on stop) by crowding and walls; "
                                      "the executed formation is classified from the recorded positions"})


# =====================================================================================================
# Partial observability (unpaired)
# =====================================================================================================

def build_partial(sc: Scene, n: int, category: str) -> None:
    rng = sc.rng
    corners = sc.route.corners()
    if category == "fov_exit":
        if corners and rng.random() < 0.6:
            sc_ = corners[0] + rng.uniform(-0.5, 1.5)
            # target on the outside of the turn: visible on approach, out of view after the robot turns
            turn = wrap(sc.route.heading(corners[0] + 0.5) - sc.route.heading(corners[0] - 0.5))
            v_side = -1 if turn > 0 else 1
            s_t = corners[0] - rng.uniform(-0.5, 1.5)
            lo, hi = sc.lat(s_t)
            v = hi if v_side > 0 else lo
            variant = "turn_at_junction"
        else:
            s_t = rng.uniform(5.0, 7.5)
            lo, hi = sc.lat(s_t)
            side = rng.choice([-1, 1])
            v = side * rng.uniform(1.0, 1.8)
            v = min(max(v, lo), hi)
            if abs(v) < 0.8:
                raise Reject("too_close")
            variant = "pass_by"
        x, y = sc.w(s_t, v)
        moving = rng.random() < 0.35
        if moving:  # slow walk in the robot's direction, robot still overtakes it
            tx, ty = sc.w(s_t + 3.0, v)
            yaw = math.atan2(ty - y, tx - x)
            sc.add("target", (x, y, yaw), {"steps": [{"wait": W0}, {"goto": [tx, ty, yaw]}]}, speed=0.4, waypoints=[(tx, ty, yaw)])
        else:
            sc.add("target", (x, y, rng.uniform(-math.pi, math.pi)), hold())
        sc.meta.update({"variant": variant, "target_s": round(s_t, 2), "target_moving": moving})
        avoid = [(x, y)]
    elif category == "fov_reentry":
        s_t = rng.uniform(8.5, 10.5)
        lo, hi = sc.lat(s_t, 0.5)
        side = rng.choice([-1, 1])
        out = (hi if side > 0 else lo)
        if abs(out) < 3.5:
            raise Reject("needs_open_space")
        out = side * min(abs(out), 5.0)
        x, y = sc.w(s_t, rng.uniform(-0.3, 0.3))
        ox, oy = sc.w(s_t + rng.uniform(-0.5, 0.5), out)
        bx, by = sc.w(s_t - 0.5, side * rng.uniform(0.6, 1.2))
        t_leave = round(rng.uniform(W0 + 0.5, W0 + 3.0), 2)
        dwell = round(rng.uniform(3.0, 5.0), 2)
        y0 = math.atan2(oy - y, ox - x)
        y1 = math.atan2(by - oy, bx - ox)
        sc.add("target", (x, y, y0), {"steps": [{"wait": t_leave}, {"goto": [ox, oy, y0]}, {"wait": dwell}, {"goto": [bx, by, y1]}]},
               speed=0.9, waypoints=[(ox, oy, y0), (bx, by, y1)])
        sc.meta.update({"variant": "lateral_exit_and_return", "target_s": round(s_t, 2), "t_leave_s": t_leave, "dwell_s": dwell,
                        "exit_lateral_m": round(out, 2)})
        avoid = [(x, y), (ox, oy)]
    elif category == "static_occlusion":
        s_o = rng.uniform(5.5, 8.5)
        lo, hi = sc.lat(s_o, 0.8)
        side = rng.choice([-1, 1])
        v_o = side * rng.uniform(0.9, 1.6)
        v_o = min(max(v_o, lo), hi)
        if abs(v_o) < 0.8:
            raise Reject("narrow")
        model = rng.choice(spec.OCCLUDERS)
        ox, oy = sc.w(s_o, v_o)
        sc.add_object("occluder", model, ox, oy, sc.h(s_o, rng.choice([0, math.pi / 2])))
        # target hidden behind the occluder on the line of sight from the robot start
        rs = sc.route.pts[0]
        ang = math.atan2(oy - rs[1], ox - rs[0])
        d = rng.uniform(1.1, 1.6)
        x, y = ox + d * math.cos(ang), oy + d * math.sin(ang)
        steps_out = rng.random() < 0.4
        if steps_out:
            tx, ty = sc.w(s_o + d + 0.5, v_o - side * 1.2)
            if not sc.free(tx, ty):
                raise Reject("step_out_wall")
            yaw = math.atan2(ty - y, tx - x)
            t_out = round(rng.uniform(W0 + 4, W0 + 12), 2)
            sc.add("target", (x, y, yaw), {"steps": [{"wait": t_out}, {"goto": [tx, ty, yaw]}]}, speed=0.7, waypoints=[(tx, ty, yaw)])
            sc.meta["t_step_out_s"] = t_out
        else:
            sc.add("target", (x, y, rng.uniform(-math.pi, math.pi)), hold())
        sc.meta.update({"variant": "occluder_object", "occluder_model": model, "occluder_s": round(s_o, 2), "target_steps_out": steps_out})
        avoid = [(x, y), (ox, oy)]
    else:  # dynamic_occlusion: a walking group passes between robot and a standing target
        s_t = rng.uniform(9.0, 11.5)
        lo, hi = sc.lat(s_t)
        x, y = sc.w(s_t, rng.uniform(max(lo, -0.8), min(hi, 0.8)))
        sc.add("target", (x, y, sc.toward_robot(s_t) + _deg(rng.uniform(-40, 40))), hold())
        k = max(1, min(3, n - 1))
        s_g = s_t - rng.uniform(1.5, 3.0)
        lo2, hi2 = sc.lat(s_g, 0.6)
        if hi2 - lo2 > 3.5:
            side = rng.choice([-1, 1])
            a, b = (hi2, lo2) if side > 0 else (lo2, hi2)
            a, b = max(min(a, 5.0), -5.0), max(min(b, 5.0), -5.0)
            start, goal = sc.w(s_g, a), sc.w(s_g, b)
            variant = "crossing_group"
        else:  # corridor: the group walks head-on past the target toward the robot
            start, goal = sc.w(s_t + rng.uniform(3.0, 5.0), 0.0), sc.w(-3.0, 0.0)
            variant = "headon_group"
        delay = rng.uniform(W0 - 2, W0 + 6) if variant == "crossing_group" else rng.uniform(W0 - 4, W0 + 2)
        _group(sc, k, start, goal, round(rng.uniform(0.7, 1.1), 2), variant, rng.choice(["side_by_side", "loose", "single_file"]), delay)
        sc.meta.update({"variant": variant, "target_s": round(s_t, 2), "occluder_group_size": k})
        avoid = [(x, y)]
        n = n - k  # remaining bystanders computed below from the target count
        sc.meta["visibility_category"] = category
        sc.bystanders(max(0, n - 1), avoid=avoid)
        return
    sc.meta["visibility_category"] = category
    sc.bystanders(n - 1, avoid=avoid)


# =====================================================================================================
# Controls (unpaired)
# =====================================================================================================

def build_control(sc: Scene, n: int, kind: str) -> None:
    rng = sc.rng
    if kind == "empty":
        pass
    elif kind == "single_static":
        s = rng.uniform(6.0, 10.5)
        lo, hi = sc.lat(s, cap=LAT_QUEUE)
        x, y = sc.w(s, rng.uniform(lo, hi))
        sc.add("pedestrian", (x, y, rng.uniform(-math.pi, math.pi)), hold())
        sc.humans[-1]["trajectory_class"] = "static"
    elif kind == "single_moving":
        sc.bystanders(1, moving_p=1.0)
    elif kind == "two_independent":
        sc.bystanders(2, moving_p=0.5)
    else:  # neutral_crowd: 3-4 people, no interactions
        sc.bystanders(n, moving_p=0.5)
    for hh in sc.humans:
        hh["role"] = "pedestrian"
    sc.meta.update({"control": kind})
