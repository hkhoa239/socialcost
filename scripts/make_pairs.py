#!/usr/bin/env python3
"""Generate the paired social scenarios (ON/OFF conditions with matched geometry).

Writes arena-scenario/2 directories socialcost_pair_<family>_<body set>_<on|off> into a world's
scenarios folder (rebuild arena_simulation_setup afterwards) and pairs.json listing every pair.
Both conditions of a pair share people, bodies, spawn poses, headings, robot start and goal;
only the social factor differs.

    python make_pairs.py --world map_empty
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import yaml

ARENA = os.environ.get("ARENA_DIR", "/opt/arena_ws/src/Arena")

# body sets: train draws from Common bodies, the held-out set uses Office and Hospital bodies
BODIES = {
    "bA": ["casual_female_african_young", "worker_male_asian_middleage", "arenian", "casual_male_caucasian_middleage"],
    "bB": ["casual_female_caucasian_young", "worker_male_caucasian_young", "casual_female_asian_middleage", "worker_male_african_young"],
    "bC": ["worker_male_caucasian_middleage", "casual_female_african_young", "worker_male_african_young", "casual_female_caucasian_young"],
    "bD": ["worker_male_african_young", "casual_male_caucasian_middleage", "casual_female_asian_middleage", "worker_male_asian_middleage"],
    "bH": ["Office/office_female_asian_young", "Office/office_male_caucasian_middleage", "Hospital/nurse_female_caucasian_young", "Office/office_male_african_young"],
}
# per body-set placement offset (m) of every person and object, so pairs differ in geometry too;
# both conditions of a pair share it. bD (training bodies, new placement) is the in-distribution test.
OFFSETS = {"bA": (0.0, 0.0), "bB": (0.6, -0.3), "bC": (-0.5, 0.4), "bD": (1.0, 0.3), "bH": (-0.8, -0.2)}
WHEELCHAIR = "casual_male_asian_middleage_wheelchair"
ROBOT = {"start": [8.0, 11.5, 0.0], "phases": [{"goto": [26.0, 11.5, 0.0]}]}


def hold(name: str, extends: str = "adult") -> dict:
    return {"name": name, "mode": "behavior_tree", "extends": extends, "initial_sequence": "stand",
            "sequences": {"stand": {"steps": {"hold": {"duration": {"mean": 1000.0, "std": 0.0}}}, "then": "stand"}}}


def bt(name: str, step: dict, extends: str = "adult") -> dict:
    return {"name": name, "mode": "behavior_tree", "extends": extends, "initial_sequence": "act",
            "sequences": {"act": {"steps": {"act": step}, "then": "act"}}}


def human(name: str, model: str, pose: list[float], agent_file: str) -> dict:
    return {"name": name, "model": model, "pose": pose, "agent": {"agent_type": f"./{agent_file}"}}


def family_conversation(bodies: list[str], on: bool) -> tuple[dict, dict]:
    files = {"talker.yaml": bt("pair_talker", {"interaction": "TALK_TO", "attention": {"gaze": "partner"}, "duration": {"mean": 1000.0, "std": 0.0}}) if on else hold("pair_still")}
    humans = [human("person_a", bodies[0], [14.4, 13.2, 0.0], "talker.yaml"), human("person_b", bodies[1], [15.6, 13.2, 3.14159], "talker.yaml")]
    return {"humans": humans}, files


def family_halt(bodies: list[str], on: bool) -> tuple[dict, dict]:
    # facing the oncoming robot (west); the raised hand points at the robot's approach, 10 deg up
    files = {"person.yaml": bt("pair_halt", {"attention": {"halt": {"azimuth": 0, "elevation": 10}}, "duration": {"mean": 1000.0, "std": 0.0}}) if on else hold("pair_still")}
    return {"humans": [human("person_a", bodies[0], [15.0, 13.0, 3.14159], "person.yaml")]}, files


def family_wheelchair(bodies: list[str], on: bool) -> tuple[dict, dict]:
    files = {"person.yaml": hold("pair_still", "wheelchair_manual" if on else "adult")}
    return {"humans": [human("person_a", WHEELCHAIR if on else "casual_male_caucasian_middleage", [15.0, 13.0, 3.14159], "person.yaml")]}, files


# queue slots where the QUEUE_USE formation settles (recorded ON run, map minus the +5.3 m scenario
# offset), facing the table (+y); spawning there keeps ON members where OFF members stand
QUEUE_SLOTS = [[14.99, 16.23], [15.40, 16.01], [14.98, 15.38], [14.93, 14.65]]


def family_queue(bodies: list[str], on: bool) -> tuple[dict, dict]:
    files = {"person.yaml": bt("pair_queuer", {"target": "desk", "interaction": "QUEUE_USE", "duration": {"mean": 1000.0, "std": 0.0}, "patience": {"mean": 2000.0, "std": 0.0}}) if on else hold("pair_still")}
    humans = [human(f"person_{c}", bodies[i], [*QUEUE_SLOTS[i], 1.5708], "person.yaml") for i, c in enumerate("abcd")]
    objects = [{"name": "service_point", "model": "table", "pose": [15.0, 17.5, 1.5708],
                "humansim": {"type": "desk", "capacity": 1, "interaction_radius": 1.0, "formation": {"type": "line", "params": {"base_step": 0.8, "front_offset": 0.6}}}}]
    return {"objects": objects, "humans": humans}, files


FAMILIES = {
    "conversation": (family_conversation, "visually observable (talk-gesture clip, partner gaze); clearer over a short history"),
    "halt": (family_halt, "visually observable (raised hand), to be confirmed in rendered frames"),
    "wheelchair": (family_wheelchair, "visually observable; the two conditions also differ in body asset (confound)"),
    "queue": (family_queue, "not visually observable from a single frame: same people, poses and table in both conditions"),
}


def shift(body: dict, dx: float, dy: float = 0.0) -> dict:
    out = {}
    for key, items in body.items():
        out[key] = [{**it, "pose": [it["pose"][0] + dx, it["pose"][1] + dy, *it["pose"][2:]]} for it in items]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="map_empty")
    ap.add_argument("--body-sets", default=",".join(BODIES))
    ap.add_argument("--dx", type=float, default=0.0, help="shift every pose along x (worlds with other free space)")
    ap.add_argument("--goal-x", type=float, default=None)
    args = ap.parse_args()
    scen_dir = os.path.join(ARENA, "arena_simulation_setup", "worlds", args.world, "scenarios")
    pairs = []
    for fam, (fn, observability) in FAMILIES.items():
        for bs in args.body_sets.split(","):
            pid = f"{fam}_{bs}"
            for on in (True, False):
                name = f"socialcost_pair_{pid}_{'on' if on else 'off'}"
                body, files = fn(BODIES[bs], on)
                d = os.path.join(scen_dir, name)
                if os.path.isdir(d):
                    shutil.rmtree(d)
                os.makedirs(d)
                ox, oy = OFFSETS[bs]
                body = shift(body, args.dx + ox, oy)
                robot = {"start": [ROBOT["start"][0] + args.dx, *ROBOT["start"][1:]],
                         "phases": [{"goto": [args.goal_x if args.goal_x is not None else ROBOT["phases"][0]["goto"][0] + args.dx, *ROBOT["phases"][0]["goto"][1:]]}]}
                doc = {"format": "arena-scenario/2", **body, "robots": [robot]}
                with open(os.path.join(d, "scenario.yaml"), "w") as f:
                    f.write(f"# SOCIALCOST paired scenario {pid}, condition {'ON' if on else 'OFF'} (generated by make_pairs.py).\n")
                    yaml.safe_dump(doc, f, sort_keys=False)
                for fname, content in files.items():
                    with open(os.path.join(d, fname), "w") as f:
                        yaml.safe_dump(content, f, sort_keys=False)
            pairs.append({"pair_id": pid if args.world == "map_empty" else f"{args.world}:{pid}", "family": fam, "body_set": bs, "held_out_bodies": bs == "bH",
                          "placement_offset": list(OFFSETS[bs]), "world": args.world,
                          "on": f"socialcost_pair_{pid}_on", "off": f"socialcost_pair_{pid}_off", "observability": observability})
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs", "dataset", f"pairs_{args.world}.json")
    json.dump(pairs, open(out, "w"), indent=2)
    print(f"{len(pairs)} pairs, {2 * len(pairs)} scenarios in {scen_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
