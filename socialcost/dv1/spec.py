"""SOCIALCOST dataset_v1 frozen specification (Stage A).

Everything the manifest generator samples from lives here: worlds and their robot routes (sites),
family quotas, split quotas, pedestrian-count targets, label versions and timing. Changing a value
after the manifest is frozen means a new dataset version.
"""

from __future__ import annotations

DATASET = "socialcost_dataset_v1"
SPEC_VERSION = "1.0"
MASTER_SEED = 20261101

# --- recording -------------------------------------------------------------------------------
SAMPLE_HZ = 5.0
DURATION_S = 25.0  # recording window; 125 samples at 5 Hz
WARMUP_S = 8.0  # window opens this long after the seeded episode start
ROBOT_SPEED = 0.29  # m/s, measured on pilot_v1 (bodies-only Nav2, Regulated Pure Pursuit)
# robot arc length along its route at window start / end (start transient included)
ROBOT_S_WINDOW = (2.3, 2.3 + ROBOT_SPEED * DURATION_S)

# --- labels ----------------------------------------------------------------------------------
LABELS = {
    "social_cost": "arena-native/1.1",  # unchanged
    "social_cost_membership": "extended-membership/1.1",  # unchanged
    "social_cost_extended": "extended/2.0",  # existing pipeline channel (membership + halt + mobility hypothesis)
    "components/combined_native": "arena-native/1.1 (recomputed, must equal social_cost)",
    "components/combined_extended": "extended/2.1 (membership + halt-gesture/1.0, no mobility hypothesis)",
    "components/personal_space": "component/personal-space/1.0",
    "components/conversation": "component/interaction-native/1.0",
    "components/queue": "component/queue-membership/1.0",
    "components/gesture": "component/halt-gesture/1.0",
    "components/appearance_or_geometry_extension": "component/mobility-aid/1.0-hypothesis",
    "components/body_occupancy": "component/body-footprint/1.0",
}

# --- worlds ----------------------------------------------------------------------------------
# role: which primary splits may use the world. D is the held-out world (Test B only).
WORLDS = {
    "A": {"name": "socialcost_office", "label": "Office corridor (lightweight; replaces office_1, see reports)", "role": "seen"},
    "B": {"name": "hospital_1", "label": "Hospital corridors / waiting area", "role": "seen"},
    "C": {"name": "airport", "label": "Open lobby with freestanding seating", "role": "seen"},
    "D": {"name": "reception", "label": "Reception (held-out)", "role": "heldout"},
}

# Robot routes in the world (scenario) frame: start is the first point, goal the last.
# kind: corridor | junction | open | aisle | door. Lateral clearance is measured from the map.
SITES = {
    # World A socialcost_office: 44 m corridor, width 3 / 4 / 3 / 2.2 m, north (2.5 m) and south (3 m) side passages, doorways
    "A_hall_e1": ("A", "corridor", [(1.5, 11.5), (17.5, 11.5)]),
    "A_hall_e2": ("A", "corridor", [(16.0, 11.5), (32.0, 11.5)]),
    "A_hall_e3": ("A", "corridor", [(27.0, 11.5), (43.0, 11.5)]),
    "A_hall_w1": ("A", "corridor", [(43.0, 11.5), (27.0, 11.5)]),
    "A_hall_w2": ("A", "corridor", [(32.0, 11.5), (16.0, 11.5)]),
    "A_hall_w3": ("A", "corridor", [(17.5, 11.5), (1.5, 11.5)]),
    "A_jx_n": ("A", "junction", [(1.5, 11.5), (9.25, 11.5), (9.25, 21.5)]),
    "A_jx_s": ("A", "junction", [(20.0, 11.5), (29.5, 11.5), (29.5, 1.5)]),
    "A_jx_up": ("A", "junction", [(29.5, 1.5), (29.5, 11.5), (17.0, 11.5)]),
    # World B hospital_1: 4 m central hallway, 3 m sub-hallway to the waiting area
    "B_hall_n1": ("B", "corridor", [(10.0, 1.5), (10.0, 17.5)]),
    "B_hall_n2": ("B", "corridor", [(10.0, 17.0), (10.0, 33.0)]),
    "B_hall_s1": ("B", "corridor", [(10.0, 33.0), (10.0, 17.0)]),
    "B_hall_s2": ("B", "corridor", [(10.0, 17.5), (10.0, 1.5)]),
    "B_jx_e": ("B", "junction", [(10.0, 1.5), (10.0, 11.55), (24.0, 11.55)]),
    "B_jx_w": ("B", "junction", [(24.0, 11.55), (10.0, 11.55), (10.0, 26.0)]),
    # World C airport: 45 x 30 m open hall, rows of seats at x 30-40
    "C_open_e1": ("C", "open", [(2.0, 9.5), (18.0, 9.5)]),
    "C_open_e2": ("C", "open", [(12.0, 22.0), (28.0, 22.0)]),
    "C_open_e3": ("C", "open", [(2.0, 19.5), (18.0, 19.5)]),
    "C_open_w1": ("C", "open", [(27.0, 8.0), (11.0, 8.0)]),
    "C_open_w2": ("C", "open", [(19.0, 11.5), (3.0, 11.5)]),
    "C_open_w3": ("C", "open", [(27.0, 20.0), (11.0, 20.0)]),
    "C_open_n1": ("C", "open", [(17.0, 1.5), (17.0, 17.5)]),
    "C_open_n2": ("C", "open", [(27.0, 4.0), (27.0, 20.0)]),
    "C_open_s1": ("C", "open", [(3.0, 28.5), (3.0, 12.5)]),
    "C_open_s2": ("C", "open", [(17.0, 28.5), (17.0, 12.5)]),
    "C_aisle_w1": ("C", "aisle", [(44.0, 12.4), (28.0, 12.4)]),
    "C_aisle_w2": ("C", "aisle", [(44.0, 17.9), (28.0, 17.9)]),
    "C_aisle_e1": ("C", "aisle", [(27.0, 12.4), (43.0, 12.4)]),
    "C_aisle_e2": ("C", "aisle", [(27.0, 17.9), (43.0, 17.9)]),
    # World D reception (held-out): open entrance hall, reception counter, 3 m door to a second room
    "D_hall_s1": ("D", "open", [(27.0, 21.5), (27.0, 5.5)]),
    "D_hall_s2": ("D", "open", [(21.5, 21.5), (21.5, 5.5)]),
    "D_hall_n1": ("D", "open", [(25.0, 3.0), (25.0, 19.0)]),
    "D_counter_w": ("D", "open", [(29.0, 12.5), (17.0, 12.5)]),
    "D_hall_e": ("D", "open", [(16.5, 19.0), (28.5, 19.0)]),
    "D_door_w": ("D", "door", [(28.0, 3.5), (15.0, 3.5), (6.0, 3.5)]),
}

# --- families ----------------------------------------------------------------------------------
# quota, paired, pedestrian-count range (inclusive), preferred world weights over A, B, C
FAMILIES = {
    "conversation": {"quota": 120, "paired": True, "peds": (2, 4), "worlds": {"A": 0.4, "B": 0.3, "C": 0.3}},
    "stop_gesture": {"quota": 100, "paired": True, "peds": (1, 4), "worlds": {"A": 0.5, "B": 0.25, "C": 0.25}},
    "wheelchair": {"quota": 100, "paired": True, "peds": (1, 4), "worlds": {"A": 0.25, "B": 0.5, "C": 0.25}},
    "queue": {"quota": 100, "paired": True, "peds": (3, 8), "worlds": {"A": 0.25, "B": 0.4, "C": 0.35}},
    "group_walking": {"quota": 60, "paired": False, "peds": (2, 6), "worlds": {"A": 0.4, "B": 0.2, "C": 0.4}},
    "partial_observability": {"quota": 80, "paired": False, "peds": (2, 6), "worlds": {"A": 0.35, "B": 0.25, "C": 0.4}},
    "controls": {"quota": 40, "paired": False, "peds": (0, 4), "worlds": {"A": 0.34, "B": 0.33, "C": 0.33}},
}
# site kinds a family may use, per world
FAMILY_SITES = {
    "conversation": ("corridor", "open", "aisle", "junction"),
    "stop_gesture": ("corridor", "open", "aisle"),
    "wheelchair": ("corridor", "open", "aisle"),
    "queue": ("corridor", "open", "aisle"),
    "group_walking": ("corridor", "open", "aisle"),
    "partial_observability": ("junction", "corridor", "open", "aisle", "door"),
    "controls": ("corridor", "open", "aisle"),
}

# --- splits ----------------------------------------------------------------------------------
SPLITS = ("train", "val", "test_a", "test_b", "test_c")
SPLIT_FRACTION = {"train": 0.6, "val": 0.1, "test_a": 0.1, "test_b": 0.1, "test_c": 0.1}
# test_b: held-out world D with known bodies; test_c: seen worlds with held-out bodies
SPLIT_WORLD_ROLE = {"train": "seen", "val": "seen", "test_a": "seen", "test_b": "heldout", "test_c": "seen"}
SPLIT_BODY_ROLE = {"train": "known", "val": "known", "test_a": "known", "test_b": "known", "test_c": "heldout"}

# --- pedestrian-count target (fraction of episodes) ------------------------------------------
PED_COUNT_BINS = {(0, 1): 0.10, (2, 2): 0.25, (3, 4): 0.30, (5, 6): 0.25, (7, 10): 0.10}

# --- stages ----------------------------------------------------------------------------------
PILOT_FRACTION = 0.2  # Stage B: 120 of 600
SUBSET_PER_FAMILY = {"conversation": 2, "stop_gesture": 1, "wheelchair": 1, "queue": 2, "group_walking": 1,
                     "partial_observability": 2, "controls": 1}  # Stage A check, units (pairs or singles)

# --- geometry limits ---------------------------------------------------------------------------
PED_WALL_CLEARANCE = 0.40  # m from walls / furniture to a pedestrian centre
PED_PED_MIN = 0.55  # m between pedestrian centres (dense queue lower bound 0.6)
ROBOT_START_PED_MIN = 2.0  # m from the robot start to any pedestrian at spawn
MATCH_POS_M = 0.02  # counterfactual matched-geometry tolerances (stationary windows)
MATCH_HEADING_DEG = 2.0

# --- camera ----------------------------------------------------------------------------------
CAMERA_HFOV_DEG = 71.6

# Per-family allocation of episodes to pedestrian-count bins. Family ranges (FAMILIES[...]["peds"]) make the
# PED_COUNT_BINS target infeasible for 5-6 (only queue, group walking and partial observability may exceed 4),
# so the allocation keeps every family inside its range and reports the shortfall:
# planned 0-1: 60 (10.0 %), 2: 148 (24.7 %), 3-4: 210 (35.0 %), 5-6: 122 (20.3 %), 7-10: 60 (10.0 %).
PED_COUNT_ALLOCATION = {
    "controls": {(0, 1): 24, (2, 2): 8, (3, 4): 8},
    "stop_gesture": {(0, 1): 18, (2, 2): 22, (3, 4): 60},
    "wheelchair": {(0, 1): 18, (2, 2): 22, (3, 4): 60},
    "conversation": {(2, 2): 82, (3, 4): 38},
    "queue": {(3, 4): 10, (5, 6): 30, (7, 10): 60},
    "group_walking": {(2, 2): 6, (3, 4): 14, (5, 6): 40},
    "partial_observability": {(2, 2): 8, (3, 4): 20, (5, 6): 52},
}

# occluder models already rendered by Isaac in Arena worlds (USD available)
OCCLUDERS = ["vending_machine", "Hospital/SM_MedicalCabinet_01a", "Hospital/SM_DrinksMachine"]
QUEUE_DESK_MODEL = "table"  # humansim desk, as in the pilot queue pairs
