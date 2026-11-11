# socialcost

SOCIALCOST data pipeline on Arena + Isaac Sim: robot RGB-D frames paired with the Arena social cost layer, recomputed offline from privileged pedestrian states.

| Path | Contents |
|---|---|
| `socialcost/labels.py`, `src/socialcost_grid.cpp` | Offline painter over `arena_social_cost_layer/cost_math.hpp`, parameters from the Nav2 overlay |
| `socialcost/live.py` | Nav2 costmap holding only the social layer (live reference) |
| `socialcost/recorder.py` | Sim-time RGB-D / pose / pedestrian recorder |
| `socialcost/dataset.py` | Labelling, membership-completed queue term, loader, splits |
| `socialcost/qa.py` | Automated dataset checks |
| `scripts/collect_dataset.py` | Headless batch collection (launch, record, label, per scenario × seed) |
| `scripts/record_episode.py` | Record one episode of a running env |
| `scripts/validate_dataset.py`, `scripts/visualize_dataset.py` | QA report, sample sheets |
| `scripts/controller_check.py` | Social layer inside Nav2 DWB / MPPI runs |
| `scripts/launch/` | `start_sibling.sh` (host), `launch_isaac.sh`, `stop_isaac.sh` (container) |
| `configs/scenarios/` | Links to `arena_simulation_setup/worlds/map_empty/scenarios/socialcost_*` |
| `configs/sensors/turtlebot_oakd.yaml` | Topics and frames of the TurtleBot4 OAK-D on Isaac |
| `configs/dataset/smoke.yaml`, `smoke_repro.yaml` | Smoke-test dataset and its determinism repeat |
| `tests/` | `test_labels.py` (no ROS), `test_live_agreement.py` (real plugin) |
| `docs/` | Audit, installation, costmap pipeline, dataset format, validation report |

Outputs: `/home/robotics/arena_ws/data/socialcost/` (`smoke_test/`, `smoke_test_repro/`, `controller_check/`, `logs/`).

Start with [docs/INSTALLATION.md](docs/INSTALLATION.md).
