# Environment audit (2026-10-29)

## Host

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| CPU / RAM | AMD Ryzen 9 5900X (24 threads), 62 GB |
| GPU | 1× RTX 3060, 12 GB, driver 580.178.04 (CUDA 13.0) |
| Disk | 916 GB, 308 GB free before Isaac; Isaac 6.1.0 image adds ~32 GB |
| Container engine | Docker 29.6.1, Compose v5.3.0, NVIDIA Container Toolkit 1.19.1 (CDI `nvidia.com/gpu=all`), `nvidia` runtime registered |
| Host ROS | `/opt/ros/humble` exists, not used. Arena runs ROS 2 Jazzy inside its containers |
| Host Python | 3.10.12, not used. Arena venv is CPython 3.12 (`src/Arena/.venv`, `/opt/venv` in the container) |

The RTX 3060 is below NVIDIA's recommended Isaac Sim GPU (RTX 4080-class) but runs Isaac 6.1 headless with one 320×240 RGB-D camera. Measured real-time factor during recording: about 0.26–0.36 (see VALIDATION_REPORT.md).

## Workspace

`/home/robotics/arena_ws`, Arena `jazzy` branch at `c9de66d1`. The simulator stack runs in Docker:

| Container | Role |
|---|---|
| `arena-arena_ws-arena-1` | Main Arena container (`arena:arena-arena_ws`). **In use by the detached mean-field benchmark** (`data/mf_bench/run_bench.sh`, MuJoCo, ROS domain 1). |
| `arena-arena_ws-isaac-1` | Isaac Sim service of the same compose project, created by `arena feature isaac install`. |
| `socialcost-arena` | Added for this work: same image contents and mounts as the main container, own PID namespace, ROS domain 7. See INSTALLATION.md for why. |

Submodules (`git submodule status`):

| Submodule | Commit | State |
|---|---|---|
| arena_isaac | `7dc86d6` (isaac6.1.0, "grounded feet", 2026-10-22) | **initialized in this work** |
| arena_assets | `be87819` | checked out |
| arena_mujoco | `fea3a8c` | checked out, modified (benchmark) |
| arena_planners (incl. `arena_social_cost_layer`) | `760b63b` | checked out |
| arena_robots | `a352f3b` | checked out |
| arena_training | `1e86279` | checked out, modified (benchmark) |
| flatsim | `d077081` | checked out |
| humansim (`arena_humansim`) | `022b183` | checked out |
| arena_auditory, arena_evaluation, arena_hearing, arena_tools | – | not initialized |

`task_generator`, `arena_bringup`, `arena_runtime`, `arena_people_msgs` live in the main Arena repository. `arena_evaluation` (the MCAP episode recorder, `record.dir:=`) is not initialized; the dataset recorder in this package does not depend on it.

Installed Arena features before this work: `docker`, `flatsim`, `training`, `mujoco`. Added: `isaac`.

## Components checked

| Component | Finding |
|---|---|
| Isaac integration | `sim:=isaac` runs `arena feature isaac launch`, which brings up the `isaac` compose service and runs `arena_isaac/run_isaacsim.py` in it. The local `arena_isaac:arena-arena_ws` image was a July build on Isaac 6.0.1; the current Dockerfile requires `nvcr.io/nvidia/isaac-sim:6.1.0`. Rebuilt. |
| Isaac version | `6.1.0-rc.26+release.49347`, Python 3.12.13, physics PhysX |
| `arena_social_cost_layer` | Nav2 `costmap_2d::Layer` plugin, registered via `arena_social_cost_layer.xml`. Tests: 20 pass, 4 skipped (`arena test arena_social_cost_layer`). |
| Overlay | `arena_robots/config/nav2/overlays/social_cost_layer.yaml`, loads the layer into **both** local and global costmaps, `robot.mobile.params_overlay:=social_cost_layer` |
| Pedestrian topic | `/arena/env_0/arena_peds`, `arena_people_msgs/Pedestrians`, published by the humansim bridge in `task_generator`, frame `map` |
| Robot with RGB-D | `turtlebot` (TurtleBot4) carries an OAK-D declared as a gz `rgbd_camera`; Isaac maps it to RGB + depth + camera_info + points. No new sensor needed. |
| Nav2 | Jazzy 1.3.13. Default controller for the turtlebot is Regulated Pure Pursuit; `robot.mobile.local_planner:=dwb` / `mppi` select DWB / MPPI. |
| Humansim | Interactions `TALK_TO`, `GROUP_CONVERSATION`, `QUEUE_USE`, `GROUP_WALK` and others; scenarios in `arena-scenario/2` format under `arena_simulation_setup/worlds/<world>/scenarios/` |

## Version and compatibility issues found

1. **`arena_isaac` not checked out, Isaac image stale.** Fixed by `arena feature isaac install` (pulls 6.1.0, rebuilds the image, registers the feature).
2. **`arena build` rebuilds dependents** (`--packages-above`). Building `arena_isaac` also rebuilt `arena_runtime`, which then failed on the missing `isaacsim_msgs` (now provided by the submodule). Nothing in `install/` changed. Fixed by `colcon build --packages-select isaacsim_msgs arena_runtime`.
3. **One Arena runtime per ROS domain and per container.** `/arena/register_env` is a ROS service, so a second stack works on another `ROS_DOMAIN_ID`. Same-container runs are not safe: the benchmark's `eval_one.sh` runs `pkill -f arena_runtime.launch.py` between steps and killed the first Isaac launch. Hence the separate `socialcost-arena` container.
4. **The main container carries ~200 apt packages installed after it was created** (Nav2, controllers, ...). A container from the bare image lacks `controller_manager_msgs`, `hri_msgs`. The sibling is therefore created from a `docker commit --pause=false` snapshot (`socialcost-arena:snapshot-20261029`).
5. **Scenario directories must be real directories.** `arena_simulation_setup` installs scenarios as per-file symlinks and skips directory symlinks; new scenarios need `colcon build --packages-select arena_simulation_setup`.
6. **Isaac needs a USD representation for objects.** `Hospital/Object/SM_BedSideTable_01b` has none and is silently not spawned; `table` (map_empty asset) and `SM_TrashCan` have one.
7. **Bare body names resolve in the `Common` domain only.** Use `Office/...`, `Hospital/...` for the other bodies.
8. **TurtleBot bumper contact sensor fails to initialise under Isaac** (`Contact Sensor needs to be created under another prim that has collision api`). Not needed for the dataset; collision events from the bumper are unavailable on Isaac.
9. Cosmetic: missing wood textures on `painting` wall decorations (`Oak_*.png`), harmless.

## Decisions

- Isaac via the supported `arena feature isaac install` path, no manual Isaac install, no simulator substitution.
- All SOCIALCOST runs on ROS domain 7 in `socialcost-arena`, leaving the benchmark on domain 1 in `arena-arena_ws-arena-1` untouched.
- New code lives in `src/deps/socialcost` (outside the Arena git checkout, picked up by `arena build`). Scenario files live where Arena reads them, `arena_simulation_setup/worlds/map_empty/scenarios/socialcost_*`, and are linked from `socialcost/configs/scenarios/`.
