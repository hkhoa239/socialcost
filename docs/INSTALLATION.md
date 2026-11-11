# Installation and reproduction

Exact steps used on this machine. Host paths are under `/home/robotics/arena_ws`, container paths under `/opt/arena_ws` (the same tree).

## 1. Isaac feature (once)

Inside the main Arena container, from `/opt/arena_ws`:

```bash
docker exec -it -u arena -w /opt/arena_ws arena-arena_ws-arena-1 bash
source /opt/arena_ws/source
arena feature isaac install        # git submodule update --init arena_isaac, builds arena_isaac:<tag> on isaac-sim:6.1.0
```

The base image is pulled once (`docker pull nvcr.io/nvidia/isaac-sim:6.1.0`, ~32 GB). `arena feature docker gpu on` was already set (`.env`: `ARENA_GPU=1`).

Build the new ROS packages. Use `--packages-select`: `arena build <pkg>` also rebuilds every dependent package.

```bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/opt/venv/bin/python \
  --base-paths /opt/arena_ws/src --build-base /opt/arena_ws/build --install-base /opt/arena_ws/install \
  --packages-select isaacsim_msgs arena_runtime arena_simulation_setup socialcost
```

`arena_simulation_setup` must be rebuilt whenever a scenario directory is added (its install tree holds per-file symlinks).

## 2. The `socialcost-arena` container

The main container also runs other jobs whose cleanup kills every `arena_runtime.launch.py` in it. SOCIALCOST therefore runs in a sibling container with the same contents and mounts, its own PID namespace and ROS domain 7:

```bash
# host; the snapshot carries the apt packages installed into the main container after its creation
docker commit --pause=false arena-arena_ws-arena-1 socialcost-arena:snapshot-20261029   # once
bash /home/robotics/arena_ws/data/socialcost/scripts/start_sibling.sh
```

`--pause=false` keeps the main container's jobs running during the commit. The sibling drives the same compose `isaac` service (`arena-arena_ws-isaac-1`); only one Isaac instance can run at a time.

If nothing else uses the main container, the same commands work there after `export ROS_DOMAIN_ID=7`.

## 3. Launch one scenario

```bash
docker exec -it socialcost-arena bash -l
cd /opt/arena_ws && source /opt/arena_ws/source       # ROS_DOMAIN_ID=7 comes from the container env
arena launch sim:=isaac headless:=true world:=map_empty robot:=turtlebot human:=arena \
  task.robots:=scenario task.obstacles:=scenario task.scenario.file:=socialcost_conversation \
  run_seed:=1 robot.mobile.params_overlay:=social_cost_layer
```

| Scenario | `task.scenario.file:=` |
|---|---|
| A conversation | `socialcost_conversation` |
| B queue | `socialcost_queue` |
| C group walking | `socialcost_group_walk` |

Optional: `robot.mobile.local_planner:=dwb` or `:=mppi` (default Regulated Pure Pursuit). Detached form with a log: `docker exec -d socialcost-arena bash -l /opt/arena_ws/data/socialcost/scripts/launch_isaac.sh <logname> <args...>`, stop with `docker exec socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/stop_isaac.sh`.

First launch after a container start compiles shaders (~3 min); later launches reach robot TF in ~40 s.

## 4. Tests

```bash
cd /opt/arena_ws/src/deps/socialcost
python -m pytest tests/test_labels.py -q                          # no ROS graph
ROS_DOMAIN_ID=9 python -m pytest tests/test_live_agreement.py -q  # starts the real plugin, ~75 s
```

## 5. Dataset

```bash
# inside socialcost-arena, sourced
cd /opt/arena_ws/src/deps/socialcost
python scripts/collect_dataset.py configs/dataset/smoke.yaml        # 3 scenarios x 2 seeds, ~15 min
python scripts/collect_dataset.py configs/dataset/smoke_repro.yaml  # seed 1 again, for the determinism check
python scripts/validate_dataset.py /opt/arena_ws/data/socialcost/smoke_test --repro /opt/arena_ws/data/socialcost/smoke_test_repro
python scripts/visualize_dataset.py /opt/arena_ws/data/socialcost/smoke_test
```

Collection is resumable: valid episodes are skipped on a rerun, and `--only <scenario>:<seed>` runs one job. A larger dataset is another yaml in `configs/dataset/` with more `seeds`, `episodes_per_seed` or `scenarios`; nothing else changes.

## 6. Controller check (DWB / MPPI)

```bash
bash /opt/arena_ws/data/socialcost/scripts/launch_isaac.sh ctrl_dwb task.robots:=scenario task.obstacles:=scenario \
  task.scenario.file:=socialcost_conversation run_seed:=1 robot.mobile.params_overlay:=social_cost_layer robot.mobile.local_planner:=dwb &
python scripts/controller_check.py --out /opt/arena_ws/data/socialcost/controller_check/dwb.json
```
