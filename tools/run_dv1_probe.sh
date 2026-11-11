#!/usr/bin/env bash
# dataset_v1 world probe: rebuild scenarios, collect one episode per candidate world.
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; S=$D/dataset_v1_STATUS.txt
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1
echo "PROBE_BEGIN $(date -Is)" >> $S
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/opt/venv/bin/python --base-paths /opt/arena_ws/src --build-base /opt/arena_ws/build --install-base /opt/arena_ws/install --packages-select arena_simulation_setup > $D/logs/dv1_probe_build.log 2>&1
rc=$?; echo "PROBE_BUILD_EXIT $rc $(date -Is)" >> $S; [ $rc -eq 0 ] || exit 1
cd $P; mkdir -p $D/datasets/dv1_world_probe
python scripts/collect_dataset.py configs/dataset/dv1_world_probe.yaml >> $D/datasets/dv1_world_probe/collect.log 2>&1
echo "PROBE_EXIT $? $(date -Is)" >> $S
