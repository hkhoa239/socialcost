#!/usr/bin/env bash
# Install the SOCIALCOST additions into an Arena checkout (https://github.com/voshch/Arena, revision in arena_base.txt).
#   bash apply.sh /path/to/arena_ws/src/Arena
# Adds: run_seed launch argument (task_generator), halt-gesture republish on reset (humansim), Isaac render-every flag
# (arena_isaac), the socialcost_estimate* Nav2 overlays (arena_robots), the socialcost_office world and every scv1_* /
# socialcost_* scenario. Then rebuild: colcon build --packages-select arena_simulation_setup arena_robots task_generator ...
set -euo pipefail
A=${1:?path to src/Arena}; P=$(cd "$(dirname "$0")" && pwd)
git -C "$A" apply --whitespace=nowarn "$P/task_generator_run_seed.patch"
git -C "$A/humansim" apply --whitespace=nowarn "$P/humansim_gesture_republish.patch"
git -C "$A/arena_isaac" apply --whitespace=nowarn "$P/arena_isaac_render_every.patch"
cp "$P"/arena_robots_overlays/*.yaml "$A/arena_robots/arena_robots/config/nav2/overlays/"
rsync -a "$P/worlds/" "$A/arena_simulation_setup/worlds/"
for w in "$P"/scenarios/*; do rsync -a "$w/" "$A/arena_simulation_setup/worlds/$(basename "$w")/scenarios/"; done
echo "applied; now rebuild arena_simulation_setup arena_robots task_generator arena_isaac arena_humansim"
