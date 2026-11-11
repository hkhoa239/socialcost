#!/usr/bin/env bash
# Inside socialcost-arena: social layer with Nav2 DWB and MPPI on the conversation scenario.
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1
OUT=/opt/arena_ws/data/socialcost/controller_check; mkdir -p $OUT
S=/opt/arena_ws/data/socialcost/scripts; P=/opt/arena_ws/src/deps/socialcost/scripts
run() {  # name, extra launch args...
  local name=$1; shift
  echo "=== $name $(date -Is)"
  bash -l $S/launch_isaac.sh ctrl_$name task.robots:=scenario task.obstacles:=scenario task.scenario.file:=socialcost_conversation run_seed:=1 "$@" &
  sleep 60
  timeout 1500 python $P/controller_check.py --out $OUT/$name.json --timeout 1400 > $OUT/$name.log 2>&1
  echo "exit $?"
  bash $S/stop_isaac.sh; sleep 10
}
run dwb_social robot.mobile.params_overlay:=social_cost_layer robot.mobile.local_planner:=dwb
run mppi_social robot.mobile.params_overlay:=social_cost_layer robot.mobile.local_planner:=mppi
run dwb_plain robot.mobile.local_planner:=dwb
echo DONE
