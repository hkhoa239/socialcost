#!/usr/bin/env bash
# Detached (docker exec -d socialcost-arena bash .../run_repro.sh): repeated launches of the
# socialcost_repro scenario with one seed, probing every episode's timing and trajectories.
D=/opt/arena_ws/data/socialcost; S=$D/scripts; P=/opt/arena_ws/src/deps/socialcost
OUT=$D/outputs/reproducibility; mkdir -p $OUT
st() { echo "$1 $(date -Is)" >> $OUT/STATUS.txt; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1
run() {  # mode, index, extra args
  local mode=$1 i=$2; shift 2
  st "LAUNCH $mode $i"
  bash -l $S/launch_isaac.sh repro_${mode}_$i task.robots:=scenario task.obstacles:=scenario task.scenario.file:=socialcost_repro \
    run_seed:=7 robot.mobile.params_overlay:=social_cost_layer "$@" &
  timeout 1500 python $P/scripts/repro_probe.py --out $OUT/${mode}_$i.jsonl --episodes 3 --timeout 1450 >> $OUT/probe_${mode}.log 2>&1
  st "PROBE_EXIT $mode $i $?"
  bash $S/stop_isaac.sh; sleep 8
}
st BEGIN
for i in 1 2 3 4 5; do run free $i; done
for i in 1 2; do run lockstep $i lockstep:=true lockstep.paused:=false; done
st DONE
