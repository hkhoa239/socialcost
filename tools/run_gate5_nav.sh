#!/usr/bin/env bash
# Detached after the experiments: Gate 5 (mask path into Nav2), then closed-loop runs and metrics.
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; S=$D/outputs/STATUS_phase1.txt; G=$D/outputs/gate5
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P; mkdir -p $G
export PYTHONPATH=$P:$PYTHONPATH
gate() {  # name method [checkpoint]
  bash -l $D/scripts/launch_isaac.sh gate5_$1 task.robots:=scenario task.obstacles:=scenario task.scenario.file:=socialcost_nav_conversation run_seed:=1 \
    robot.mobile.params_overlay:=socialcost_estimate robot.mobile.local_planner:=dwb &
  python -m socialcost.estimator_node --method $2 ${3:+--checkpoint $3} --log $G/$1_estimator.jsonl > $G/$1_estimator.log 2>&1 &
  EST=$!
  sleep 60
  timeout 1500 python scripts/controller_check.py --out $G/$1.json --timeout 1400 --tolerance 3 --window 10 > $G/$1.log 2>&1
  st "GATE5_$1_EXIT $?"
  kill -INT $EST; bash $D/scripts/stop_isaac.sh; sleep 8
}
st GATE5_BEGIN
gate oracle_ext oracle_ext
gate rgbd rgbd $D/experiments/pilot_v1/pilot_rgbd_extended/best.pt
python - <<PY >> $S
import json
r = json.load(open("$G/oracle_ext.json"))
ok = r.get("merged_ge_social_fraction_min") is not None and r["merged_ge_social_fraction_min"] >= 0.999 and "social_estimate_layer" in r.get("local_plugins", "")
print("GATE5_RESULT", "PASS" if ok else "FAIL", r.get("merged_ge_social_fraction_min"), r.get("costmaps_checked"))
PY
if grep -q "GATE5_RESULT PASS" $S; then
  st NAV_DWB_BEGIN; python scripts/run_nav_eval.py --config configs/navigation/nav_v1_dwb.yaml >> $D/experiments/navigation_dwb.log 2>&1; st "NAV_DWB_EXIT $?"
  st NAV_MPPI_BEGIN; python scripts/run_nav_eval.py --config configs/navigation/nav_v1_mppi.yaml >> $D/experiments/navigation_mppi.log 2>&1; st "NAV_MPPI_EXIT $?"
  python scripts/nav_metrics.py $D/experiments/navigation/nav_v1 --out $D/outputs/evaluation/navigation_v1.json >> $D/experiments/navigation_metrics.log 2>&1; st "NAV_METRICS_EXIT $?"
fi
st NAV_DONE
