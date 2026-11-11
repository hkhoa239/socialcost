#!/usr/bin/env bash
# Gate 5 v2 (RGB-D diagnosis, Phases A/B/E): per-estimator audit of the costmap pipeline.
# Each arm: fresh launch of socialcost_nav_conversation, run_seed 1, DWB, overlay socialcost_estimate;
# estimator with intermediate-grid dumps (+ oracle shadow for learned arms); pipeline_audit compares every
# local/global costmap with the grids actually published, from bring-up (static) through the episode.
# usage (inside socialcost-arena): bash run_gate5_v2.sh [arm ...]   default: all arms
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; O=$D/outputs/rgbd_diagnosis/gate5_v2
S=$D/outputs/rgbd_diagnosis/STATUS.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source; cd $P; mkdir -p $O
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
CK=$D/experiments/pilot_v1
declare -A ARGS=(
  [oracle_ext]="--method oracle_ext"
  [oracle_ext_replay]="--method oracle_ext_replay"
  [rgbd]="--method rgbd --checkpoint $CK/pilot_rgbd_extended/best.pt --shadow"
  [rgbd_t13]="--method rgbd --checkpoint $CK/pilot_rgbd_extended/best.pt --shadow --post $D/outputs/rgbd_diagnosis/specs/rgbd_threshold13.json"
  [depth]="--method depth --checkpoint $CK/pilot_depth_extended/best.pt --shadow"
  [pos]="--method pos --checkpoint $CK/pilot_pos_extended/best.pt --shadow"
)
ARMS=${@:-oracle_ext oracle_ext_replay rgbd rgbd_t13 depth pos}
st GATE5V2_BEGIN
for a in $ARMS; do
  R=$O/$a; rm -rf $R; mkdir -p $R
  bash -l $D/scripts/launch_isaac.sh gate5v2_$a task.robots:=scenario task.obstacles:=scenario task.scenario.file:=socialcost_nav_conversation run_seed:=1 \
    robot.mobile.params_overlay:=socialcost_estimate robot.mobile.local_planner:=dwb &
  python -m socialcost.estimator_node ${ARGS[$a]} --log $R/estimator.jsonl --dump-dir $R/dumps --dump-every 10 > $R/estimator.log 2>&1 &
  EST=$!
  timeout 1700 python scripts/pipeline_audit.py --out $R/audit.jsonl --summary $R/audit_summary.json --estimator-log $R/estimator.jsonl \
    --until-episode-end --min-episode-id 2 --static-test 30 --duration 1600 --snapshots $R/snapshots > $R/audit.log 2>&1 &
  AUD=$!
  timeout 600 python -m socialcost.warmup --timeout 500 > $R/warmup.log 2>&1
  st "GATE5V2_${a}_WARMUP_EXIT $?"
  wait $AUD
  AX=$?
  kill -INT $EST 2>/dev/null; sleep 3; kill -9 $EST 2>/dev/null
  python scripts/pipeline_audit.py --out $R/audit.jsonl --summary $R/audit_summary.json --estimator-log $R/estimator.jsonl --resummarise >> $R/audit.log 2>&1
  st "GATE5V2_${a}_AUDIT_EXIT $AX $(python -c "import json;print(json.load(open('$R/audit_summary.json'))['verdict'])" 2>/dev/null)"
  bash $D/scripts/stop_isaac.sh; sleep 10
done
st GATE5V2_DONE
