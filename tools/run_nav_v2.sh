#!/usr/bin/env bash
# RGB-D diagnosis Phase F: controlled navigation re-evaluation (nav_v2), DWB (5 seeds + robot-free references)
# then MPPI (3 seeds), then metrics. Resumable: run_nav_eval.py skips runs whose run.json says ok/timeout.
# usage (inside socialcost-arena, detached): bash run_nav_v2.sh
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; S=$D/outputs/rgbd_diagnosis/STATUS.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source; cd $P
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
st NAV_V2_BEGIN
st NAV_V2_DWB_BEGIN; python scripts/run_nav_eval.py --config configs/navigation/nav_v2_dwb.yaml >> $D/experiments/navigation_v2_dwb.log 2>&1; st "NAV_V2_DWB_EXIT $?"
python scripts/nav_metrics_v2.py $D/experiments/navigation/nav_v2 --out $D/outputs/rgbd_diagnosis/navigation_v2.json >> $D/experiments/navigation_v2_metrics.log 2>&1; st "NAV_V2_METRICS_DWB_EXIT $?"
st NAV_V2_MPPI_BEGIN; python scripts/run_nav_eval.py --config configs/navigation/nav_v2_mppi.yaml >> $D/experiments/navigation_v2_mppi.log 2>&1; st "NAV_V2_MPPI_EXIT $?"
python scripts/nav_metrics_v2.py $D/experiments/navigation/nav_v2 --out $D/outputs/rgbd_diagnosis/navigation_v2.json >> $D/experiments/navigation_v2_metrics.log 2>&1; st "NAV_V2_METRICS_EXIT $?"
st NAV_V2_DONE
