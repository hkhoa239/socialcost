#!/usr/bin/env bash
# Controller diagnosis, then (if a setup qualifies) the full re-run of all arms with it.
#   docker exec -d -u arena socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/run_ctrl_diag.sh
# 1. waits until challenge_v2 has finished (V2_DONE) and no runner is alive (one Isaac at a time)
# 2. ground-truth + bodies arms on 24 configurations, 4 setups: social cost in the global costmap only, DWB scale 0.1 + patient progress checker, DWB scale 0.1, MPPI
# 3. scripts/ctrl_select.py picks one by a fixed rule -> reports/navigation_controller_diagnosis.md, configs/navigation/challenge_v3.yaml
# 4. challenge_v3: all six arms + rgb / rgbd camera arms, 36 configurations, 1 repeat (no qualifier: camera arms on the v2 setup) -> reports/navigation_challenge_v3_results.md
# Resumable (finished runs are skipped). Status: experiments/navigation/STATUS_ctrl.txt
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; N=$D/experiments/navigation; S=$N/STATUS_ctrl.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
CH=$D/navigation_challenge_v1/navigation_challenge_v1.json
mkdir -p $N/ctrl_diag
st WAIT_FOR_V2
until grep -q V2_DONE $N/STATUS_challenge.txt 2>/dev/null && ! pgrep -f run_nav_challenge.py >/dev/null; do sleep 300; done
# camera arms (rgb / rgbd on the extended label) are trained on the host right after V2 (data/costmap_bench/run_cam_ext.sh);
# wait for them so training never shares the GPU with Isaac (at most 3 h)
st WAIT_FOR_CAM_MODELS
for i in $(seq 36); do grep -q CAM_DONE /opt/arena_ws/data/costmap_bench/STATUS_cam_ext.txt 2>/dev/null && break; sleep 300; done
st "CAM_MODELS $(ls /opt/arena_ws/data/costmap_bench/runs/main/rgb*_extended_s*/best.pt 2>/dev/null | wc -l)"
st DIAG_BEGIN
for v in global obs01_patient obs01 mppi; do
  st "DIAG_${v}_BEGIN"
  python scripts/run_nav_challenge.py --config configs/navigation/challenge_ctrl_$v.yaml >> $N/ctrl_diag/$v.log 2>&1; st "DIAG_${v}_EXIT $?"
  python scripts/nav_challenge_metrics.py $N/ctrl_diag/$v --challenge $CH --out $N/ctrl_diag/$v.json >> $N/ctrl_diag/metrics.log 2>&1
done
python scripts/ctrl_select.py --diag $N/ctrl_diag --baseline $N/challenge_v2.json --report $D/reports/navigation_controller_diagnosis.md >> $N/ctrl_diag/select.log 2>&1
rc=$?; st "SELECT_EXIT $rc"
if [ $rc -ne 0 ]; then
  # no controller fix: camera arms on the challenge_v2 setup instead (compare with v2 repeat 0)
  st V2CAM_BEGIN
  python scripts/run_nav_challenge.py --config configs/navigation/challenge_v2cam.yaml >> $N/challenge_v2cam.log 2>&1; st "V2CAM_EXIT $?"
  python scripts/nav_challenge_metrics.py $N/challenge_v2cam --challenge $CH --out $N/challenge_v2cam.json >> $N/challenge_v2cam_metrics.log 2>&1
  st CTRL_DONE_NO_RERUN; exit 0
fi
metrics() { python scripts/nav_challenge_metrics.py $N/challenge_v3 --challenge $CH --out $N/challenge_v3.json --report $D/reports/navigation_challenge_v3_results.md >> $N/challenge_v3_metrics.log 2>&1; }
( while sleep 1800; do metrics; done ) &
REFRESH=$!
st V3_BEGIN
python scripts/run_nav_challenge.py --config configs/navigation/challenge_v3.yaml >> $N/challenge_v3.log 2>&1; st "V3_EXIT $?"
kill $REFRESH 2>/dev/null
metrics; st "V3_METRICS_EXIT $?"
st CTRL_DONE
