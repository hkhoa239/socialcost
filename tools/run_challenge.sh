#!/usr/bin/env bash
# Closed-loop navigation_challenge_v1 evaluation of the costmap estimators (inside socialcost-arena, detached):
#   docker exec -d -u arena socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/run_challenge.sh
# 1. waits for the 9 extended-label models (metrics.json of pos_u / typemix_gt / typemix_cffullraw x 3 seeds) and no running training
# 2. pilot: N02 N15 N23 N30 (reception) + N09 (airport, raised hand), all 6 arms, then metrics
# 3. gate: no infrastructure failure and every arm >= 4 completed pilot runs -> full 36 configurations, metrics, report
# Resumable (run_nav_challenge.py skips finished runs). Status: experiments/navigation/STATUS_challenge.txt
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; N=$D/experiments/navigation; S=$N/STATUS_challenge.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
CFG=configs/navigation/challenge_v1.yaml; CH=$D/navigation_challenge_v1/navigation_challenge_v1.json
st WAIT_MODELS
R=/opt/arena_ws/data/costmap_bench/runs/main
until [ "$(ls $R/{pos_u,typemix_gt,typemix_cffullraw}_extended_s{0,1,2}/metrics.json 2>/dev/null | wc -l)" -eq 9 ] && ! pgrep -f bench_train.py >/dev/null; do sleep 60; done
st MODELS_READY
st PILOT_BEGIN
python scripts/run_nav_challenge.py --config $CFG --only N02,N15,N23,N30,N09 >> $N/challenge_v1.log 2>&1; st "PILOT_EXIT $?"
python scripts/nav_challenge_metrics.py $N/challenge_v1 --challenge $CH --out $N/challenge_v1_pilot.json --report $D/reports/navigation_challenge_pilot.md >> $N/challenge_v1_metrics.log 2>&1
GATE=$(python - <<PY
import json; r=json.load(open("$N/challenge_v1_pilot.json")); pm=r["per_method"]
ok = not r["infrastructure_failures"] and len(pm)==6 and all(v["n"]>=4 for v in pm.values())
print("PASS" if ok else "FAIL")
PY
)
st "PILOT_GATE $GATE"
[ "$GATE" = PASS ] || { st STOPPED_AT_GATE; exit 0; }
st FULL_BEGIN
python scripts/run_nav_challenge.py --config $CFG >> $N/challenge_v1.log 2>&1; st "FULL_EXIT $?"
python scripts/nav_challenge_metrics.py $N/challenge_v1 --challenge $CH --out $N/challenge_v1.json --report $D/reports/navigation_challenge_results.md >> $N/challenge_v1_metrics.log 2>&1
st "METRICS_EXIT $?"
st CHALLENGE_DONE
