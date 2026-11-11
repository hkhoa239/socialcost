#!/usr/bin/env bash
# Full closed-loop navigation_challenge_v1 evaluation, v2 (threshold 12 on learned arms, abort-ends-run, 2 repeats).
#   docker exec -d -u arena socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/run_challenge_v2.sh
# Resumable. Metrics and the report are refreshed every 30 minutes and at the end. Status: experiments/navigation/STATUS_challenge.txt
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; N=$D/experiments/navigation; S=$N/STATUS_challenge.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
CH=$D/navigation_challenge_v1/navigation_challenge_v1.json
metrics() { python scripts/nav_challenge_metrics.py $N/challenge_v2 --challenge $CH --out $N/challenge_v2.json --report $D/reports/navigation_challenge_results.md >> $N/challenge_v2_metrics.log 2>&1; }
( while sleep 1800; do metrics; done ) &
REFRESH=$!
st V2_BEGIN
python scripts/run_nav_challenge.py --config configs/navigation/challenge_v2.yaml >> $N/challenge_v2.log 2>&1; st "V2_EXIT $?"
kill $REFRESH 2>/dev/null
metrics; st "V2_METRICS_EXIT $?"
st V2_DONE
