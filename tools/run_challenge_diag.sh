#!/usr/bin/env bash
# Diagnostic closed-loop runs (inside socialcost-arena, detached). Status: experiments/navigation/STATUS_challenge.txt
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; N=$D/experiments/navigation; S=$N/STATUS_challenge.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
st DIAG_BEGIN
python scripts/run_nav_challenge.py --config configs/navigation/challenge_diag.yaml >> $N/challenge_diag.log 2>&1; st "DIAG_EXIT $?"
