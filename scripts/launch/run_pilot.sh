#!/usr/bin/env bash
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; S=$D/outputs/STATUS_phase1.txt
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
mkdir -p $D/datasets/pilot_v1
echo "PILOT_BEGIN $(date -Is)" >> $S
python scripts/collect_dataset.py configs/dataset/pilot_v1.yaml >> $D/datasets/pilot_v1/collect.log 2>&1
echo "PILOT_EXIT $? $(date -Is)" >> $S
