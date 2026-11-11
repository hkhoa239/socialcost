#!/usr/bin/env bash
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; OUT=$D/outputs/STATUS_phase1.txt
st() { echo "$1 $(date -Is)" >> $OUT; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
mkdir -p $D/datasets/repro_gate1 $D/datasets/pairs_probe
st GATE1_COLLECT_BEGIN
python scripts/collect_dataset.py configs/dataset/repro_gate1.yaml >> $D/datasets/repro_gate1/collect.log 2>&1; st "GATE1_COLLECT_EXIT $?"
st GATE2_COLLECT_BEGIN
python scripts/collect_dataset.py configs/dataset/pairs_probe.yaml >> $D/datasets/pairs_probe/collect.log 2>&1; st "GATE2_COLLECT_EXIT $?"
st DONE
