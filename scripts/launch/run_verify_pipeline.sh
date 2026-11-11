#!/usr/bin/env bash
# Detached verification pipeline, run inside socialcost-arena with `docker exec -d`:
#   docker exec -d socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/run_verify_pipeline.sh [PID_TO_WAIT_FOR]
# Waits for an already running collector (optional pid), then records the verification scenarios
# (resumable: valid episodes are skipped), writes label manifests, runs the verification with the
# plugin replay, the semantic report and the visualizations. Progress in $OUT/STATUS.txt.
D=/opt/arena_ws/data/socialcost
P=/opt/arena_ws/src/deps/socialcost
OUT=$D/outputs
STATUS=$OUT/STATUS.txt
LOG=$OUT/pipeline.log
mkdir -p $OUT
st() { echo "$1 $(date -Is)" >> $STATUS; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1
cd $P

if [ -n "$1" ]; then
  st "WAIT_FOR_COLLECTOR pid=$1"
  while kill -0 "$1" 2>/dev/null; do sleep 10; done
fi
bash $D/scripts/stop_isaac.sh >> $LOG 2>&1

st COLLECT_VERIFY_BEGIN
python scripts/collect_dataset.py configs/dataset/verify.yaml >> $D/verify_scenarios/collect.log 2>&1
st "COLLECT_VERIFY_EXIT $?"
bash $D/scripts/stop_isaac.sh >> $LOG 2>&1

DS="--dataset $D/smoke_test --dataset $D/smoke_test_repro --dataset $D/verify_scenarios"
python scripts/label_manifest.py $D/smoke_test $D/smoke_test_repro $D/verify_scenarios >> $LOG 2>&1
st "LABEL_MANIFEST_EXIT $?"
ROS_DOMAIN_ID=9 python scripts/verify_dataset.py $DS --output $OUT/verification --replay 10 >> $LOG 2>&1
st "VERIFY_EXIT $?"
SOCIALCOST_DATASETS=$D/smoke_test:$D/smoke_test_repro:$D/verify_scenarios python -m pytest tests/test_dataset_geometry.py tests/test_labels.py -q >> $LOG 2>&1
st "PYTEST_EXIT $?"
python scripts/semantic_report.py $DS --output $OUT/semantics >> $LOG 2>&1
st "SEMANTIC_REPORT_EXIT $?"
python scripts/visualize_semantics.py --dataset $D/smoke_test --dataset $D/verify_scenarios --output $OUT/semantics/visualizations >> $LOG 2>&1
st "VIS_SEMANTICS_EXIT $?"
python scripts/visualize_dataset.py --dataset $D/smoke_test --output $OUT/visualizations >> $LOG 2>&1
python scripts/visualize_dataset.py --dataset $D/verify_scenarios --output $OUT/visualizations >> $LOG 2>&1
st "VIS_DATASET_EXIT $?"
st PIPELINE_DONE
