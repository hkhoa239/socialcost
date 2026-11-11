#!/usr/bin/env bash
# Detached: rerun verification, tests, semantic report and lifecycle visualizations after the 1.1 relabel.
D=/opt/arena_ws/data/socialcost; OUT=$D/outputs; P=/opt/arena_ws/src/deps/socialcost
st() { echo "$1 $(date -Is)" >> $OUT/STATUS.txt; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
DS="--dataset $D/smoke_test --dataset $D/smoke_test_repro --dataset $D/verify_scenarios"
st TAIL_1_1_BEGIN
rm -rf $OUT/verification/failure_samples
ROS_DOMAIN_ID=9 python scripts/verify_dataset.py $DS --output $OUT/verification --replay 10 >> $OUT/pipeline.log 2>&1; st "VERIFY_EXIT $?"
SOCIALCOST_DATASETS=$D/smoke_test:$D/smoke_test_repro:$D/verify_scenarios python -m pytest tests/test_dataset_geometry.py tests/test_labels.py -q >> $OUT/pipeline.log 2>&1; st "PYTEST_EXIT $?"
python scripts/semantic_report.py $DS --output $OUT/semantics >> $OUT/pipeline.log 2>&1; st "SEMANTIC_REPORT_EXIT $?"
python scripts/visualize_semantics.py --dataset $D/verify_scenarios --episode episode_000004 --output $OUT/semantics/visualizations >> $OUT/pipeline.log 2>&1
python scripts/visualize_dataset.py --dataset $D/verify_scenarios --episode 4 --output $OUT/visualizations >> $OUT/pipeline.log 2>&1; st "VIS_EXIT $?"
st TAIL_1_1_DONE
