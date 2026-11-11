#!/usr/bin/env bash
# Navigation on v2 labels (native + extended): position U-Net vs ours (+ memory), ground truth and no social cost as references.
#   docker exec -d -u arena socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/run_labels_v2_nav.sh
# Waits for the v2 retraining queue to finish (GPU), then runs configs/navigation/challenge_labels_v2.yaml (resumable, speed guard),
# then scores native arms against v2 native and extended arms against v2 extended ground truth.
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; N=$D/experiments/navigation; S=$N/STATUS_labels_v2.txt
CH=$D/navigation_challenge_v1/navigation_challenge_v1.json; Q=/opt/arena_ws/data/costmap_bench/STATUS_v2_train.txt
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
export PYTHONPATH=$P:$PYTHONPATH ROS_DOMAIN_ID=7
st WAIT_FOR_TRAINING
until [ "$(tail -n 1 $Q | cut -d' ' -f1)" = "DONE" ]; do sleep 120; done
st BEGIN
metrics() {
  python scripts/nav_challenge_metrics.py $N/challenge_labels_v2 --challenge $CH --meter-label v2e --oracle gt_ext --methods bodies gt_ext ours_ext oursmem_ext posu_ext \
    --out $N/challenge_labels_v2_extended.json --report $D/reports/navigation_labels_v2_extended.md >> $N/challenge_labels_v2_metrics.log 2>&1
  python scripts/nav_challenge_metrics.py $N/challenge_labels_v2 --challenge $CH --meter-label v2n --oracle gt_nat --methods bodies gt_nat ours_nat posu_nat \
    --out $N/challenge_labels_v2_native.json --report $D/reports/navigation_labels_v2_native.md >> $N/challenge_labels_v2_metrics.log 2>&1
}
( while sleep 1800; do metrics; done ) &
R=$!
python scripts/run_nav_challenge.py --config configs/navigation/challenge_labels_v2.yaml >> $N/challenge_labels_v2.log 2>&1; st "RUN_EXIT $?"
kill $R 2>/dev/null
metrics; st DONE
