#!/usr/bin/env bash
# usage: run_viz_predictions.sh <out name>
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
O=$D/outputs/visualizations/$1
python scripts/visualize_predictions.py --experiments $D/experiments/pilot_v1 --split $D/experiments/splits_pilot_v1.json --roots $D/datasets/pilot_v1 \
  --models pilot_rgbd_extended pilot_depth_extended pilot_rgb_extended pilot_pos_extended pilot_pos_fov_extended rule_extended \
  --label extended --sets test_A test_B test_C --output $O > $O.log 2>&1
echo "VIZ_$1_EXIT $? $(date -Is)" >> $D/outputs/STATUS_phase1.txt
