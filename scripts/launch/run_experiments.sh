#!/usr/bin/env bash
# Detached after the pilot: split, QA, baselines, training, paired evaluation.
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; S=$D/outputs/STATUS_phase1.txt
X=$D/experiments/pilot_v1; R=$D/datasets/pilot_v1; SPLIT=$D/experiments/splits_pilot_v1.json
st() { echo "$1 $(date -Is)" >> $S; }
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; cd $P
mkdir -p $X $D/outputs/paired_scenarios $D/outputs/evaluation
st EXP_BEGIN
python scripts/label_manifest.py $R >> $X/pipeline.log 2>&1; st "MANIFEST_EXIT $?"
python scripts/make_experiment_split.py $R --out $SPLIT >> $X/pipeline.log 2>&1; st "SPLIT_EXIT $?"
python scripts/verify_dataset.py --dataset $R --output $D/outputs/verification_pilot >> $X/pipeline.log 2>&1; st "VERIFY_EXIT $?"
PYTHONPATH=$P:$PYTHONPATH python $D/tools/pairs_check.py $R $D/outputs/paired_scenarios/pilot_v1.json >> $X/pipeline.log 2>&1; st "PAIRS_EXIT $?"
for L in extended native; do python scripts/eval_baselines.py --split $SPLIT --roots $R --label $L --out $X >> $X/pipeline.log 2>&1; st "BASELINES_${L}_EXIT $?"; done
for c in pilot_pos_extended pilot_pos_fov_extended pilot_rgbd_extended pilot_depth_extended pilot_rgb_extended pilot_pos_native pilot_rgbd_native; do
  python scripts/train_estimator.py --config configs/experiments/$c.yaml --save-predictions >> $X/train_$c.log 2>&1; st "TRAIN_${c}_EXIT $?"
done
for m in pos_extended pos_fov_extended rgbd_extended depth_extended rgb_extended rule_extended oracle_native_extended oracle_extended_extended; do
  python scripts/eval_pairs.py --pred $X/pilot_${m}/predictions.npz --split $SPLIT --label extended --roots $R --out $D/outputs/evaluation/pairs_${m}.json >> $X/pipeline.log 2>&1 \
   || python scripts/eval_pairs.py --pred $X/${m}/predictions.npz --split $SPLIT --label extended --roots $R --out $D/outputs/evaluation/pairs_${m}.json >> $X/pipeline.log 2>&1
done
for m in pos_native rgbd_native; do python scripts/eval_pairs.py --pred $X/pilot_${m}/predictions.npz --split $SPLIT --label native --roots $R --out $D/outputs/evaluation/pairs_${m}.json >> $X/pipeline.log 2>&1; done
for m in rule_native oracle_native_native; do python scripts/eval_pairs.py --pred $X/${m}/predictions.npz --split $SPLIT --label native --roots $R --out $D/outputs/evaluation/pairs_${m}.json >> $X/pipeline.log 2>&1; done
st EXP_DONE
