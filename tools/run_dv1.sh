#!/usr/bin/env bash
# dataset_v1 stage runner (detached): collect a stage, then validate it.
# usage: run_dv1.sh <subset|pilot|remaining|all>
STAGE=${1:-subset}
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; R=$D/dataset_v1; S=$D/dataset_v1_STATUS.txt
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1
export ROS_DOMAIN_ID=7
cd $P
echo "STAGE_BEGIN $STAGE $(date -Is)" >> $S
python scripts/dv1_collect.py --root $R --stage $STAGE >> $R/logs/collect_$STAGE.log 2>&1
echo "STAGE_COLLECT_EXIT $STAGE $? $(date -Is)" >> $S
if [ -f scripts/dv1_validate.py ]; then
  python scripts/dv1_validate.py --root $R --stage $STAGE --out $D/outputs/dataset_v1_validation/$STAGE >> $R/logs/validate_$STAGE.log 2>&1
  echo "STAGE_VALIDATE_EXIT $STAGE $? $(date -Is)" >> $S
fi
echo "STAGE_DONE $STAGE $(date -Is)" >> $S
