#!/usr/bin/env bash
# Re-record listed episodes, then the navigation references (dv1_nav_refs.py) if present.
D=/opt/arena_ws/data/socialcost; P=/opt/arena_ws/src/deps/socialcost; R=$D/dataset_v1; S=$D/dataset_v1_STATUS.txt
cd /opt/arena_ws && source /opt/arena_ws/source >/dev/null 2>&1; export ROS_DOMAIN_ID=7; cd $P
echo "RERECORD_BEGIN $(date -Is)" >> $S
python scripts/dv1_collect.py --root $R --stage all --only $(cat $R/metadata/rerecord_ids.txt) --retries 3 >> $R/logs/collect_rerecord.log 2>&1
echo "RERECORD_EXIT $? $(date -Is)" >> $S
if [ -f scripts/dv1_nav_refs.py ]; then
  python scripts/dv1_nav_refs.py --root $D/navigation_challenge_v1 >> $D/navigation_challenge_v1/references.log 2>&1
  echo "NAVREFS_EXIT $? $(date -Is)" >> $S
fi
echo "RERECORD_DONE $(date -Is)" >> $S
