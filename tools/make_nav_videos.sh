#!/usr/bin/env bash
# Bird's-eye MP4 (ground-truth vs predicted costmap) for every navigation run that saved its published grids (dump/).
# Host side: frames are rendered in socialcost-arena (scripts/nav_video.py), then transcoded to H.264 with the host ffmpeg.
#   bash make_nav_videos.sh <runs dir under experiments/navigation, e.g. challenge_v3> [arm filter, default typemix]
# Output: experiments/navigation/videos/<runs dir>/<config>_<arm>_<outcome>.mp4 (existing files are skipped)
set -u
R=${1:?runs dir, e.g. challenge_v3}; ARM=${2:-typemix}
H=/home/robotics/arena_ws/data/socialcost/experiments/navigation; C=/opt/arena_ws/data/socialcost/experiments/navigation
mkdir -p $H/videos/$R
case $ARM in
  typemix) T="ours: counterfactual stage 1 + TypeMix";; typemix_ucb) T="ours, mu + 1 sigma";; typemix_gt) T="TypeMix + true types (privileged)";;
  pos_u) T="position-only U-Net";; rgb) T="RGB image U-Net";; rgbd) T="RGB-D image U-Net";; *) T=$ARM;;
esac
for d in $H/$R/*__${ARM}__*; do
  [ -d $d/dump ] && [ -f $d/run.json ] || continue
  tag=$(basename $d); cid=${tag%%__*}
  [ -f /home/robotics/arena_ws/data/socialcost/navigation_challenge_v1/references/$cid/pedestrian_states.jsonl ] || { echo "skip $tag (no reference recording)"; continue; }
  st=$(python3 -c "import json;print(json.load(open('$d/run.json'))['status'].split(':')[0])")
  out=$H/videos/$R/${tag}_${st}.mp4
  [ -f $out ] && continue
  docker exec -u arena -w /opt/arena_ws socialcost-arena bash -c "cd /opt/arena_ws/src/deps/socialcost && PYTHONPATH=/opt/arena_ws/src/deps/socialcost /opt/venv/bin/python scripts/nav_video.py --run $C/$R/$tag --config $cid --out $C/videos/$R/${tag}_${st}.mp4 --title \"$T\"" >/dev/null 2>&1 \
    && ffmpeg -loglevel error -y -i $out.raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart $out && rm -f $out.raw.mp4 && echo "$out"
done
