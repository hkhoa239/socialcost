#!/usr/bin/env bash
# Host-side: start `socialcost-arena`, an Arena container isolated from arena-arena_ws-arena-1.
# Same image layer contents (snapshot of the live container, which carries apt packages
# installed after its creation), same mounts, but its own PID namespace (other jobs'
# `pkill -f arena_runtime.launch.py` cannot reach it) and ROS_DOMAIN_ID 7.
# It reuses the compose project's `isaac` service (arena-arena_ws-isaac-1).
set -euo pipefail
IMAGE="${SOCIALCOST_IMAGE:-socialcost-arena:snapshot-20261029}"
WS=/home/robotics/arena_ws
docker rm -f socialcost-arena >/dev/null 2>&1 || true
docker run -d --name socialcost-arena --network host --ipc private --gpus all \
  --user 1004 --group-add sudo --group-add audio --group-add 999 --group-add 110 \
  --device /dev/dri --shm-size 2g -w /opt/arena_ws --entrypoint /entrypoint.sh \
  -v $WS/data:/opt/arena_ws/data -v $WS/build:/opt/arena_ws/build \
  -v $WS/src/Arena/.venv:/opt/venv -v $WS/data/screenshots:/opt/arena_ws/screenshots \
  -v $WS/.uv-python:/opt/arena_ws/.uv-python -v $WS/install:/opt/arena_ws/install \
  -v $WS/log:/opt/arena_ws/log -v $WS/.env:/opt/arena_ws/.env \
  -v $WS/src:/opt/arena_ws/src -v /var/run/docker.sock:/var/run/docker.sock \
  -v arena-arena_ws_shared:/tmp/shared -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e QT_X11_NO_MITSHM=1 -e TMPDIR=/tmp/shared -e UV_LINK_MODE=copy -e HOST_GID=999 \
  -e HOST_ARENA_SCREENSHOTS_DIR=$WS/data/screenshots -e ARENA_SCREENSHOTS_DIR=/opt/arena_ws/screenshots \
  -e AMENT_CMAKE_VENV_UV_OVERRIDES=/opt/arena_ws/src/Arena/_meta/docker/uv-overrides.txt -e ROSNAV_AGENTS_DIR=/opt/arena_ws/data/agents \
  -e ARENA_ROS_DISTRO=jazzy -e ARENA_PROJECT_NAME=arena-arena_ws -e HOST_ARENA_DATA_DIR=$WS/data \
  -e HOST_ARENA_WS_DIR=$WS -e UV_PYTHON_INSTALL_DIR=/opt/arena_ws/.uv-python -e ARENA_IMAGE_TAG=arena-arena_ws \
  -e HF_HOME=/opt/arena_ws/data/hf -e ROS_DOMAIN_ID="${SOCIALCOST_DOMAIN:-7}" -e ARENA_DATA_DIR=/opt/arena_ws/data -e GZ_PARTITION=socialcost-7 \
  -e ARENA_IMAGE=arena:arena-arena_ws -e GZ_IP=127.0.0.1 -e HOST_UID=1004 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e UV_CACHE_DIR=/opt/arena_ws/build/.uv-cache -e CYCLONEDDS_URI=file:///opt/arena_ws/src/Arena/_meta/docker/cyclonedds.local.xml \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/opt/arena_ws/src/Arena/_meta/docker/fastdds.local.xml -e HOME=/home/arena \
  "$IMAGE" sleep infinity
