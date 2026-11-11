# costmap benchmark environment: torch 2.1.2+cu121, dgl 2.0.0+cu121 (SNGNN2D-v2 pins), host python 3.11
export LD_LIBRARY_PATH=$(ls -d /home/robotics/arena_ws/data/costmap_bench/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}
export PYTHONPATH=/home/robotics/arena_ws/src/deps/socialcost:${PYTHONPATH}
export DGLBACKEND=pytorch
PY=/home/robotics/arena_ws/data/costmap_bench/.venv/bin/python
