#!/usr/bin/env bash
# Launch Arena + Isaac Sim on an isolated ROS domain (default 7) so it cannot
# interfere with other Arena runtimes on domain 1.
# usage: launch_isaac.sh <logname> [extra launch args...]   (world:=map_empty unless a world:= is given)
cd /opt/arena_ws
source /opt/arena_ws/source >/dev/null 2>&1
export ROS_DOMAIN_ID="${SOCIALCOST_DOMAIN:-7}"
log="/opt/arena_ws/data/socialcost/logs/$1.log"; shift
world="world:=map_empty"
for a in "$@"; do case "$a" in world:=*) world="";; esac; done
echo "=== $(date -Is) domain=$ROS_DOMAIN_ID args: $world $*" >> "$log"
arena launch sim:=isaac headless:=true $world robot:=turtlebot human:=arena "$@" >> "$log" 2>&1
