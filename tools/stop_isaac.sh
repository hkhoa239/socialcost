#!/usr/bin/env bash
# Inside socialcost-arena: stop the arena launch (its finally stops the isaac service).
pkill -INT -f "arena_bringup.supervisor|arena launch" 2>/dev/null
for i in $(seq 1 30); do pgrep -f "arena_runtime.launch.py|task_generator.launch.py" >/dev/null || break; sleep 1; done
pkill -KILL -f "arena_runtime.launch.py|task_generator.launch.py" 2>/dev/null
true
