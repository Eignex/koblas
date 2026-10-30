#!/bin/sh
# Choose a lightly loaded allowed physical core without reserving a worker.
set -eu

bench=$(CDPATH= cd "$(dirname "$0")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' 0
trap 'exit 1' HUP INT TERM

allowed=$(awk '/^Cpus_allowed_list:/ {print $2; found=1; exit} END {if (!found) exit 1}' /proc/self/status)
cp /proc/stat "$temporary/before"
sleep 0.2
cp /proc/stat "$temporary/after"

# Second-resolution seeds would make simultaneous captures choose the same CPU.
seed=$(od -An -N4 -tu4 /dev/urandom)
awk -v allowed="$allowed" -v topology=/sys/devices/system/cpu -v seed="$seed" \
  -f "$bench/select-cpu.awk" "$temporary/before" "$temporary/after"
