#!/usr/bin/env bash
set -euo pipefail
for pass in 1 2 3; do
  if [[ "$pass" == 2 ]]; then arms=(after before); else arms=(before after); fi
  for arm in "${arms[@]}"; do
    date -u +%FT%TZ
    taskset -c 4 "/tmp/scal-openblas-source/native-$arm.kexe" > "/tmp/scal-openblas-source/native-$arm-$pass.csv"
  done
done
