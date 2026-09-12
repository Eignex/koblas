#!/usr/bin/env bash
set -euo pipefail
report=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=${1:?usage: reproduce.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e $output ]] || { echo "output already exists: $output" >&2; exit 2; }
mkdir -p "$output"
output=$(cd "$output" && pwd)
for mode in Scalar C Simd; do
  taskset -c 4 ./gradlew --no-daemon ":koblas-bench:jvm${mode}Benchmark" \
    "-Pbench.cases=$report/cases.txt" -Pbench.operation=all -Pbench.warmups=5 -Pbench.samples=5 \
    -Pbench.targetMs=200 -Pbench.forks=2 "-Pbench.output=$output/jvm-$mode.csv"
done
taskset -c 4 ./gradlew --no-daemon :koblas-bench:nativeBenchmark \
  "-Pbench.cases=$report/cases.txt" -Pbench.operation=all -Pbench.warmups=5 -Pbench.samples=5 \
  -Pbench.targetMs=200 "-Pbench.output=$output/native.csv"
taskset -c 4 koblas-bench/reference.sh --libraries openblas,onemkl --cases "$report/cases.txt" \
  --output "$output/vendors" --warmups 5 --samples 5 --target-ms 200
