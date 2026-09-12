#!/usr/bin/env bash
set -euo pipefail
out=$1
mkdir -p "$out"
cp /tmp/scal-gather-cases.txt "$out/cases.txt"
git rev-parse HEAD > "$out/commit.txt"
date -u > "$out/times.txt"
java koblas-bench/tools/CpuSampler.java "$out/cpu.csv" "$out/ready" "$out/stop" > "$out/cpu.log" 2>&1 &
sampler=$!
trap 'touch "$out/stop"; wait "$sampler" || true' EXIT
while [[ ! -f "$out/ready" ]]; do sleep 1; done
date -u >> "$out/times.txt"
taskset -c 4 koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe --mode=native --operation=all --cases=/tmp/scal-gather-cases.txt "--output=$out/native.csv" --warmups=5 --samples=5 --target-ms=200 "--source-commit=$(git rev-parse HEAD)" --dirty=false > "$out/native.log" 2>&1
for mode in Scalar C Simd; do
 date -u >> "$out/times.txt"
 taskset -c 4 ./gradlew --no-daemon ":koblas-bench:jvm${mode}Benchmark" -Pbench.cases=/tmp/scal-gather-cases.txt -Pbench.operation=all -Pbench.warmups=5 -Pbench.samples=5 -Pbench.targetMs=200 -Pbench.forks=2 "-Pbench.output=$out/jvm-$mode.csv" > "$out/jvm-$mode.log" 2>&1
done
date -u >> "$out/times.txt"
taskset -c 4 koblas-bench/reference.sh --libraries openblas,onemkl --cases /tmp/scal-gather-cases.txt --output "$out/vendors" --warmups 5 --samples 5 --target-ms 200 > "$out/vendors.log" 2>&1
date -u >> "$out/times.txt"
