#!/bin/bash
#
# Runs one gate suite so that two runs can be compared. Usage:
#
#   koblas-bench/measure.sh jvm    jvmSweepGateBenchmark          out.txt
#   koblas-bench/measure.sh jvm    jvmScalarKernelsGateBenchmark  out.txt -Pkoblas.noSimd=true
#   koblas-bench/measure.sh native linuxX64SweepGateBenchmark     out.txt
#
# Compare a change by running the same suite before and after and reading the control row first: if the
# control moved, the machine moved, and only differences larger than that mean anything.
#
# Each step below exists because leaving it out produced a wrong answer:
#
#   Build and measure in separate invocations. Compile load bleeding into a measurement moved a sweep row
#   by 2x between two runs of identical code.
#
#   Measure with --no-daemon under taskset. The JMH fork inherits the affinity of its parent, and with a
#   daemon its parent is the daemon rather than this script, so the pinning is silently lost.
#
#   Delete the benchmark report before measuring. The native benchmark task is up-to-date-checkable, so a
#   second invocation reports success without executing anything.
#
# CORES pins the measurement, so choose cores nothing else is using. Set it to the whole machine when idle.
set -u

if [ "$#" -lt 3 ]; then
    sed -n '2,20p' "$0"
    exit 2
fi

target=$1
task=$2
out=$3
shift 3

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
CORES=${CORES:-16-19}

if [ "$target" = native ]; then
    # No compile-only task exists for the native benchmarks, so a discarded run does the build and link.
    ./gradlew ":koblas-bench:$task" "$@" > /dev/null 2>&1 || { echo "build failed"; exit 1; }
else
    ./gradlew ":koblas-bench:jvmBenchmarkJar" "$@" > /dev/null 2>&1 || { echo "build failed"; exit 1; }
fi

rm -rf "$root/koblas-bench/build/reports/benchmarks"
echo "cores=$CORES load=$(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c "$CORES" ./gradlew --no-daemon ":koblas-bench:$task" "$@" > "$out" 2>&1
echo "exit=$?"
# Match any class, since naming the known ones once hid a new suite's whole result behind an empty table.
grep -aE "^Benchmark +\(|^[A-Za-z0-9_]+Benchmark\.[A-Za-z0-9_]+ " "$out"
