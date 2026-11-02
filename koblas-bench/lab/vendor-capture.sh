#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output=
operation=all
requested_vendor=
cases="$root/koblas-bench/lab/cases-vendor-comparison.txt"
samples=5
warmups=20
target_ms=200
forks=2
rounds=3
targets=
native_executable=
while (($#)); do
  case "$1" in
    --output) output=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --vendor) requested_vendor=${2:?}; shift 2 ;;
    # Placement reads the architecture; the host it lands on already is that architecture.
    --arch) shift 2 ;;
    --cases) cases=${2:?}; shift 2 ;;
    --samples) samples=${2:?}; shift 2 ;;
    --warmups) warmups=${2:?}; shift 2 ;;
    --target-ms) target_ms=${2:?}; shift 2 ;;
    --forks) forks=${2:?}; shift 2 ;;
    --rounds) rounds=${2:?}; shift 2 ;;
    --targets) targets=${2:?}; shift 2 ;;
    --native-executable) native_executable=${2:?}; shift 2 ;;
    --profile-gc) export KOBLAS_BENCH_PROFILE_GC=${2:?}; shift 2 ;;
    *) echo "usage: vendor-capture.sh --output DIR [--operation NAME] [--vendor NAME]" >&2; exit 2 ;;
  esac
done
[[ -n $output ]] || { echo "--output is required" >&2; exit 2; }
mkdir -p "$output"
cd "$root"

if [[ $(uname -s) == Darwin ]]; then
  vendor=accelerate
elif [[ $requested_vendor == openblas ]]; then
  # The worker image ships OpenBLAS; the lab already holds it to one thread through OPENBLAS_NUM_THREADS.
  vendor=openblas
else
  vendor=onemkl
  directory="$root/koblas-bench/build/vendor-reference"
  [[ -d $directory/lib ]] || { echo "oneMKL must be prepared before timing" >&2; exit 2; }
  export MKLROOT="$directory"
  export LD_LIBRARY_PATH="$directory/lib:${LD_LIBRARY_PATH:-}"
  export MKL_THREADING_LAYER=SEQUENTIAL
  cp "$directory/vendor-package.txt" "$output/vendor-package.txt"
fi
[[ -z $requested_vendor || $requested_vendor == "$vendor" ]] || { echo "vendor requires a different capture host" >&2; exit 2; }
if [[ -z $targets ]]; then targets="jvm-scalar,jvm-simd,$vendor-jvm,$vendor"; fi

native_args=()
if [[ ,$targets, =~ ,(native|native-default|onemkl|openblas|accelerate), ]]; then
  [[ -n $native_executable && -x $native_executable ]] || { echo "Native captures require a prepared executable" >&2; exit 2; }
  native_args=(--native-executable "$native_executable")
fi
export KOBLAS_BENCH_JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
export KOBLAS_BENCH_CLASSPATH="$root/koblas-bench/build/lab/classes:$root/koblas-bench/build/lab/lib/*"
export KOBLAS_BENCH_HEAP_MIB=512
libraries=$vendor
if [[ $vendor == onemkl ]]; then libraries=onemkl,openblas; fi
exec bash koblas-bench/capture-report.sh --prepared --retain-incomplete \
  "${native_args[@]}" \
  --libraries "$libraries" --targets "$targets" \
  --cases "$cases" \
  --operation "$operation" \
  --rounds "$rounds" --samples "$samples" --warmups "$warmups" --target-ms "$target_ms" --forks "$forks" \
  --jvm-gc serial --jvm-heap 512m --output "$output/report"
