#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output=
operation=all
requested_vendor=
cases="$root/koblas-bench/lab/cases-vendor-comparison.txt"
samples=5
warmups=5
target_ms=200
forks=2
rounds=3
shard=1/1
targets=
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
    --shard) shard=${2:?}; shift 2 ;;
    --targets) targets=${2:?}; shift 2 ;;
    --profile-gc) export KOBLAS_BENCH_PROFILE_GC=${2:?}; shift 2 ;;
    *) echo "usage: vendor-capture.sh --output DIR [--operation NAME] [--vendor NAME]" >&2; exit 2 ;;
  esac
done
[[ -n $output ]] || { echo "--output is required" >&2; exit 2; }
mkdir -p "$output"
cd "$root"

# Bound the build separately from the measured JVM, and end it before measuring.
export GRADLE_OPTS="${GRADLE_OPTS:-} -Dorg.gradle.jvmargs=-Xmx2g -Dorg.gradle.workers.max=2"
if [[ $(uname -s) == Darwin ]]; then
  vendor=accelerate
  target=MacosArm64
  pin=()
elif [[ $requested_vendor == openblas ]]; then
  # The worker image ships OpenBLAS; the lab already holds it to one thread through OPENBLAS_NUM_THREADS.
  vendor=openblas
  case $(uname -m) in
    aarch64 | arm64) target=LinuxArm64 ;;
    *) target=LinuxX64 ;;
  esac
  pin=(--cpu spread)
else
  vendor=onemkl
  target=LinuxX64
  pin=(--cpu spread)
  # Extract the same checksum-pinned reference used in earlier lab comparisons, without installing globally.
  package=intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb
  directory="$root/koblas-bench/build/vendor-reference"
  mkdir -p "$directory"
  python3 - "$directory/$package" <<'PY'
import hashlib
from pathlib import Path
import sys
import urllib.request

package = Path(sys.argv[1])
expected = "3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93"
if not package.exists():
    urllib.request.urlretrieve("https://apt.repos.intel.com/oneapi/pool/main/" + package.name, package)
if hashlib.sha256(package.read_bytes()).hexdigest() != expected:
    raise RuntimeError("oneMKL reference package checksum mismatch")
PY
  dpkg-deb --extract "$directory/$package" "$directory"
  export MKLROOT="$directory/opt/intel/oneapi/mkl/2026.1"
  export LD_LIBRARY_PATH="$MKLROOT/lib:${LD_LIBRARY_PATH:-}"
  export MKL_THREADING_LAYER=SEQUENTIAL
  printf '%s\n' "$package 3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93" > "$output/vendor-package.txt"
fi
[[ -z $requested_vendor || $requested_vendor == "$vendor" ]] || { echo "vendor requires a different capture host" >&2; exit 2; }
if [[ -z $targets ]]; then targets="jvm-scalar,jvm-simd,$vendor-jvm,$vendor"; fi

build_tasks=(:koblas-bench:prepareLabBenchmark)
native_args=()
if [[ ,$targets, == *,$vendor,* || ,$targets, == *,native,* || ,$targets, == *,native-default,* ]]; then
  build_tasks+=(":koblas-bench:linkReleaseExecutable$target")
  case $target in
    MacosArm64) native_target=macosArm64 ;;
    LinuxArm64) native_target=linuxArm64 ;;
    *) native_target=linuxX64 ;;
  esac
  native_args=(--native-executable "$root/koblas-bench/build/bin/$native_target/releaseExecutable/koblas-bench.kexe")
fi
./gradlew --no-daemon -Ptargets.full -Pkotlin.compiler.execution.strategy=in-process "${build_tasks[@]}" > "$output/build.log" 2>&1
export KOBLAS_BENCH_JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
export KOBLAS_BENCH_CLASSPATH="$root/koblas-bench/build/lab/classes:$root/koblas-bench/build/lab/lib/*"
export KOBLAS_BENCH_HEAP_MIB=512
exec bash koblas-bench/capture-report.sh --prepared --retain-incomplete \
  "${native_args[@]}" \
  --libraries "$vendor" --targets "$targets" \
  --cases "$cases" \
  --operation "$operation" \
  --rounds "$rounds" --shard "$shard" --samples "$samples" --warmups "$warmups" --target-ms "$target_ms" --forks "$forks" \
  --jvm-gc serial --jvm-heap 512m "${pin[@]}" --output "$output/report"
