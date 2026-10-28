#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output=
operation=all
requested_vendor=
while (($#)); do
  case "$1" in
    --output) output=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --vendor) requested_vendor=${2:?}; shift 2 ;;
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
else
  vendor=onemkl
  target=LinuxX64
  pin=(--cpu auto)
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

./gradlew --no-daemon -Pkotlin.compiler.execution.strategy=in-process :koblas-bench:prepareLabBenchmark ":koblas-bench:linkReleaseExecutable$target" > "$output/build.log" 2>&1
export KOBLAS_BENCH_JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
export KOBLAS_BENCH_CLASSPATH="$root/koblas-bench/build/lab/classes:$root/koblas-bench/build/lab/lib/*"
export KOBLAS_BENCH_HEAP_MIB=512
if [[ $target == MacosArm64 ]]; then native_target=macosArm64; else native_target=linuxX64; fi
exec bash koblas-bench/capture-report.sh --prepared --retain-incomplete \
  --native-executable "$root/koblas-bench/build/bin/$native_target/releaseExecutable/koblas-bench.kexe" \
  --libraries "$vendor" --targets "jvm-scalar,jvm-simd,$vendor-jvm,$vendor" \
  --cases "$root/koblas-bench/lab/cases-vendor-comparison.txt" \
  --operation "$operation" \
  --rounds 3 --samples 3 --warmups 3 --target-ms 100 --forks 1 \
  --jvm-gc serial --jvm-heap 512m "${pin[@]}" --output "$output/report"
