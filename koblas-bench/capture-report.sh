#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench="$root/koblas-bench"
reports="$bench/reports"
libraries=all
operation=all
samples=5
warmups=3
target_ms=1000
forks=2
pass=1

usage() {
  echo "usage: koblas-bench/capture-report.sh [--libraries openblas,onemkl|all] [--operation NAME|all] [--samples N] [--warmups N] [--target-ms N] [--forks N] [--pass N]" >&2
}

while (($#)); do
  case "$1" in
    --libraries) libraries=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --samples) samples=${2:?}; shift 2 ;;
    --warmups) warmups=${2:?}; shift 2 ;;
    --target-ms) target_ms=${2:?}; shift 2 ;;
    --forks) forks=${2:?}; shift 2 ;;
    --pass) pass=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

temporary=$(mktemp -d "${TMPDIR:-/tmp}/koblas-bench-report.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
"$bench/tools/hardware.sh" >"$temporary/hardware.txt"

cases="$bench/cases.txt"
if [[ $operation != all ]]; then
  cases="$temporary/cases.txt"
  awk -F+ -v operation="$operation" '
    /^[[:space:]]*($|#)/ || $1 == operation { print }
  ' "$bench/cases.txt" >"$cases"
fi

if command -v sha256sum >/dev/null 2>&1; then
  hardware_hash=$(sha256sum "$temporary/hardware.txt" | awk '{ print $1 }')
else
  hardware_hash=$(shasum -a 256 "$temporary/hardware.txt" | awk '{ print $1 }')
fi

commit=$(git -C "$root" rev-parse --short=12 HEAD)
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$commit"
hardware_report="$reports/$hardware_hash"
mkdir -p "$hardware_report"
if [[ -e $hardware_report/hardware.txt ]]; then
  cmp -s "$temporary/hardware.txt" "$hardware_report/hardware.txt" || {
    echo "hardware hash collision" >&2; exit 1;
  }
else
  cp "$temporary/hardware.txt" "$hardware_report/hardware.txt"
fi
run="$hardware_report/$run_id"
[[ ! -e $run ]] || { echo "report already exists: $run" >&2; exit 1; }
mkdir "$run"
echo "incomplete" >"$run/status.txt"
cp "$cases" "$run/cases.txt"
cases="$run/cases.txt"
{
  date -u
  uname -a
  cc --version
  java --version
  git -C "$root" rev-parse HEAD
  git -C "$root" status --porcelain
  echo "operation=$operation warmups=$warmups samples=$samples target_ms=$target_ms forks=$forks pass=$pass libraries=$libraries"
  echo "JVM benchmark runtime and VM flags: see the JMH logs and CSV runtime fields."
  echo "Native: Kotlin 2.4.10 release executable; build flags are in the source commit and native-build.log."
  env | LC_ALL=C sort | awk '/^KOBLAS_(DENSE|SPARSE)_/ { print }'
} >"$run/provenance.txt"
git -C "$root" diff --binary HEAD >"$run/source.patch"

common=("-Pbench.operation=$operation" "-Pbench.cases=$cases" "-Pbench.warmups=$warmups" "-Pbench.samples=$samples" "-Pbench.targetMs=$target_ms" "-Pbench.pass=$pass")
(cd "$root" && ./gradlew :koblas-bench:jvmScalarBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$run/jvm-scalar.csv") 2>&1 | tee "$run/jvm-scalar.log"
(cd "$root" && ./gradlew :koblas-bench:jvmCBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$run/jvm-c.csv") 2>&1 | tee "$run/jvm-c.log"
(cd "$root" && ./gradlew :koblas-bench:jvmSimdBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$run/jvm-simd.csv") 2>&1 | tee "$run/jvm-simd.log"
(cd "$root" && ./gradlew :koblas-bench:nativeBenchmark "${common[@]}" "-Pbench.output=$run/native.csv") 2>&1 | tee "$run/native-build.log"
"$bench/reference.sh" --libraries "$libraries" --output "$run/vendor" --cases "$cases" --samples "$samples" --warmups "$warmups" --target-ms "$target_ms" --pass "$pass" 2>&1 | tee "$run/vendor.log"

echo "complete" >"$run/status.txt"
echo "$run"
