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
run="$temporary/report"
mkdir "$run"

common=("-Pbench.operation=$operation" "-Pbench.cases=$cases" "-Pbench.warmups=$warmups" "-Pbench.samples=$samples" "-Pbench.targetMs=$target_ms" "-Pbench.pass=$pass")
(cd "$root" && ./gradlew :koblas-bench:jvmScalarBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$run/jvm-scalar.csv")
(cd "$root" && ./gradlew :koblas-bench:jvmCBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$run/jvm-c.csv")
(cd "$root" && ./gradlew :koblas-bench:jvmSimdBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$run/jvm-simd.csv")
(cd "$root" && ./gradlew :koblas-bench:nativeBenchmark "${common[@]}" "-Pbench.output=$run/native.csv")
"$bench/reference.sh" --libraries "$libraries" --output "$run/vendor" --cases "$cases" --samples "$samples" --warmups "$warmups" --target-ms "$target_ms" --pass "$pass"

hardware_report="$reports/$hardware_hash"
if [[ -e $hardware_report/hardware.txt ]]; then
  cmp -s "$temporary/hardware.txt" "$hardware_report/hardware.txt" || {
    echo "hardware hash collision: $hardware_report/hardware.txt differs" >&2
    exit 1
  }
else
  mkdir -p "$hardware_report"
  mv "$temporary/hardware.txt" "$hardware_report/hardware.txt"
fi

destination="$hardware_report/$run_id"
[[ ! -e $destination ]] || { echo "report already exists: $destination" >&2; exit 1; }
mv "$run" "$destination"
echo "$destination"
