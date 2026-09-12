#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench="$root/koblas-bench"
reports="$bench/reports"
libraries=all
operation=all
suite=all
samples=5
warmups=3
target_ms=1000
forks=2
pass=1

usage() {
  echo "usage: koblas-bench/capture-report.sh [--libraries openblas,accelerate,onemkl|all] [--operation NAME|all] [--suite all|packed] [--samples N] [--warmups N] [--target-ms N] [--forks N] [--pass N]" >&2
}

while (($#)); do
  case "$1" in
    --libraries) libraries=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --suite) suite=${2:?}; shift 2 ;;
    --samples) samples=${2:?}; shift 2 ;;
    --warmups) warmups=${2:?}; shift 2 ;;
    --target-ms) target_ms=${2:?}; shift 2 ;;
    --forks) forks=${2:?}; shift 2 ;;
    --pass) pass=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ $suite == all || $suite == packed ]] || { usage; exit 2; }

temporary=$(mktemp -d "${TMPDIR:-/tmp}/koblas-bench-report.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$bench/tools/hardware.sh" >"$temporary/hardware.txt"

cases="$bench/cases.txt"
if [[ $operation != all || $suite == packed ]]; then
  cases="$temporary/cases.txt"
  awk -F+ -v operation="$operation" -v suite="$suite" '
    /^[[:space:]]*($|#)/ || ((operation == "all" || $1 == operation) && (suite == "all" || /\+packed=/)) { print }
  ' "$bench/cases.txt" >"$cases"
fi

if command -v sha256sum >/dev/null 2>&1; then
  hardware_hash=$(sha256sum "$temporary/hardware.txt" | awk '{ print $1 }')
else
  hardware_hash=$(shasum -a 256 "$temporary/hardware.txt" | awk '{ print $1 }')
fi

commit=$(git -C "$root" rev-parse --short=12 HEAD)
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$commit"
mkdir -p "$reports/$hardware_hash"
run="$reports/$hardware_hash/$run_id"
[[ ! -e $run ]] || { echo "report already exists: $run" >&2; exit 1; }
results="$temporary/results"
mkdir "$results"
cp "$cases" "$temporary/selected-cases.txt"
cases="$temporary/selected-cases.txt"
common=("-Pbench.operation=$operation" "-Pbench.cases=$cases" "-Pbench.warmups=$warmups" "-Pbench.samples=$samples" "-Pbench.targetMs=$target_ms" "-Pbench.pass=$pass")
(cd "$root" && ./gradlew :koblas-bench:jvmScalarBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$results/jvm-scalar.csv")
(cd "$root" && ./gradlew :koblas-bench:jvmCBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$results/jvm-c.csv")
(cd "$root" && ./gradlew :koblas-bench:jvmSimdBenchmark "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$results/jvm-simd.csv")
(cd "$root" && ./gradlew :koblas-bench:nativeBenchmark "${common[@]}" "-Pbench.output=$results/native.csv")
"$bench/reference.sh" --libraries "$libraries" --output "$temporary/vendor" --cases "$cases" --samples "$samples" --warmups "$warmups" --target-ms "$target_ms" --pass "$pass"

mv "$temporary/vendor/"*.csv "$results/"
mkdir "$run"
mv "$results/"*.csv "$run/"
echo "$run"
