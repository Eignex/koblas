#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-jvm}
profile=${2:-report}
output=${3:-"$root/koblas-bench/build/reports/submissions"}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ $target == native ]]; then
    case "$(uname -s):$(uname -m)" in
        Linux:x86_64|Linux:amd64) target=linuxX64 ;;
        Darwin:arm64) target=macosArm64 ;;
        *) echo "native reports require Linux x86-64 or Apple Silicon" >&2; exit 1 ;;
    esac
fi
case $target in
    jvm|linuxX64|macosArm64) ;;
    *) echo "target must be jvm, native, linuxX64, or macosArm64" >&2; exit 1 ;;
esac
case $profile in
    report) profile_task=Report ;;
    openblas) profile_task=Openblas ;;
    oneMkl) profile_task=OneMkl ;;
    *) echo "profile must be report, openblas, or oneMkl" >&2; exit 1 ;;
esac
mkdir -p "$output"
marker1=$(mktemp "$output/.benchmark-marker-1.XXXXXX")
marker2=$(mktemp "$output/.benchmark-marker-2.XXXXXX")
log1=$(mktemp "$output/.benchmark-log-1.XXXXXX")
log2=$(mktemp "$output/.benchmark-log-2.XXXXXX")
raw1=$(mktemp "$output/.benchmark-result-1.XXXXXX")
raw2=$(mktemp "$output/.benchmark-result-2.XXXXXX")
trap 'rm -f "$marker1" "$marker2" "$log1" "$log2" "$raw1" "$raw2"' EXIT
task=":koblas-bench:${target}${profile_task}Benchmark"
command=("$root/gradlew" "$task" --no-daemon)
if [[ -n ${CORES:-} && $(uname -s) == Linux && -x $(command -v taskset) ]]; then
    command=(taskset -c "$CORES" "${command[@]}")
    affinity="taskset -c $CORES"
else affinity=none; fi
if command -v rg &>/dev/null; then resolved_grep() { rg "$@"; }
else resolved_grep() { grep -E "$@"; }
fi
cd "$root"
"$root/gradlew" ":koblas-bench:${target}BenchmarkCompile"
run_pass() {
    local marker=$1 log=$2 destination=$3
    rm -rf "$root/koblas-bench/build/reports/benchmarks/$profile"
    touch "$marker"
    if ! "${command[@]}" --rerun-tasks 2>&1 | tee "$log" >&2; then
        if resolved_grep -q '/tmp/jmh\.lock|jmh\.lock' "$log"; then echo "A stale JMH lock was detected; inspect it before removing it." >&2; fi
        exit 1
    fi
    if resolved_grep -q '<failure>|EXCEPTION: <ERROR>' "$log"; then
        echo "Benchmark fork reported a setup or execution failure." >&2
        exit 1
    fi
    fresh=()
    while IFS= read -r result; do fresh+=("$result"); done < <(
        find "$root/koblas-bench/build/reports" -name '*.json' -newer "$marker" -print
    )
    if ((${#fresh[@]} != 1)); then echo "Expected exactly one fresh benchmark JSON, found ${#fresh[@]}." >&2; exit 1; fi
    cp "${fresh[0]}" "$destination"
}
run_pass "$marker1" "$log1" "$raw1"
run_pass "$marker2" "$log2" "$raw2"
pass1=$raw1
pass2=$raw2
if cmp -s "$pass1" "$pass2"; then echo "Independent passes produced identical JSON; refusing a possibly stale report." >&2; exit 1; fi
if [[ $profile == report ]]; then
    python3 "$root/koblas-bench/tools/check-benchmark-coverage.py" "$root/koblas-bench/benchmark-coverage.tsv" "$pass1" "$pass2"
fi
stage=$(mktemp -d "$output/.koblas-report-${target}-${timestamp}.XXXXXX")
archive="$output/koblas-report-${target}-${timestamp}.tar.gz"
git_commit=$(git rev-parse HEAD) || git_commit=unknown
gradle_version=$("$root/gradlew" --version --quiet | awk '/Gradle / { print $2; exit }') || gradle_version=unknown
{
    echo "timestamp_utc=$timestamp"
    echo "git_commit=$git_commit"
    echo "git_dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
    echo "os=$(uname -srm)"
    echo "architecture=$(uname -m)"
    echo "gradle_version=$gradle_version"
    echo "target=$target"
    echo "profile=$profile"
    echo "command=${command[*]}"
    echo "cpu_model=$(lscpu 2>/dev/null | awk -F: '/Model name/ { sub(/^ +/, "", $2); print $2; exit }' || true)"
    echo "logical_cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || true)"
    echo "affinity=$affinity"
    if [[ $target == jvm ]]; then echo "jvm_version=$(java -version 2>&1 | head -n 1)"; fi
    echo "resolved_backends:"
    resolved_grep '^resolved:' "$log1" || true
    resolved_grep '^resolved:' "$log2" || true
} > "$stage/metadata.txt"
cp "$log1" "$stage/benchmark-pass1.log"
cp "$log2" "$stage/benchmark-pass2.log"
cp "$root/koblas-bench/benchmark-coverage.tsv" "$stage/"
cp "$root/koblas-bench/comparator-coverage.tsv" "$stage/"
cp "$pass1" "$stage/results-pass1.json"
cp "$pass2" "$stage/results-pass2.json"
tar -C "$(dirname "$stage")" -czf "$archive" "$(basename "$stage")"
rm -rf "$stage"
echo "$(cd "$(dirname "$archive")" && pwd)/$(basename "$archive")"
