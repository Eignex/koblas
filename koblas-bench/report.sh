#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-jvm}
output=${2:-"$root/koblas-bench/build/reports/submissions"}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ $target == native ]]; then
    case "$(uname -s):$(uname -m)" in
        Linux:x86_64|Linux:amd64) target=linuxX64 ;;
        Darwin:arm64) target=macosArm64 ;;
        *) echo "native reports require Linux x86-64 or Apple Silicon" >&2; exit 1 ;;
    esac
fi
case $target in jvm|linuxX64|macosArm64) ;; *) echo "target must be jvm or native" >&2; exit 1 ;; esac
mkdir -p "$output"
marker=$(mktemp "$output/.benchmark-marker.XXXXXX")
log=$(mktemp "$output/.benchmark-log.XXXXXX")
trap 'rm -f "$marker" "$log"' EXIT
task=":koblas-bench:${target}ReportBenchmark"
command=("$root/gradlew" "$task" --no-daemon)
if [[ -n ${CORES:-} && $(uname -s) == Linux && -x $(command -v taskset) ]]; then
    command=(taskset -c "$CORES" "${command[@]}")
    affinity="taskset -c $CORES"
else affinity=none; fi
cd "$root"
"$root/gradlew" ":koblas-bench:${target}BenchmarkCompile" --no-daemon
if ! "${command[@]}" 2>&1 | tee "$log"; then
    if rg -q '/tmp/jmh\.lock|jmh\.lock' "$log"; then echo "A stale JMH lock was detected; inspect it before removing it." >&2; fi
    exit 1
fi
mapfile -t reports < <(find "$root/koblas-bench/build/reports" -name '*.json' -newer "$marker" -print)
if ((${#reports[@]} == 0)); then echo "No fresh benchmark JSON was produced; a stale JMH lock may have prevented measurement." >&2; exit 1; fi
python3 "$root/koblas-bench/tools/check-benchmark-coverage.py" "$root/koblas-bench/benchmark-coverage.tsv" "${reports[@]}"
stage=$(mktemp -d "$output/.koblas-report-${target}-${timestamp}.XXXXXX")
archive="$output/koblas-report-${target}-${timestamp}.tar.gz"
{
    echo "timestamp_utc=$timestamp"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
    echo "os=$(uname -srm)"
    echo "architecture=$(uname -m)"
    echo "gradle_version=$("$root/gradlew" --version --quiet | awk '/Gradle / { print $2; exit }')"
    echo "target=$target"
    echo "command=${command[*]}"
    echo "cpu_model=$(lscpu 2>/dev/null | awk -F: '/Model name/ { sub(/^ +/, "", $2); print $2; exit }' || true)"
    echo "logical_cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || true)"
    echo "affinity=$affinity"
    if [[ $target == jvm ]]; then echo "jvm_version=$(java -version 2>&1 | head -n 1)"; fi
    echo "resolved_backends:"
    rg '^resolved:' "$log" || true
} > "$stage/metadata.txt"
cp "$log" "$stage/benchmark.log"
cp "$root/koblas-bench/benchmark-coverage.tsv" "$stage/"
cp "${reports[@]}" "$stage/"
tar -C "$(dirname "$stage")" -czf "$archive" "$(basename "$stage")"
rm -rf "$stage"
echo "$(cd "$(dirname "$archive")" && pwd)/$(basename "$archive")"
