#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench="$root/koblas-bench"
libraries=all
operation=all
suite=default
native_variant=
cases="$bench/cases.txt"
output=
samples=5
warmups=3
target_ms=1000
forks=2
pass=1
vendors_only=false
smoke=false

usage() {
  echo "usage: capture-report.sh [--libraries openblas,accelerate,onemkl|all] [--native-variant scalar|sse2|avx2|neon] [--vendors-only] [--smoke] [--output DIR] [--operation NAME|all] [--suite default|sweep] [--samples N] [--warmups N] [--target-ms N] [--forks N] [--pass N]" >&2
}
while (($#)); do
  case "$1" in
    --libraries) libraries=${2:?}; shift 2 ;;
    --native-variant) native_variant=${2:?}; shift 2 ;;
    --suite) suite=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --samples) samples=${2:?}; shift 2 ;;
    --warmups) warmups=${2:?}; shift 2 ;;
    --target-ms) target_ms=${2:?}; shift 2 ;;
    --forks) forks=${2:?}; shift 2 ;;
    --pass) pass=${2:?}; shift 2 ;;
    --vendors-only) vendors_only=true; shift ;;
    --smoke) smoke=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -z $native_variant || $native_variant == scalar || $native_variant == sse2 || $native_variant == avx2 || $native_variant == neon ]] || { usage; exit 2; }
[[ $suite == default || $suite == sweep ]] || { echo "suite must be default or sweep" >&2; exit 2; }
[[ $suite != sweep || $operation != all ]] || { echo "suite sweep requires a specific operation" >&2; exit 2; }
if $smoke; then samples=1; warmups=0; target_ms=1; forks=1; fi
platform=$(uname -s)
if [[ $libraries == all ]]; then
  if [[ $platform == Darwin ]]; then libraries=openblas,accelerate; else libraries=openblas,onemkl; fi
fi
IFS=, read -r -a vendors <<<"$libraries"
for vendor in "${vendors[@]}"; do
  [[ $vendor == openblas || $vendor == accelerate || $vendor == onemkl ]] || { echo "unknown library: $vendor" >&2; exit 2; }
done

temporary=$(mktemp -d "${TMPDIR:-/tmp}/koblas-bench-report.XXXXXX")
publication=
cleanup() {
  if [[ -n $publication ]]; then
    if [[ ! -e $output && -d $publication/previous ]]; then
      mv "$publication/previous" "$output" || return
    fi
    rm -rf "$publication"
  fi
  rm -rf "$temporary"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
{
  echo "architecture=$(uname -m)"
  if [[ $platform == Linux ]]; then
    LC_ALL=C lscpu | awk -F: '
      /^[[:space:]]*(Vendor ID|Model name|CPU family|Model|Stepping|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|.*[Cc]ache|NUMA node\(s\)):/ {
        key=$1; value=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print tolower(key) "=" value
      }'
    awk '/^MemTotal:/ { print "memory_kib=" $2 }' /proc/meminfo
  elif [[ $platform == Darwin ]]; then
    sysctl -n machdep.cpu.brand_string | sed 's/^/cpu_model=/'
    sysctl -n hw.physicalcpu | sed 's/^/physical_cpus=/'
    sysctl -n hw.physicalcpu_max | sed 's/^/physical_cpus_max=/'
    sysctl -n hw.memsize | sed 's/^/memory_bytes=/'
  else
    echo "platform=$platform"
  fi
} >"$temporary/hardware.txt"
# Linux usable memory varies with kernel reservations; retain it only as run metadata.
sed '/^memory_kib=/d' "$temporary/hardware.txt" | LC_ALL=C sort >"$temporary/hardware-key.txt"
if command -v sha256sum >/dev/null 2>&1; then
  hardware_hash=$(sha256sum "$temporary/hardware-key.txt" | awk '{ print $1 }')
else
  hardware_hash=$(shasum -a 256 "$temporary/hardware-key.txt" | awk '{ print $1 }')
fi
commit=$(git -C "$root" rev-parse HEAD)
dirty=false
[[ -z $(git -C "$root" status --porcelain) ]] || dirty=true
if [[ -z $output ]]; then
  if $vendors_only || $smoke || [[ $operation != all ]]; then output="$bench/build/benchmarks/$hardware_hash";
  else output="$bench/reports/$hardware_hash"; fi
fi
[[ $output == /* ]] || output="$PWD/$output"
[[ ! -L $output && ( ! -e $output || -f $output/metadata.txt ) ]] || {
  echo "output must be a report directory: $output" >&2; exit 2;
}
results="$temporary/results"
mkdir "$results"
metadata="$results/metadata.txt"
awk -v operation="$operation" -v suite="$suite" -v smoke="$smoke" \
  -f "$bench/select-cases.awk" "$cases" >"$temporary/cases.txt"
cases="$temporary/cases.txt"
{
  echo "status=incomplete"
  printf '\n[capture]\n'
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "completed_at=pending"
  echo "source_commit=$commit"
  echo "dirty=$dirty"
  echo "platform=$(uname -a)"
  printf '%s\n' "operation=$operation" "suite=$suite" "selected_cases=$(wc -l <"$cases" | tr -d ' ')" "warmups=$warmups" "requested_samples=$samples" \
    "target_ns=$((target_ms * 1000000))" "pass=$pass" "libraries=$libraries" "vendors_only=$vendors_only"
  echo "requested_jvm_forks=$forks"
  if ! $vendors_only; then
    [[ -z $native_variant ]] || echo "requested_native_variant=$native_variant"
  fi
  env | LC_ALL=C sort | awk '/^KOBLAS_(DENSE|SPARSE)_/ { print }'
  echo "threads=1"
  printf '\n[toolchain]\n'
  # Only the koblas C kernel arms still involve a C compiler; a vendor-only capture compiles no C at all.
  if ! $vendors_only; then cc --version | head -n 1; fi
  printf '\n[hardware]\n'
  cat "$temporary/hardware.txt"
} >"$metadata"

run_target() {
  local target=$1
  shift
  printf '\n[%s]\n' "$target" >>"$metadata"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$metadata"
  "$@" 2>&1 | tee "$temporary/$target.log"
  awk '/^resolved implementation=/ {
    sub(/^resolved /, ""); sub(/ runtime=/, "\nruntime="); sub(/ harness=/, "\nharness="); print
  }' "$temporary/$target.log" >>"$metadata"
  if [[ $target == jvm-* || $target == *-jvm ]]; then
    echo "warmup_target_ns=$((target_ms * 1000000))"
  else
    echo "warmup_target_ns=$((target_ms > 4 ? target_ms * 250000 : 1000000))"
  fi >>"$metadata"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$metadata"
}

cd "$root"
common=("-Pbench.operation=$operation" "-Pbench.suite=$suite" "-Pbench.cases=$cases" "-Pbench.warmups=$warmups" "-Pbench.samples=$samples" "-Pbench.targetMs=$target_ms")
if ! $vendors_only; then
  c_target=jvm-c
  native_target=native
  if [[ -n $native_variant ]]; then
    c_target=jvm-c-raw-$native_variant
    native_target=native-raw-$native_variant
    common+=("-Pbench.variant=$native_variant")
  fi
  for target in jvm-scalar "$c_target" jvm-simd "$native_target"; do
    case "$target" in
      jvm-scalar) task=jvmScalarBenchmark ;;
      jvm-c) task=jvmCBenchmark ;;
      jvm-c-raw-*) task=jvmCRawBenchmark ;;
      jvm-simd) task=jvmSimdBenchmark ;;
      native*) task=nativeBenchmark ;;
    esac
    run_target "$target" ./gradlew --no-daemon ":koblas-bench:$task" "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$results/$target.csv"
  done
fi
# Vendor arms run through the production Koblas bindings on both runtimes. There is one implementation of each
# vendor call and one description of it, so what a row claims about the library that ran is the binding's own
# answer rather than a second program's. Nothing here sets a thread count: the binding holds each library to one
# compute thread at load, before any arithmetic, and refuses one it cannot hold, so an environment variable
# would be a weaker second answer to a question that is already settled.
#
# The plain vendor target keeps its name and reaches the library the way the C program did, from native code
# with no runtime between the caller's storage and the call. The JVM target is the same operation through the
# JVM binding, which copies operands into native memory and reports that transfer as part of its route.
for vendor in "${vendors[@]}"; do
  [[ $vendor != accelerate || $platform == Darwin ]] || { echo "Accelerate requires macOS" >&2; exit 2; }
  run_target "$vendor" ./gradlew --no-daemon ":koblas-bench:nativeVendorBenchmark" "${common[@]}" \
    "-Pbench.vendor=$vendor" "-Pbench.output=$results/$vendor.csv"
  run_target "$vendor-jvm" ./gradlew --no-daemon ":koblas-bench:jvmVendorBenchmark" "${common[@]}" \
    "-Pbench.vendor=$vendor" "-Pbench.forks=$forks" "-Pbench.output=$results/$vendor-jvm.csv"
done
sed -e '1s/status=incomplete/status=complete/' \
  -e "s/^completed_at=pending$/completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)/" \
  "$metadata" >"$temporary/metadata.txt"
mv "$temporary/metadata.txt" "$metadata"
mkdir -p "$(dirname "$output")"
publication=$(mktemp -d "$(dirname "$output")/.koblas-report.XXXXXX")
mv "$results" "$publication/current"
if [[ -d $output ]]; then mv "$output" "$publication/previous"; fi
mv "$publication/current" "$output"
echo "$output"
