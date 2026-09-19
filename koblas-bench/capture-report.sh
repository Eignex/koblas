#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench="$root/koblas-bench"
libraries=all
operation=all
suite=default
cases="$bench/cases.txt"
output=
native_executable=
samples=5
warmups=3
target_ms=1000
forks=2
pass=1
vendors_only=false
smoke=false

usage() {
  echo "usage: capture-report.sh [--libraries openblas,accelerate,onemkl,aocl,armpl|all] [--vendors-only] [--smoke] [--output DIR] [--native-executable PATH] [--operation NAME|all] [--suite default|sweep] [--samples N] [--warmups N] [--target-ms N] [--forks N] [--pass N]" >&2
}
while (($#)); do
  case "$1" in
    --libraries) libraries=${2:?}; shift 2 ;;
    --suite) suite=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --native-executable) native_executable=${2:?}; shift 2 ;;
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
[[ $suite == default || $suite == sweep ]] || { echo "suite must be default or sweep" >&2; exit 2; }
[[ $suite != sweep || $operation != all ]] || { echo "suite sweep requires a specific operation" >&2; exit 2; }
if $smoke; then samples=1; warmups=0; target_ms=1; forks=1; fi
platform=$(uname -s)
# Whichever production vendor this platform selects, beside OpenBLAS as the reference arm. oneMKL has no
# ARM64 build and ArmPL no x86-64 one, so the architecture decides here as much as the operating system does.
if [[ $libraries == all ]]; then
  if [[ $platform == Darwin ]]; then libraries=openblas,accelerate
  elif [[ $(uname -m) == aarch64 || $(uname -m) == arm64 ]]; then libraries=openblas,armpl
  else libraries=openblas,onemkl; fi
fi
# The Kotlin/Native compiler ships no linux-aarch64 host, so an ARM64 Linux machine cannot build the native
# executable at all and its capture is the JVM arms only. Recorded in metadata.txt so a report with no native
# rows says which it is, a host that could not build them or a run that was asked for less.
#
# That limit is on building, not on measuring. The compiler cross-compiles to linuxArm64 from an x86-64 host,
# so --native-executable takes a binary built there and carried over, and the native arms become reachable on
# a machine that could never have produced one.
native_capable=true
if [[ $platform == Linux && ( $(uname -m) == aarch64 || $(uname -m) == arm64 ) ]]; then native_capable=false; fi
if [[ -n $native_executable ]]; then
  [[ -x $native_executable ]] || { echo "not an executable: $native_executable" >&2; exit 2; }
  native_executable=$(cd "$(dirname "$native_executable")" && pwd)/$(basename "$native_executable")
  native_capable=true
fi
IFS=, read -r -a vendors <<<"$libraries"
for vendor in "${vendors[@]}"; do
  [[ $vendor == openblas || $vendor == accelerate || $vendor == onemkl || $vendor == aocl || $vendor == armpl ]] ||
    { echo "unknown library: $vendor" >&2; exit 2; }
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
  echo "native_capable=$native_capable"
  # Which binary the native rows came from, since a supplied one was built on another machine and is the only
  # way an ARM64 host has any. A report that does not say this cannot be told apart from one that linked here.
  echo "native_executable=${native_executable:-built on this host}"
  echo "requested_jvm_forks=$forks"
  env | LC_ALL=C sort | awk '/^KOBLAS_(DENSE|SPARSE)_/ { print }'
  # Koblas arms are single-threaded by construction. Each vendor arm records what its binding read back from
  # the library it opened, in that target's own section, rather than being covered by a claim made here.
  echo "koblas_threads=1"
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

# One native arm, reached either way. A supplied executable is run as it stands, since the machine holding it
# is by definition one that could not have linked it; otherwise Gradle links the host's own and runs that.
# The arguments are the same either way because the Gradle task only passes these through to the binary.
run_native() {
  local target=$1 mode=$2 task=$3
  shift 3
  if [[ -n $native_executable ]]; then
    run_target "$target" "$native_executable" "--mode=$mode" "--operation=$operation" "--suite=$suite" \
      "--cases=$cases" "--warmups=$warmups" "--samples=$samples" "--target-ms=$target_ms" \
      "--output=$results/$target.csv"
  else
    run_target "$target" ./gradlew --no-daemon ":koblas-bench:$task" "${common[@]}" "$@" \
      "-Pbench.output=$results/$target.csv"
  fi
}
if ! $vendors_only; then
  for target in jvm-scalar jvm-simd jvm-default; do
    case "$target" in
      jvm-scalar) task=jvmScalarBenchmark ;;
      jvm-simd) task=jvmSimdBenchmark ;;
      jvm-default) task=jvmDefaultBenchmark ;;
    esac
    run_target "$target" ./gradlew --no-daemon ":koblas-bench:$task" "${common[@]}" "-Pbench.forks=$forks" "-Pbench.output=$results/$target.csv"
  done
  if $native_capable; then run_native native native nativeBenchmark; fi
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
  if $native_capable; then
    run_native "$vendor" "native-vendor-$vendor" nativeVendorBenchmark "-Pbench.vendor=$vendor"
  fi
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
