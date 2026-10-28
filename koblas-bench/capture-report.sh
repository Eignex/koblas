#!/usr/bin/env bash
set -euo pipefail

harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
source "$(dirname "$harness")/script-helpers.sh"
root=$(cd "${KOBLAS_CAPTURE_ROOT:-$(dirname "$harness")/..}" && pwd)
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
rounds=1
cpu=
jvm_gc=
jvm_heap=
original_args=("$@")
vendors_only=false
smoke=false
selected_targets=
custom_cases=false
prepared=false
retain_incomplete=false

usage() {
  echo "usage: capture-report.sh [--libraries openblas,accelerate,onemkl,aocl,armpl|all] [--targets NAME,...] [--prepared] [--retain-incomplete] [--vendors-only] [--smoke] [--output DIR] [--native-executable PATH] [--cases FILE] [--operation NAME|all] [--suite default|sweep] [--samples N] [--warmups N] [--target-ms N] [--forks N] [--pass N] [--rounds N] [--cpu N|auto|spread] [--jvm-gc g1|serial|zgc] [--jvm-heap SIZE]" >&2
}
while (($#)); do
  case "$1" in
    --prepared) prepared=true; shift ;;
    --retain-incomplete) retain_incomplete=true; shift ;;
    --libraries) libraries=${2:?}; shift 2 ;;
    --targets) selected_targets=${2:?}; shift 2 ;;
    --suite) suite=${2:?}; shift 2 ;;
    --operation) operation=${2:?}; shift 2 ;;
    --cases) cases=${2:?}; custom_cases=true; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --native-executable) native_executable=${2:?}; shift 2 ;;
    --samples) samples=${2:?}; shift 2 ;;
    --warmups) warmups=${2:?}; shift 2 ;;
    --target-ms) target_ms=${2:?}; shift 2 ;;
    --forks) forks=${2:?}; shift 2 ;;
    --pass) pass=${2:?}; shift 2 ;;
    --rounds) rounds=${2:?}; shift 2 ;;
    --cpu) cpu=${2:?}; shift 2 ;;
    --jvm-gc) jvm_gc=${2:?}; shift 2 ;;
    --jvm-heap) jvm_heap=${2:?}; shift 2 ;;
    --vendors-only) vendors_only=true; shift ;;
    --smoke) smoke=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ $rounds =~ ^[1-9][0-9]*$ ]] || { echo "rounds must be a positive integer" >&2; exit 2; }
[[ -z $jvm_gc || $jvm_gc == g1 || $jvm_gc == serial || $jvm_gc == zgc ]] || { echo "jvm-gc must be g1, serial or zgc" >&2; exit 2; }
[[ -z $jvm_heap || $jvm_heap =~ ^[1-9][0-9]*[mg]$ ]] || { echo "jvm-heap must be a positive size ending in m or g" >&2; exit 2; }
if [[ -n $cpu && ${KOBLAS_BENCH_PINNED:-false} != true ]]; then
  [[ $(uname -s) == Linux ]] || { echo "--cpu requires Linux taskset" >&2; exit 2; }
  cpu_policy=$cpu
  if [[ $cpu == spread ]]; then cpu=$(python3 "$bench/select-cpu.py"); fi
  if [[ $cpu == auto ]]; then cpu=$(first_allowed_cpu); fi
  [[ $cpu =~ ^[0-9]+$ ]] || { echo "cpu must be a logical CPU number, auto or spread" >&2; exit 2; }
  exec taskset -c "$cpu" env KOBLAS_BENCH_PINNED=true KOBLAS_BENCH_CPU_POLICY="$cpu_policy" "$0" "${original_args[@]}"
fi
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

available_targets=()
if ! $vendors_only; then
  available_targets+=(jvm-scalar jvm-simd jvm-default)
  if $native_capable; then available_targets+=(native native-default); fi
fi
for vendor in "${vendors[@]}"; do
  if $native_capable; then available_targets+=("$vendor"); fi
  available_targets+=("$vendor-jvm")
done
targets=("${available_targets[@]}")
if [[ -n $selected_targets ]]; then
  IFS=, read -r -a targets <<<"$selected_targets"
  seen_targets=,
  for target in "${targets[@]}"; do
    found=false
    for available in "${available_targets[@]}"; do
      if [[ $target == "$available" ]]; then found=true; break; fi
    done
    $found || { echo "target is not available with these libraries and platform: $target" >&2; exit 2; }
    [[ $seen_targets != *",$target,"* ]] || { echo "duplicate target: $target" >&2; exit 2; }
    seen_targets+="$target,"
  done
fi

temporary=$(mktemp -d "${TMPDIR:-/tmp}/koblas-bench-report.XXXXXX")
publication=
cleanup() {
  if $retain_incomplete && [[ -n ${results:-} && -d $results && -n $output && ! -e $output ]]; then
    mkdir -p "$(dirname "$output")"
    mv "$results" "$output"
  fi
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
jvm_init_args=()
if [[ -n $jvm_gc || -n $jvm_heap ]]; then
  # An init script configures old revision tasks too, without changing their source or build files.
  {
    echo 'allprojects { tasks.withType(JavaExec).configureEach {'
    echo "if (name in ['jvmSimdBenchmark', 'jvmScalarBenchmark', 'jvmDefaultBenchmark', 'jvmVendorBenchmark']) {"
    case "$jvm_gc" in
      g1) echo "jvmArgs('-XX:+UseG1GC')" ;;
      serial) echo "jvmArgs('-XX:+UseSerialGC')" ;;
      zgc) echo "jvmArgs('-XX:+UseZGC')" ;;
    esac
    if [[ -n $jvm_heap ]]; then printf "minHeapSize = '%s'\nmaxHeapSize = '%s'\n" "$jvm_heap" "$jvm_heap"; fi
    echo '} } }'
  } >"$temporary/runtime.init.gradle"
  jvm_init_args=(--init-script "$temporary/runtime.init.gradle")
fi
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
source_kind=git
if [[ $(git -C "$root" rev-parse --show-toplevel 2>/dev/null || true) == "$root" ]]; then
  commit=$(git -C "$root" rev-parse HEAD)
  dirty=false
  [[ -z $(git -C "$root" status --porcelain) ]] || dirty=true
else
  # Lab workers receive source archives without Git metadata; the job records their revision.
  source_kind=archive
  commit=unavailable
  dirty=unavailable
fi
if [[ -z $output ]]; then
  if $vendors_only || $smoke || $custom_cases || [[ $operation != all || -n $selected_targets ]]; then output="$bench/build/benchmarks/$hardware_hash";
  else output="$bench/reports/$hardware_hash"; fi
fi
[[ $output == /* ]] || output="$PWD/$output"
[[ ! -L $output && ( ! -e $output || -f $output/metadata.txt ) ]] || {
  echo "output must be a report directory: $output" >&2; exit 2;
}
if $retain_incomplete; then
  [[ ! -e $output ]] || { echo "--retain-incomplete requires an absent output directory" >&2; exit 2; }
fi
if ((rounds > 1)); then
  [[ ! -e $output ]] || { echo "multiple rounds require a new output directory: $output" >&2; exit 2; }
  mkdir -p "$(dirname "$output")"
  mkdir "$output"
  metadata="$output/metadata.txt"
  printf 'status=incomplete\n\n[capture]\nsource_commit=%s\nsource_kind=%s\nrounds=%s\naggregation=retain round medians for downstream aggregation\ncompleted_at=pending\n' \
    "$commit" "$source_kind" "$rounds" >"$metadata"
  printf 'jvm_gc=%s\njvm_heap=%s\n' "${jvm_gc:-default}" "${jvm_heap:-default}" >>"$metadata"
  fingerprint "$root" sources "$source_kind" >"$output/source-hashes.txt"
  for ((round=1; round<=rounds; round++)); do
    fingerprint "$root" sources "$source_kind" >"$temporary/current-hashes.txt"
    cmp "$output/source-hashes.txt" "$temporary/current-hashes.txt"
    offset=$(((round - 1) % ${#targets[@]}))
    ordered=("${targets[@]:offset}" "${targets[@]:0:offset}")
    order=$(IFS=,; echo "${ordered[*]}")
    printf 'round_%s_targets=%s\n' "$round" "$order" >>"$metadata"
    args=(--libraries "$libraries" --targets "$order" --cases "$cases" --suite "$suite" --operation "$operation"
      --samples "$samples" --warmups "$warmups" --target-ms "$target_ms" --forks "$forks"
      --pass "$round" --output "$output/pass-$round")
    if [[ -n $native_executable ]]; then args+=(--native-executable "$native_executable"); fi
    if [[ -n $jvm_gc ]]; then args+=(--jvm-gc "$jvm_gc"); fi
    if [[ -n $jvm_heap ]]; then args+=(--jvm-heap "$jvm_heap"); fi
    if $prepared; then args+=(--prepared); fi
    if $retain_incomplete; then args+=(--retain-incomplete); fi
    if $vendors_only; then args+=(--vendors-only); fi
    if $smoke; then args+=(--smoke); fi
    "$0" "${args[@]}" 2>&1 | tee "$output/pass-$round.log"
    fingerprint "$root" sources "$source_kind" >"$temporary/current-hashes.txt"
    cmp "$output/source-hashes.txt" "$temporary/current-hashes.txt"
    fingerprint "$root" binaries >"$output/pass-$round-binary-hashes.txt"
    if ((round > 1)); then cmp "$output/pass-1-binary-hashes.txt" "$output/pass-$round-binary-hashes.txt"; fi
  done
  sed -e '1s/status=incomplete/status=complete/' \
    -e "s/^completed_at=pending$/completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)/" \
    "$metadata" >"$temporary/completed-metadata.txt"
  mv "$temporary/completed-metadata.txt" "$metadata"
  echo "$output"
  exit 0
fi

results="$temporary/results"
mkdir "$results"
metadata="$results/metadata.txt"
awk -v operation="$operation" -v suite="$suite" -v smoke="$smoke" \
  -f "$bench/select-cases.awk" "$cases" >"$temporary/cases.txt"
cases="$temporary/cases.txt"
cp "$cases" "$results/cases.txt"
{
  echo "status=incomplete"
  printf '\n[capture]\n'
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "completed_at=pending"
  echo "source_commit=$commit"
  echo "source_kind=$source_kind"
  echo "dirty=$dirty"
  printf 'jvm_gc=%s\njvm_heap=%s\n' "${jvm_gc:-default}" "${jvm_heap:-default}"
  echo "platform=$(uname -a)"
  printf '%s\n' "operation=$operation" "suite=$suite" "selected_cases=$(wc -l <"$cases" | tr -d ' ')" "warmups=$warmups" "requested_samples=$samples" \
    "target_ns=$((target_ms * 1000000))" "pass=$pass" "libraries=$libraries" "vendors_only=$vendors_only"
  echo "native_capable=$native_capable"
  # Which binary the native rows came from, since a supplied one was built on another machine and is the only
  # way an ARM64 host has any. A report that does not say this cannot be told apart from one that linked here.
  echo "native_executable=${native_executable:-not supplied}"
  echo "requested_jvm_forks=$forks"
  echo "cpu_policy=${KOBLAS_BENCH_CPU_POLICY:-unbound}"
  if [[ $platform == Linux ]]; then awk '/^Cpus_allowed_list:/ { print "cpu_affinity=" $2 }' /proc/self/status; fi
  echo "targets=$(IFS=,; echo "${targets[*]}")"
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
  "$@" 2>&1 | tee "$results/$target.log"
  awk '/^resolved implementation=/ {
    sub(/^resolved /, ""); sub(/ runtime=/, "\nruntime="); sub(/ harness=/, "\nharness="); print
  }' "$results/$target.log" >>"$metadata"
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
  if $prepared && [[ -z $native_executable ]]; then
    echo "Prepared Native capture requires --native-executable" >&2; return 2
  fi
  if [[ -n $native_executable ]]; then
    run_target "$target" "$native_executable" "--mode=$mode" "--operation=$operation" "--suite=$suite" \
      "--cases=$cases" "--warmups=$warmups" "--samples=$samples" "--target-ms=$target_ms" \
      "--output=$results/$target.csv"
  else
    run_target "$target" ./gradlew --no-daemon ":koblas-bench:$task" "${common[@]}" "$@" \
      "-Pbench.output=$results/$target.csv"
  fi
}
run_jvm() {
  local target=$1 mode=$2 task=$3
  shift 3
  if $prepared; then
    local jvm_args=(--enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=1)
    if [[ -n $jvm_heap ]]; then
      jvm_args+=("-Xms$jvm_heap" "-Xmx$jvm_heap")
    else
      jvm_args+=("-Xmx${KOBLAS_BENCH_HEAP_MIB:-512}m")
    fi
    case "$jvm_gc" in
      g1) jvm_args+=(-XX:+UseG1GC) ;;
      serial) jvm_args+=(-XX:+UseSerialGC) ;;
      zgc) jvm_args+=(-XX:+UseZGC) ;;
    esac
    [[ $mode != jvm-simd && $mode != jvm-default ]] || jvm_args+=(--add-modules=jdk.incubator.vector)
    run_target "$target" "${KOBLAS_BENCH_JAVA:?}" "${jvm_args[@]}" -cp "${KOBLAS_BENCH_CLASSPATH:?}" \
      com.eignex.koblas.bench.JvmRunnerKt "--mode=$mode" "--operation=$operation" "--suite=$suite" \
      "--cases=$cases" "--warmups=$warmups" "--samples=$samples" "--target-ms=$target_ms" \
      "--forks=$forks" "--output=$results/$target.csv"
  else
    run_target "$target" ./gradlew --no-daemon "${jvm_init_args[@]}" ":koblas-bench:$task" "${common[@]}" "$@" \
      "-Pbench.forks=$forks" "-Pbench.output=$results/$target.csv"
  fi
}
# Vendor arms run through the production Koblas bindings on both runtimes. There is one implementation of each
# vendor call and one description of it, so what a row claims about the library that ran is the binding's own
# answer rather than a second program's. Nothing here sets a thread count: the binding holds each library to one
# compute thread at load, before any arithmetic, and refuses one it cannot hold, so an environment variable
# would be a weaker second answer to a question that is already settled.
#
# The plain vendor target keeps its name and reaches the library the way the C program did, from native code
# with no runtime between the caller's storage and the call. The JVM target is the same operation through the
# JVM binding, which copies operands into native memory and reports that transfer as part of its route.
for target in "${targets[@]}"; do
  case "$target" in
    jvm-scalar|jvm-simd|jvm-default)
      case "$target" in
        jvm-scalar) task=jvmScalarBenchmark ;;
        jvm-simd) task=jvmSimdBenchmark ;;
        jvm-default) task=jvmDefaultBenchmark ;;
      esac
      run_jvm "$target" "$target" "$task"
      ;;
    native) run_native native native nativeBenchmark ;;
    native-default) run_native native-default native-default nativeDefaultBenchmark ;;
    *)
      vendor=${target%-jvm}
      [[ $vendor != accelerate || $platform == Darwin ]] || { echo "Accelerate requires macOS" >&2; exit 2; }
      if [[ $target == *-jvm ]]; then
        run_jvm "$target" "jvm-vendor-$vendor" jvmVendorBenchmark "-Pbench.vendor=$vendor"
      else
        run_native "$target" "native-vendor-$vendor" nativeVendorBenchmark "-Pbench.vendor=$vendor"
      fi
      ;;
  esac
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
