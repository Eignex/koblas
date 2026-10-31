#!/usr/bin/env bash
set -euo pipefail
bench=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$bench/script-helpers.sh"
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
tree="$temporary/source archive"
mkdir -p "$tree/src" "$tree/build" "$tree/nested/.gradle" "$tree/koblas/build/libs" "$tree/koblas-bench/build/classes"
echo source >"$tree/src/a file.kt"
echo ignored >"$tree/build/generated"
echo ignored >"$tree/nested/.gradle/cache"
echo binary >"$tree/koblas/build/libs/kernel.jar"
echo binary >"$tree/koblas-bench/build/classes/Runner.class"
fingerprint "$tree" sources archive >"$temporary/sources"
[[ $(wc -l <"$temporary/sources" | tr -d ' ') == 1 ]]
grep -Fxq "$(sha256 "$tree/src/a file.kt") src/a file.kt" "$temporary/sources"
ln -s 'a file.kt' "$tree/src/linked.kt"
fingerprint "$tree" sources archive >"$temporary/linked"
grep -Fxq "$(sha256 "$tree/src/a file.kt") src/linked.kt" "$temporary/linked"
rm "$tree/src/linked.kt"
echo changed >"$tree/build/generated"
fingerprint "$tree" sources archive >"$temporary/unchanged"
cmp "$temporary/sources" "$temporary/unchanged"
echo changed >"$tree/src/a file.kt"
fingerprint "$tree" sources archive >"$temporary/changed"
if cmp -s "$temporary/sources" "$temporary/changed"; then echo 'Source changes must change the fingerprint' >&2; exit 1; fi
fingerprint "$tree" binaries archive >"$temporary/binaries"
fingerprint "$tree" binaries >"$temporary/default-binaries"
cmp "$temporary/binaries" "$temporary/default-binaries"
[[ $(wc -l <"$temporary/binaries" | tr -d ' ') == 2 ]]
grep -Fxq "$(sha256 "$tree/koblas/build/libs/kernel.jar") koblas/build/libs/kernel.jar" "$temporary/binaries"
git -C "$tree" init -q
git -C "$tree" add src
echo untracked >"$tree/untracked"
fingerprint "$tree" sources git >"$temporary/git"
cmp "$temporary/changed" "$temporary/git"
if [[ $(uname -s) == Linux ]]; then
  cpu=$(first_allowed_cpu)
  [[ $cpu =~ ^[0-9]+$ ]]
  taskset -c "$cpu" true
  spread=$(sh "$bench/select-cpu.sh")
  taskset -c "$spread" true
  [[ $(taskset -c "$cpu" sh "$bench/select-cpu.sh") == "$cpu" ]]

  cpu_fixture="$temporary/cpu"
  mkdir -p "$cpu_fixture"
  for cpu_index in 0 1 2 3; do
    printf 'cpu%s 0 0 0 100 0 0 0 0 0 0\n' "$cpu_index"
    mkdir -p "$cpu_fixture/cpu$cpu_index/topology"
  done >"$cpu_fixture/before"
  cat >"$cpu_fixture/after" <<'STAT'
cpu0 0 0 0 120 0 0 0 0 0 0
cpu1 0 0 0 120 0 0 0 0 0 0
cpu2 20 0 0 100 0 0 0 0 0 0
cpu3 0 0 0 120 0 0 0 0 0 0
STAT
  echo '0,2' >"$cpu_fixture/cpu0/topology/thread_siblings_list"
  echo '1,3' >"$cpu_fixture/cpu1/topology/thread_siblings_list"
  select_cpu() {
    awk -v allowed="$1" -v topology="$cpu_fixture" -v seed="${2:-0}" \
      -f "$bench/select-cpu.awk" "$cpu_fixture/before" "$cpu_fixture/after"
  }
  # An idle CPU loses to an idle physical core when its disallowed sibling is busy.
  [[ $(select_cpu 0-1) == 1 ]]
  # Equal idle cores must both be reachable within a restricted affinity mask.
  for seed in {0..19}; do select_cpu 1,3 "$seed"; done | sort -u >"$cpu_fixture/ties"
  printf '1\n3\n' >"$cpu_fixture/expected-ties"
  cmp "$cpu_fixture/expected-ties" "$cpu_fixture/ties"
  # Missing topology falls back to the CPU itself; missing counters count as busy.
  [[ $(select_cpu 3-4) == 3 ]]
  echo 0 >"$cpu_fixture/cpu0/topology/thread_siblings_list"
  echo 1 >"$cpu_fixture/cpu1/topology/thread_siblings_list"
  cat >"$cpu_fixture/after" <<'STAT'
cpu0 0 0 0 100 0 0 0 0 0 0
cpu1 0 0 0 120 0 0 0 0 0 0
STAT
  [[ $(select_cpu 0-1) == 1 ]]
  # Guest time is already accounted for in user/nice; I/O wait also counts as idle.
  cat >"$cpu_fixture/after" <<'STAT'
cpu0 0 0 0 100 20 0 0 0 1000 0
cpu1 2 0 0 118 0 0 0 0 0 0
STAT
  [[ $(select_cpu 0-1) == 0 ]]
fi
bash "$bench/capture-report.sh" --help >/dev/null 2>&1

# Two passes of two cells; the failed row and the case only one target measured are left out.
report="$temporary/report"
header='case,status,comparison_kind,timing_mode,actual_kernel,samples,forks,median_ns,min_ns,max_ns'
mkdir -p "$report/cases/0/pass-1" "$report/cases/0/pass-2" "$report/cases/1/pass-1"
printf '%s\n%s\n%s\n' "$header" 'dot+64+uniform,ok,d,a,k,5,1,10,9,11' 'dot+8192+uniform,ok,d,a,k,5,1,100,99,101' \
  >"$report/cases/0/pass-1/vendor.csv"
printf '%s\n%s\n' "$header" 'dot+64+uniform,ok,d,a,k,5,1,30,29,31' >"$report/cases/0/pass-2/vendor.csv"
printf '%s\n%s\n%s\n' "$header" 'dot+64+uniform,ok,d,a,k,5,1,8,7,9' 'dot+8192+uniform,ok,d,a,k,5,1,150,149,151' \
  >"$report/cases/0/pass-1/jvm.csv"
printf '%s\n%s\n%s\n' "$header" 'dot+64+uniform,failed,d,a,k,5,1,1,1,1' 'gemv+4x4+uniform,ok,d,a,k,5,1,1,1,1' \
  >"$report/cases/1/pass-1/jvm.csv"
bash "$bench/lab/compare.sh" "$report" jvm vendor >"$temporary/compared"
printf '%s\n' 'dot+64+uniform 20.0 0.4000' 'dot+8192+uniform 100.0 1.5000' | cmp - "$temporary/compared"
printf '%s\n' 'dot+64+uniform 40.0 0.6000' >"$temporary/second"
printf '%s\n' 'dot+64+uniform 60.0 0.8000' >"$temporary/third"
bash "$bench/lab/median.sh" "$temporary/compared" "$temporary/second" "$temporary/third" >"$temporary/median"
printf '%s\n' 'dot+64+uniform 40.0 0.6000' 'dot+8192+uniform 100.0 1.5000' | cmp - "$temporary/median"
# One losing row sinks a small cell; a mid cell wins on its geometric mean despite a losing row.
printf '%s\n' 'axpy+64+uniform 1.0 0.5' 'axpy+128+uniform 1.0 1.01' 'axpy+4096+uniform 1.0 1.2' \
  'axpy+8192+uniform 1.0 0.5' >"$temporary/cells"
bash "$bench/lab/score.sh" openblas "$temporary/cells" >"$temporary/scored"
grep -Eq '^axpy +small +lose ' "$temporary/scored"
grep -Eq '^axpy +mid +win ' "$temporary/scored"
# Only Accelerate cells whose reference outruns the vector unit on every row are set aside.
printf '%s\n' 'dot+65536+uniform 655.36 2.0' >"$temporary/fast"
bash "$bench/lab/score.sh" accelerate "$temporary/fast" | grep -Eq '^dot +large +coprocessor '
bash "$bench/lab/score.sh" openblas "$temporary/fast" | grep -Eq '^dot +large +lose '
echo 'Benchmark script checks passed'
