#!/usr/bin/env bash
set -euo pipefail
report=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(git rev-parse --show-toplevel)
output=${1:?usage: reproduce-diagnostics.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 2; }
mkdir -p "$output"
output=$(cd "$output" && pwd)
compiler=${CC:-/home/rasmus/.konan/dependencies/llvm-21-x86_64-linux-essentials-116/bin/clang}
core=${BENCH_CPU:-4}
baseline=ef35f7ca6871efddc43194c55b0d92b63c42af88
candidate=10128c807d7b21990a9a043b34709d4c8f3d839e
cleanup() {
  for arm in before after; do
    if [[ -d "$output/worktree-$arm" ]]; then
      git -C "$root" worktree remove --force "$output/worktree-$arm"
    fi
  done
}
trap cleanup EXIT
for arm in before after; do
  revision=$baseline
  if [[ "$arm" == after ]]; then revision=$candidate; fi
  tree="$output/worktree-$arm"
  git -C "$root" worktree add --detach "$tree" "$revision"
  source_dir="$tree/koblas-bench/src/linuxX64Main/kotlin/com/eignex/koblas/bench"
  mkdir -p "$source_dir"
  cp "$report/ScalDiagnostic.kt" "$source_dir/ScalDiagnostic.kt"
  python3 - "$tree/koblas-bench/build.gradle.kts" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
path.write_text(path.read_text().replace('entryPoint = "com.eignex.koblas.bench.main"',
                                       'entryPoint = "com.eignex.koblas.bench.scalDiagnostic"'))
PY
  (cd "$tree" && ./gradlew :koblas-bench:linkReleaseExecutableLinuxX64 > "$output/build-$arm.log" 2>&1)
  cp "$tree/koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe" "$output/native-$arm.kexe"
done
for probe in probe blocks; do
  "$compiler" -std=c11 -O3 -DNDEBUG -I "$output/worktree-before" "$report/$probe.c" -lm -ldl -o "$output/$probe"
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 \
    taskset -c "$core" "$output/$probe" > "$output/$probe.csv"
done
for pass in 1 2 3; do
  arms=(before after)
  if [[ "$pass" == 2 ]]; then arms=(after before); fi
  for arm in "${arms[@]}"; do
    taskset -c "$core" "$output/native-$arm.kexe" > "$output/native-$arm-$pass.csv"
  done
done
python3 "$report/summarize-results.py" "$output" > "$output/tables.md"
