#!/usr/bin/env bash
set -euo pipefail
out=/tmp/scal-gather-paired
mkdir -p "$out"
cat > "$out/cases.txt" <<'CASES'
scal+4096+uniform+timing=arithmetic
spgather+4096+sparse-uniform+density=0.25+timing=arithmetic
spgather+65536+sparse-uniform+density=0.25+timing=arithmetic
CASES
mkl=/home/rasmus/.local/share/koblas-onemkl/venv/lib/libmkl_rt.so.3
cc -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror koblas-bench/reference/vendor_runner.c -lopenblas -lm -o "$out/openblas-runner"
cc -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror -DUSE_MKL koblas-bench/reference/vendor_runner.c "$mkl" -Wl,-rpath,"$(dirname "$mkl")" -lpthread -lm -ldl -o "$out/onemkl-runner"
commit=$(git rev-parse HEAD)
printf '%s\n' "$commit" > "$out/commit.txt"
java koblas-bench/tools/CpuSampler.java "$out/cpu.csv" "$out/ready" "$out/stop" > "$out/cpu.log" 2>&1 &
sampler=$!
trap 'touch "$out/stop"; wait "$sampler" || true' EXIT
while [[ ! -f "$out/ready" ]]; do sleep 1; done
for pass in 1 2 3; do
 case $pass in
  1) order='scalar native onemkl openblas';;
  2) order='openblas onemkl native scalar';;
  3) order='native scalar openblas onemkl';;
 esac
 for mode in $order; do
  printf '%s pass=%s mode=%s\n' "$(date -u +%FT%TZ)" "$pass" "$mode" >> "$out/times.txt"
  args=("--cases=$out/cases.txt" "--output=$out/p$pass-$mode.csv" --samples=5 --warmups=3 --target-ms=50 "--pass=$pass" "--source-commit=$commit" --dirty=false)
  case $mode in
   scalar) KOBLAS_SPARSE_INDEXED_NATIVE_CROSSOVER=2147483647 taskset -c 4 koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe --mode=native --operation=all "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
   native) taskset -c 4 koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe --mode=native --operation=all "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
   onemkl) MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 taskset -c 4 "$out/onemkl-runner" "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
   openblas) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 4 "$out/openblas-runner" "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
  esac
 done
done
