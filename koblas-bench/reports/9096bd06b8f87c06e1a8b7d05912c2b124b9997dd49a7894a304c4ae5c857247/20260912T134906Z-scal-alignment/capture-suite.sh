#!/usr/bin/env bash
set -euo pipefail
out=/tmp/scal-followup/suite
mkdir -p "$out"
cat > "$out/cases.txt" <<'CASES'
scal+64+uniform+timing=arithmetic
scal+256+uniform+timing=arithmetic
scal+4096+uniform+timing=arithmetic
scal+65536+uniform+timing=arithmetic
scal+4096+uniform
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
  1) order='before native onemkl openblas';;
  2) order='openblas onemkl native before';;
  3) order='native before openblas onemkl';;
 esac
 for mode in $order; do
  printf '%s pass=%s mode=%s\n' "$(date -u +%FT%TZ)" "$pass" "$mode" >> "$out/times.txt"
  args=("--cases=$out/cases.txt" "--output=$out/p$pass-$mode.csv" --samples=5 --warmups=5 --target-ms=200 "--pass=$pass" "--source-commit=$commit" --dirty=false)
  case $mode in
   before) taskset -c 4 /tmp/scal-alignment/before.kexe --mode=native --operation=all "${args[@]/--source-commit=$commit/--source-commit=5e5a7f1f1de41b1824299822f66314b6af1574ef}" > "$out/p$pass-$mode.log" 2>&1;;
   native) taskset -c 4 koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe --mode=native --operation=all "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
   onemkl) MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 taskset -c 4 "$out/onemkl-runner" "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
   openblas) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 4 "$out/openblas-runner" "${args[@]}" > "$out/p$pass-$mode.log" 2>&1;;
  esac
 done
done
