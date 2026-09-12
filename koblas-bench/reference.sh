#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench="$root/koblas-bench"
libraries=all
output=
cases="$bench/cases.txt"
samples=5
warmups=3
target_ms=1000
pass=1

usage() {
  echo "usage: koblas-bench/reference.sh [--libraries openblas,onemkl,scalar-slices|all] [--output DIR] [--cases FILE] [--samples N] [--warmups N] [--target-ms N] [--pass LABEL]" >&2
}

while (($#)); do
  case "$1" in
    --libraries) libraries=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --cases) cases=${2:?}; shift 2 ;;
    --samples) samples=${2:?}; shift 2 ;;
    --warmups) warmups=${2:?}; shift 2 ;;
    --target-ms) target_ms=${2:?}; shift 2 ;;
    --pass) pass=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ $libraries == all ]]; then libraries=openblas,onemkl; fi
IFS=, read -r -a requested <<<"$libraries"
for library in "${requested[@]}"; do
  [[ $library == openblas || $library == onemkl || $library == scalar-slices ]] || { echo "unknown library: $library" >&2; exit 2; }
done

if [[ -z $output ]]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  output="$bench/build/reference/$stamp-$$"
fi
[[ ! -e $output ]] || { echo "output directory already exists: $output" >&2; exit 2; }
mkdir -p "$output/bin"
commit=$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)
if [[ -z $(git -C "$root" status --porcelain --untracked-files=normal 2>/dev/null) ]]; then dirty=false; else dirty=true; fi

run_openblas() {
  cc -std=c11 -O3 -ffp-contract=off -DNDEBUG -Wall -Wextra -Werror "$bench/reference/vendor_runner.c" -lopenblas -lm -o "$output/bin/openblas-runner"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$output/bin/openblas-runner" \
    --cases="$cases" --output="$output/openblas.csv" --samples="$samples" \
    --warmups="$warmups" --target-ms="$target_ms" --pass="$pass" --source-commit="$commit" --dirty="$dirty"
}

run_scalar_slices() {
  cc -std=c11 -O3 -ffp-contract=off -DNDEBUG -Wall -Wextra -Werror "$bench/reference/vendor_runner.c" -lopenblas -lm -o "$output/bin/scalar-slices-runner"
  cc -std=c11 -O3 -ffp-contract=off -DNDEBUG -Wall -Wextra -Werror "$bench/reference/sparse_slices_test.c" -lopenblas -lm -o "$output/bin/scalar-slices-test"
  "$output/bin/scalar-slices-test"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$output/bin/scalar-slices-runner" --slices-backend=scalar \
    --cases="$cases" --output="$output/scalar-slices.csv" --samples="$samples" \
    --warmups="$warmups" --target-ms="$target_ms" --pass="$pass" --source-commit="$commit" --dirty="$dirty"
}

run_onemkl() {
  local library=${ONEMKL_LIBRARY:-/home/rasmus/.local/share/koblas-onemkl/venv/lib/libmkl_rt.so.3}
  [[ -f $library ]] || { echo "requested oneMKL library is missing: $library" >&2; exit 2; }
  local directory
  directory=$(dirname "$library")
  cc -std=c11 -O3 -ffp-contract=off -DNDEBUG -Wall -Wextra -Werror -DUSE_MKL "$bench/reference/vendor_runner.c" \
    "$library" -Wl,-rpath,"$directory" -lpthread -lm -ldl -o "$output/bin/onemkl-runner"
  cc -std=c11 -O3 -ffp-contract=off -DNDEBUG -Wall -Wextra -Werror -DUSE_MKL "$bench/reference/sparse_slices_test.c" \
    "$library" -Wl,-rpath,"$directory" -lpthread -lm -ldl -o "$output/bin/slices-test"
  MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 "$output/bin/slices-test"
  LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 \
    "$output/bin/onemkl-runner" --cases="$cases" --output="$output/onemkl.csv" \
    --samples="$samples" --warmups="$warmups" --target-ms="$target_ms" --pass="$pass" --source-commit="$commit" --dirty="$dirty"
}

for library in "${requested[@]}"; do
  case "$library" in
    openblas) run_openblas ;;
    onemkl) run_onemkl ;;
    scalar-slices) run_scalar_slices ;;
  esac
done

rm -r "$output/bin"
echo "$output"
