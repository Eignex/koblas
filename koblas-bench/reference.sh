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
  echo "usage: koblas-bench/reference.sh [--libraries openblas,accelerate,onemkl|all] [--output DIR] [--cases FILE] [--samples N] [--warmups N] [--target-ms N] [--pass LABEL]" >&2
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

if [[ $libraries == all ]]; then
  if [[ $(uname) == Darwin ]]; then libraries=openblas,accelerate; else libraries=openblas,onemkl; fi
fi
IFS=, read -r -a requested <<<"$libraries"
for library in "${requested[@]}"; do
  [[ $library == openblas || $library == accelerate || $library == onemkl ]] || { echo "unknown library: $library" >&2; exit 2; }
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
  local -a flags=()
  if [[ $(uname) == Darwin ]] && command -v brew >/dev/null 2>&1; then
    local prefix
    prefix=$(brew --prefix openblas 2>/dev/null || true)
    if [[ -f $prefix/include/cblas.h ]]; then
      flags=("-I$prefix/include" "-L$prefix/lib" "-Wl,-rpath,$prefix/lib")
    fi
  fi
  cc -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror "${flags[@]}" "$bench/reference/vendor_runner.c" -lopenblas -lm -o "$output/bin/openblas-runner"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$output/bin/openblas-runner" \
    --cases="$cases" --output="$output/openblas.csv" --samples="$samples" \
    --warmups="$warmups" --target-ms="$target_ms" --pass="$pass" --source-commit="$commit" --dirty="$dirty"
}

run_accelerate() {
  [[ $(uname) == Darwin ]] || { echo "Accelerate is available only on macOS" >&2; exit 2; }
  cc -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK \
    "$bench/reference/vendor_runner.c" -framework Accelerate -lm -o "$output/bin/accelerate-runner"
  VECLIB_MAXIMUM_THREADS=1 "$output/bin/accelerate-runner" --cases="$cases" --output="$output/accelerate.csv" --samples="$samples" \
    --warmups="$warmups" --target-ms="$target_ms" --pass="$pass" --source-commit="$commit" --dirty="$dirty"
}

run_onemkl() {
  local library=${ONEMKL_LIBRARY:-/home/rasmus/.local/share/koblas-onemkl/venv/lib/libmkl_rt.so.3}
  [[ -f $library ]] || { echo "requested oneMKL library is missing: $library" >&2; exit 2; }
  local directory
  directory=$(dirname "$library")
  cc -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror -DUSE_MKL "$bench/reference/vendor_runner.c" \
    "$library" -Wl,-rpath,"$directory" -lpthread -lm -ldl -o "$output/bin/onemkl-runner"
  LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 \
    "$output/bin/onemkl-runner" --cases="$cases" --output="$output/onemkl.csv" \
    --samples="$samples" --warmups="$warmups" --target-ms="$target_ms" --pass="$pass" --source-commit="$commit" --dirty="$dirty"
}

for library in "${requested[@]}"; do
  case "$library" in
    openblas) run_openblas ;;
    accelerate) run_accelerate ;;
    onemkl) run_onemkl ;;
  esac
done

rm -r "$output/bin"
echo "$output"
