#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
temporary=$(mktemp -d)
trap 'rm -r "$temporary"' EXIT
cc -std=c11 -O2 -Wall -Wextra -Werror "$root/koblas-bench/reference/vendor_runner.c" -lopenblas -lm -o "$temporary/runner"
OPENBLAS_NUM_THREADS=1 "$temporary/runner" --cases="$root/koblas-bench/smoke-cases.txt" --output="$temporary/result.csv" --warmups=0 --samples=1 --target-ms=1
test "$(wc -l <"$temporary/result.csv")" -gt 2

reject_case() {
  local name=$1
  local contents=$2
  printf '%s\n' "$contents" >"$temporary/$name.txt"
  if "$temporary/runner" --cases="$temporary/$name.txt" --output="$temporary/bad.csv" --warmups=0 --samples=1 --target-ms=1 2>/dev/null; then
    echo "$name case unexpectedly accepted" >&2
    exit 1
  fi
}

reject_case duplicate $'dot+4+uniform\ndot+4+uniform'
reject_case empty-field 'dot++4+uniform'
reject_case irrelevant-option 'dot+4+uniform+transA=T'
reject_case wrong-fixture 'gemm+4x4x4+triangular'
reject_case oversized-packed 'gemm-tile+9x4x32+uniform+physical=8x4'
reject_case unsupported-mode 'spsymv+8x8+sparse-uniform+density=0.25+mode=prepared+uplo=L'
reject_case long-operation 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa+4+uniform'

sed 's/,arithmetic,/,incompatible,/' "$temporary/result.csv" >"$temporary/incompatible.csv"
if python3 "$root/koblas-bench/tools/compare.py" --require-compatible "$temporary/result.csv" "$temporary/incompatible.csv" >"$temporary/compare-output.txt" 2>"$temporary/compare-error.txt"; then
  echo "incompatible supported cases unexpectedly joined" >&2
  exit 1
fi
grep -q '^incompatible case=' "$temporary/compare-error.txt"
