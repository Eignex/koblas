#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
temporary=$(mktemp -d)
trap 'rm -r "$temporary"' EXIT

smoke_output="$temporary/smoke"
"$root/koblas-bench/reference-smoke.sh" --libraries openblas --output "$smoke_output" >/dev/null
result="$smoke_output/openblas.csv"
test "$(wc -l <"$result")" -eq 12
test ! -d "$smoke_output/bin"

# Parser rejection checks use a private test binary; smoke coverage above goes through the production entry point.
cc -std=c11 -O2 -Wall -Wextra -Werror "$root/koblas-bench/reference/vendor_runner.c" -lopenblas -lm -o "$temporary/runner"

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

sed 's/,arithmetic,/,incompatible,/' "$result" >"$temporary/incompatible.csv"
if "$root/koblas-bench/tools/compare.sh" --require-compatible --mode logical "$result" "$temporary/incompatible.csv" >"$temporary/compare-output.txt" 2>"$temporary/compare-error.txt"; then
  echo "incompatible supported cases unexpectedly joined" >&2
  exit 1
fi
grep -q '^incompatible case=' "$temporary/compare-error.txt"

"$root/koblas-bench/tools/compare_test.sh"

# Remove each required field independently from a valid canonical case.
packed=$(awk '/^gemm-block/ { print; exit }' "$root/koblas-bench/cases.txt")
IFS=+ read -r -a fields <<<"$packed"
for field in "${fields[@]:3}"; do
  reject_case missing-field "${packed/+${field}/}"
done
reject_case stride "${packed/leftStride=4/leftStride=8}"
reject_case alignment "${packed/alignment=8/alignment=64}"
reject_case panel "${packed/panel=31/panel=32}"
reject_case batch "${packed/batch=1/batch=2}"
reject_case layout "${packed/leftLayout=depth-rows-v1/leftLayout=depth-rows-v2}"

# Parse and execute the full shared workload, including explicit unsupported vendor cases.
"$temporary/runner" --cases="$root/koblas-bench/cases.txt" --output="$temporary/all.csv" --warmups=0 --samples=1 --target-ms=1
