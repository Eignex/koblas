#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
temporary=$(mktemp -d)
trap 'rm -r "$temporary"' EXIT

smoke_output="$temporary/smoke"
"$root/koblas-bench/reference-smoke.sh" --libraries openblas --output "$smoke_output" >/dev/null
result="$smoke_output/openblas.csv"
test "$(grep -c '^case,[0-9]' "$result")" -eq 11
test "$(grep -c '^sample,[0-9]' "$result")" -gt 0
test ! -d "$smoke_output/bin"

# Parser rejection checks use a private test binary; smoke coverage above goes through the production entry point.
cc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror "$root/koblas-bench/reference/vendor_runner.c" -lopenblas -lm -o "$temporary/runner"

reject_case() {
  local name=$1
  local contents=$2
  printf '%s\n' "$contents" >"$temporary/$name.txt"
  if "$temporary/runner" --cases="$temporary/$name.txt" --output="$temporary/bad.csv" --warmups=0 --samples=1 --target-ms=1 2>/dev/null; then
    echo "$name case unexpectedly accepted" >&2
    exit 1
  fi
}

cc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror "$root/koblas-bench/reference/sparse_slices_test.c" -lopenblas -lm -o "$temporary/slices-test"
"$temporary/slices-test"
reject_case slices-missing-timing 'sparse-slices-cycle+64+sparse-uniform+density=0.25'
reject_case slices-wrong-timing 'sparse-slices-gather+64+sparse-uniform+density=0.25+timing=arithmetic'
reject_case slices-compact-clear 'sparse-slices-clear+64+sparse-uniform+density=0.25+timing=reuse+compact=T'
reject_case slices-invalid-locality 'sparse-slices-cycle+64+sparse-uniform+density=0.25+timing=reuse+locality=random'
reject_case slices-legacy-locality 'sparse-slices-gather+64+sparse-uniform+density=0.25+locality=shuffled'
reject_case dense-reuse 'scal+64+uniform+timing=reuse'

reject_case scal-packed-timing 'scal+64+uniform+timing=prepacked-compute'
reject_case gather-invalid-timing 'spgather+64+sparse-uniform+density=0.25+timing=reset-and-arithmetic'
reject_case block-arithmetic-timing 'gemm-block+4x4x4+uniform+packed=4x4+timing=arithmetic'
reject_case duplicate $'dot+4+uniform\ndot+4+uniform'
reject_case empty-field 'dot++4+uniform'
reject_case irrelevant-option 'dot+4+uniform+transA=T'
reject_case wrong-fixture 'gemm+4x4x4+triangular'
reject_case oversized-packed 'gemm-tile+9x4x32+uniform+packed=8x4'
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
for override in leftStride=8 alignment=64 panel=32 batch=2 leftLayout=custom-layout; do
  reject_case override "$packed+$override"
done
reject_case recipe "${packed/packed=4x4/packed=4x4-custom}"
reject_case duplicate-option "$packed+packed=4x4"
reordered=${packed/+packed=4x4/}+packed=4x4
reject_case duplicate-order "$packed"$'\n'"$reordered"
printf '%s\n' "$reordered" >"$temporary/reordered.txt"
"$temporary/runner" --cases="$temporary/reordered.txt" --output="$temporary/reordered.csv" --warmups=0 --samples=1 --target-ms=1
grep -Fq "$packed" "$temporary/reordered.csv"

# Parse and execute the full shared workload, including explicit unsupported vendor cases.
"$temporary/runner" --cases="$root/koblas-bench/cases.txt" --output="$temporary/all.csv" --warmups=0 --samples=1 --target-ms=1

# The local reference is an explicit arm, including operations without a matching oneMKL composition.
"$temporary/runner" --slices-backend=scalar --cases="$root/koblas-bench/sparse-slices-cases.txt" --output="$temporary/slices.csv" --warmups=0 --samples=1 --target-ms=1
test "$(grep -c '^case,[0-9].*,ok,composed,' "$temporary/slices.csv")" -eq 26
grep -q '^run,1,scalar-slices,' "$temporary/slices.csv"
grep -q ',scalar-c-validated-slices$' "$temporary/slices.csv"
"$temporary/runner" --cases="$root/koblas-bench/sparse-slices-cases.txt" --output="$temporary/slices-unsupported.csv" --warmups=0 --samples=1 --target-ms=1
test "$(grep -c '^case,[0-9].*,unsupported,' "$temporary/slices-unsupported.csv")" -eq 26
