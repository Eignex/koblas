#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
temporary=$(mktemp -d)
trap 'rm -r "$temporary"' EXIT
cc -std=c11 -O2 -Wall -Wextra -Werror "$root/koblas-bench/reference/vendor_runner.c" -lopenblas -lm -o "$temporary/runner"
OPENBLAS_NUM_THREADS=1 "$temporary/runner" --cases="$root/koblas-bench/smoke-cases.txt" --output="$temporary/result.csv" --warmups=0 --samples=1 --target-ms=1
test "$(wc -l <"$temporary/result.csv")" -gt 2
printf 'dot+4+uniform\ndot+4+uniform\n' >"$temporary/duplicate.txt"
if "$temporary/runner" --cases="$temporary/duplicate.txt" --output="$temporary/bad.csv" --warmups=0 --samples=1 --target-ms=1 2>/dev/null; then
  echo "duplicate case unexpectedly accepted" >&2
  exit 1
fi
