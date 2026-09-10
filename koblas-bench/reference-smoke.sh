#!/usr/bin/env bash
set -euo pipefail

bench=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$bench/reference.sh" "$@" \
  --cases "$bench/smoke-cases.txt" \
  --warmups 0 \
  --samples 1 \
  --target-ms 1
