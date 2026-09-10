#!/usr/bin/env bash
set -euo pipefail

echo "collected_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "uname=$(uname -a)"
compiler_version=$(cc --version)
echo "compiler=${compiler_version%%$'\n'*}"
if command -v lscpu >/dev/null 2>&1; then
  LC_ALL=C lscpu
elif command -v sysctl >/dev/null 2>&1; then
  sysctl -a 2>/dev/null | grep -E 'machdep.cpu|hw.(model|machine|ncpu|memsize)' || true
fi
