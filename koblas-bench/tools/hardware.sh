#!/usr/bin/env bash
set -euo pipefail

echo "schema=1"
echo "architecture=$(uname -m)"

if [[ $(uname -s) == Linux ]]; then
  LC_ALL=C lscpu | awk -F: '
    /^[[:space:]]*(Vendor ID|Model name|CPU family|Model|Stepping|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|.*[Cc]ache|NUMA node\(s\)):/ {
      key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      value = $2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      print tolower(key) "=" value
    }
  '
  awk '/^MemTotal:/ { print "memory_kib=" $2 }' /proc/meminfo
elif [[ $(uname -s) == Darwin ]]; then
  sysctl -n machdep.cpu.brand_string 2>/dev/null | sed 's/^/cpu_model=/' || true
  sysctl -n hw.physicalcpu 2>/dev/null | sed 's/^/physical_cpus=/' || true
  sysctl -n hw.physicalcpu_max 2>/dev/null | sed 's/^/physical_cpus_max=/' || true
  sysctl -n hw.memsize 2>/dev/null | sed 's/^/memory_bytes=/' || true
else
  echo "platform=$(uname -s)"
fi
