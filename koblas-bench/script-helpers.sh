#!/usr/bin/env bash

# Native processes on Windows need drive paths rather than Git Bash's /c/... mount paths.
native_path() {
  if [[ ${platform:-} == Windows ]]; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'
}

first_allowed_cpu() {
  awk '/^Cpus_allowed_list:/ {split($2, cpus, /[,-]/); print cpus[1]; found=1; exit} END {if (!found) exit 1}' /proc/self/status
}

fingerprint() (
  set -euo pipefail
  local tree=$1 kind=$2 source_kind=${3:-archive} path directory
  cd "$tree"
  {
    if [[ $kind == sources && $source_kind == git ]]; then
      git ls-files -z | while IFS= read -r -d '' path; do printf '%s\n' "$path"; done
    elif [[ $kind == sources ]]; then
      find . -type d \( -name .git -o -name .gradle -o -name .kotlin -o -name .idea -o -name build -o -name node_modules \) \
        -prune -o \( -type f -o -type l \) -print | sed 's|^./||'
    else
      for directory in koblas/build/libs koblas-bench/build/classes koblas-bench/build/bin koblas-bench/jmh/build/libs; do
        if [[ -d $directory ]]; then find "$directory" \( -type f -o -type l \); fi
      done
    fi
  } | LC_ALL=C sort | while IFS= read -r path; do
    [[ -f $path ]] || continue
    printf '%s %s\n' "$(sha256 "$path")" "$path"
  done
)
