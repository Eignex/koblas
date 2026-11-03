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

fingerprint_paths() {
  local kind=$1 source_kind=$2 path directory
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
  } | LC_ALL=C sort
}

fingerprint() (
  set -euo pipefail
  local tree=$1 kind=$2 source_kind=${3:-archive} path temporary=
  cd "$tree"
  # Buffer a helper result so a missing runtime or failed read cannot publish a partial fingerprint.
  if [[ ( ${OSTYPE:-} == darwin* || ${OSTYPE:-} == linux* ) &&
        -n ${LAB_WORKSPACE_CACHE:-} && -n ${KOBLAS_BENCH_CLASSPATH:-} && -x ${KOBLAS_BENCH_JAVA:-} &&
        -f koblas-bench/build/lab/classes/com/eignex/koblas/bench/PreparedFingerprints.class ]]; then
    if temporary=$(mktemp -d "${TMPDIR:-/tmp}/koblas-fingerprint.XXXXXX"); then
      trap 'rm -rf "$temporary"' EXIT
      if fingerprint_paths "$kind" "$source_kind" >"$temporary/inventory" &&
          "$KOBLAS_BENCH_JAVA" -Xms16m -Xmx64m -cp "$KOBLAS_BENCH_CLASSPATH" \
            com.eignex.koblas.bench.PreparedFingerprints "$PWD" \
            <"$temporary/inventory" >"$temporary/result" 2>/dev/null; then
        cat "$temporary/result"
        return
      fi
    fi
  fi
  fingerprint_paths "$kind" "$source_kind" | while IFS= read -r path; do
    [[ -f $path ]] || continue
    printf '%s %s\n' "$(sha256 "$path")" "$path"
  done
)
