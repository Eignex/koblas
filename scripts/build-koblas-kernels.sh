#!/usr/bin/env bash
set -euo pipefail

platform=""
output=""
while (($#)); do
  case "$1" in
    --platform) platform="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$platform" && -n "$output" ]] || { echo "--platform and --output are required" >&2; exit 2; }

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../koblas/src/nativeInterop/cinterop" && pwd)"
resource_dir="$output/com/eignex/koblas/internal/kernels/$platform"
mkdir -p "$resource_dir"

case "$platform" in
  linux-x86_64|linux-arm64)
    library="$resource_dir/libkoblas_kernels.so"
    "${CC:-cc}" -std=c11 -O3 -fPIC -fvisibility=hidden -shared \
      "$source_dir/koblas_kernels.c" -Wl,-soname,libkoblas_kernels.so -lm -o "$library"
    ;;
  macosx-arm64)
    library="$resource_dir/libkoblas_kernels.dylib"
    "${CC:-cc}" -std=c11 -O3 -fPIC -fvisibility=hidden -dynamiclib \
      "$source_dir/koblas_kernels.c" -Wl,-install_name,@rpath/libkoblas_kernels.dylib -o "$library"
    ;;
  *) echo "unsupported platform: $platform" >&2; exit 2 ;;
esac
