#!/usr/bin/env bash
set -euo pipefail

platform=
output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) platform="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$platform" && -n "$output" ]] || {
  echo "usage: $0 --platform <platform> --output <directory>" >&2
  exit 2
}

case "$platform" in
  linux-x86_64|linux-arm64|macosx-arm64) ;;
  *) echo "unsupported OpenBLAS platform: $platform" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
lock_file="$project_dir/koblas-openblas/openblas.lock"
version=$(sed -n 's/^version=//p' "$lock_file")
url=$(sed -n 's/^url=//p' "$lock_file")
expected_sha=$(sed -n 's/^sha256=//p' "$lock_file")
c_compiler="${CC:-cc}"
fortran_compiler="${FC:-gfortran}"
cache_dir="${OPENBLAS_DOWNLOAD_CACHE:-$project_dir/koblas-openblas/build/openblas/downloads}"
archive="$cache_dir/openblas-$version.tar.gz"
mkdir -p "$cache_dir"

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --silent --show-error "$url" --output "$archive"
fi
actual_sha=$(sha256sum "$archive" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "OpenBLAS archive checksum mismatch" >&2
  exit 1
}

case "$platform" in
  linux-x86_64)
    [[ "$(uname -s)" == Linux && "$(uname -m)" =~ ^(x86_64|amd64)$ ]] || {
      echo "$platform must be built on Linux x86_64" >&2; exit 1
    }
    library_name=libopenblas.so.0
    ;;
  linux-arm64)
    [[ "$(uname -s)" == Linux ]] || { echo "$platform must be built on Linux" >&2; exit 1; }
    if [[ ! "$(uname -m)" =~ ^(aarch64|arm64)$ ]] &&
      ! "$c_compiler" -dumpmachine | grep -Eq '^aarch64.*-linux'; then
      echo "$platform requires Linux ARM64 or an aarch64 Linux cross-compiler" >&2
      exit 1
    fi
    library_name=libopenblas.so.0
    ;;
  macosx-arm64)
    [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
      echo "$platform must be built on macOS ARM64" >&2; exit 1
    }
    library_name=libopenblas.0.dylib
    ;;
esac

resource_dir="$output/org/bytedeco/openblas/$platform"
license_name="LICENSE.openblas-$version.txt"
case "$platform" in
  linux-*) required_files=("$library_name" libgfortran.so.5 libquadmath.so.0 libgcc_s.so.1 "$license_name") ;;
  macosx-arm64) required_files=("$library_name" libgfortran.dylib libgfortran.5.dylib libquadmath.0.dylib libgcc_s.1.1.dylib "$license_name") ;;
esac
if [[ -f "$resource_dir/.openblas-source-sha256" ]] &&
  [[ "$(<"$resource_dir/.openblas-source-sha256")" == "$expected_sha" ]]; then
  complete=true
  for file in "${required_files[@]}"; do
    [[ -s "$resource_dir/$file" ]] || { complete=false; break; }
  done
  "$complete" && exit 0
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/koblas-openblas.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
tar -xzf "$archive" -C "$work_dir"
source_dir="$work_dir/OpenBLAS-$version"
if [[ "$platform" == linux-* ]]; then
  # The extracted Fortran runtimes live beside libopenblas, so the dynamic loader must search that directory
  # before falling back to a host installation.
  printf '%s\n' "EXTRALIB += -Wl,-rpath,'\$\$ORIGIN'" >> "$source_dir/Makefile.system"
fi
build_args=(shared NO_AFFINITY=1 USE_OPENMP=0 LAPACKE=1 CFLAGS=-w FFLAGS=-w "CC=$c_compiler" "FC=$fortran_compiler" "HOSTCC=${HOSTCC:-cc}")
if [[ "$platform" == linux-arm64 ]] && ! [[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  build_args+=(TARGET=ARMV8 DYNAMIC_ARCH=0)
else
  build_args+=(DYNAMIC_ARCH=1)
fi
make -s -C "$source_dir" -j"${OPENBLAS_JOBS:-2}" "${build_args[@]}"

rm -rf "$resource_dir"
mkdir -p "$resource_dir"
library=$(find "$source_dir" -type f \( -name 'libopenblas*.so' -o -name 'libopenblas*.dylib' \) -print -quit)
[[ -n "$library" ]] || { echo "OpenBLAS build did not produce a shared library" >&2; exit 1; }
cp "$library" "$resource_dir/$library_name"
cp "$source_dir/LICENSE" "$resource_dir/LICENSE.openblas-$version.txt"

if [[ "$platform" == linux-* ]]; then
  for runtime in \
    "$("$fortran_compiler" -print-file-name=libgfortran.so.5)" \
    "$("$fortran_compiler" -print-file-name=libquadmath.so.0)" \
    "$("$c_compiler" -print-file-name=libgcc_s.so.1)"; do
    [[ -f "$runtime" ]] || { echo "missing Linux runtime $runtime" >&2; exit 1; }
    cp "$runtime" "$resource_dir/$(basename "$runtime")"
  done
else
  command -v install_name_tool >/dev/null || {
    echo "install_name_tool is required to bundle macOS Fortran runtimes" >&2
    exit 1
  }
  while read -r dependency; do
    [[ -f "$dependency" ]] && cp "$dependency" "$resource_dir/$(basename "$dependency")"
  done < <(otool -L "$resource_dir/$library_name" | awk '/\/.*\/(libgfortran|libquadmath|libgcc_s).*\.dylib/ { print $1 }')
  for binary in "$resource_dir"/*.dylib; do
    while read -r dependency; do
      [[ -f "$dependency" ]] && cp -n "$dependency" "$resource_dir/$(basename "$dependency")"
      install_name_tool -change "$dependency" "@loader_path/$(basename "$dependency")" "$binary"
    done < <(otool -L "$binary" | awk 'NR > 1 && /\/.*\/(libgfortran|libquadmath|libgcc_s).*\.dylib/ { print $1 }')
  done
  [[ -f "$resource_dir/libgfortran.5.dylib" ]] &&
    cp -n "$resource_dir/libgfortran.5.dylib" "$resource_dir/libgfortran.dylib"
fi

for file in "${required_files[@]}"; do
  [[ -s "$resource_dir/$file" ]] || { echo "missing bundled runtime $file" >&2; exit 1; }
done
printf '%s\n' "$expected_sha" > "$resource_dir/.openblas-source-sha256"
