#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 --platform <platform> --output <directory> --blas <library>" >&2; exit 2; }
platform=""; output=""; blas=""
while (($#)); do
    case "$1" in
        --platform) platform="${2:-}"; shift 2 ;;
        --output) output="${2:-}"; shift 2 ;;
        --blas) blas="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$platform" && -n "$output" && -n "$blas" ]] || usage

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/koblas-superlu/superlu.lock"
version="$(sed -n 's/^version=//p' "$lock")"
url="$(sed -n 's/^url=//p' "$lock")"
expected="$(sed -n 's/^sha256=//p' "$lock")"
cache="$output/../downloads/superlu-$version.tar.gz"
mkdir -p "$(dirname "$cache")"
if [[ ! -f "$cache" ]]; then curl --fail --location --silent --show-error "$url" -o "$cache"; fi
actual="$(sha256sum "$cache" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || { echo "SuperLU source checksum mismatch" >&2; exit 1; }

case "$platform" in
    linux-x86_64) [[ "$(uname -s)" == Linux ]] || { echo "linux-x86_64 requires Linux" >&2; exit 1; } ;;
    linux-arm64) [[ "$(uname -s)" == Linux ]] || { echo "linux-arm64 requires Linux" >&2; exit 1; } ;;
    macosx-arm64) [[ "$(uname -s)" == Darwin ]] || { echo "macosx-arm64 requires macOS" >&2; exit 1; } ;;
    *) echo "unsupported SuperLU platform $platform" >&2; exit 1 ;;
esac

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tar -xzf "$cache" -C "$work"
source="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
install="$work/install"
cmake -S "$source" -B "$work/build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$install" \
    -DTPL_BLAS_LIBRARIES="$blas" -Denable_internal_blaslib=NO -Denable_single=NO -Denable_complex=NO \
    -Denable_complex16=NO -Denable_examples=NO -Denable_tests=NO -Denable_fortran=NO \
    -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_C_COMPILER="${CC:-cc}" >/dev/null
cmake --build "$work/build" --parallel "${SUPERLU_JOBS:-2}" >/dev/null
cc -shared -fPIC "$root/koblas-superlu/native/koblas_superlu.c" \
    -I"$source/SRC" -I"$work/build/SRC" -L"$work/build/SRC" -lsuperlu -Wl,-rpath,'$ORIGIN' \
    -o "$work/libkoblas-superlu.$([[ "$platform" == linux-* ]] && echo so || echo dylib)"
cmake --install "$work/build" >/dev/null

destination="$output/org/eignex/superlu/$platform"
rm -rf "$destination"; mkdir -p "$destination"
find "$install/lib" -maxdepth 1 -type f \( -name '*.so*' -o -name '*.dylib' \) -exec cp {} "$destination" \;
cp "$work/libkoblas-superlu.$([[ "$platform" == linux-* ]] && echo so || echo dylib)" "$destination/"
if [[ "$platform" == linux-* ]]; then
    while IFS= read -r library; do
        soname="$(readelf -d "$library" | sed -n '/SONAME/s/.*\[\(.*\)\]/\1/p')"
        [[ -z "$soname" || "$(basename "$library")" == "$soname" ]] || cp "$library" "$destination/$soname"
    done < <(find "$destination" -maxdepth 1 -type f -name '*.so.*')
else
    for library in "$destination"/*.dylib; do cp "$library" "$destination/$(basename "$library" | sed 's/\.[0-9].*\.dylib$/.dylib/')"; done
fi
find "$destination" -maxdepth 1 -type f -printf '%f\n' | sort > "$destination/.libraries"
cp "$source/License.txt" "$destination/LICENSE.superlu-$version.txt"
printf '%s\n' "$expected" > "$destination/.superlu-source-sha256"
