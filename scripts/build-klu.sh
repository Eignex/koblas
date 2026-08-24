#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 --platform <platform> --output <directory>" >&2; exit 2; }
platform=""; output=""
while (($#)); do
    case "$1" in
        --platform) platform="${2:-}"; shift 2 ;;
        --output) output="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$platform" && -n "$output" ]] || usage

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/third-party-notices.sh"
lock="$root/koblas-klu/klu.lock"
version="$(sed -n 's/^version=//p' "$lock")"
url="$(sed -n 's/^url=//p' "$lock")"
expected="$(sed -n 's/^sha256=//p' "$lock")"
cache="$output/../downloads/suitesparse-$version.tar.gz"
mkdir -p "$(dirname "$cache")"
if [[ ! -f "$cache" ]]; then curl --fail --location --silent --show-error "$url" -o "$cache"; fi
actual="$(sha256sum "$cache" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || { echo "SuiteSparse source checksum mismatch" >&2; exit 1; }

case "$platform" in
    linux-x86_64) [[ "$(uname -s)" == Linux ]] || { echo "linux-x86_64 requires Linux" >&2; exit 1; } ;;
    linux-arm64) [[ "$(uname -s)" == Linux ]] || { echo "linux-arm64 requires Linux" >&2; exit 1; } ;;
    macosx-arm64) [[ "$(uname -s)" == Darwin ]] || { echo "macosx-arm64 requires macOS" >&2; exit 1; } ;;
    *) echo "unsupported SuiteSparse KLU platform $platform" >&2; exit 1 ;;
esac

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tar -xzf "$cache" -C "$work"
source="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
install="$work/install"
cmake -S "$source" -B "$work/build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$install" \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;btf;klu" \
    -DBLA_VENDOR=Generic -DKLU_USE_CHOLMOD=OFF -DSUITESPARSE_REQUIRE_BLAS=OFF -DSUITESPARSE_USE_OPENMP=OFF \
    -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_C_COMPILER="${CC:-cc}" >/dev/null
cmake --build "$work/build" --parallel "${KLU_JOBS:-2}" >/dev/null
cmake --install "$work/build" >/dev/null

destination="$output/org/eignex/klu/$platform"
notices="$output/THIRD-PARTY-NOTICES.txt"
rm -rf "$destination"; mkdir -p "$destination"
find "$install/lib" -maxdepth 1 -type f \( -name '*.so*' -o -name '*.dylib' \) -exec cp {} "$destination" \;
if [[ "$platform" == linux-* ]]; then
    while IFS= read -r library; do
        soname="$(readelf -d "$library" | sed -n '/SONAME/s/.*\[\(.*\)\]/\1/p')"
        [[ -z "$soname" || "$(basename "$library")" == "$soname" ]] || cp "$library" "$destination/$soname"
    done < <(find "$destination" -maxdepth 1 -type f -name '*.so.*')
else
    for library in "$destination"/*.dylib; do cp "$library" "$destination/$(basename "$library" | sed 's/\.[0-9].*\.dylib$/.dylib/')"; done
fi
find "$destination" -maxdepth 1 -type f -exec basename {} \; | sort > "$destination/.libraries"
notices_init "$notices" "koblas-klu" "scripts/build-klu.sh"
notices_append_file "$notices" "SuiteSparse KLU $version — LGPL-2.1-or-later" "SuiteSparse/KLU/Doc/License.txt" "$source/KLU/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse KLU LGPL-2.1 license text" "SuiteSparse/KLU/Doc/lesser.txt" "$source/KLU/Doc/lesser.txt"
notices_append_file "$notices" "SuiteSparse AMD — BSD-3-Clause" "SuiteSparse/AMD/Doc/License.txt" "$source/AMD/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse BTF — LGPL-2.1-or-later" "SuiteSparse/BTF/Doc/License.txt" "$source/BTF/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse COLAMD — BSD-3-Clause" "SuiteSparse/COLAMD/Doc/License.txt" "$source/COLAMD/Doc/License.txt"
notices_append_suite_sparse_config "$notices" "$source" "$work"
printf '%s\n' "$expected" > "$destination/.suitesparse-source-sha256"
