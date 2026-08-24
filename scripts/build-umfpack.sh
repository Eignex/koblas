#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 --platform <platform> --output <directory>" >&2; exit 2; }
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
source "$root/scripts/third-party-notices.sh"
lock="$root/koblas-umfpack/umfpack.lock"
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
    *) echo "unsupported SuiteSparse platform $platform" >&2; exit 1 ;;
esac

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tar -xzf "$cache" -C "$work"
source="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
openblas_lock="$root/koblas-openblas/openblas.lock"
openblas_version="$(sed -n 's/^version=//p' "$openblas_lock")"
openblas_url="$(sed -n 's/^url=//p' "$openblas_lock")"
openblas_sha256="$(sed -n 's/^sha256=//p' "$openblas_lock")"
openblas_cache="${OPENBLAS_DOWNLOAD_CACHE:-$root/koblas-openblas/build/openblas/downloads}"
openblas_archive="$openblas_cache/openblas-$openblas_version.tar.gz"
mkdir -p "$openblas_cache"
if [[ ! -f "$openblas_archive" ]]; then
    curl --fail --location --retry 3 --silent --show-error "$openblas_url" -o "$openblas_archive"
fi
openblas_actual_sha256="$(sha256sum "$openblas_archive" | awk '{print $1}')"
[[ "$openblas_actual_sha256" == "$openblas_sha256" ]] || {
    echo "OpenBLAS archive checksum mismatch" >&2
    exit 1
}
openblas_work="$work/openblas"
mkdir -p "$openblas_work"
tar -xzf "$openblas_archive" -C "$openblas_work"
openblas_source="$openblas_work/OpenBLAS-$openblas_version"
install="$work/install"
if [[ "$platform" == macosx-arm64 ]]; then
    export DYLD_LIBRARY_PATH="$(dirname "$blas")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
cmake -S "$source" -B "$work/build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$install" \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;umfpack" \
    -DUMFPACK_USE_CHOLMOD=OFF \
    -DBLAS_LIBRARIES="$blas" \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_C_COMPILER="${CC:-cc}" >/dev/null
cmake --build "$work/build" --parallel "${SUITESPARSE_JOBS:-2}" >/dev/null
cmake --install "$work/build" >/dev/null

destination="$output/org/eignex/umfpack/$platform"
notices="$output/THIRD-PARTY-NOTICES.txt"
rm -rf "$destination"; mkdir -p "$destination"
find "$install/lib" -maxdepth 1 -type f \( -name '*.so*' -o -name '*.dylib' \) -exec cp {} "$destination" \;
if [[ "$platform" == linux-* ]]; then
    while IFS= read -r library; do
        soname="$(readelf -d "$library" | sed -n '/SONAME/s/.*\[\(.*\)\]/\1/p')"
        [[ -z "$soname" || "$(basename "$library")" == "$soname" ]] || cp "$library" "$destination/$soname"
    done < <(find "$destination" -maxdepth 1 -type f -name '*.so.*')
else
    for library in "$destination"/libumfpack.*.dylib; do cp "$library" "$destination/libumfpack.dylib"; done
fi
find "$destination" -maxdepth 1 -type f -printf '%f\n' | sort > "$destination/.libraries"
notices_init "$notices" "koblas-umfpack" "scripts/build-umfpack.sh"
notices_append_file "$notices" "SuiteSparse UMFPACK $version — GPL-2.0-or-later" "SuiteSparse/UMFPACK/Doc/License.txt" "$source/UMFPACK/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse UMFPACK GPL-2.0 license text" "SuiteSparse/UMFPACK/Doc/gpl.txt" "$source/UMFPACK/Doc/gpl.txt"
notices_append_file "$notices" "SuiteSparse AMD — BSD-3-Clause" "SuiteSparse/AMD/Doc/License.txt" "$source/AMD/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse COLAMD — BSD-3-Clause" "SuiteSparse/COLAMD/Doc/License.txt" "$source/COLAMD/Doc/License.txt"
notices_append_suite_sparse_config "$notices" "$source" "$work"
notices_append_file "$notices" "OpenBLAS $openblas_version — BSD-3-Clause" "OpenBLAS/LICENSE" "$openblas_source/LICENSE"
notices_append_gcc_runtime_licenses "$notices" "$work"
printf '%s\n' "$expected" > "$destination/.suitesparse-source-sha256"
