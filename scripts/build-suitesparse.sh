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
source "$root/scripts/third-party-notices.sh"
lock="$root/koblas-suitesparse/suitesparse.lock"
version="$(sed -n 's/^version=//p' "$lock")"
url="$(sed -n 's/^url=//p' "$lock")"
expected="$(sed -n 's/^sha256=//p' "$lock")"
cache="$output/../downloads/suitesparse-$version.tar.gz"
cache_dir="$(dirname "$cache")"
mkdir -p "$cache_dir"
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
# CHOLMOD and SPQR want LAPACK where UMFPACK wanted only BLAS, and the bundled OpenBLAS carries both, so
# both variables point at it. Naming LAPACK_LIBRARIES also settles where it comes from: SuiteSparseLAPACK
# takes a supplied one as-is and returns, where falling through to cmake's own FindLAPACK searches the host
# and takes whatever it finds, which on a machine with no system LAPACK is nothing.
#
# CUDA is opt-out upstream, so a build host that happens to have it would otherwise put a CUDA runtime
# dependency into CHOLMOD and SPQR. Static archives are never copied into the bundle, so building them
# would only cost compile time.
cmake -S "$source" -B "$work/build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX="$install" \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;colamd;ccolamd;btf;klu;umfpack;cholmod;spqr" \
    -DKLU_USE_CHOLMOD=OFF -DUMFPACK_USE_CHOLMOD=OFF \
    -DBLAS_LIBRARIES="$blas" -DLAPACK_LIBRARIES="$blas" \
    -DBLA_VENDOR=OpenBLAS \
    -DSUITESPARSE_USE_OPENMP=OFF -DSUITESPARSE_USE_CUDA=OFF \
    -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-c++}" >/dev/null
cmake --build "$work/build" --parallel "${SUITESPARSE_JOBS:-2}" >/dev/null
cmake --install "$work/build" >/dev/null

destination="$output/org/eignex/suitesparse/$platform"
notices="$output/THIRD-PARTY-NOTICES.txt"
rm -rf "$destination"; mkdir -p "$destination"
find "$install/lib" -maxdepth 1 -type f \( -name '*.so*' -o -name '*.dylib' \) -exec cp {} "$destination" \;
if [[ "$platform" == linux-* ]]; then
    while IFS= read -r library; do
        soname="$(readelf -d "$library" | sed -n '/SONAME/s/.*\[\(.*\)\]/\1/p')"
        [[ -z "$soname" || "$(basename "$library")" == "$soname" ]] || cp "$library" "$destination/$soname"
    done < <(find "$destination" -maxdepth 1 -type f -name '*.so.*')
else
    # cmake installs the fully versioned dylib as the only real file and records dependencies against the
    # major-versioned name, so both that name and the unversioned one are laid down beside it.
    for library in "$destination"/*.dylib; do
        [[ -e "$library" ]] || continue
        base="$(basename "$library")"
        major="$(printf '%s' "$base" | sed 's/\.\([0-9][0-9]*\)\.[0-9][0-9.]*\.dylib$/.\1.dylib/')"
        unversioned="$(printf '%s' "$base" | sed 's/\.[0-9][0-9.]*\.dylib$/.dylib/')"
        [[ "$major" == "$base" ]] || cp "$library" "$destination/$major"
        [[ "$unversioned" == "$base" ]] || cp "$library" "$destination/$unversioned"
    done
fi
find "$destination" -maxdepth 1 -type f -exec basename {} \; | sort > "$destination/.libraries"
notices_init "$notices" "koblas-suitesparse" "scripts/build-suitesparse.sh"
notices_append_file "$notices" "SuiteSparse UMFPACK $version — GPL-2.0-or-later" "SuiteSparse/UMFPACK/Doc/License.txt" "$source/UMFPACK/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse UMFPACK GPL-2.0 license text" "SuiteSparse/UMFPACK/Doc/gpl.txt" "$source/UMFPACK/Doc/gpl.txt"
notices_append_file "$notices" "SuiteSparse SPQR $version — GPL-2.0-or-later" "SuiteSparse/SPQR/Doc/License.txt" "$source/SPQR/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse SPQR GPL-2.0 license text" "SuiteSparse/SPQR/Doc/gpl.txt" "$source/SPQR/Doc/gpl.txt"
notices_append_file "$notices" "SuiteSparse CHOLMOD $version — LGPL-2.1-or-later and GPL-2.0-or-later, per module" "SuiteSparse/CHOLMOD/Doc/License.txt" "$source/CHOLMOD/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse CHOLMOD bundled METIS — Apache-2.0" "SuiteSparse/CHOLMOD/SuiteSparse_metis/LICENSE.txt" "$source/CHOLMOD/SuiteSparse_metis/LICENSE.txt"
notices_append_file "$notices" "SuiteSparse KLU $version — LGPL-2.1-or-later" "SuiteSparse/KLU/Doc/License.txt" "$source/KLU/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse KLU LGPL-2.1 license text" "SuiteSparse/KLU/Doc/lesser.txt" "$source/KLU/Doc/lesser.txt"
notices_append_file "$notices" "SuiteSparse BTF — LGPL-2.1-or-later" "SuiteSparse/BTF/Doc/License.txt" "$source/BTF/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse AMD — BSD-3-Clause" "SuiteSparse/AMD/Doc/License.txt" "$source/AMD/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse CAMD — BSD-3-Clause" "SuiteSparse/CAMD/Doc/License.txt" "$source/CAMD/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse COLAMD — BSD-3-Clause" "SuiteSparse/COLAMD/Doc/License.txt" "$source/COLAMD/Doc/License.txt"
notices_append_file "$notices" "SuiteSparse CCOLAMD — BSD-3-Clause" "SuiteSparse/CCOLAMD/Doc/License.txt" "$source/CCOLAMD/Doc/License.txt"
notices_append_suite_sparse_config "$notices" "$source" "$work"
notices_append_file "$notices" "OpenBLAS $openblas_version — BSD-3-Clause" "OpenBLAS/LICENSE" "$openblas_source/LICENSE"
notices_append_gcc_runtime_licenses "$notices" "$cache_dir"
printf '%s\n' "$expected" > "$destination/.suitesparse-source-sha256"
