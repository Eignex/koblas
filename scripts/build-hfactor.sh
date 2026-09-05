#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/native-build-common.sh"
source "$root/scripts/third-party-notices.sh"
native="$root/koblas-hfactor/native"

parse_native_build_args "usage: $0 --platform <platform> --output <directory>" "$@"
read_native_lock "$root/koblas-hfactor/hfactor.lock"
cache="$output/../downloads/highs-$version.tar.gz"
fetch_verified_archive "$url" "$cache" "$expected" HiGHS
require_platform_host "$platform" HFactor

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tar -xzf "$cache" -C "$work"
source_root="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
# HiGHS moved its sources from src to highs in 1.11, so the lock pins the layout along with the version.
sources="$source_root/highs"
[[ -f "$sources/util/HFactor.cpp" ]] || { echo "HiGHS $version has no highs/util/HFactor.cpp" >&2; exit 1; }
destination="$output/org/eignex/hfactor/$platform"
notices="$output/THIRD-PARTY-NOTICES.txt"
rm -rf "$destination"; mkdir -p "$destination"

# HFactor compiles standalone; only these four upstream units are needed to link it, with the shim
# supplying the C entry points and hfactor_log.cpp standing in for the logging HighsIO.cpp would bring.
units=(
    "$sources/util/HFactor.cpp"
    "$sources/util/HFactorRefactor.cpp"
    "$sources/util/HFactorDebug.cpp"
    "$sources/util/HVectorBase.cpp"
    "$native/hfactor_log.cpp"
    "$native/hfactor_shim.cpp"
)

# -fPIC on every unit, not only the link: HighsTimer's vtable otherwise lands in a read-only relocation
# that a shared object cannot take.
flags=(-std=c++17 -O3 -DNDEBUG -fPIC -fvisibility=hidden -I"$native" -I"$sources" -I"$source_root/extern")

if [[ "$platform" == linux-* ]]; then
    library="libkoblas_hfactor.so.1"
    "${CXX:-c++}" "${flags[@]}" -shared -Wl,-soname,"$library" "${units[@]}" -o "$destination/$library"
    cp "$destination/$library" "$destination/libkoblas_hfactor.so"
else
    library="libkoblas_hfactor.1.dylib"
    "${CXX:-c++}" "${flags[@]}" -dynamiclib -Wl,-install_name,@rpath/"$library" "${units[@]}" \
        -o "$destination/$library"
    cp "$destination/$library" "$destination/libkoblas_hfactor.dylib"
fi
find "$destination" -maxdepth 1 -type f -exec basename {} \; | sort > "$destination/.libraries"
notices_init "$notices" "koblas-hfactor" "scripts/build-hfactor.sh"
printf '\nHiGHS Copyright (c) 2026 HiGHS\nUsed in Koblas under the MIT license.\n' >> "$notices"
notices_append_file "$notices" "HiGHS $version — MIT" "HiGHS/LICENSE.txt" "$source_root/LICENSE.txt"
notices_append_file "$notices" "pdqsort — MIT" "HiGHS/extern/pdqsort/pdqsort.h" "$source_root/extern/pdqsort/license.txt"
printf '%s\n' "$expected" > "$destination/.hfactor-source-sha256"
