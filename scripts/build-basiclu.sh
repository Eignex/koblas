#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/native-build-common.sh"
source "$root/scripts/third-party-notices.sh"

parse_native_build_args "usage: $0 --platform <platform> --output <directory>" "$@"
read_native_lock "$root/koblas-basiclu/basiclu.lock"
cache="$output/../downloads/basiclu-$version.tar.gz"
fetch_verified_archive "$url" "$cache" "$expected" BASICLU
require_platform_host "$platform" BASICLU

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tar -xzf "$cache" -C "$work"
source="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
destination="$output/org/eignex/basiclu/$platform"
notices="$output/THIRD-PARTY-NOTICES.txt"
rm -rf "$destination"; mkdir -p "$destination"

if [[ "$platform" == linux-* ]]; then
    library="libkoblas_basiclu.so.1"
    "${CC:-cc}" -std=c99 -O3 -DNDEBUG -fPIC -shared -Wl,-soname,"$library" \
        -I"$source/include" "$source"/src/*.c -lm \
        -o "$destination/$library"
    cp "$destination/$library" "$destination/libkoblas_basiclu.so"
else
    library="libkoblas_basiclu.1.dylib"
    "${CC:-cc}" -std=c99 -O3 -DNDEBUG -fPIC -DCLOCK_MONOTONIC_RAW=CLOCK_MONOTONIC -dynamiclib -Wl,-install_name,@rpath/"$library" \
        -I"$source/include" "$source"/src/*.c -lm \
        -o "$destination/$library"
    cp "$destination/$library" "$destination/libkoblas_basiclu.dylib"
fi
find "$destination" -maxdepth 1 -type f -exec basename {} \; | sort > "$destination/.libraries"
notices_init "$notices" "koblas-basiclu" "scripts/build-basiclu.sh"
printf '\nBASICLU Copyright (c) 2016-2020 ERGO-Code\nUsed in Koblas under the MIT license.\n' >> "$notices"
notices_append_file "$notices" "BASICLU $version — MIT" "BASICLU/LICENSE" "$source/LICENSE"
printf '%s\n' "$expected" > "$destination/.basiclu-source-sha256"
