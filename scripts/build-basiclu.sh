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
lock="$root/koblas-basiclu/basiclu.lock"
version="$(sed -n 's/^version=//p' "$lock")"
url="$(sed -n 's/^url=//p' "$lock")"
expected="$(sed -n 's/^sha256=//p' "$lock")"
cache="$output/../downloads/basiclu-$version.tar.gz"
mkdir -p "$(dirname "$cache")"
if [[ ! -f "$cache" ]]; then curl --fail --location --silent --show-error "$url" -o "$cache"; fi
actual="$(sha256sum "$cache" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || { echo "BASICLU source checksum mismatch" >&2; exit 1; }

case "$platform" in
    linux-x86_64) [[ "$(uname -s)" == Linux ]] || { echo "linux-x86_64 requires Linux" >&2; exit 1; } ;;
    linux-arm64) [[ "$(uname -s)" == Linux ]] || { echo "linux-arm64 requires Linux" >&2; exit 1; } ;;
    macosx-arm64) [[ "$(uname -s)" == Darwin ]] || { echo "macosx-arm64 requires macOS" >&2; exit 1; } ;;
    *) echo "unsupported BASICLU platform $platform" >&2; exit 1 ;;
esac

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tar -xzf "$cache" -C "$work"
source="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
destination="$output/org/eignex/basiclu/$platform"
notices="$output/THIRD-PARTY-NOTICES.txt"
rm -rf "$destination"; mkdir -p "$destination"

if [[ "$platform" == linux-* ]]; then
    library="libkoblas_basiclu.so.1"
    "${CC:-cc}" -std=c99 -O3 -DNDEBUG -fPIC -shared -Wl,-soname,"$library" \
        -I"$source/include" "$source"/src/*.c "$root/koblas-basiclu/native/basiclu_shim.c" -lm \
        -o "$destination/$library"
    cp "$destination/$library" "$destination/libkoblas_basiclu.so"
else
    library="libkoblas_basiclu.1.dylib"
    "${CC:-cc}" -std=c99 -O3 -DNDEBUG -fPIC -DCLOCK_MONOTONIC_RAW=CLOCK_MONOTONIC -dynamiclib -Wl,-install_name,@rpath/"$library" \
        -I"$source/include" "$source"/src/*.c "$root/koblas-basiclu/native/basiclu_shim.c" -lm \
        -o "$destination/$library"
    cp "$destination/$library" "$destination/libkoblas_basiclu.dylib"
fi
find "$destination" -maxdepth 1 -type f -exec basename {} \; | sort > "$destination/.libraries"
notices_init "$notices" "koblas-basiclu" "scripts/build-basiclu.sh"
printf '\nBASICLU Copyright (c) 2016-2020 ERGO-Code\nUsed in Koblas under the MIT license.\n' >> "$notices"
notices_append_file "$notices" "BASICLU $version — MIT" "BASICLU/LICENSE" "$source/LICENSE"
printf '%s\n' "$expected" > "$destination/.basiclu-source-sha256"
