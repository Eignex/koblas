#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$1"
platform="$2"
compiler="${CC:-cc}"
checks="${build}.checks/$platform"
mkdir -p "$checks"
case "$platform" in
    linux-*) library="$build/com/eignex/koblas/internal/kernels/$platform/libkoblas_kernels.so" ;;
    macosx-*) library="$build/com/eignex/koblas/internal/kernels/$platform/libkoblas_kernels.dylib" ;;
    *) echo "unsupported platform $platform" >&2; exit 1 ;;
esac
"$compiler" -std=c11 -Wall -Wextra -Werror -I"$root/koblas/src/nativeInterop/kernels" \
    "$root/koblas/src/nativeInterop/tests/probe_test.c" "$library" -Wl,-rpath,"$(dirname "$library")" -o "$checks/probe-test"
"$checks/probe-test"
# Exact ABI exports replace all former clone/resolver and header implementation symbols.
if [[ "$platform" == macosx-* ]]; then
    nm -gU "$library" | awk '{print $NF}' | sed 's/^_//' | sort > "$checks/exports.txt"
else
    nm -D --defined-only "$library" | awk '{print $NF}' | sort > "$checks/exports.txt"
fi
{ sed -n 's/.*int32_t \(koblas_[a-z0-9_]*\)(.*/\1/p' "$root/koblas/src/nativeInterop/kernels/koblas_kernels.h"; echo koblas_probe_v1; } | sort > "$checks/expected-exports.txt"
diff -u "$checks/expected-exports.txt" "$checks/exports.txt"

# Loader fixtures distinguish an absent library, an old ABI, missing symbols and incompatible versions.
fixture_dir="$checks/fixtures"
mkdir -p "$fixture_dir"
printf 'int koblas_dense_dot(void) { return 0; }\n' > "$fixture_dir/old.c"
printf 'int koblas_probe_v1(void) { return 2; }\n' > "$fixture_dir/incomplete.c"
{
    cat "$fixture_dir/incomplete.c"
    sed -n 's/.*int32_t \(koblas_[a-z0-9_]*\)(.*/int \1(void) { return 2; }/p' "$root/koblas/src/nativeInterop/kernels/koblas_kernels.h"
} > "$fixture_dir/incompatible.c"
for fixture in old incomplete incompatible; do
    case "$platform" in
        linux-*) "$compiler" -shared -fPIC "$fixture_dir/$fixture.c" -o "$fixture_dir/$fixture.so" ;;
        macosx-*) "$compiler" -dynamiclib "$fixture_dir/$fixture.c" -o "$fixture_dir/$fixture.so" ;;
    esac
done
