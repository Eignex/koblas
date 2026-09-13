#!/usr/bin/env bash
set -euo pipefail

platform="" output="" kind=shared compiler="${CC:-cc}" archiver="${AR:-ar}"
cflags=()
while (($#)); do
    case "$1" in
        --platform) platform="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --kind) kind="$2"; shift 2 ;;
        --compiler) compiler="$2"; shift 2 ;;
        --archiver) archiver="$2"; shift 2 ;;
        --cflag) cflags+=("$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$platform" && -n "$output" ]] || { echo "--platform and --output are required" >&2; exit 2; }
[[ "$kind" == shared || "$kind" == static ]] || { echo "invalid library kind: $kind" >&2; exit 2; }
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../koblas/src/nativeInterop/kernels" && pwd)"
build_dir="$output"
[[ "$kind" != shared ]] || build_dir="${output}.build/$platform"
mkdir -p "$build_dir/objects"
triple="$("$compiler" ${cflags[@]+"${cflags[@]}"} -dumpmachine)"
case "$platform:$triple" in
    linux-x86_64:x86_64*linux*)
        baseline=(-march=x86-64 -mno-avx -mno-fma)
        variants=(sse2 avx2)
        defines=(-DKOBLAS_BUILD_SSE2 -DKOBLAS_BUILD_AVX2)
        ;;
    linux-arm64:aarch64*linux*|macosx-arm64:arm64*apple*|macosx-arm64:aarch64*apple*)
        baseline=(-march=armv8-a)
        variants=(neon)
        defines=(-DKOBLAS_BUILD_NEON)
        ;;
    *) echo "compiler target $triple does not match requested platform $platform" >&2; exit 1 ;;
esac
version="$("$compiler" --version)"
if [[ "$version" == *clang* ]]; then
    scalar_flags=(-fno-vectorize -fno-slp-vectorize)
else
    scalar_flags=(-fno-tree-vectorize -fno-tree-slp-vectorize)
fi
common=(-std=c11 -O3 -fPIC -fvisibility=hidden -fno-lto -ffp-contract=off -I"$source_dir" ${cflags[@]+"${cflags[@]}"})
{
    printf 'target=%s\nplatform=%s\nkind=%s\ncompiler=%s\n' "$triple" "$platform" "$kind" "$compiler"
    printf '%s\n' "$version"
    printf 'common='; printf '%q ' "${common[@]}"; printf '\nbaseline='; printf '%q ' "${baseline[@]}"
    printf '\nscalar='; printf '%q ' "${scalar_flags[@]}"; printf '\nvariants=%s\n' "${variants[*]}"
    if [[ "$kind" == static ]]; then "$archiver" --version; fi
} > "$build_dir/toolchain.txt"
objects=()
compile() {
    local name="$1" source="$2"; shift 2
    "$compiler" "${common[@]}" "${baseline[@]}" "$@" -c "$source_dir/$source" -o "$build_dir/objects/$name.o"
    objects+=("$build_dir/objects/$name.o")
}
compile scalar ordinary.c "${scalar_flags[@]}" -DKOBLAS_SCALAR_IMPL -DKOBLAS_SUFFIX=_scalar
for variant in "${variants[@]}"; do
    flags=()
    [[ "$variant" != avx2 ]] || flags=(-mavx2 -mprefer-vector-width=256)
    compile "$variant" ordinary.c ${flags[@]+"${flags[@]}"} "-DKOBLAS_SUFFIX=_$variant"
done
compile probe probe.c "${scalar_flags[@]}" "${defines[@]}"
compile execute execute.c "${scalar_flags[@]}" "${defines[@]}"
if [[ "$kind" == static ]]; then
    rm -f "$output/libkoblas_kernels.a"
    "$archiver" rcs "$output/libkoblas_kernels.a" "${objects[@]}"
else
    resource_dir="$output/com/eignex/koblas/internal/kernels/$platform"
    mkdir -p "$resource_dir"
    case "$platform" in
        linux-*)
            # Older target linkers otherwise publish their CRT boundary symbols as part of our ABI.
            {
                printf '{ global:\n'
                sed -n 's/.*int32_t \(koblas_[a-z0-9_]*\)(.*/    \1;/p' "$source_dir/koblas_kernels.h"
                printf '    koblas_probe_v1;\n  local: *;\n};\n'
            } > "$build_dir/exports.map"
            "$compiler" ${cflags[@]+"${cflags[@]}"} -shared "${objects[@]}" \
                -Wl,-soname,libkoblas_kernels.so -Wl,--version-script,"$build_dir/exports.map" \
                -lm -o "$resource_dir/libkoblas_kernels.so"
            ;;
        macosx-*) "$compiler" ${cflags[@]+"${cflags[@]}"} -dynamiclib "${objects[@]}" -Wl,-install_name,@rpath/libkoblas_kernels.dylib -o "$resource_dir/libkoblas_kernels.dylib" ;;
    esac
fi
