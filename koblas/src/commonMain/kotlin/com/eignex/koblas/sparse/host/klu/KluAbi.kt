package com.eignex.koblas.sparse.host.klu

/**
 * KLU 2's ABI as offsets and codes, shared by the bindings that reach it: the JVM reads a `MemorySegment`
 * at these offsets and the native targets read a pointer. Both hold the `klu_common` and `klu_numeric`
 * structs as opaque blocks, since neither has a header to generate from at build time.
 */
internal val KLU_SONAMES: List<String> = listOf("libklu.so.2", "libklu.2.dylib")

internal const val KLU_COMMON_BYTES = 160L
internal const val KLU_NUMERIC_BYTES = 168L
internal const val KLU_COMMON_TOL = 0L
internal const val KLU_COMMON_MEMGROW = 8L
internal const val KLU_COMMON_INITMEM_AMD = 16L
internal const val KLU_COMMON_INITMEM = 24L
internal const val KLU_COMMON_MAXWORK = 32L
internal const val KLU_COMMON_BTF = 40L
internal const val KLU_COMMON_ORDERING = 44L
internal const val KLU_COMMON_SCALE = 48L
internal const val KLU_COMMON_HALT_IF_SINGULAR = 72L
internal const val KLU_COMMON_STATUS = 76L
internal const val KLU_COMMON_RCOND = 112L
internal const val KLU_NUMERIC_LNZ = 8L
internal const val KLU_NUMERIC_UNZ = 12L
internal const val KLU_SCALE_NONE = 0
internal const val KLU_SCALE_MAX = 2
internal const val KLU_SINGULAR = 1

/** KLU takes its flags as ints. */
internal fun Boolean.asNativeKluBoolean(): Int = if (this) 1 else 0

internal val KluOrdering.nativeValue: Int
    get() = when (this) {
        KluOrdering.AMD -> 0
        KluOrdering.COLAMD -> 1
    }

internal val KluScaling.nativeValue: Int
    get() = when (this) {
        KluScaling.NONE -> KLU_SCALE_NONE
        KluScaling.SUM -> 1
        KluScaling.MAX -> KLU_SCALE_MAX
    }
