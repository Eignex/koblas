package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.internal.host.NativeBlock

/**
 * KLU 2's ABI as offsets and codes, shared by the bindings that reach it: the JVM reads a `MemorySegment`
 * at these offsets and the native targets read a pointer. Both hold the `klu_common` and `klu_numeric`
 * structs as opaque blocks, since neither has a header to generate from at build time.
 */
internal val KLU_SONAMES: List<String> = listOf(
    "libklu.so.2",
    "libklu.so",
    "libklu.2.dylib",
    "libklu.dylib",
    "/opt/homebrew/opt/suite-sparse/lib/libklu.dylib", // Homebrew is keg-only
    "/opt/homebrew/opt/suite-sparse/lib/libklu.2.dylib",
    "/usr/local/opt/suite-sparse/lib/libklu.dylib",
)

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
internal const val KLU_NUMERIC_NBLOCKS = 4L
internal const val KLU_NUMERIC_LNZ = 8L
internal const val KLU_NUMERIC_UNZ = 12L

/** Entries outside the diagonal blocks, the last field of a `klu_numeric` and the size `F` needs. */
internal const val KLU_NUMERIC_NZOFF = 160L
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

/**
 * Writes [config] into a `klu_common` KLU has already filled with its own defaults, leaving every field the
 * configuration says nothing about as KLU set it.
 *
 * [equilibrate] is the backend's policy rather than the configuration's, so the scaling it selects is the
 * one field written unconditionally: KLU's default scales, and a backend that does not equilibrate has to
 * say so.
 */
internal fun NativeBlock.applyKluConfig(config: KluConfig, equilibrate: Boolean) {
    config.pivotTolerance?.let { putDouble(KLU_COMMON_TOL, it) }
    config.memoryGrowth?.let { putDouble(KLU_COMMON_MEMGROW, it) }
    config.amdInitialMemoryFactor?.let { putDouble(KLU_COMMON_INITMEM_AMD, it) }
    config.initialMemoryFactor?.let { putDouble(KLU_COMMON_INITMEM, it) }
    config.maxBtfWork?.let { putDouble(KLU_COMMON_MAXWORK, it) }
    config.useBtf?.let { putInt(KLU_COMMON_BTF, it.asNativeKluBoolean()) }
    config.ordering?.let { putInt(KLU_COMMON_ORDERING, it.nativeValue) }
    config.haltIfSingular?.let { putInt(KLU_COMMON_HALT_IF_SINGULAR, it.asNativeKluBoolean()) }
    putInt(KLU_COMMON_SCALE, if (equilibrate) config.equilibratedScaling.nativeValue else KLU_SCALE_NONE)
}
