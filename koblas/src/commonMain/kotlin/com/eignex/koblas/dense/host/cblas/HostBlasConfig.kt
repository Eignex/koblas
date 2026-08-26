package com.eignex.koblas.dense.host.cblas

/**
 * Whether an OpenBLAS reporting this `openblas_get_config` string takes 64-bit integers. Such a build
 * exports the same unsuffixed symbols as LP64 and then reads the wrong halves of every dimension.
 */
internal fun isIlp64OpenBlas(config: String): Boolean =
    config.split(' ', '\t', '\n').any { it == "USE64BITINT" || it == "INTERFACE64" }

/**
 * Whether the pivots a probing `LAPACKE_dgetrf` wrote are 64 bits wide, read as the first three 32-bit
 * [words] of a zeroed buffer. The probe factorizes a 2x2 whose both pivots are row 2, so an LP64 build
 * leaves `2, 2` in the first two words and an ILP64 build spreads the same two pivots over four, putting a
 * zero where the second pivot would sit.
 *
 * A vendor that reports no `openblas_get_config` string is judged here instead, since ILP64 exports the same
 * unsuffixed symbols and the width of what it writes is the only evidence left. Anything but the exact ILP64
 * arrangement counts as LP64, so an unrecognized library keeps working rather than being turned away.
 */
internal fun isIlp64PivotWidth(words: IntArray): Boolean =
    words.size >= PROBE_WORDS && words[0] == PROBE_PIVOT && words[1] == 0 && words[2] == PROBE_PIVOT

/** Words of the pivot buffer [isIlp64PivotWidth] reads, enough to cover two 64-bit pivots. */
internal const val PROBE_WORDS: Int = 3

/** The pivot both steps of the probe's 2x2 factorization select, in LAPACK's 1-based row numbering. */
internal const val PROBE_PIVOT: Int = 2

/** The order of the matrix the pivot-width probe factorizes. */
internal const val PROBE_ORDER: Int = 2

/** Names used by the platform loader to locate a host OpenBLAS. */
internal val OPENBLAS_SONAMES = listOf(
    "libopenblas.so.0",
    "libopenblas.so",
    "libopenblas.dylib",
    "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
    "/usr/local/opt/openblas/lib/libopenblas.dylib",
    "openblas.dll",
)

/** Names used when LAPACKE is not exported by OpenBLAS itself. */
internal val LAPACKE_SONAMES = listOf(
    "liblapacke.so.3",
    "liblapacke.so",
    "liblapacke.dylib",
    "/opt/homebrew/opt/decompositions/lib/liblapacke.dylib",
    "/usr/local/opt/decompositions/lib/liblapacke.dylib",
    "lapacke.dll",
)

/** Policy for one OpenBLAS and optional LAPACKE backend instance. */
public data class HostBlasConfig(
    /** An absolute OpenBLAS library path, or the deployment lookup chain when null. */
    val libraryPath: String? = null,
    /** An absolute LAPACKE library path, or the deployment lookup chain when null. */
    val lapackeLibraryPath: String? = null,
    /** OpenBLAS thread count; setting it, including to one, requires a higher JVM thread-memory cap. */
    val threadCount: Int? = 1,
    /** Smallest run length routed to native level-1 BLAS; null keeps the platform default. */
    val level1Min: Int? = null,
    /** Smallest dimension routed to native level-2 BLAS; null keeps the platform default. */
    val level2Min: Int? = null,
    /** Smallest dimension routed to native level-3 BLAS; null keeps the platform default. */
    val level3Min: Int? = null,
    /** Smallest dimension routed to the native factorizations; null keeps the platform default. */
    val factorizeMin: Int? = null,
) {
    init {
        require(threadCount == null || threadCount > 0) { "threadCount must be positive" }
        require(level1Min == null || level1Min >= 0) { "level1Min must not be negative" }
        require(level2Min == null || level2Min >= 0) { "level2Min must not be negative" }
        require(level3Min == null || level3Min >= 0) { "level3Min must not be negative" }
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}
