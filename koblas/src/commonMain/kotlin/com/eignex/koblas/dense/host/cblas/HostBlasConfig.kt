package com.eignex.koblas.dense.host.cblas

/**
 * Whether an OpenBLAS reporting this `openblas_get_config` string takes 64-bit integers. Such a build
 * exports the same unsuffixed symbols as LP64 and then reads the wrong halves of every dimension.
 */
internal fun isIlp64OpenBlas(config: String): Boolean =
    config.split(' ', '\t', '\n').any { it == "USE64BITINT" || it == "INTERFACE64" }

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
    /** Smallest number of right-hand sides routed to a blocked native solve; null keeps the default. */
    val factorizeRhsMin: Int? = null,
) {
    init {
        require(threadCount == null || threadCount > 0) { "threadCount must be positive" }
        require(level1Min == null || level1Min >= 0) { "level1Min must not be negative" }
        require(level2Min == null || level2Min >= 0) { "level2Min must not be negative" }
        require(level3Min == null || level3Min >= 0) { "level3Min must not be negative" }
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
        require(factorizeRhsMin == null || factorizeRhsMin >= 0) {
            "factorizeRhsMin must not be negative"
        }
    }
}
