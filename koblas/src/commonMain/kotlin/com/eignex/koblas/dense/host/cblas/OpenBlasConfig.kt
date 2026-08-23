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
    "/opt/homebrew/opt/lapack/lib/liblapacke.dylib",
    "/usr/local/opt/lapack/lib/liblapacke.dylib",
    "lapacke.dll",
)

/** Policy for one OpenBLAS and optional LAPACKE backend instance. */
public data class OpenBlasConfig(
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
    /** Smallest dimension routed to native LAPACK; null keeps the platform default. */
    val lapackMin: Int? = null,
    /** Smallest dimension routed to the native pivoted QR routine. */
    val pivotedQrMin: Int = 8,
    /** Smallest number of right-hand sides routed to native triangular solve. */
    val triangularSolveMinRhs: Int = 4,
    /** Smallest order routed to native Cholesky factorization. */
    val choleskyMin: Int = 32,
    /** Smallest order routed to native SPD inversion. */
    val spdInvertMin: Int = 16,
) {
    init {
        require(threadCount == null || threadCount > 0) { "threadCount must be positive" }
        require(level1Min == null || level1Min >= 0) { "level1Min must not be negative" }
        require(level2Min == null || level2Min >= 0) { "level2Min must not be negative" }
        require(level3Min == null || level3Min >= 0) { "level3Min must not be negative" }
        require(lapackMin == null || lapackMin >= 0) { "lapackMin must not be negative" }
        require(pivotedQrMin >= 0) { "pivotedQrMin must not be negative" }
        require(triangularSolveMinRhs >= 0) { "triangularSolveMinRhs must not be negative" }
        require(choleskyMin >= 0) { "choleskyMin must not be negative" }
        require(spdInvertMin >= 0) { "spdInvertMin must not be negative" }
    }
}
