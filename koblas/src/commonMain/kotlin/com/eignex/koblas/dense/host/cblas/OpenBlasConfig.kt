package com.eignex.koblas.dense.host.cblas

/**
 * Whether an OpenBLAS reporting this `openblas_get_config` string takes 64-bit integers. Such a build
 * exports the same unsuffixed symbols as LP64 and then reads the wrong halves of every dimension.
 */
internal fun isIlp64OpenBlas(config: String): Boolean =
    // Whole tokens, so a hypothetical USE64BITINT_OFF is not read as the marker.
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
