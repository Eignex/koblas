package com.eignex.koblas.dense

/**
 * Whether an OpenBLAS reporting this `openblas_get_config` string takes 64-bit integers. Such a build
 * exports the same unsuffixed symbols as LP64 and then reads the wrong halves of every dimension.
 */
internal fun isIlp64OpenBlas(config: String): Boolean =
    // Whole tokens rather than substrings, so a hypothetical USE64BITINT_OFF is not read as the marker.
    config.split(' ', '\t', '\n').any { it == "USE64BITINT" || it == "INTERFACE64" }
