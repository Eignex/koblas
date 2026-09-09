package com.eignex.koblas.internal.configuration

/**
 * Stable implementation names used for diagnostics and benchmark attribution.
 */
internal object ImplementationNames {
    /** The exact scalar reference implementation. */
    const val REFERENCE = "reference"

    /** The compiled-in C kernels used by Native and by a JVM without the Vector API. */
    const val C = "c"

    /** The compiled-in C kernels for sparse vectors. */
    const val C_SPARSE = "c-sparse"

    /** Scalar kernels used while cross-compiling for a foreign host or when JVM C kernels are unavailable. */
    const val SCALAR = "scalar"

    /** The compiled-in SIMD kernels, which report a lane count after this prefix. */
    const val SIMD = "simd"

    /** The compiled-in SIMD kernels for sparse vectors, which have no lane count to report. */
    const val SIMD_SPARSE = "simd-sparse"
}
