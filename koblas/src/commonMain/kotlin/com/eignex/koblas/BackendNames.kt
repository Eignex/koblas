package com.eignex.koblas

/**
 * The [Backend.name] values koblas's own backends report. They are also what a caller writes into
 * [ConfigurationKeys.BACKEND_PROPERTY] to pin one backend, so a name is a shared identifier and not a label
 * any one implementation is free to reword.
 */
internal object BackendNames {
    /** The portable implementations, and the [ConfigurationKeys.BACKEND_PROPERTY] value that registers none. */
    const val REFERENCE = "reference"

    /** The JVM's `java.lang.foreign` binding to a host OpenBLAS. */
    const val OPENBLAS = "openblas"

    /** The Kotlin/Native `dlopen` binding to a host OpenBLAS. */
    const val CBLAS = "cblas"

    /** The binding to SuiteSparse's UMFPACK. */
    const val UMFPACK = "umfpack"

    /** The compiled-in scalar kernels. */
    const val SCALAR = "scalar"

    /** The compiled-in SIMD kernels, which report a lane count after this prefix. */
    const val SIMD = "simd"

    /** The compiled-in SIMD kernels for sparse vectors, which have no lane count to report. */
    const val SIMD_SPARSE = "simd-sparse"
}
