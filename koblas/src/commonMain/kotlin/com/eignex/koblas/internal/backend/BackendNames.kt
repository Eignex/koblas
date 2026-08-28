package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
/**
 * The [Backend.name] values koblas's own backends report. They are also what a caller writes into
 * [ConfigurationKeys.BACKENDS] to pin one semantic role, so a name is a shared identifier and not a label
 * any one implementation is free to reword.
 */
internal object BackendNames {
    /** The portable implementations, and a backend-selection value that registers none. */
    const val REFERENCE = "reference"

    /** The JVM's `java.lang.foreign` binding to a host OpenBLAS. */
    const val OPENBLAS = "openblas"

    /** The Kotlin/Native `dlopen` binding to a host OpenBLAS. */
    const val CBLAS = "cblas"

    /** The binding to SuiteSparse's UMFPACK. */
    const val UMFPACK = "umfpack"

    /** The binding to SuiteSparse's KLU. */
    const val KLU = "klu"

    /** The binding to BASICLU, whose basis updates no other sparse backend offers. */
    const val BASICLU = "basiclu"

    /** The binding to HiGHS's HFactor, which solves a simplex basis hypersparsely and updates it. */
    const val HFACTOR = "hfactor"

    /** SuiteSparse CHOLMOD, which fills the sparse matrix products. */
    const val CHOLMOD = "cholmod"

    /** The compiled-in scalar kernels. */
    const val SCALAR = "scalar"

    /** The compiled-in SIMD kernels, which report a lane count after this prefix. */
    const val SIMD = "simd"

    /** The compiled-in SIMD kernels for sparse vectors, which have no lane count to report. */
    const val SIMD_SPARSE = "simd-sparse"
}
