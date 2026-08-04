package com.eignex.koblas.dense

/**
 * Both halves of the compute seam at once: the [Blas] routines and the [Lapack] factorizations built on
 * them. [koblas] is one of these, composed from whichever backend won each half.
 *
 * Implement this when a backend provides both, which is the usual case for a host library. Implement
 * [Blas] or [Lapack] alone when it does not — the two are ranked and installed independently, so a host
 * with CBLAS but no LAPACKE still accelerates its level-2 and level-3 work.
 */
interface LinearAlgebra :
    Blas,
    Lapack
