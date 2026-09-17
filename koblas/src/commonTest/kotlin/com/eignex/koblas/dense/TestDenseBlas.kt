package com.eignex.koblas.dense

import com.eignex.koblas.vendor.openBlas

/**
 * The dense Level 2 and 3 seam under test, or null when this host has no CBLAS at all.
 *
 * Production selection, which ends in OpenBLAS on Linux. A host with a tuned library tests the one it would
 * actually use, and a host with only the distribution's BLAS tests the same CBLAS entry points, which is what
 * the contract these tests state is about.
 *
 * Without that last resort the whole dense contract would go untested on every machine that has only OpenBLAS
 * installed, which is most of them, and would do it by passing.
 */
internal val testDenseBlas: DenseBlas? by lazy {
    openBlas()?.let { VendorDenseBlas(it) }
}

/**
 * Runs [body] against that seam, or reports that it did not run.
 *
 * A test that quietly passes because nothing was installed is worse than no test, so the skip says what was
 * missing on the way past rather than leaving a green result to be read as evidence.
 */
internal fun withDenseBlas(body: (DenseBlas) -> Unit) {
    val blas = testDenseBlas
    if (blas == null) {
        println("SKIPPED: no CBLAS library installed; the dense level 2 and 3 contract was not verified here")
        return
    }
    body(blas)
}
