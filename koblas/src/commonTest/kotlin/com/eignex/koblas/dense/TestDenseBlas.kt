package com.eignex.koblas.dense

import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.openVendorBlas

/**
 * The dense Level 2 and 3 seam under test, or null when this host has no CBLAS at all.
 *
 * Production selection is tried first, so a host with a supported vendor tests the one it would actually use.
 * OpenBLAS is the fallback because it exports the same CBLAS entry points, and the contract these tests state
 * is about those calls rather than about which library serves them; it stays unselectable in production, where
 * the choice is also about how a library is held to one thread.
 *
 * Without this fallback the whole dense contract would go untested on every machine that has only OpenBLAS
 * installed, which is most of them, and would do it by passing.
 */
internal val testDenseBlas: DenseBlas? by lazy {
    (openVendorBlas() ?: openVendorBlas(Vendor.OpenBlas))?.let { VendorDenseBlas(it) }
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
