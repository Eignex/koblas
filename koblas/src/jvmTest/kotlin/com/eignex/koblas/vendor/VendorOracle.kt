package com.eignex.koblas.vendor

import kotlin.math.abs
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * A vendor to test against, or null when this host has none installed.
 *
 * Production selection is tried first so that a host with a real vendor exercises the one it would actually
 * use. OpenBLAS is the fallback because it shares the CBLAS transport, which is what these tests are about;
 * it is never selected in production.
 */
internal val testVendor: VendorBlas? by lazy {
    openVendorBlas() ?: openVendorBlas(Vendor.OpenBlas)
}

/**
 * Runs [body] against an installed vendor, or reports that it did not run.
 *
 * A test that quietly passes because nothing was installed is worse than no test, so the skip says what was
 * missing on the way past rather than leaving a green result to be read as evidence.
 */
internal fun withVendor(body: (VendorBlas) -> Unit) {
    val blas = testVendor
    if (blas == null) {
        println("SKIPPED: no CBLAS library installed; vendor execution was not verified on this host")
        return
    }
    body(blas)
}

/**
 * Asserts agreement with `ReferenceBlas` at a tolerance scaled to the magnitudes involved.
 *
 * The expected side comes from that oracle rather than from anything defined here: it already reads each
 * operand through its structure and transpose, so a second definition beside it would only be another chance
 * to encode the same mistake twice and call the agreement evidence.
 */
internal fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray, what: String) {
    if (expected.size != actual.size) fail("$what: size ${actual.size}, expected ${expected.size}")
    for (index in expected.indices) {
        val scale = maxOf(1.0, abs(expected[index]), abs(actual[index]))
        val difference = abs(expected[index] - actual[index])
        assertTrue(
            difference <= TOLERANCE * scale,
            "$what: entry $index was ${actual[index]}, expected ${expected[index]}",
        )
    }
}

/** Loose enough for a different summation order, tight enough to catch a wrong index or flag. */
private const val TOLERANCE = 1e-12
