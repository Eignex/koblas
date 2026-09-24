package com.eignex.koblas.vendor

import kotlin.math.abs
import kotlin.test.assertTrue
import kotlin.test.fail

/** A vendor to test against through production selection, or null when this host has none installed. */
internal val testVendor: Blas? by lazy { openBlas() }

/**
 * Runs [body] against an installed vendor, or reports that it did not run: a green result from a host with
 * nothing installed must not be read as evidence.
 */
internal fun withVendor(body: (Blas) -> Unit) {
    val blas = testVendor
    if (blas == null) {
        println("SKIPPED: no CBLAS library installed; vendor execution was not verified on this host")
        return
    }
    body(blas)
}

/**
 * Asserts agreement with `ReferenceBlas` at a tolerance scaled to the magnitudes involved. The expected side
 * comes from that oracle rather than a second definition here, which would only repeat the same mistake.
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
