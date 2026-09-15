package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
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

/** The plain definition of the product, read through the windows' own structure and strides. */
internal fun oracleGemm(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow): DoubleArray {
    val result = DoubleArray(c.rows * c.columns)
    for (column in 0 until c.columns) {
        for (row in 0 until c.rows) {
            var sum = 0.0
            for (k in 0 until a.columns) sum += a[row, k] * b[k, column]
            val previous = if (beta == 0.0) 0.0 else beta * c[row, column]
            result[row + column * c.rows] = previous + alpha * sum
        }
    }
    return result
}

/** `y = alpha · A · x + beta · y` read the same way. */
internal fun oracleGemv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow): DoubleArray =
    DoubleArray(y.size) { row ->
        var sum = 0.0
        for (k in 0 until a.columns) sum += a[row, k] * x[k]
        (if (beta == 0.0) 0.0 else beta * y[row]) + alpha * sum
    }

/** Reads a window into a packed column-major array, so a result can be compared entry by entry. */
internal fun MatrixWindow.toPacked(): DoubleArray {
    val packed = DoubleArray(rows * columns)
    for (column in 0 until columns) for (row in 0 until rows) packed[row + column * rows] = this[row, column]
    return packed
}

/** Reads a vector window into a packed array. */
internal fun VectorWindow.toPacked(): DoubleArray = DoubleArray(size) { this[it] }

/** Asserts agreement with the oracle at a tolerance scaled to the magnitudes involved. */
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
