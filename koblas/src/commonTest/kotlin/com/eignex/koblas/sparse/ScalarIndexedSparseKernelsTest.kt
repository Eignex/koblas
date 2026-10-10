package com.eignex.koblas.sparse

import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScalarIndexedSparseKernelsTest {
    @Test
    fun `indexed norm agrees with reference across scaling and tails`() {
        for (count in COUNTS) {
            for (scale in listOf(0.0, 1.0, 1e200, 1e-200)) {
                val indices = IntArray(count + 5) { -1 }
                val values = DoubleArray(37) { (it % 11 - 5) * scale }
                for (k in 0 until count) indices[k + 2] = (k * 13) % values.size
                val gathered = DoubleArray(count) { values[indices[it + 2]] }
                val expected = euclideanNorm(gathered, 0, 1, count)

                val actual = ScalarIndexedSparseKernels.nrm2(indices, 2, count, values)

                assertNormAgreesWithReference(expected, actual, "count=$count scale=$scale")
            }
        }
    }

    @Test
    fun `indexed norm evaluates exceptional values in every accumulator and tail`() {
        for (position in 0 until 67) {
            for (exceptional in listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY)) {
                val indices = IntArray(67) { it }
                val values = DoubleArray(67) { 1.0 }.also { it[position] = exceptional }
                val expectedNorm = euclideanNorm(values, 0, 1, 67)

                val actualNorm = ScalarIndexedSparseKernels.nrm2(indices, 0, 67, values)

                assertNormAgreesWithReference(expectedNorm, actualNorm, "position=$position")
            }
        }
    }

    private fun assertNormAgreesWithReference(expected: Double, actual: Double, context: String) {
        when {
            expected.isNaN() -> assertTrue(actual.isNaN(), context)
            expected.isInfinite() -> assertEquals(expected, actual, context)
            else -> assertTrue(abs(expected - actual) <= 1e-12 * abs(expected), context)
        }
    }

    private companion object {
        val COUNTS = (0..33).toList() + listOf(63, 64, 65, 127, 128, 129, 1023, 1024, 1025)
    }
}
