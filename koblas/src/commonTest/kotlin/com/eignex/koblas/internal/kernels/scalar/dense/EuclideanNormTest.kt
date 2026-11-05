package com.eignex.koblas.internal.kernels.scalar.dense

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals

class EuclideanNormTest {
    @Test
    fun `the norm rescales components whose squares exceed the exponent range`() {
        for (scale in doubleArrayOf(1e200, 1e-200)) {
            val input = doubleArrayOf(3.0 * scale, 4.0 * scale)

            val actual = euclideanNorm(input, 0, 1, input.size)

            assertEquals(5.0 * scale, actual, absoluteTolerance = scale * 1e-12)
        }
    }

    @Test
    fun `the norm preserves equal components across lengths`() {
        for (length in intArrayOf(16, 33, 64)) {
            for (scale in doubleArrayOf(1e200, 1e-200)) {
                val input = DoubleArray(length) { scale }
                val expected = sqrt(length.toDouble()) * scale

                val actual = euclideanNorm(input, 0, 1, length)

                assertEquals(expected, actual, absoluteTolerance = expected * 1e-12)
            }
        }
    }
}
