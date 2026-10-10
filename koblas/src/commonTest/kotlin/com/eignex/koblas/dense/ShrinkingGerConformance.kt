package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import kotlin.math.abs
import kotlin.math.ulp
import kotlin.test.assertTrue

/** Finite rank updates through every trailing support of a six-by-six buffer. */
internal fun assertShrinkingGerAgreesWithReference(update: (DoubleArray, DoubleArray, DenseMatrix) -> Unit) {
    val n = 6
    for (fixture in 0 until 5) {
        for (start in 1 until n) {
            val x = DoubleArray(n) { i ->
                if (i < start) {
                    0.0
                } else {
                    val sign = if (i % 2 == 0) -1.0 else 1.0
                    sign * when (fixture) {
                        // 2^500; paired products stay below overflow.
                        2 -> Double.fromBits(1523L shl 52)

                        3 -> Double.MIN_VALUE

                        4 -> Double.MAX_VALUE

                        else -> 1.0 + i / 8.0
                    }
                }
            }
            val y = DoubleArray(n) { j ->
                if (j < start) {
                    0.0
                } else {
                    val sign = if (j % 3 == 0) -1.0 else 1.0
                    sign * when (fixture) {
                        2 -> Double.fromBits(1523L shl 52)
                        3, 4 -> 1.0
                        else -> 0.1 + j / 16.0
                    }
                }
            }
            val initial = DoubleArray(n * n) { at ->
                val i = at % n
                val j = at / n
                when (fixture) {
                    0 -> (i - j) / 8.0
                    4 -> 0.0
                    else -> x[i] * y[j]
                }
            }
            val expected = DenseMatrix.wrap(n, n, initial.copyOf())
            ReferenceBlas.ger(-1.0, x, y, expected)
            val actual = DenseMatrix.wrap(n, n, initial.copyOf())

            update(x, y, actual)

            for (at in initial.indices) {
                // Cancellation exposes rounding of an unfused product, so scale to the terms rather than the residual.
                val magnitude = maxOf(abs(initial[at]), abs(x[at % n] * y[at / n]))
                val error = abs(expected.values[at] - actual.values[at])
                assertTrue(
                    error <= 32.0 * magnitude.ulp,
                    "GER fixture=$fixture support=${n - start} at=$at: " +
                        "${actual.values[at]} expected ${expected.values[at]}",
                )
            }
        }
    }
}
