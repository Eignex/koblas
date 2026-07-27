package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class TriangularMultiplyTest {

    /**
     * A poisoned triangular input and its explicit dense equivalent: the selected triangle holds the
     * values, the opposite triangle (and, with unitDiag, the diagonal) holds NaN that trmv/trmm must
     * never read, and the dense twin holds zeros/ones there instead.
     */
    private fun poisonedTriangle(rng: Random, n: Int, lower: Boolean, unitDiag: Boolean): Pair<DenseMatrix, DenseMatrix> {
        val poisoned = DenseMatrix(n)
        val dense = DenseMatrix(n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (lower) j < i else j > i
                when {
                    selected -> {
                        val v = rng.nextDouble(-1.0, 1.0)
                        poisoned[i, j] = v
                        dense[i, j] = v
                    }

                    i == j -> {
                        if (unitDiag) {
                            poisoned[i, j] = Double.NaN
                            dense[i, j] = 1.0
                        } else {
                            val v = rng.nextDouble(0.5, 1.5)
                            poisoned[i, j] = v
                            dense[i, j] = v
                        }
                    }

                    else -> poisoned[i, j] = Double.NaN
                }
            }
        }
        return poisoned to dense
    }

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, context: String) {
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= 1e-12 * maxOf(1.0, abs(expected[i])),
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
    }

    @Test
    fun `trmv matches gemv on the explicit triangle for all flag combos`() {
        val rng = Random(20260915)
        for (n in intArrayOf(1, 2, 7, 16)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (poisoned, dense) = poisonedTriangle(rng, n, lower, unitDiag)
                        val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                        val expected = koblas.gemv(dense, x, transpose)
                        val actual = x.copyOf()
                        trmv(poisoned, actual, lower, transpose, unitDiag)
                        assertClose(expected, actual, "trmv n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmm matches gemm on the explicit triangle for all flag combos`() {
        val rng = Random(20260916)
        for (n in intArrayOf(1, 5, 12)) {
            val p = 3
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (poisoned, dense) = poisonedTriangle(rng, n, lower, unitDiag)
                        val b = DenseMatrix(n, p)
                        for (i in 0 until n) for (j in 0 until p) b[i, j] = rng.nextDouble(-1.0, 1.0)
                        val expected = DenseMatrix(n, p)
                        koblas.gemm(1.0, dense, transpose, b, false, 0.0, expected)
                        val actual = DenseMatrix(n, p, b.data.copyOf())
                        trmm(poisoned, actual, lower, transpose, unitDiag)
                        assertClose(expected.data, actual.data, "trmm n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmv round-trips with trsv`() {
        val rng = Random(20260917)
        val n = 9
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val (poisoned, _) = poisonedTriangle(rng, n, lower, unitDiag = false)
                val x0 = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                val x = x0.copyOf()
                trmv(poisoned, x, lower, transpose)
                trsv(poisoned, x, lower, transpose)
                assertClose(x0, x, "roundtrip lower=$lower t=$transpose")
            }
        }
    }
}
