package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class RightSideOpsTest {

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, context: String) {
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= 1e-11 * maxOf(1.0, abs(expected[i])),
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
    }

    /** A triangular matrix with the unselected strict triangle NaN-poisoned, plus its explicit
     *  (zero-filled, unit-diag resolved) twin for dense comparison. */
    private fun poisonedTriangle(
        rng: Random,
        n: Int,
        lower: Boolean,
        unitDiag: Boolean,
    ): Pair<DenseMatrix, DenseMatrix> {
        val t = DenseMatrix(n)
        val explicit = DenseMatrix(n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (lower) j <= i else j >= i
                when {
                    !selected -> t[i, j] = Double.NaN

                    i == j -> {
                        val v = rng.nextDouble(1.0, 2.0)
                        t[i, j] = v
                        explicit[i, j] = if (unitDiag) 1.0 else v
                    }

                    else -> {
                        val v = rng.nextDouble(-1.0, 1.0)
                        t[i, j] = v
                        explicit[i, j] = v
                    }
                }
            }
        }
        return t to explicit
    }

    @Test
    fun `trmm right matches gemm across all flag combos`() {
        val rng = Random(20260940)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = DenseMatrix(rows, n)
                    for (i in 0 until rows) for (j in 0 until n) b[i, j] = rng.nextDouble(-1.0, 1.0)
                    val expected = DenseMatrix(rows, n)
                    koblas.gemm(1.0, b, false, explicit, transpose, 0.0, expected)
                    val actual = DenseMatrix.wrap(rows, n, b.data.copyOf())
                    trmm(t, actual, lower, transpose, unitDiag, right = true)
                    assertClose(expected.data, actual.data, "trmm right l=$lower t=$transpose u=$unitDiag")
                }
            }
        }
    }

    @Test
    fun `trsm right inverts trmm right`() {
        val rng = Random(20260941)
        val n = 6
        val rows = 3
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val x = DenseMatrix(rows, n)
                    for (i in 0 until rows) for (j in 0 until n) x[i, j] = rng.nextDouble(-1.0, 1.0)
                    val b = DenseMatrix.wrap(rows, n, x.data.copyOf())
                    trmm(t, b, lower, transpose, unitDiag, right = true)
                    trsm(t, b, lower, transpose, unitDiag, right = true)
                    assertClose(x.data, b.data, "trsm right l=$lower t=$transpose u=$unitDiag")
                }
            }
        }
    }

    @Test
    fun `symm right matches gemm and reads only the selected triangle`() {
        val rng = Random(20260942)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (alpha in doubleArrayOf(0.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val full = DenseMatrix(n)
                    val poisoned = DenseMatrix(n)
                    for (i in 0 until n) {
                        for (j in 0..i) {
                            val v = rng.nextDouble(-1.0, 1.0)
                            full[i, j] = v
                            full[j, i] = v
                            poisoned[if (lower) i else j, if (lower) j else i] = v
                            if (j != i) poisoned[if (lower) j else i, if (lower) i else j] = Double.NaN
                        }
                    }
                    val b = DenseMatrix(rows, n)
                    for (i in 0 until rows) for (j in 0 until n) b[i, j] = rng.nextDouble(-1.0, 1.0)
                    val c0 = DoubleArray(rows * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.gemm(alpha, b, false, full, false, beta, expected)
                    val actual = DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.symm(alpha, poisoned, b, beta, actual, lower, right = true)
                    assertClose(expected.data, actual.data, "symm right l=$lower a=$alpha b=$beta")
                }
            }
        }
    }
}
