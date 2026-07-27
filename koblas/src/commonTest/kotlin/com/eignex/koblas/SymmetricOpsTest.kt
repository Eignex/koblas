package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class SymmetricOpsTest {

    /** Random symmetric matrix with the strictly upper triangle NaN-poisoned: symv/symm must not read it. */
    private fun poisonedSymmetric(rng: Random, n: Int): Pair<DenseMatrix, DenseMatrix> {
        val full = DenseMatrix(n)
        val poisoned = DenseMatrix(n)
        for (i in 0 until n) {
            for (j in 0..i) {
                val v = rng.nextDouble(-1.0, 1.0)
                full[i, j] = v
                full[j, i] = v
                poisoned[i, j] = v
                if (j != i) poisoned[j, i] = Double.NaN
            }
        }
        return full to poisoned
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
    fun `symv matches gemv on the full matrix and never reads the upper triangle`() {
        val rng = Random(20260910)
        for (n in intArrayOf(1, 2, 7, 16)) {
            for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n)
                    val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                    val y0 = DoubleArray(n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = y0.copyOf()
                    koblas.gemv(alpha, full, x, beta, expected)
                    val actual = y0.copyOf()
                    koblas.symv(alpha, poisoned, x, beta, actual)
                    assertClose(expected, actual, "symv n=$n a=$alpha b=$beta")
                }
            }
        }
    }

    @Test
    fun `symm matches gemm on the full matrix and never reads the upper triangle`() {
        val rng = Random(20260911)
        for (n in intArrayOf(1, 5, 12)) {
            val p = 4
            for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n)
                    val b = DenseMatrix(n, p)
                    for (i in 0 until n) for (j in 0 until p) b[i, j] = rng.nextDouble(-1.0, 1.0)
                    val c0 = DoubleArray(n * p) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = DenseMatrix(n, p, c0.copyOf())
                    koblas.gemm(alpha, full, transposeA = false, b, transposeB = false, beta, expected)
                    val actual = DenseMatrix(n, p, c0.copyOf())
                    koblas.symm(alpha, poisoned, b, beta, actual)
                    assertClose(expected.data, actual.data, "symm n=$n a=$alpha b=$beta")
                }
            }
        }
    }

    @Test
    fun `symm handles empty shapes`() {
        koblas.symm(1.0, DenseMatrix(0, 0), DenseMatrix(0, 3), 0.0, DenseMatrix(0, 3))
        val c = DenseMatrix(2, 0)
        koblas.symm(1.0, DenseMatrix(2, 2), DenseMatrix(2, 0), 0.0, c)
        val y = DoubleArray(2) { Double.NaN }
        koblas.symv(0.0, DenseMatrix(2, 2), DoubleArray(2), 0.0, y)
        assertTrue(y.all { it == 0.0 }, "symv alpha=0 beta=0 left ${y.toList()}")
    }
}
