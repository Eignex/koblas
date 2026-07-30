package com.eignex.koblas

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The symmetric routines: `symv`, `symm` on either side, and the `syrk` rank-k update.
 *
 * Each is checked against the general routine that computes the same thing from a fully written-out
 * matrix, over the alpha and beta values BLAS treats specially. A poisoned operand (see
 * [poisonedSymmetric]) proves the unselected triangle is never read, and a NaN-filled destination
 * proves `beta == 0` overwrites rather than reading what was there.
 */
class SymmetricOpsTest {

    @Test
    fun `symv matches gemv on the full matrix and reads only the selected triangle`() {
        val rng = Random(20260910)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 2, 7, 16)) {
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val x = randomVector(n, rng)
                        val y0 = DoubleArray(n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = y0.copyOf()
                        koblas.gemv(alpha, full, x, beta, expected)
                        val actual = y0.copyOf()
                        koblas.symv(alpha, poisoned, x, beta, actual, lower)
                        assertClose(expected, actual, "symv n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `symm matches gemm on the full matrix and reads only the selected triangle`() {
        val rng = Random(20260911)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 5, 12)) {
                val p = 4
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val b = randomMatrix(n, p, rng)
                        val c0 = DoubleArray(n * p) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = DenseMatrix(n, p, c0.copyOf())
                        koblas.gemm(alpha, full, transposeA = false, b, transposeB = false, beta, expected)
                        val actual = DenseMatrix(n, p, c0.copyOf())
                        koblas.symm(alpha, poisoned, b, beta, actual, lower)
                        assertClose(expected, actual, "symm n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `symm right multiplies from the right and reads only the selected triangle`() {
        val rng = Random(20260942)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (alpha in doubleArrayOf(0.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                    val b = randomMatrix(rows, n, rng)
                    val c0 = DoubleArray(rows * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.gemm(alpha, b, false, full, false, beta, expected)
                    val actual = DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.symm(alpha, poisoned, b, beta, actual, lower, right = true)
                    assertClose(expected, actual, "symm right l=$lower a=$alpha b=$beta", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `syrk triangle modes write only the selected triangle with strict beta semantics`() {
        val rng = Random(20260933)
        for (uplo in listOf(Uplo.LOWER, Uplo.UPPER)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        checkSyrk(rng, uplo, transpose, alpha, beta)
                    }
                }
            }
        }
    }

    @Suppress("CyclomaticComplexMethod")
    private fun checkSyrk(rng: Random, uplo: Uplo, transpose: Boolean, alpha: Double, beta: Double) {
        val a = randomMatrix(6, 4, rng)
        val n = if (transpose) a.cols else a.rows
        // The alpha term alone, from the FULL mode with a clean output.
        val term = DenseMatrix(n, n)
        koblas.syrk(1.0, a, transpose, 0.0, term)
        // Selected entries are NaN when beta == 0 (overwrite without reading); the unselected strict
        // triangle is always NaN and must survive untouched.
        val c = DenseMatrix(n, n)
        val c0 = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (uplo == Uplo.LOWER) j <= i else j >= i
                val v = if (!selected || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                c[i, j] = v
                c0[i, j] = v
            }
        }
        koblas.syrk(alpha, a, transpose, beta, c, uplo)
        val context = "syrk $uplo t=$transpose a=$alpha b=$beta"
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (uplo == Uplo.LOWER) j <= i else j >= i
                if (selected) {
                    val expected = (if (beta == 0.0) 0.0 else beta * c0[i, j]) + alpha * term[i, j]
                    assertClose(expected, c[i, j], "$context ($i;$j)")
                } else {
                    assertTrue(c[i, j].isNaN(), "$context ($i;$j): untouched triangle was written")
                }
            }
        }
    }

    @Test
    fun `symmetric ops handle empty shapes`() {
        koblas.symm(1.0, DenseMatrix(0, 0), DenseMatrix(0, 3), 0.0, DenseMatrix(0, 3))
        val c = DenseMatrix(2, 0)
        koblas.symm(1.0, DenseMatrix(2, 2), DenseMatrix(2, 0), 0.0, c)
        val y = DoubleArray(2) { Double.NaN }
        koblas.symv(0.0, DenseMatrix(2, 2), DoubleArray(2), 0.0, y)
        assertTrue(y.all { it == 0.0 }, "symv alpha=0 beta=0 left ${y.toList()}")
    }
}
