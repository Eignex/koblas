package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class SyrkUploTest {

    private fun assertClose(expected: Double, actual: Double, context: String) {
        assertTrue(
            abs(expected - actual) <= 1e-12 * maxOf(1.0, abs(expected)),
            "$context: expected $expected actual $actual",
        )
    }

    @Test
    fun `triangle modes write only the selected triangle with strict beta semantics`() {
        val rng = Random(20260933)
        for (uplo in listOf(Uplo.LOWER, Uplo.UPPER)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        checkCase(rng, uplo, transpose, alpha, beta)
                    }
                }
            }
        }
    }

    @Suppress("CyclomaticComplexMethod")
    private fun checkCase(rng: Random, uplo: Uplo, transpose: Boolean, alpha: Double, beta: Double) {
        val a = DenseMatrix(6, 4)
        for (i in 0 until 6) for (j in 0 until 4) a[i, j] = rng.nextDouble(-1.0, 1.0)
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
}
