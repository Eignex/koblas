package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.koblas
import com.eignex.koblas.times
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DenseBlasTest {

    private val eps = 2.220446049250313e-16

    private fun infNorm(v: DoubleArray): Double {
        var m = 0.0
        for (x in v) m = maxOf(m, abs(x))
        return m
    }

    private fun infNorm(a: DenseMatrix): Double {
        var m = 0.0
        for (i in 0 until a.rows) {
            var r = 0.0
            for (j in 0 until a.cols) r += abs(a[i, j])
            m = maxOf(m, r)
        }
        return m
    }

    @Test
    fun `full gemv agrees with the scalar oracle across alpha beta and transpose`() = withDenseBlas { blas ->
        val rng = Random(20260727)
        for (n in intArrayOf(1, 4, 17)) {
            val m = n + 3 // non-square to catch row/col mixups
            val a = DenseMatrix(m, n, DoubleArray(m * n) { rng.nextDouble(-1.0, 1.0) })
            for (transpose in booleanArrayOf(false, true)) {
                val xLen = if (transpose) m else n
                val yLen = if (transpose) n else m
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        val x = DoubleArray(xLen) { rng.nextDouble(-2.0, 2.0) }
                        // beta == 0 must overwrite without reading, so y starts poisoned with NaN.
                        val y0 = DoubleArray(yLen) { if (beta == 0.0) Double.NaN else rng.nextDouble(-2.0, 2.0) }
                        val expected = y0.copyOf()
                        ReferenceBlas.gemv(alpha, a, x, beta, expected, transpose)
                        val y = y0.copyOf()
                        blas.gemv(alpha, a, x, beta, y, transpose)
                        val bound = 100.0 * maxOf(m, n) * eps * (infNorm(a) * infNorm(x) + infNorm(expected))
                        for (i in 0 until yLen) {
                            assertTrue(
                                abs(y[i] - expected[i]) <= bound + 1e-12,
                                "gemv n=$n t=$transpose a=$alpha b=$beta at $i: ${y[i]} vs ${expected[i]}",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `gemv and gemm reject incompatible shapes`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix(2, 3) * DenseMatrix(2, 2) }
        assertFailsWith<DimensionMismatch> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<DimensionMismatch> {
            koblas.gemv(DenseMatrix(2, 3), DoubleArray(3), transpose = true)
        }
    }

    @Test
    fun `gemm agrees with the scalar oracle for a tall result in every mode`() = withDenseBlas { blas ->
        checkGemmShape(blas, Random(20260728), m = 5, k = 7, n = 4)
    }

    @Test
    fun `gemm agrees with the scalar oracle for a wide result in every mode`() = withDenseBlas { blas ->
        checkGemmShape(blas, Random(20260728), m = 3, k = 7, n = 6)
    }

    private fun checkGemmShape(blas: DenseBlas, rng: Random, m: Int, k: Int, n: Int) {
        for (tA in booleanArrayOf(false, true)) {
            for (tB in booleanArrayOf(false, true)) {
                val a = if (tA) DenseMatrix(k, m) else DenseMatrix(m, k)
                val b = if (tB) DenseMatrix(n, k) else DenseMatrix(k, n)
                for (idx in a.values.indices) a.values[idx] = rng.nextDouble(-1.0, 1.0)
                for (idx in b.values.indices) b.values[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -2.0)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // beta == 0 must overwrite without reading, so C starts poisoned with NaN.
                        val c0 = DenseMatrix(m, n)
                        for (idx in c0.values.indices) {
                            c0.values[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                        }
                        val expected = DenseMatrix(m, n, c0.values.copyOf())
                        ReferenceBlas.gemm(alpha, a, tA, b, tB, beta, expected)
                        val c = DenseMatrix(m, n, c0.values.copyOf())
                        blas.gemm(alpha, a, tA, b, tB, beta, c)
                        val bound = 100.0 * k * eps * (infNorm(a) * infNorm(b) + infNorm(expected))
                        for (idx in c.values.indices) {
                            assertTrue(
                                abs(c.values[idx] - expected.values[idx]) <= bound + 1e-12,
                                "gemm tA=$tA tB=$tB a=$alpha b=$beta at $idx: " +
                                    "${c.values[idx]} vs ${expected.values[idx]}",
                            )
                        }
                    }
                }
            }
        }
    }
}
