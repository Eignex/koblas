package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.koblas
import com.eignex.koblas.times
import com.eignex.koblas.wellConditioned
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BlasTest {

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
    fun `full gemv matches the naive reference across alpha beta and transpose`() = withDenseBlas { blas ->
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
                        val expected = DoubleArray(yLen) { i ->
                            var s = 0.0
                            for (k in 0 until xLen) s += (if (transpose) a[k, i] else a[i, k]) * x[k]
                            alpha * s + (if (beta == 0.0) 0.0 else beta * y0[i])
                        }
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
    fun `gemm reproduces the identity and is associative within tolerance`() = withDenseBlas { blas ->
        fun product(x: DenseMatrix, y: DenseMatrix) =
            DenseMatrix(x.rows, y.cols).also { blas.gemm(1.0, x, false, y, false, 0.0, it) }

        val rng = Random(3)
        for (n in intArrayOf(1, 4, 16)) {
            val a = wellConditioned(n, rng)
            val id = DenseMatrix.diagonal(n)
            assertEquals(product(a, id), a, "A·I != A at n=$n")
            val b = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val c = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val left = product(product(a, b), c)
            val right = product(a, product(b, c))
            val bound = 100.0 * n * eps * infNorm(a) * infNorm(b) * infNorm(c)
            var maxDiff = 0.0
            for (k in left.values.indices) maxDiff = maxOf(maxDiff, abs(left.values[k] - right.values[k]))
            assertTrue(maxDiff <= bound + 1e-9, "gemm associativity n=$n: $maxDiff > $bound")
        }
    }

    @Test
    fun `gemm preserves a rectangular result with zero depth`() = withDenseBlas { blas ->
        val c = DenseMatrix(2, 3)
        blas.gemm(1.0, DenseMatrix(2, 0), false, DenseMatrix(0, 3), false, 0.0, c)
        assertEquals(DenseMatrix(2, 3), c)
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
    fun `full gemm matches the naive reference across transpose flags alpha and beta`() = withDenseBlas { blas ->
        val rng = Random(20260728)
        // Two shapes, so a row/column mixup that happens to be harmless in one of them still shows in the
        // other, which one square shape on its own would hide.
        for ((m, k, n) in listOf(Triple(5, 7, 4), Triple(3, 7, 6))) {
            checkGemmShape(blas, rng, m, k, n)
        }
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
                        val expected = DenseMatrix(m, n)
                        for (i in 0 until m) {
                            for (j in 0 until n) {
                                var s = 0.0
                                for (p in 0 until k) {
                                    s += (if (tA) a[p, i] else a[i, p]) * (if (tB) b[j, p] else b[p, j])
                                }
                                expected[i, j] = alpha * s + (if (beta == 0.0) 0.0 else beta * c0[i, j])
                            }
                        }
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

    @Test
    fun `syrk matches gemm over the selected triangle`() = withDenseBlas { blas ->
        val rng = Random(20260731)
        for ((n, k) in listOf(1 to 1, 4 to 7, 9 to 3)) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = if (transpose) DenseMatrix(k, n) else DenseMatrix(n, k)
                for (idx in a.values.indices) a.values[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // C is asymmetric on purpose, since only syrk's alpha term is symmetric.
                        val c0 = DenseMatrix(n, n)
                        for (idx in c0.values.indices) {
                            c0.values[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                        }
                        val expected = DenseMatrix(n, n, if (beta == 0.0) DoubleArray(n * n) else c0.values.copyOf())
                        blas.gemm(alpha, a, transpose, a, !transpose, if (beta == 0.0) 0.0 else beta, expected)
                        val c = DenseMatrix(n, n, c0.values.copyOf())
                        blas.syrk(alpha, a, transpose, beta, c)
                        val bound = 100.0 * k * eps * (infNorm(a) * infNorm(a) + infNorm(expected)) + 1e-12
                        for (j in 0 until n) {
                            for (i in j until n) {
                                assertTrue(
                                    abs(c[i, j] - expected[i, j]) <= bound,
                                    "syrk n=$n k=$k t=$transpose a=$alpha b=$beta at ($i,$j)",
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
