package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
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
    fun `full gemv matches the naive reference across alpha beta and transpose`() {
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
                        koblas.gemv(alpha, a, x, beta, y, transpose, workspace = Workspace())
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
    fun `gemv forms zero products from nonzero alpha`() {
        val a = DenseMatrix(2, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0, 2.0, 3.0))
        val y = DoubleArray(2)

        ReferenceBlas.gemv(1.0, a, doubleArrayOf(0.0, 1.0), 0.0, y)

        assertTrue(y[0].isNaN(), "zero times infinity was ${y[0]}")
        assertEquals(3.0, y[1])
    }

    @Test
    fun `gemm reproduces the identity and is associative within tolerance`() {
        val rng = Random(3)
        for (n in intArrayOf(1, 4, 16)) {
            val a = wellConditioned(n, rng)
            val id = DenseMatrix.diagonal(n)
            assertEquals(a * id, a, "A·I != A at n=$n")
            val b = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val c = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val left = a * b * c
            val right = a * (b * c)
            val bound = 100.0 * n * eps * infNorm(a) * infNorm(b) * infNorm(c)
            var maxDiff = 0.0
            for (k in left.data.indices) maxDiff = maxOf(maxDiff, abs(left.data[k] - right.data[k]))
            assertTrue(maxDiff <= bound + 1e-9, "gemm associativity n=$n: $maxDiff > $bound")
        }
    }

    @Test
    fun `gemm preserves a rectangular result with zero depth`() {
        assertEquals(DenseMatrix(2, 3), DenseMatrix(2, 0) * DenseMatrix(0, 3))
    }

    @Test
    fun `gemv and gemm reject incompatible shapes`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3) * DenseMatrix(2, 2) }
        assertFailsWith<IllegalArgumentException> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<IllegalArgumentException> {
            koblas.gemv(DenseMatrix(2, 3), DoubleArray(3), transpose = true)
        }
    }

    @Test
    fun `gemm forms zero products in every transpose mode`() {
        val a = DenseMatrix.diagonal(2).also { it[0, 0] = Double.POSITIVE_INFINITY }
        val b = DenseMatrix(2, 2)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val c = DenseMatrix(2, 2)

                ReferenceBlas.gemm(1.0, a, transposeA, b, transposeB, 0.0, c)

                assertTrue(c[0, 0].isNaN(), "transposeA=$transposeA transposeB=$transposeB produced ${c[0, 0]}")
            }
        }
    }

    @Test
    fun `full gemm matches the naive reference across transpose flags alpha and beta`() {
        val rng = Random(20260728)
        // Two shapes, so the doubly-transposed case transposes A in one and B in the other: it copies
        // whichever operand is smaller, and one shape alone would only ever reach one of those.
        for ((m, k, n) in listOf(Triple(5, 7, 4), Triple(3, 7, 6))) {
            checkGemmShape(rng, m, k, n)
        }
    }

    @Test
    fun `gemm agrees with a naive product across every cache tile boundary`() {
        val rng = Random(20261031)
        val m = LEVEL3_BLOCK_ROWS + 7
        val k = LEVEL3_BLOCK_DEPTH + 3
        val n = LEVEL3_BLOCK_COLUMNS + 3
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val a = if (transposeA) randomMatrix(k, m, rng) else randomMatrix(m, k, rng)
                val b = if (transposeB) randomMatrix(n, k, rng) else randomMatrix(k, n, rng)
                val expected = DenseMatrix(m, n)
                for (j in 0 until n) {
                    for (i in 0 until m) {
                        var sum = 0.0
                        for (p in 0 until k) {
                            val av = if (transposeA) a[p, i] else a[i, p]
                            val bv = if (transposeB) b[j, p] else b[p, j]
                            sum += av * bv
                        }
                        expected[i, j] = sum
                    }
                }
                val actual = DenseMatrix(m, n)
                ReferenceBlas.gemm(
                    1.0,
                    a,
                    transposeA,
                    b,
                    transposeB,
                    0.0,
                    actual,
                    workspace = Workspace(),
                )
                assertClose(expected, actual, "gemm tA=$transposeA tB=$transposeB", tolerance = 1e-10)
            }
        }
    }

    private fun checkGemmShape(rng: Random, m: Int, k: Int, n: Int) {
        for (tA in booleanArrayOf(false, true)) {
            for (tB in booleanArrayOf(false, true)) {
                val a = if (tA) DenseMatrix(k, m) else DenseMatrix(m, k)
                val b = if (tB) DenseMatrix(n, k) else DenseMatrix(k, n)
                for (idx in a.data.indices) a.data[idx] = rng.nextDouble(-1.0, 1.0)
                for (idx in b.data.indices) b.data[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -2.0)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // beta == 0 must overwrite without reading, so C starts poisoned with NaN.
                        val c0 = DenseMatrix(m, n)
                        for (idx in c0.data.indices) {
                            c0.data[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
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
                        val c = DenseMatrix(m, n, c0.data.copyOf())
                        koblas.gemm(alpha, a, tA, b, tB, beta, c)
                        val bound = 100.0 * k * eps * (infNorm(a) * infNorm(b) + infNorm(expected))
                        for (idx in c.data.indices) {
                            assertTrue(
                                abs(c.data[idx] - expected.data[idx]) <= bound + 1e-12,
                                "gemm tA=$tA tB=$tB a=$alpha b=$beta at $idx: ${c.data[idx]} vs ${expected.data[idx]}",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `syrk matches gemm over the selected triangle`() {
        val rng = Random(20260731)
        for ((n, k) in listOf(1 to 1, 4 to 7, 9 to 3)) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = if (transpose) DenseMatrix(k, n) else DenseMatrix(n, k)
                for (idx in a.data.indices) a.data[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // C is asymmetric on purpose, since only syrk's alpha term is symmetric.
                        val c0 = DenseMatrix(n, n)
                        for (idx in c0.data.indices) {
                            c0.data[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                        }
                        val expected = DenseMatrix(n, n, if (beta == 0.0) DoubleArray(n * n) else c0.data.copyOf())
                        koblas.gemm(alpha, a, transpose, a, !transpose, if (beta == 0.0) 0.0 else beta, expected)
                        val c = DenseMatrix(n, n, c0.data.copyOf())
                        koblas.syrk(alpha, a, transpose, beta, c)
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
