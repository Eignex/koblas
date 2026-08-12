package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.gemv
import com.eignex.koblas.sparse.lu
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class BlasConformanceTest {

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

    private fun hilbert(n: Int): DenseMatrix =
        DenseMatrix(n, n).also { for (i in 0 until n) for (j in 0 until n) it[i, j] = 1.0 / (i + j + 1.0) }

    private fun diagonal(n: Int, rng: Random): DenseMatrix =
        DenseMatrix(n, n).also { for (i in 0 until n) it[i, i] = rng.nextDouble(1.0, 5.0) }

    private fun randomWellConditioned(n: Int, rng: Random): DenseMatrix =
        DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            .also { for (i in 0 until n) it[i, i] = it[i, i] + n } // diagonally dominant

    private fun spd(n: Int, rng: Random): DenseMatrix {
        val a = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        val m = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                var s = 0.0
                for (k in 0 until n) s += a[k, i] * a[k, j]
                m[i, j] = s + if (i == j) n.toDouble() else 0.0
            }
        }
        return m
    }

    // The bound scales the residual by the norms of A, x and b and by n times the unit roundoff. Only the
    // residual is checked, not the forward error, which the condition number of a Hilbert matrix inflates.
    private fun assertSolveResidual(a: DenseMatrix, x: DoubleArray, b: DoubleArray, name: String) {
        val ax = koblas.gemv(a, x)
        val residual = DoubleArray(a.rows) { ax[it] - b[it] }
        val bound = 100.0 * a.rows * eps * (infNorm(a) * infNorm(x) + infNorm(b))
        assertTrue(infNorm(residual) <= bound + 1e-12, "$name: residual ${infNorm(residual)} > bound $bound")
    }

    @Test
    fun `dense LU solve has a small residual on the standard test matrices`() {
        val rng = Random(20260716)
        for (n in intArrayOf(1, 2, 5, 12, 40)) {
            val matrices = listOf(
                "identity" to DenseMatrix.diagonal(n),
                "diagonal" to diagonal(n, rng),
                "hilbert" to hilbert(n),
                "random" to randomWellConditioned(n, rng),
                "spd" to spd(n, rng),
            )
            for ((label, a) in matrices) {
                val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                val b = koblas.gemv(a, xTrue)
                val lu = a.lu()
                if (lu.singular) continue
                val x = lu.solve(b)
                assertSolveResidual(a, x, b, "LU/$label/n=$n")
            }
        }
    }

    @Test
    fun `dense LU transpose-solve has a small residual`() {
        val rng = Random(99)
        for (n in intArrayOf(2, 7, 25)) {
            val a = randomWellConditioned(n, rng)
            val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            val b = koblas.gemv(a, xTrue, transpose = true)
            val x = a.lu().solve(b, transpose = true)
            val atx = koblas.gemv(a, x, transpose = true)
            val residual = DoubleArray(n) { atx[it] - b[it] }
            val bound = 100.0 * n * eps * (infNorm(a) * infNorm(x) + infNorm(b))
            assertTrue(infNorm(residual) <= bound + 1e-12, "transpose-solve n=$n: ${infNorm(residual)} > $bound")
        }
    }

    @Test
    fun `cholesky solve has a small residual on SPD matrices`() {
        val rng = Random(7)
        for (n in intArrayOf(1, 3, 10, 30)) {
            val a = spd(n, rng)
            val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            val b = koblas.gemv(a, xTrue)
            val l = a.cholesky()
            val x = l.solve(b)
            assertSolveResidual(a, x, b, "cholesky/n=$n")
        }
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
                        koblas.gemv(alpha, a, x, beta, y, transpose)
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
    fun `invertSpd produces an inverse with a small identity residual`() {
        val rng = Random(20260805)
        for (n in intArrayOf(1, 3, 10, 30)) {
            val a = spd(n, rng)
            val inv = a.cholesky().invert()
            val prod = a.matMul(inv)
            var maxOffIdentity = 0.0
            for (i in 0 until n) {
                for (j in 0 until n) {
                    maxOffIdentity = maxOf(maxOffIdentity, abs(prod[i, j] - if (i == j) 1.0 else 0.0))
                }
            }
            val bound = 100.0 * n * eps * infNorm(a) * infNorm(inv)
            assertTrue(maxOffIdentity <= bound + 1e-12, "invertSpd n=$n: $maxOffIdentity > $bound")
        }
    }

    @Test
    fun `gemm reproduces the identity and is associative within tolerance`() {
        val rng = Random(3)
        for (n in intArrayOf(1, 4, 16)) {
            val a = randomWellConditioned(n, rng)
            val id = DenseMatrix.diagonal(n)
            assertTrue(a.matMul(id) == a, "A·I != A at n=$n")
            val b = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val c = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val left = a.matMul(b).matMul(c)
            val right = a.matMul(b.matMul(c))
            val bound = 100.0 * n * eps * infNorm(a) * infNorm(b) * infNorm(c)
            var maxDiff = 0.0
            for (k in left.data.indices) maxDiff = maxOf(maxDiff, abs(left.data[k] - right.data[k]))
            assertTrue(maxDiff <= bound + 1e-9, "gemm associativity n=$n: $maxDiff > $bound")
        }
    }

    @Test
    fun `full gemm matches the naive reference across transpose flags alpha and beta`() {
        val rng = Random(20260728)
        val m = 5
        val k = 7
        val n = 4
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
    fun `syrk matches gemm with a transposed operand and stays symmetric`() {
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
                        for (idx in c.data.indices) {
                            assertTrue(
                                abs(c.data[idx] - expected.data[idx]) <= bound,
                                "syrk n=$n k=$k t=$transpose a=$alpha b=$beta at $idx",
                            )
                        }
                        if (beta == 0.0 && alpha != 0.0) { // pure alpha term must be exactly symmetric
                            for (i in 0 until n) {
                                for (j in 0 until i) {
                                    assertTrue(c[i, j] == c[j, i], "syrk asymmetry at ($i,$j)")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `sparse LU solve has a small residual on standard matrices`() {
        val rng = Random(20260101)
        for (n in intArrayOf(2, 8, 30)) {
            val dense = randomWellConditioned(n, rng)
            val cols = List(n) { j ->
                (0 until n).mapNotNull { i -> if (dense[i, j] != 0.0) i to dense[i, j] else null }
            }
            val sparse = SparseMatrix.ofColumns(n, n, cols)
            val lu = sparse.lu(equilibrate = true)
            if (lu.singular) continue
            val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            val b = sparse.gemv(xTrue)
            val x = lu.solve(b)
            assertSolveResidual(dense, x, b, "sparseLU/n=$n")
        }
    }
}
