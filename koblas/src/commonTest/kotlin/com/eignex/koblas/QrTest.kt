package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class QrTest {

    private fun randomMatrix(rng: Random, rows: Int, cols: Int) =
        DenseMatrix(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, tol: Double, context: String) {
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= tol * maxOf(1.0, abs(expected[i])),
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
    }

    /** Column j of R (rows 0 until min(j+1, k)) padded to length m. */
    private fun rColumn(qr: QrDecomposition, j: Int): DoubleArray {
        val col = DoubleArray(qr.m)
        for (i in 0 until minOf(j + 1, qr.tau.size)) col[i] = qr.qr[i * qr.n + j]
        return col
    }

    @Test
    fun `Q times R reconstructs the matrix for square tall and wide shapes`() {
        val rng = Random(20260920)
        for ((m, n) in listOf(1 to 1, 5 to 5, 9 to 4, 4 to 9, 12 to 7)) {
            val a = randomMatrix(rng, m, n)
            val f = koblas.qr(a)
            for (j in 0 until n) {
                val rebuilt = koblas.applyQ(f, rColumn(f, j))
                val expected = DoubleArray(m) { i -> a[i, j] }
                assertClose(expected, rebuilt, 1e-11, "reconstruct ${m}x$n col $j")
            }
        }
    }

    @Test
    fun `applying Q then Q transpose round-trips`() {
        val rng = Random(20260921)
        val a = randomMatrix(rng, 8, 5)
        val f = koblas.qr(a)
        val y = DoubleArray(8) { rng.nextDouble(-1.0, 1.0) }
        val roundTrip = koblas.applyQ(f, koblas.applyQ(f, y, transpose = true))
        assertClose(y, roundTrip, 1e-12, "orthogonality")
    }

    @Test
    fun `least squares satisfies the normal equations`() {
        val rng = Random(20260922)
        val m = 10
        val n = 4
        val a = randomMatrix(rng, m, n)
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val x = koblas.solveLeastSquares(koblas.qr(a), b)
        // Residual must be orthogonal to the column space: Aᵀ(A·x − b) ≈ 0.
        val ax = koblas.gemv(a, x)
        val residual = DoubleArray(m) { ax[it] - b[it] }
        val normalResidual = koblas.gemv(a, residual, transpose = true)
        for (v in normalResidual) assertTrue(abs(v) <= 1e-11, "normal equations residual $v")
    }

    @Test
    fun `least squares on a square system matches the LU solve`() {
        val rng = Random(20260923)
        val n = 7
        val a = randomMatrix(rng, n, n)
        for (i in 0 until n) a[i, i] = a[i, i] + n
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val viaQr = koblas.solveLeastSquares(koblas.qr(a), b)
        val viaLu = koblas.solve(a.lu(), b)
        assertClose(viaLu, viaQr, 1e-10, "square system")
    }

    @Test
    fun `exact solutions are recovered on tall systems`() {
        val rng = Random(20260924)
        val m = 9
        val n = 3
        val a = randomMatrix(rng, m, n)
        val xTrue = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val b = koblas.gemv(a, xTrue)
        val x = koblas.solveLeastSquares(koblas.qr(a), b)
        assertClose(xTrue, x, 1e-11, "consistent system")
    }

    @Test
    fun `degenerate shapes and wide rejection`() {
        val empty = koblas.qr(DenseMatrix(0, 0))
        assertTrue(koblas.applyQ(empty, DoubleArray(0)).isEmpty())
        assertTrue(koblas.solveLeastSquares(empty, DoubleArray(0)).isEmpty())
        val wide = koblas.qr(DenseMatrix(2, 4))
        assertFailsWith<IllegalArgumentException> { koblas.solveLeastSquares(wide, DoubleArray(2)) }
    }
}
