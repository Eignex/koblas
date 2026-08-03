package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The QR factorization and the two solves built on it: least squares for overdetermined systems, and the
 * minimum-norm solve for underdetermined ones.
 *
 * The factorization keeps `Q` implicit as Householder reflectors, so it is verified through what it does
 * rather than by materializing it: reflectors applied to the columns of `R` must rebuild the original
 * matrix, and applying `Q` then `Q-transpose` must be the identity. The solves are checked against their
 * defining properties — an orthogonal residual for least squares, the pseudoinverse for minimum norm —
 * since a wrong implementation can still return a plausible-looking vector that is not the optimal one.
 */
class LinearAlgebraQrTest {

    /** Column j of R (rows 0 until min(j+1, k)) padded to length m. */
    private fun rColumn(qr: QrDecomposition, j: Int): DoubleArray {
        val col = DoubleArray(qr.m)
        for (i in 0 until minOf(j + 1, qr.tau.size)) col[i] = qr.qr[i + j * qr.m]
        return col
    }

    @Test
    fun `Q times R reconstructs the matrix for square tall and wide shapes`() {
        val rng = Random(20260920)
        for ((m, n) in listOf(1 to 1, 5 to 5, 9 to 4, 4 to 9, 12 to 7)) {
            val a = randomMatrix(m, n, rng)
            val f = koblas.qr(a)
            for (j in 0 until n) {
                val rebuilt = koblas.applyQ(f, rColumn(f, j))
                val expected = DoubleArray(m) { i -> a[i, j] }
                assertClose(expected, rebuilt, "reconstruct ${m}x$n col $j", tolerance = 1e-11)
            }
        }
    }

    @Test
    fun `applying Q then Q transpose round-trips`() {
        val rng = Random(20260921)
        val f = koblas.qr(randomMatrix(8, 5, rng))
        val y = randomVector(8, rng)
        val roundTrip = koblas.applyQ(f, koblas.applyQ(f, y, transpose = true))
        assertClose(y, roundTrip, "orthogonality")
    }

    @Test
    fun `least squares satisfies the normal equations`() {
        val rng = Random(20260922)
        val m = 10
        val n = 4
        val a = randomMatrix(m, n, rng)
        val b = randomVector(m, rng)
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
        val a = wellConditioned(n, rng)
        val b = randomVector(n, rng)
        val viaQr = koblas.solveLeastSquares(koblas.qr(a), b)
        val viaLu = koblas.solve(a.lu(), b)
        assertClose(viaLu, viaQr, "square system", tolerance = 1e-10)
    }

    @Test
    fun `exact solutions are recovered on tall systems`() {
        val rng = Random(20260924)
        val m = 9
        val n = 3
        val a = randomMatrix(m, n, rng)
        val xTrue = randomVector(n, rng)
        val b = koblas.gemv(a, xTrue)
        val x = koblas.solveLeastSquares(koblas.qr(a), b)
        assertClose(xTrue, x, "consistent system", tolerance = 1e-11)
    }

    @Test
    fun `minimum norm solves wide systems exactly with the smallest norm`() {
        val rng = Random(20260960)
        for ((m, n) in listOf(2 to 4, 3 to 7, 5 to 6)) {
            val a = randomMatrix(m, n, rng)
            val b = randomVector(m, rng)
            val x = koblas.solveMinimumNorm(koblas.qr(a.transpose()), b)
            // Exact solution: A · x = b.
            assertClose(b, koblas.gemv(a, x), "residual ${m}x$n", tolerance = 1e-11)
            // Minimum norm: x equals the pseudoinverse solution Aᵀ·(A·Aᵀ)⁻¹·b.
            val g = DenseMatrix(m, m)
            koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = g)
            val z = koblas.solve(koblas.factor(g), b)
            val expected = koblas.gemv(a, z, transpose = true)
            assertClose(expected, x, "min-norm ${m}x$n", tolerance = 1e-10)
        }
    }

    @Test
    fun `minimum norm on a square system agrees with the LU solve`() {
        val rng = Random(20260961)
        val n = 6
        val a = wellConditioned(n, rng)
        val b = randomVector(n, rng)
        val viaLu = koblas.solve(koblas.factor(a), b)
        val viaMinNorm = koblas.solveMinimumNorm(koblas.qr(a.transpose()), b)
        assertClose(viaLu, viaMinNorm, "square", tolerance = 1e-10)
    }

    @Test
    fun `degenerate shapes and wide rejection`() {
        val empty = koblas.qr(DenseMatrix(0, 0))
        assertTrue(koblas.applyQ(empty, DoubleArray(0)).isEmpty())
        assertTrue(koblas.solveLeastSquares(empty, DoubleArray(0)).isEmpty())
        val wide = koblas.qr(DenseMatrix(2, 4))
        assertFailsWith<IllegalArgumentException> { koblas.solveLeastSquares(wide, DoubleArray(2)) }
        // The minimum-norm solve needs the QR of Aᵀ (tall); the wide QR of A itself is a misuse.
        assertFailsWith<IllegalArgumentException> { koblas.solveMinimumNorm(wide, DoubleArray(4)) }
    }
}
