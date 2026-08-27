package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.host.rColumn
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

class LinearAlgebraQrTest {

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
    fun `a column already in triangular form needs no reflector`() {
        val f = koblas.qr(F64DenseMatrix.diagonal(3))
        assertTrue(f.tau.all { it == 0.0 }, "expected no reflectors, got ${f.tau.toList()}")
        assertClose(F64DenseMatrix.diagonal(3), F64DenseMatrix(3, 3, f.qr), "R", tolerance = 1e-15)
        assertClose(
            doubleArrayOf(1.0, 0.0, 0.0),
            koblas.applyQ(f, doubleArrayOf(1.0, 0.0, 0.0)),
            "Q is the identity",
            tolerance = 1e-15,
        )
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
        val x = koblas.solve(koblas.qr(a), b)
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
        val viaQr = koblas.solve(koblas.qr(a), b)
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
        val x = koblas.solve(koblas.qr(a), b)
        assertClose(xTrue, x, "consistent system", tolerance = 1e-11)
    }

    @Test
    fun `minimum norm solves wide systems exactly with the smallest norm`() {
        val rng = Random(20260960)
        for ((m, n) in listOf(2 to 4, 3 to 7, 5 to 6)) {
            val a = randomMatrix(m, n, rng)
            val b = randomVector(m, rng)
            val x = koblas.solve(koblas.qr(a.transpose()), b, minimumNorm = true)
            assertClose(b, koblas.gemv(a, x), "residual ${m}x$n", tolerance = 1e-11)
            // The minimum-norm solution is the pseudoinverse solution At (A At)^-1 b.
            val g = F64DenseMatrix(m, m)
            koblas.gemm(1.0, a, false, a, true, 0.0, g)
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
        val viaMinNorm = koblas.solve(koblas.qr(a.transpose()), b, minimumNorm = true)
        assertClose(viaLu, viaMinNorm, "square", tolerance = 1e-10)
    }

    @Test
    fun `degenerate shapes and wide rejection`() {
        val empty = koblas.qr(F64DenseMatrix(0, 0))
        assertTrue(koblas.applyQ(empty, DoubleArray(0)).isEmpty())
        assertTrue(koblas.solve(empty, DoubleArray(0)).isEmpty())
        val wide = koblas.qr(F64DenseMatrix(2, 4))
        assertFailsWith<DimensionMismatch> { koblas.solve(wide, DoubleArray(2)) }
        assertFailsWith<DimensionMismatch> { koblas.solve(wide, DoubleArray(4), minimumNorm = true) }
    }
}
