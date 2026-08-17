package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.poisonedIndefinite
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearAlgebraLdlTest {

    @Test
    fun `ldl solve matches LU solve on indefinite matrices and reads only the lower triangle`() {
        val rng = Random(20260930)
        for (n in intArrayOf(1, 2, 3, 8, 21)) {
            val (full, poisoned) = poisonedIndefinite(rng, n)
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val viaLu = koblas.solve(full.lu(), b)
            val f = koblas.ldl(poisoned)
            assertTrue(!f.singular, "n=$n flagged singular")
            val viaLdl = koblas.solve(f, b)
            assertClose(viaLu, viaLdl, "ldl n=$n", tolerance = 1e-9)
        }
    }

    @Test
    fun `zero diagonal forces a two-by-two pivot`() {
        val a = DenseMatrix(2)
        a[1, 0] = 1.0
        a[0, 1] = Double.NaN
        val f = koblas.ldl(a)
        assertTrue(!f.singular)
        assertTrue(f.ipiv[0] < 0 && f.ipiv[1] == f.ipiv[0], "expected a 2x2 block, got ${f.ipiv.toList()}")
        val x = koblas.solve(f, doubleArrayOf(2.0, 3.0))
        assertClose(doubleArrayOf(3.0, 2.0), x, "antidiagonal solve", tolerance = 1e-14)
    }

    @Test
    fun `spd matrices agree with the Cholesky solve`() {
        val rng = Random(20260931)
        val n = 9
        val a = DenseMatrix(n)
        for (i in 0 until n) {
            for (j in 0..i) {
                val v = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
                a[i, j] = v
                a[j, i] = v
            }
        }
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val viaCholesky = a.cholesky().solve(b)
        val viaLdl = koblas.solve(koblas.ldl(a), b)
        assertClose(viaCholesky, viaLdl, "spd", tolerance = 1e-10)
    }

    @Test
    fun `singular and empty conventions`() {
        assertTrue(koblas.ldl(DenseMatrix(3, 3)).singular)
        val empty = koblas.ldl(DenseMatrix(0, 0))
        assertTrue(!empty.singular)
        assertTrue(koblas.solve(empty, DoubleArray(0)).isEmpty())
        assertEquals(0, koblas.ldl(DenseMatrix(3, 3)).failedAt, "the first of three zero pivots")
        assertEquals(NOT_SINGULAR, empty.failedAt)
    }
}
