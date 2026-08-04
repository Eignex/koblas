package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class LinearAlgebraLdlTest {

    /** Random symmetric indefinite matrix (mixed-sign diagonal shifts force 2×2 pivots) with the
     *  strictly upper triangle NaN-poisoned, plus its clean full-symmetric twin. */
    private fun poisonedIndefinite(rng: Random, n: Int): Pair<DenseMatrix, DenseMatrix> {
        val full = DenseMatrix(n)
        val poisoned = DenseMatrix(n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var v = rng.nextDouble(-1.0, 1.0)
                if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
                full[i, j] = v
                full[j, i] = v
                poisoned[i, j] = v
                if (j != i) poisoned[j, i] = Double.NaN
            }
        }
        return full to poisoned
    }

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
        // [[0, 1], [1, 0]] has no nonzero 1x1 pivot: Bunch-Kaufman must take the 2x2 block.
        val a = DenseMatrix(2)
        a[1, 0] = 1.0
        a[0, 1] = Double.NaN
        val f = koblas.ldl(a)
        assertTrue(!f.singular)
        assertTrue(f.ipiv[0] < 0 && f.ipiv[1] == f.ipiv[0], "expected a 2x2 block, got ${f.ipiv.toList()}")
        // A · x = (2, 3) has the exact solution (3, 2).
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
        val viaCholesky = solveSpd(a.cholesky(), b)
        val viaLdl = koblas.solve(koblas.ldl(a), b)
        assertClose(viaCholesky, viaLdl, "spd", tolerance = 1e-10)
    }

    @Test
    fun `singular and empty conventions`() {
        assertTrue(koblas.ldl(DenseMatrix(3, 3)).singular)
        val empty = koblas.ldl(DenseMatrix(0, 0))
        assertTrue(!empty.singular)
        assertTrue(koblas.solve(empty, DoubleArray(0)).isEmpty())
    }
}
