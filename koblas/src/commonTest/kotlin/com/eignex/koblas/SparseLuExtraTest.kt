package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseLuExtraTest {

    private fun square(vararg rows: DoubleArray): SparseMatrix {
        val n = rows.size
        val cols = List(n) { j -> (0 until n).mapNotNull { i -> if (rows[i][j] != 0.0) i to rows[i][j] else null } }
        return SparseMatrix.ofColumns(n, n, cols)
    }

    @Test
    fun `bounded pivot search stays accurate and sparse on a larger random system`() {
        val rng = Random(20260727)
        val n = 300
        val columns = List(n) { j ->
            buildList {
                add(j to (rng.nextDouble(-1.0, 1.0) + n)) // dominant diagonal
                for (i in 0 until n) if (i != j && rng.nextDouble() < 0.02) add(i to rng.nextDouble(-1.0, 1.0))
            }
        }
        val a = SparseMatrix.ofColumns(n, n, columns)
        val lu = assertNotNull(SparseLu.factorize(a, equilibrate = true))
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val x = lu.ftran(b)
        val ax = a.gemv(x)
        for (i in 0 until n) assertTrue(abs(ax[i] - b[i]) < 1e-8, "ftran residual at $i: ${ax[i] - b[i]}")
        val y = lu.btran(b)
        val aty = a.gemv(y, transpose = true)
        for (i in 0 until n) assertTrue(abs(aty[i] - b[i]) < 1e-8, "btran residual at $i: ${aty[i] - b[i]}")
        // The bounded candidate search must keep fill near the input's sparsity, far from dense.
        assertTrue(lu.nnz < 6 * a.nnz, "fill blew up: factor nnz=${lu.nnz} vs input nnz=${a.nnz}")
    }

    @Test
    fun `determinant matches the hand value and tracks sign under row order`() {
        // [[2, 1], [1, 3]] → det 5.
        val det = assertNotNull(
            SparseLu.factorize(square(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))),
        ).determinant()
        assertTrue(abs(det - 5.0) < 1e-9, "det=$det")
        // A row swap negates the determinant: [[1,3],[2,1]] → det = 1 - 6 = -5.
        val det2 = assertNotNull(
            SparseLu.factorize(square(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 1.0))),
        ).determinant()
        assertTrue(abs(det2 + 5.0) < 1e-9, "det2=$det2")
    }

    @Test
    fun `determinant is invariant to equilibration`() {
        val rng = Random(11)
        repeat(30) {
            val n = rng.nextInt(2, 6)
            val rows = Array(n) { DoubleArray(n) { rng.nextDouble(-3.0, 3.0) } }
            for (i in 0 until n) rows[i][i] += n * 4.0
            val a = square(*rows)
            val d0 = assertNotNull(SparseLu.factorize(a, equilibrate = false)).determinant()
            val d1 = assertNotNull(SparseLu.factorize(a, equilibrate = true)).determinant()
            assertTrue(abs(d0 - d1) <= 1e-6 * (abs(d0) + 1.0), "det differs: $d0 vs $d1")
        }
    }

    @Test
    fun `solve is correct with equilibration off`() {
        val rng = Random(22)
        repeat(60) {
            val n = rng.nextInt(1, 8)
            val rows = Array(n) { DoubleArray(n) { rng.nextDouble(-2.0, 2.0) } }
            for (i in 0 until n) rows[i][i] += n * 5.0
            val a = square(*rows)
            val lu = assertNotNull(SparseLu.factorize(a, equilibrate = false))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val solved = lu.ftran(a.gemv(x))
            for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-7)
        }
    }

    @Test
    fun `factorize rejects a non-square matrix`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))
        assertFailsWith<IllegalArgumentException> { SparseLu.factorize(a) }
    }

    @Test
    fun `a duplicated column is singular`() {
        assertNull(SparseLu.factorize(square(doubleArrayOf(1.0, 1.0), doubleArrayOf(2.0, 2.0))))
    }

    @Test
    fun `EtaBasis solves and transpose-solves after multiple rank-1 updates`() {
        val n = 4
        val base = assertNotNull(SparseLu.factorize(SparseMatrix.ofColumns(n, n, List(n) { listOf(it to 1.0) })))
        val eta = EtaBasis.of(base, n)

        // Track the dense basis alongside the eta chain: start from identity, replace columns 1 then 3.
        val basis = Array(n) { j -> DoubleArray(n) { i -> if (i == j) 1.0 else 0.0 } }
        val updates = listOf(1 to doubleArrayOf(2.0, 1.0, 0.0, -1.0), 3 to doubleArrayOf(0.5, 0.0, 3.0, 2.0))
        for ((p, q) in updates) {
            eta.update(p, eta.ftran(q))
            for (i in 0 until n) basis[i][p] = q[i]
        }
        assertEquals(2, eta.etaCount)

        val x = doubleArrayOf(1.0, -2.0, 4.0, 0.5)
        val b = DoubleArray(n) { i -> (0 until n).sumOf { j -> basis[i][j] * x[j] } } // B x
        val solved = eta.ftran(b)
        for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-9, "ftran component $i")

        val bt = DoubleArray(n) { j -> (0 until n).sumOf { i -> basis[i][j] * x[i] } } // Bᵀ x
        val solvedT = eta.btran(bt)
        for (i in 0 until n) assertTrue(abs(solvedT[i] - x[i]) < 1e-9, "btran component $i")
    }
}
