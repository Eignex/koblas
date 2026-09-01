package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.random.Random
import kotlin.test.*

class LinearAlgebraMultiRhsSolveTest {

    @Test
    fun `lu block solve matches column-by-column vector solves`() {
        val rng = Random(20260950)
        for (n in intArrayOf(1, 4, 12)) {
            val a = F64DenseMatrix(n)
            for (i in 0 until n) {
                for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
            }
            val lu = koblas.factor(a)
            val nrhs = 3
            val b = F64DenseMatrix(n, nrhs)
            for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
            for (transpose in booleanArrayOf(false, true)) {
                val block = koblas.solve(lu, b, transpose)
                assertEquals(n, block.rows)
                assertEquals(nrhs, block.cols)
                for (c in 0 until nrhs) {
                    val col = DoubleArray(n) { b[it, c] }
                    val x = koblas.solve(lu, col, transpose)
                    for (i in 0 until n) {
                        assertClose(
                            doubleArrayOf(x[i]),
                            doubleArrayOf(block[i, c]),
                            "lu n=$n t=$transpose col=$c",
                            tolerance = 1e-11,
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `ldl block solve matches column-by-column vector solves on indefinite matrices`() {
        val rng = Random(20260951)
        for (n in intArrayOf(1, 2, 5, 14)) {
            val (_, a) = poisonedIndefinite(rng, n)
            val f = koblas.pivotedSymmetricIndefinite(a)
            assertTrue(!f.singular, "n=$n flagged singular")
            val nrhs = 3
            val b = F64DenseMatrix(n, nrhs)
            for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
            val block = koblas.solve(f, b)
            for (c in 0 until nrhs) {
                val col = DoubleArray(n) { b[it, c] }
                val x = koblas.solve(f, col)
                for (i in 0 until n) {
                    assertClose(doubleArrayOf(x[i]), doubleArrayOf(block[i, c]), "ldl n=$n col=$c", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `antidiagonal two-by-two pivot solves exactly for a block`() {
        val a = F64DenseMatrix(2)
        a[1, 0] = 1.0
        a[0, 1] = Double.NaN
        val f = koblas.pivotedSymmetricIndefinite(a)
        // A swaps the rows, so the column-major columns (2, 3) and (1, -1) map to (3, 2) and (-1, 1).
        val b = F64DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 1.0, -1.0))
        val x = koblas.solve(f, b)
        assertClose(doubleArrayOf(3.0, 2.0, -1.0, 1.0), x.data, "antidiagonal block", tolerance = 1e-11)
    }

    @Test
    fun `degenerate shapes round-trip`() {
        val empty = koblas.factor(F64DenseMatrix(0, 0))
        val x0 = koblas.solve(empty, F64DenseMatrix(0, 3))
        assertEquals(0, x0.rows)
        assertEquals(3, x0.cols)
        val lu = koblas.factor(F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0))))
        val xn = koblas.solve(lu, F64DenseMatrix(1, 0))
        assertEquals(1, xn.rows)
        assertEquals(0, xn.cols)
        val ldl0 = koblas.pivotedSymmetricIndefinite(F64DenseMatrix(0, 0))
        assertEquals(0, koblas.solve(ldl0, F64DenseMatrix(0, 2)).rows)
    }

    /**
     * The per-column callback is the caller's, and a native solve in it reports failure by throwing, so the
     * two borrows must come back. A stranded borrow is invisible: its pool can neither lend that buffer
     * again nor be reclaimed, so this asks for both back.
     */
    @Test
    fun `a column solve that throws hands its borrows back`() {
        val ws = Workspace()
        val n = 4
        val lent = listOf(ws.take(n), ws.take(n))
        assertTrue(lent[0] !== lent[1], "two outstanding borrows shared a buffer")
        lent.forEach { ws.release(it) }
        assertFailsWith<IllegalStateException> {
            solveColumnwise(F64DenseMatrix(n, 2), F64DenseMatrix(n, 2), n, 2, ws) { _, _ ->
                error("the native solve failed")
            }
        }
        for (buffer in listOf(ws.take(n), ws.take(n))) {
            assertTrue(buffer === lent[0] || buffer === lent[1], "the failed solve kept a borrow")
        }
    }
}
