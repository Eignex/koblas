package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearAlgebraMultiRhsSolveTest {

    @Test
    fun `lu block solve matches column-by-column vector solves`() {
        val rng = Random(20260950)
        for (n in intArrayOf(1, 4, 12)) {
            val a = DenseMatrix(n)
            for (i in 0 until n) {
                for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
            }
            val lu = koblas.factor(a)
            val nrhs = 3
            val b = DenseMatrix(n, nrhs)
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
            val a = DenseMatrix(n)
            for (i in 0 until n) {
                for (j in 0..i) {
                    var v = rng.nextDouble(-1.0, 1.0)
                    if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
                    a[i, j] = v
                    if (j != i) a[j, i] = Double.NaN
                }
            }
            val f = koblas.ldl(a)
            assertTrue(!f.singular, "n=$n flagged singular")
            val nrhs = 3
            val b = DenseMatrix(n, nrhs)
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
        val a = DenseMatrix(2)
        a[1, 0] = 1.0
        a[0, 1] = Double.NaN
        val f = koblas.ldl(a)
        // A swaps the rows, so the column-major columns (2, 3) and (1, -1) map to (3, 2) and (-1, 1).
        val b = DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 1.0, -1.0))
        val x = koblas.solve(f, b)
        assertClose(doubleArrayOf(3.0, 2.0, -1.0, 1.0), x.data, "antidiagonal block", tolerance = 1e-11)
    }

    @Test
    fun `degenerate shapes round-trip`() {
        val empty = koblas.factor(DenseMatrix(0, 0))
        val x0 = koblas.solve(empty, DenseMatrix(0, 3))
        assertEquals(0, x0.rows)
        assertEquals(3, x0.cols)
        val lu = koblas.factor(DenseMatrix.of(arrayOf(doubleArrayOf(2.0))))
        val xn = koblas.solve(lu, DenseMatrix(1, 0))
        assertEquals(1, xn.rows)
        assertEquals(0, xn.cols)
        val ldl0 = koblas.ldl(DenseMatrix(0, 0))
        assertEquals(0, koblas.solve(ldl0, DenseMatrix(0, 2)).rows)
    }
}
