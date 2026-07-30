package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The CSC matrix and its sparse LU: products, solves in both directions, the determinant, and the
 * singular and malformed cases.
 *
 * Sparse factorization is judged on two axes at once. Accuracy is checked by residual, since the
 * factorization is not exact: pivots come from a bounded candidate list chosen to limit fill rather than
 * purely for stability. Fill itself is checked too, because a factorization that quietly densifies is
 * correct and useless — the sparsity is what makes a simplex basis update cheap.
 */
class SparseTest {

    /** A square CSC matrix from dense rows, dropping the zeros. */
    private fun square(vararg rows: DoubleArray): SparseMatrix {
        val n = rows.size
        val cols = List(n) { j -> (0 until n).mapNotNull { i -> if (rows[i][j] != 0.0) i to rows[i][j] else null } }
        return SparseMatrix.ofColumns(n, n, cols)
    }

    /**
     * A random square matrix with [density] fill off the diagonal and a diagonal dominant by a factor of
     * [dominance], which makes it non-singular. Lower dominance leaves the pivot search more work to do.
     */
    private fun randomSparseSquare(n: Int, rng: Random, density: Double = 0.3, dominance: Double = 5.0): SparseMatrix {
        val columns = List(n) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-2.0, 2.0) + n * dominance))
            for (i in 0 until n) if (i != j && rng.nextDouble() < density) entries.add(i to rng.nextDouble(-2.0, 2.0))
            entries
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    @Test
    fun `CSC mat-vec multiplies a matrix and its transpose`() {
        // [[1, 0, 2], [0, 3, 0]]  (2x3), CSC by column.
        val a = SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(
                listOf(0 to 1.0),
                listOf(1 to 3.0),
                listOf(0 to 2.0),
            ),
        )
        assertTrue(doubleArrayOf(7.0, 9.0).contentEquals(a.gemv(doubleArrayOf(1.0, 3.0, 3.0))))
        assertTrue(doubleArrayOf(1.0, 6.0, 2.0).contentEquals(a.gemv(doubleArrayOf(1.0, 2.0), transpose = true)))
    }

    @Test
    fun `SparseLu ftran and btran solve a random sparse system`() {
        val rng = Random(20260716)
        repeat(150) {
            val n = rng.nextInt(1, 10)
            val a = randomSparseSquare(n, rng)
            val lu = assertNotNull(SparseLu.factorize(a, equilibrate = true))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.ftran(a.gemv(x)), "ftran n=$n", tolerance = 1e-7)
            assertClose(x, lu.btran(a.gemv(x, transpose = true)), "btran n=$n", tolerance = 1e-7)
        }
    }

    @Test
    fun `bounded pivot search stays accurate and sparse on a larger random system`() {
        val rng = Random(20260727)
        val n = 300
        val a = randomSparseSquare(n, rng, density = 0.02, dominance = 1.0)
        val lu = assertNotNull(SparseLu.factorize(a, equilibrate = true))
        val b = randomVector(n, rng)
        assertClose(b, a.gemv(lu.ftran(b)), "ftran residual", tolerance = 1e-8)
        assertClose(b, a.gemv(lu.btran(b), transpose = true), "btran residual", tolerance = 1e-8)
        // The bounded candidate search must keep fill near the input's sparsity, far from dense.
        assertTrue(lu.nnz < 6 * a.nnz, "fill blew up: factor nnz=${lu.nnz} vs input nnz=${a.nnz}")
    }

    @Test
    fun `solve is correct with equilibration off`() {
        val rng = Random(22)
        repeat(60) {
            val n = rng.nextInt(1, 8)
            val a = randomSparseSquare(n, rng, density = 1.0)
            val lu = assertNotNull(SparseLu.factorize(a, equilibrate = false))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.ftran(a.gemv(x)), "unequilibrated n=$n", tolerance = 1e-7)
        }
    }

    @Test
    fun `determinant matches the hand value and tracks sign under row order`() {
        // [[2, 1], [1, 3]] → det 5.
        val det = assertNotNull(
            SparseLu.factorize(square(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))),
        ).determinant()
        assertClose(5.0, det, "det", tolerance = 1e-9)
        // A row swap negates the determinant: [[1,3],[2,1]] → det = 1 - 6 = -5.
        val det2 = assertNotNull(
            SparseLu.factorize(square(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 1.0))),
        ).determinant()
        assertClose(-5.0, det2, "det after row swap", tolerance = 1e-9)
    }

    @Test
    fun `determinant is invariant to equilibration`() {
        val rng = Random(11)
        repeat(30) {
            val n = rng.nextInt(2, 6)
            val a = randomSparseSquare(n, rng, density = 1.0, dominance = 4.0)
            val d0 = assertNotNull(SparseLu.factorize(a, equilibrate = false)).determinant()
            val d1 = assertNotNull(SparseLu.factorize(a, equilibrate = true)).determinant()
            assertTrue(abs(d0 - d1) <= 1e-6 * (abs(d0) + 1.0), "det differs: $d0 vs $d1")
        }
    }

    @Test
    fun `SparseLu reports a singular matrix`() {
        // A column of zeros ⇒ singular.
        assertNull(SparseLu.factorize(SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf()))))
        // So is a duplicated column.
        assertNull(SparseLu.factorize(square(doubleArrayOf(1.0, 1.0), doubleArrayOf(2.0, 2.0))))
    }

    @Test
    fun `factorize rejects a non-square matrix`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))
        assertFailsWith<IllegalArgumentException> { SparseLu.factorize(a) }
    }

    @Test
    fun `SparseMatrix rejects structurally invalid CSC`() {
        // colPtr must have cols + 1 entries.
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 2, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(1.0))
        }
        // rowIdx out of range.
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(5), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `ofColumns sums duplicate entries and sorts rows`() {
        val a = SparseMatrix.ofColumns(3, 1, listOf(listOf(2 to 1.0, 0 to 2.0, 2 to 3.0)))
        assertTrue(intArrayOf(0, 2).contentEquals(a.rowIdx)) // ascending
        assertTrue(doubleArrayOf(2.0, 4.0).contentEquals(a.values)) // 1.0 + 3.0 summed at row 2
    }
}
