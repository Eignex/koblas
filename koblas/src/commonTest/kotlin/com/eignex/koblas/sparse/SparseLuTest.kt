package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.gemv
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The sparse LU: factorization, the FTRAN and BTRAN solves, the determinant, and the singular and
 * malformed cases.
 *
 * This factorization is judged on two axes at once. Accuracy is checked by residual, since pivots come
 * from a bounded candidate list chosen to limit fill rather than purely for stability. Fill itself is
 * checked too, because a factorization that quietly densifies is correct and useless — the sparsity is
 * what makes a simplex basis update cheap. Equilibration must not change the answer, only the arithmetic
 * getting there, so the determinant is compared across both settings.
 */
class SparseLuTest {

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
    fun `ftran and btran solve a random sparse system`() {
        val rng = Random(20260716)
        repeat(150) {
            val n = rng.nextInt(1, 10)
            val a = randomSparseSquare(n, rng)
            val lu = assertNotNull(a.lu(equilibrate = true))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.solve(a.gemv(x)), "forward solve n=$n", tolerance = 1e-7)
            assertClose(
                x,
                lu.solve(a.gemv(x, transpose = true), transpose = true),
                "transposed solve n=$n",
                tolerance = 1e-7,
            )
        }
    }

    @Test
    fun `bounded pivot search stays accurate and sparse on a larger random system`() {
        val rng = Random(20260727)
        val n = 300
        val a = randomSparseSquare(n, rng, density = 0.02, dominance = 1.0)
        val lu = assertNotNull(a.lu(equilibrate = true))
        val b = randomVector(n, rng)
        assertClose(b, a.gemv(lu.solve(b)), "ftran residual", tolerance = 1e-8)
        assertClose(b, a.gemv(lu.solve(b, transpose = true), transpose = true), "btran residual", tolerance = 1e-8)
        // The bounded candidate search must keep fill near the input's sparsity, far from dense.
        assertTrue(lu.nnz < 6 * a.nnz, "fill blew up: factor nnz=${lu.nnz} vs input nnz=${a.nnz}")
    }

    @Test
    fun `solve is correct with equilibration off`() {
        val rng = Random(22)
        repeat(60) {
            val n = rng.nextInt(1, 8)
            val a = randomSparseSquare(n, rng, density = 1.0)
            val lu = assertNotNull(a.lu(equilibrate = false))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.solve(a.gemv(x)), "unequilibrated n=$n", tolerance = 1e-7)
        }
    }

    @Test
    fun `determinant matches the hand value and tracks sign under row order`() {
        // [[2, 1], [1, 3]] → det 5.
        val det = square(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)).lu().determinant()
        assertClose(5.0, det, "det", tolerance = 1e-9)
        // A row swap negates the determinant: [[1,3],[2,1]] → det = 1 - 6 = -5.
        val det2 = square(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 1.0)).lu().determinant()
        assertClose(-5.0, det2, "det after row swap", tolerance = 1e-9)
    }

    @Test
    fun `determinant is invariant to equilibration`() {
        val rng = Random(11)
        repeat(30) {
            val n = rng.nextInt(2, 6)
            val a = randomSparseSquare(n, rng, density = 1.0, dominance = 4.0)
            val d0 = assertNotNull(a.lu(equilibrate = false)).determinant()
            val d1 = assertNotNull(a.lu(equilibrate = true)).determinant()
            assertTrue(abs(d0 - d1) <= 1e-6 * (abs(d0) + 1.0), "det differs: $d0 vs $d1")
        }
    }

    /**
     * A singular matrix comes back as a factorization reporting `singular`, not as null.
     *
     * The nullable result this replaced forced an unwrap for a condition the dense side reports with a
     * flag, and discarded the one useful fact — which pivot position had no candidate. Solving against it
     * throws rather than returning nonsense, because Markowitz aborts with no factors at all; a dense
     * `getrf` leaves partial ones behind, so there the garbage is at least LAPACK's own.
     */
    @Test
    fun `a singular matrix factors to a singular factorization`() {
        // A column of zeros, and a duplicated column: neither has a full set of acceptable pivots.
        for (a in listOf(
            SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf())),
            square(doubleArrayOf(1.0, 1.0), doubleArrayOf(2.0, 2.0)),
        )) {
            val f = a.lu()
            assertTrue(f.singular, "expected singular for $a")
            assertEquals(2, f.n)
            assertEquals(0, f.nnz, "a singular factorization has no fill")
            assertEquals(0.0, f.determinant())
            assertFailsWith<IllegalStateException> { f.solve(doubleArrayOf(1.0, 2.0)) }
        }
        // The failing pivot position is reported, which a null result could not do.
        val singular = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf())).lu()
        assertTrue(singular is SingularSparseFactorization)
        assertEquals(1, singular.failedAt, "the second pivot is the one with no candidate")
    }

    @Test
    fun `factorize rejects a non-square matrix`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))
        assertFailsWith<IllegalArgumentException> { a.lu() }
    }
}
