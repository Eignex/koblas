package com.eignex.koblas.sparse.factorization.ldl

import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.randomVector
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.sparseSymmetricConformanceSystem
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseLdlTest {

    private fun ldl(a: F64SparseMatrix) = F64QuasiDefiniteUpLookingLdl.factorLower(a)

    /** The full symmetric matrix a stored lower triangle stands for, for taking a residual against. */
    private fun multiplySymmetric(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
        val y = DoubleArray(a.rows)
        for (j in 0 until a.cols) {
            a.forEachInColumn(j) { i, v ->
                y[i] += v * x[j]
                if (i != j) y[j] += v * x[i]
            }
        }
        return y
    }

    @Test
    fun `the solution reproduces the right-hand side`() {
        val rng = Random(20260923)
        val n = 14
        val a = sparseSymmetricConformanceSystem(n, rng)
        val b = randomVector(n, rng)

        val x = ldl(a).solve(b)

        assertClose(b, multiplySymmetric(a, x), "residual", tolerance = 1e-9)
    }

    @Test
    fun `it agrees with the Cholesky on a positive definite matrix`() {
        val rng = Random(20260924)
        for (n in intArrayOf(1, 2, 6, 17)) {
            val a = sparseSymmetricConformanceSystem(n, rng)
            val b = randomVector(n, rng)

            val fromLdl = ldl(a).solve(b)
            val fromCholesky = F64ReferenceSparseLinearAlgebra.cholesky(a).solve(b)

            assertClose(fromCholesky, fromLdl, "n=$n", tolerance = 1e-9)
        }
    }

    /** The whole reason this exists beside the Cholesky, which refuses the same matrix. */
    @Test
    fun `an indefinite matrix factors where a Cholesky would not`() {
        // Determinant is negative, so the matrix is symmetric and indefinite.
        val indefinite = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 4.0, 2 to 3.0), listOf(1 to 4.0), listOf(2 to 2.0)),
        )
        val b = doubleArrayOf(1.0, 2.0, 3.0)

        val f = ldl(indefinite)

        assertTrue(!f.singular, "an indefinite matrix has an L D Lt")
        assertTrue(f.d.any { it < 0.0 }, "D should carry the negative pivot, got ${f.d.toList()}")
        assertClose(b, multiplySymmetric(indefinite, f.solve(b)), "residual", tolerance = 1e-9)
    }

    @Test
    fun `a zero pivot is reported singular at its column`() {
        // Column 1 is empty off the diagonal and its diagonal is zero, so the pivot there is exactly zero.
        val singular = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 2.0), listOf(1 to 0.0), listOf(2 to 2.0)),
        )

        val f = ldl(singular)

        assertTrue(f.singular, "a zero pivot should be reported")
        assertEquals(1, f.failedAt)
        assertEquals(0, f.nnz, "a singular factorization has no fill")
        assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 1.0, 1.0)) }
    }

    @Test
    fun `the unit diagonal of L is not among its stored entries`() {
        val rng = Random(20260925)
        val n = 9
        val a = sparseSymmetricConformanceSystem(n, rng)

        val l = ldl(a).l

        for (j in 0 until n) {
            l.forEachInColumn(j) { i, _ ->
                assertTrue(i > j, "column $j stores row $i, which is not strictly below the diagonal")
            }
        }
    }

    @Test
    fun `an identity factors to a unit L and a unit D`() {
        val n = 5
        val identity = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })

        val f = ldl(identity)

        assertEquals(0, f.l.nnz, "an identity fills nowhere below the diagonal")
        assertTrue(f.d.all { it == 1.0 }, "D should be ones, got ${f.d.toList()}")
        assertEquals(1.0, f.rcond)
    }

    @Test
    fun `a solve may write into the vector it read`() {
        val rng = Random(20260926)
        val n = 8
        val a = sparseSymmetricConformanceSystem(n, rng)
        val b = randomVector(n, rng)
        val f = ldl(a)
        val expected = f.solve(b)

        val inPlace = b.copyOf()
        f.solveInto(inPlace, inPlace)

        assertClose(expected, inPlace, "an aliased solve", tolerance = 0.0)
    }
}
