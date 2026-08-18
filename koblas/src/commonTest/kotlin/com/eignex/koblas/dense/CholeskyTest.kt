@file:Suppress("PropertyName") // math convention: single-letter matrices (A, L) in tests

package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@Suppress("VariableNaming") // single-letter matrix/vector names track math conventions
class CholeskyTest {

    private fun spdExample() = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(4.0, 2.0, 0.0),
            doubleArrayOf(2.0, 5.0, 1.0),
            doubleArrayOf(0.0, 1.0, 3.0),
        ),
    )

    /** Computes `(L Lt)(i, j)` from the stored lower triangle only. */
    private fun reconstruct(l: F64DenseMatrix, i: Int, j: Int): Double {
        var s = 0.0
        for (k in 0..minOf(i, j)) s += l[i, k] * l[j, k]
        return s
    }

    @Test
    fun `cholesky reconstructs A as L Lt for a non-diagonal SPD matrix`() {
        val A = spdExample()
        val L = A.cholesky()
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(A[i, j], reconstruct(L.l, i, j), 1e-10, "L LT mismatch at [$i,$j]")
            }
        }
        for (i in 0 until 3) for (j in i + 1 until 3) assertEquals(0.0, L.l[i, j], "L[$i,$j] should be zero")
    }

    @Test
    fun `solveSpd inverts A b via Cholesky factor`() {
        val A = spdExample()
        val L = A.cholesky()
        val b = doubleArrayOf(1.0, 0.5, -1.0)
        val x = L.solve(b)
        for (i in 0 until 3) {
            var s = 0.0
            for (j in 0 until 3) s += A[i, j] * x[j]
            assertEquals(b[i], s, 1e-10, "A*x reproduce b at $i")
        }
        assertEquals(3, L.n)
        assertEquals(A.rows, L.l.rows, "the factor stays reachable as an ordinary matrix")
    }

    @Test
    fun `invertSpd produces A inverse for a non-diagonal matrix`() {
        val A = spdExample()
        val Ainv = A.cholesky().invert()
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                var s = 0.0
                for (k in 0 until 3) s += A[i, k] * Ainv[k, j]
                assertEquals(if (i == j) 1.0 else 0.0, s, 1e-9, "A*Ainv mismatch at [$i,$j]")
            }
        }
    }

    @Test
    fun `cholesky rejects a non positive definite pivot by default`() {
        val failure = assertFailsWith<NotPositiveDefinite> { notPositiveDefinite().cholesky() }
        val message = failure.message!!
        assertTrue("pivot 1" in message, "the message should name the position: $message")
        assertTrue("Regularize" in message, "the message should name the way out: $message")
        assertFailsWith<IllegalArgumentException> { koblas.cholesky(notPositiveDefinite()) }
    }

    @Test
    fun `cholesky regularizes when the policy asks for it`() {
        val l = notPositiveDefinite().cholesky(CholeskyPolicy.Regularize())
        assertEquals(1e-5, l.l[1, 1], 1e-18, "the default pivot floor puts 1e-5 on L")

        val loose = notPositiveDefinite().cholesky(CholeskyPolicy.Regularize(minimumPivot = 4e-4))
        assertEquals(2e-2, loose.l[1, 1], 1e-12, "L's diagonal is the square root of the pivot floor")

        val good = F64DenseMatrix.of(arrayOf(doubleArrayOf(4.0, 2.0), doubleArrayOf(2.0, 5.0)))
        val strict = good.cholesky()
        val regularized = good.cholesky(CholeskyPolicy.Regularize())
        for (i in 0 until 2) {
            for (j in 0 until 2) assertEquals(strict.l[i, j], regularized.l[i, j], 1e-15, "[$i,$j]")
        }
    }

    @Test
    fun `the regularization floor must be positive`() {
        assertFailsWith<IllegalArgumentException> { CholeskyPolicy.Regularize(minimumPivot = 0.0) }
        assertFailsWith<IllegalArgumentException> { CholeskyPolicy.Regularize(minimumPivot = -1.0) }
        assertFailsWith<IllegalArgumentException> { CholeskyPolicy.Regularize(minimumPivot = Double.NaN) }
    }

    private fun notPositiveDefinite() = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(1.0, 0.0),
            doubleArrayOf(0.0, -0.5),
        ),
    )

    @Test
    fun `cholesky reads only the lower triangle`() {
        // Values above the diagonal must not reach the factor, so an upper-only caller has to mirror first.
        val a = spdExample()
        val defaced = F64DenseMatrix.of(a.toArray())
        for (i in 0 until defaced.rows) {
            for (j in i + 1 until defaced.cols) defaced[i, j] = Double.NaN
        }
        val expected = a.cholesky().l
        val actual = defaced.cholesky().l
        for (i in 0 until expected.rows) {
            for (j in 0..i) assertEquals(expected[i, j], actual[i, j], 0.0, "factor differs at ($i,$j)")
        }
    }
}
