@file:Suppress("PropertyName") // math convention: single-letter matrices (A, L) in tests

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The Cholesky family: factorization, the solve and inverse built on it, and the rank-1 update and downdate that keep
 * a factor current without refactorizing.
 */
@Suppress("VariableNaming") // single-letter matrix/vector names track math conventions
class CholeskyTest {

    /** The worked SPD example: non-diagonal, so convention errors cannot hide. */
    private fun spdExample() = DenseMatrix.of(
        arrayOf(
            doubleArrayOf(4.0, 2.0, 0.0),
            doubleArrayOf(2.0, 5.0, 1.0),
            doubleArrayOf(0.0, 1.0, 3.0),
        ),
    )

    /** `(L Lt)[i, j]`, summing only the stored part of a lower-triangular factor. */
    private fun reconstruct(l: DenseMatrix, i: Int, j: Int): Double {
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
        // Strict lower triangular: upper entries zero.
        for (i in 0 until 3) for (j in i + 1 until 3) assertEquals(0.0, L.l[i, j], "L[$i,$j] should be zero")
    }

    @Test
    fun `solveSpd inverts A b via Cholesky factor`() {
        val A = spdExample()
        val L = A.cholesky()
        val b = doubleArrayOf(1.0, 0.5, -1.0)
        val x = L.solve(b)
        // A * x should reproduce b.
        for (i in 0 until 3) {
            var s = 0.0
            for (j in 0 until 3) s += A[i, j] * x[j]
            assertEquals(b[i], s, 1e-10, "A*x reproduce b at $i")
        }
    }

    @Test
    fun `invertSpd produces A inverse for a non-diagonal matrix`() {
        val A = spdExample()
        val Ainv = A.cholesky().invert()
        // A * Ainv should be identity.
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                var s = 0.0
                for (k in 0 until 3) s += A[i, k] * Ainv[k, j]
                assertEquals(if (i == j) 1.0 else 0.0, s, 1e-9, "A*Ainv mismatch at [$i,$j]")
            }
        }
    }

    /** A negative diagonal pivot is a failure by default. */
    @Test
    fun `cholesky rejects a non positive definite pivot by default`() {
        val failure = assertFailsWith<IllegalArgumentException> { notPositiveDefinite().cholesky() }
        val message = failure.message!!
        assertTrue("pivot 1" in message, "the message should name the position: $message")
        assertTrue("Regularize" in message, "the message should name the way out: $message")
        assertFailsWith<IllegalArgumentException> { koblas.cholesky(notPositiveDefinite()) }
    }

    /** Regularizing is available, but only when asked for, and it factors a *nearby* matrix. */
    @Test
    fun `cholesky regularizes when the policy asks for it`() {
        val l = notPositiveDefinite().cholesky(CholeskyPolicy.Regularize())
        assertEquals(1e-5, l.l[1, 1], 1e-18, "the default floor is the historical 1e-5 on L")

        val loose = notPositiveDefinite().cholesky(CholeskyPolicy.Regularize(minimumPivot = 4e-4))
        assertEquals(2e-2, loose.l[1, 1], 1e-12, "L's diagonal is the square root of the pivot floor")

        // A positive-definite matrix is untouched by the policy: the same factor either way.
        val good = DenseMatrix.of(arrayOf(doubleArrayOf(4.0, 2.0), doubleArrayOf(2.0, 5.0)))
        val strict = good.cholesky()
        val regularized = good.cholesky(CholeskyPolicy.Regularize())
        for (i in 0 until 2) {
            for (j in 0 until 2) assertEquals(strict.l[i, j], regularized.l[i, j], 1e-15, "[$i,$j]")
        }
    }

    /** A non-positive floor would produce a NaN or a zero diagonal, so it is rejected at construction. */
    @Test
    fun `the regularization floor must be positive`() {
        assertFailsWith<IllegalArgumentException> { CholeskyPolicy.Regularize(minimumPivot = 0.0) }
        assertFailsWith<IllegalArgumentException> { CholeskyPolicy.Regularize(minimumPivot = -1.0) }
        assertFailsWith<IllegalArgumentException> { CholeskyPolicy.Regularize(minimumPivot = Double.NaN) }
    }

    private fun notPositiveDefinite() = DenseMatrix.of(
        arrayOf(
            doubleArrayOf(1.0, 0.0),
            doubleArrayOf(0.0, -0.5),
        ),
    )

    /**
     * The reason the type exists: the factor and the matrix it came from are no longer interchangeable.
     *
     * `solveSpd(L, b)` took any DenseMatrix, so passing the unfactored A compiled and returned nonsense.
     * The equivalent mistake is now a type error, which is checked here by confirming the two carry
     * different types and that the solve reads the factor rather than the input.
     */
    @Test
    fun `the factorization is a distinct type from the matrix it factors`() {
        val a = spdExample()
        val chol = a.cholesky()
        val b = doubleArrayOf(1.0, 2.0, 3.0)
        val x = chol.solve(b)
        // A·x = b, so the solve used the factor and not something that merely had the right shape.
        for (i in 0 until 3) {
            var s = 0.0
            for (j in 0 until 3) s += a[i, j] * x[j]
            assertEquals(b[i], s, 1e-9, "residual at $i")
        }
        assertEquals(3, chol.n)
        assertEquals(a.rows, chol.l.rows, "the factor stays reachable as an ordinary matrix")
    }

    @Test
    fun `cholesky reads only the lower triangle`() {
        // The documented contract, pinned: garbage above the diagonal must not reach the factor. A
        // caller holding an upper-only matrix has to mirror it, and this is what says so.
        val a = spdExample()
        val defaced = DenseMatrix.of(a.toArray())
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
