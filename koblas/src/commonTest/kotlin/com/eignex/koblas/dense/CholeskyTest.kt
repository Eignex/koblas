@file:Suppress("PropertyName") // math convention: single-letter matrices (A, L) in tests

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The Cholesky family: factorization, the solve and inverse built on it, and the rank-1 update and
 * downdate that keep a factor current without refactorizing.
 *
 * The worked examples are non-diagonal and asymmetric in their off-diagonal structure, which is where a
 * transposed index or a swapped triangle shows up; a diagonal example would pass either way. Each
 * property is checked by reconstructing `L Lt` and comparing against the matrix it should equal, so a
 * factor that is self-consistent but wrong still fails.
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
                assertEquals(A[i, j], reconstruct(L, i, j), 1e-10, "L LT mismatch at [$i,$j]")
            }
        }
        // Strict lower triangular: upper entries zero.
        for (i in 0 until 3) for (j in i + 1 until 3) assertEquals(0.0, L[i, j], "L[$i,$j] should be zero")
    }

    @Test
    fun `solveSpd inverts A b via Cholesky factor`() {
        val A = spdExample()
        val L = A.cholesky()
        val b = doubleArrayOf(1.0, 0.5, -1.0)
        val x = solveSpd(L, b)
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
        val Ainv = invertSpd(A.cholesky())
        // A * Ainv should be identity.
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                var s = 0.0
                for (k in 0 until 3) s += A[i, k] * Ainv[k, j]
                assertEquals(if (i == j) 1.0 else 0.0, s, 1e-9, "A*Ainv mismatch at [$i,$j]")
            }
        }
    }

    @Test
    fun `cholesky strict mode throws on a non positive definite pivot`() {
        // Negative diagonal pivot - immediately rejected with regularizeNonPD=false.
        val bad = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(0.0, -0.5),
            ),
        )
        assertFailsWith<IllegalArgumentException> { bad.cholesky(regularizeNonPD = false) }
        // Default regularising path still succeeds (clamps the pivot to 1e-5).
        val L = bad.cholesky()
        assertTrue(L[1, 1] > 0.0)
    }
}
