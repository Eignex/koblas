@file:Suppress("PropertyName") // math convention: single-letter matrices (A, L) in tests

package com.eignex.koblas

import kotlin.random.Random
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

    /** A random SPD matrix built by symmetrizing and then dominating the diagonal. */
    private fun randomSpd(n: Int, rng: Random): DenseMatrix {
        val a = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        for (i in 0 until n) for (j in 0 until i) a[j, i] = a[i, j]
        for (i in 0 until n) a[i, i] = a[i, i] + n
        return a
    }

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
    fun `cholesky update then reconstruct equals A plus x xt`() {
        val rng = Random(20260801)
        for (n in intArrayOf(1, 3, 8, 20)) {
            val a = randomSpd(n, rng)
            val l = a.cholesky(regularizeNonPD = false)
            val x = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            l.choleskyUpdateInPlace(DenseVector.of(x))
            for (i in 0 until n) {
                for (j in 0 until n) {
                    assertEquals(a[i, j] + x[i] * x[j], reconstruct(l, i, j), 1e-9, "update at [$i,$j] n=$n")
                }
            }
            // The factor must stay a valid lower-triangular Cholesky: positive diagonal.
            for (i in 0 until n) assertTrue(l[i, i] > 0.0, "non-positive diagonal at $i")
        }
    }

    @Test
    fun `cholesky update then downdate round-trips to the original factor`() {
        val rng = Random(20260802)
        val n = 10
        val l = randomSpd(n, rng).cholesky(regularizeNonPD = false)
        val original = l.data.copyOf()
        val x = DenseVector.of(DoubleArray(n) { rng.nextDouble(-2.0, 2.0) })
        l.choleskyUpdateInPlace(x)
        assertEquals(0.0, l.choleskyDowndateInPlace(x), "downdating a just-updated factor must succeed")
        for (idx in l.data.indices) {
            assertEquals(original[idx], l.data[idx], 1e-9, "round-trip mismatch at $idx")
        }
    }

    @Test
    fun `cholesky update with the zero vector is a no-op`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(4.0, 1.0),
                doubleArrayOf(1.0, 3.0),
            ),
        )
        val l = a.cholesky()
        val before = l.data.copyOf()
        l.choleskyUpdateInPlace(DenseVector.zero(2))
        for (idx in before.indices) assertEquals(before[idx], l.data[idx], 0.0)
        l.choleskyUpdateInPlace(SparseVector.of(2, IntArray(0), DoubleArray(0)))
        for (idx in before.indices) assertEquals(before[idx], l.data[idx], 0.0)
    }

    @Test
    fun `cholesky downdate then reconstruct equals A minus x xt`() {
        // Build an SPD A with enough headroom to absorb the downdate.
        val A = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(10.0, 2.0, 1.0),
                doubleArrayOf(2.0, 8.0, 3.0),
                doubleArrayOf(1.0, 3.0, 7.0),
            ),
        )
        val L = A.cholesky()
        val x = doubleArrayOf(0.5, 1.0, -0.5)
        assertEquals(0.0, L.choleskyDowndateInPlace(DenseVector.of(x)), "downdate should stay in the SPD cone")
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(A[i, j] - x[i] * x[j], reconstruct(L, i, j), 1e-9, "downdate at [$i,$j]")
            }
        }
    }

    @Test
    fun `cholesky downdate refuses to exit the SPD cone`() {
        val A = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, 0.0),
                doubleArrayOf(0.0, 2.0),
            ),
        )
        val L = A.cholesky()
        // x with ||L^-1 x|| >= 1 - pick x so that x*xT swamps A.
        val norm = L.choleskyDowndateInPlace(DenseVector.of(doubleArrayOf(3.0, 0.0)))
        assertTrue(norm >= 1.0, "expected norm >= 1 for an infeasible downdate, got $norm")
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
