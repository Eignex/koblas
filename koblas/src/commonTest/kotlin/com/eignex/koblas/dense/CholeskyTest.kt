@file:Suppress("PropertyName") // math convention: single-letter matrices (A, L) in tests

package com.eignex.koblas.dense

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas
import kotlin.test.*

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
    fun `cholesky destination solves support aliases and blocks`() {
        val factor = spdExample().cholesky()
        val rhs = doubleArrayOf(1.0, 0.5, -1.0)
        val distinct = DoubleArray(3)

        assertSame(distinct, koblas.solveInto(factor, rhs, distinct))
        assertContentEquals(factor.solve(rhs), distinct)
        assertSame(rhs, koblas.solveInto(factor, rhs, rhs))
        assertContentEquals(distinct, rhs)

        val block = F64DenseMatrix(3, 2, rhs + doubleArrayOf(2.0, 1.0, 0.0))
        val out = F64DenseMatrix(3, 2)
        assertSame(out, koblas.solveInto(factor, block, out))
        assertContentEquals(factor.solve(block).data, out.data)
        assertEquals(0, koblas.solve(factor, F64DenseMatrix(3, 0)).cols)
        assertFailsWith<IllegalArgumentException> { koblas.solveInto(factor, DoubleArray(2), DoubleArray(3)) }
        assertFailsWith<IllegalArgumentException> { koblas.solveInto(factor, F64DenseMatrix(2, 1), out) }
    }

    @Test
    fun `logAbsDeterminant matches the LU form on the same matrix`() {
        val A = spdExample()
        // 4*(5*3 - 1) - 2*(2*3) = 44
        assertEquals(kotlin.math.ln(44.0), A.cholesky().logAbsDeterminant(), 1e-10, "log det")
        assertEquals(A.lu().logAbsDeterminant(), A.cholesky().logAbsDeterminant(), 1e-10, "against LU")
    }

    @Test
    fun `logAbsDeterminant survives a matrix whose determinant would overflow`() {
        // A product of diagonals would leave the double range long before this many terms.
        val n = 400
        val scaled = F64DenseMatrix.zero(n, n)
        for (k in 0 until n) scaled[k, k] = 1e30
        val logDet = scaled.cholesky().logAbsDeterminant()
        assertEquals(n * kotlin.math.ln(1e30), logDet, 1e-6, "log det of the scaled identity")
        assertTrue(logDet.isFinite(), "the accumulated form stays finite")
    }

    @Test
    fun `logAbsDeterminant reports a zero diagonal as negative infinity`() {
        val chol = spdExample().cholesky()
        chol.l[1, 1] = 0.0
        assertEquals(Double.NEGATIVE_INFINITY, chol.logAbsDeterminant(), "a zero diagonal")
    }

    @Test
    fun `invertInto fills the caller's matrix and returns it`() {
        val A = spdExample()
        val chol = A.cholesky()
        val allocated = chol.invert()
        // Poisoned, because every entry is written and nothing may survive from before.
        val out = F64DenseMatrix.wrap(3, 3, DoubleArray(9) { Double.NaN })
        val returned = chol.invertInto(out)
        assertSame(out, returned, "the destination is returned")
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(allocated[i, j], out[i, j], 1e-12, "($i,$j)")
            }
        }
    }

    @Test
    fun `invertInto rejects a destination of the wrong shape`() {
        val chol = spdExample().cholesky()
        assertFailsWith<DimensionMismatch> { chol.invertInto(F64DenseMatrix.zero(2, 2)) }
        assertFailsWith<DimensionMismatch> { chol.invertInto(F64DenseMatrix.zero(3, 4)) }
    }

    @Test
    fun `invertInto rejects the factor's own matrix`() {
        val chol = spdExample().cholesky()
        // Both halves read the factor while writing the inverse, so this would give a wrong answer quietly.
        assertFailsWith<IllegalArgumentException> { chol.invertInto(chol.l) }
    }

    @Test
    fun `invertInto handles an empty factorization`() {
        val chol = F64DenseMatrix.zero(0, 0).cholesky()
        val out = F64DenseMatrix.zero(0, 0)
        assertSame(out, chol.invertInto(out), "the empty destination is returned")
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
        assertFailsWith<NotPositiveDefinite> { koblas.cholesky(notPositiveDefinite()) }
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
    fun `regularizing keeps the factor near the input instead of inflating it`() {
        // Substituting the floor alone divided this column's tail by 1e-5, so the trailing entry came back
        // as 6.25e10 against an input of 9. Raising the pivot to bound the column leaves every entry but the
        // regularized diagonal reproduced exactly.
        val a = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(4.0, 2.0, 1.0),
                doubleArrayOf(2.0, 1.0, 3.0),
                doubleArrayOf(1.0, 3.0, 9.0),
            ),
        )

        val l = a.cholesky(CholeskyPolicy.Regularize()).l

        val reconstructed = { i: Int, j: Int -> (0 until 3).sumOf { k -> l[i, k] * l[j, k] } }
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                if (i == 1 && j == 1) continue
                assertEquals(a[i, j], reconstructed(i, j), 1e-12, "L·Lᵀ[$i,$j] should reproduce the input")
            }
        }
        assertTrue(reconstructed(1, 1) < 100.0, "the regularized diagonal inflated to ${reconstructed(1, 1)}")
    }

    @Test
    fun `the regularization floor lifts a tiny positive pivot`() {
        // The near-singular case is the common one, and it used to pass straight through because the clamp
        // only looked at non-positive pivots.
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1e-300, 0.0), doubleArrayOf(0.0, 1.0)))

        val l = a.cholesky(CholeskyPolicy.Regularize()).l

        assertEquals(1e-5, l[0, 0], 1e-18, "the floor should have lifted a 1e-300 pivot")
    }

    @Test
    fun `a NaN pivot is refused even when regularizing`() {
        // Regularizing overwrote the diagonal and left the column below it NaN, so the factor read as finite
        // while being nonsense. A NaN is corrupt input rather than indefiniteness.
        val a = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0),
                doubleArrayOf(0.0, 1.0, 0.0),
                doubleArrayOf(Double.NaN, 0.0, 1.0),
            ),
        )

        assertFailsWith<NotPositiveDefinite> { a.cholesky(CholeskyPolicy.Regularize()) }
        assertFailsWith<NotPositiveDefinite> { a.cholesky() }
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
        val actual = defaced.cholesky(lower = true).l
        for (i in 0 until expected.rows) {
            for (j in 0..i) assertEquals(expected[i, j], actual[i, j], 0.0, "factor differs at ($i,$j)")
        }
    }
}
