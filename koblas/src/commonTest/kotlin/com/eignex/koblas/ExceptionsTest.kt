package com.eignex.koblas

import com.eignex.koblas.dense.CholeskyPolicy
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.invert
import com.eignex.koblas.dense.lu
import com.eignex.koblas.sparse.SparseLdlPolicy
import com.eignex.koblas.sparse.ldl
import com.eignex.koblas.sparse.trsv
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ExceptionsTest {

    private val singular = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
    private val indefinite = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 1.0)))

    @Test
    fun `a shape mismatch is a DimensionMismatch`() {
        val a = DenseMatrix.zero(2, 3)
        assertFailsWith<DimensionMismatch> { koblas.factor(a) }
        assertFailsWith<DimensionMismatch> { DenseVector.zero(2) dot DenseVector.zero(3) }
        assertFailsWith<DimensionMismatch> { koblas.gemm(a, DenseMatrix.zero(2, 2)) }
    }

    @Test
    fun `inverting a singular factorization reports the pivot`() {
        val e = assertFailsWith<SingularMatrix> { singular.lu().invert() }
        assertEquals(singular.lu().failedAt, e.position, "position should match the factorization's failedAt")
        assertTrue(e.position >= 0)
    }

    @Test
    fun `a singular sparse triangle reports the column`() {
        val t = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 1),
            colIdx = intArrayOf(0, 0, 1),
            values = doubleArrayOf(1.0, 3.0, 0.0),
        )
        val e = assertFailsWith<SingularMatrix> { t.trsv(doubleArrayOf(1.0, 1.0), lower = true) }
        assertEquals(1, e.position)
    }

    @Test
    fun `an indefinite matrix reports the pivot and its value`() {
        val e = assertFailsWith<NotPositiveDefinite> { indefinite.cholesky() }
        assertEquals(1, e.position, "the second pivot is the one that goes negative")
        assertTrue(e.pivot <= 0.0, "the reported pivot should be the offending one, was ${e.pivot}")
        indefinite.cholesky(CholeskyPolicy.Regularize())
    }

    @Test
    fun `a strict sparse ldl reports a non-positive pivot`() {
        val s = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 0, 1),
            colIdx = intArrayOf(0, 0, 1, 1),
            values = doubleArrayOf(1.0, 2.0, 2.0, 1.0),
        )
        val e = assertFailsWith<NotPositiveDefinite> { s.ldl(SparseLdlPolicy.Strict) }
        assertTrue(e.pivot < 0.0, "the reported pivot should be negative, was ${e.pivot}")
    }

    @Test
    fun `every koblas failure stays an IllegalArgumentException`() {
        for (block in listOf<() -> Unit>(
            { koblas.factor(DenseMatrix.zero(2, 3)) },
            { singular.lu().invert() },
            { indefinite.cholesky() },
        )) {
            val e = assertFailsWith<IllegalArgumentException> { block() }
            assertTrue(e is KoblasException, "expected a typed koblas failure, got ${e::class.simpleName}")
        }
    }
}
