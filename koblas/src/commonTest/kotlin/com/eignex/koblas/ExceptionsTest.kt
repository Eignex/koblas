package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import kotlin.test.*

class ExceptionsTest {

    private val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
    private val indefinite = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 1.0)))

    @Test
    fun `a shape mismatch is a DimensionMismatch`() {
        val a = F64DenseMatrix.zero(2, 3)
        assertFailsWith<DimensionMismatch> { koblas.invert(koblas.factor(a)) }
        assertFailsWith<DimensionMismatch> { F64DenseVector.zero(2) dot F64DenseVector.zero(3) }
        assertFailsWith<DimensionMismatch> { koblas.gemm(a, F64DenseMatrix.zero(2, 2)) }
    }

    @Test
    fun `inverting a singular factorization reports the pivot`() {
        val e = assertFailsWith<SingularMatrix> { singular.lu().invert() }
        assertEquals(singular.lu().failedAt, e.position, "position should match the factorization's failedAt")
        assertTrue(e.position >= 0)
        assertTrue("pivot 1" in e.message!!, "the message should name the failing pivot: ${e.message}")
    }

    @Test
    fun `a factorization without a pivot position reports unknown singularity`() {
        val factorization = F64SingularSparseFactorization(2, SINGULAR_POSITION_UNKNOWN)

        val e = assertFailsWith<SingularMatrix> { factorization.solve(doubleArrayOf(1.0, 2.0)) }

        assertEquals(SINGULAR_POSITION_UNKNOWN, e.position)
        assertTrue("factorization is singular" in e.message!!, "the message should omit a made-up pivot")
    }

    @Test
    fun `an indefinite matrix reports the pivot and its value`() {
        val e = assertFailsWith<NotPositiveDefinite> { indefinite.cholesky() }
        assertEquals(1, e.position, "the second pivot is the one that goes negative")
        assertTrue(e.pivot <= 0.0, "the reported pivot should be the offending one, was ${e.pivot}")
        indefinite.cholesky(CholeskyPolicy.Regularize())
    }

    @Test
    fun `every koblas failure stays an IllegalArgumentException`() {
        for (block in listOf<() -> Unit>(
            { koblas.invert(koblas.factor(F64DenseMatrix.zero(2, 3))) },
            { singular.lu().invert() },
            { indefinite.cholesky() },
        )) {
            val e = assertFailsWith<IllegalArgumentException> { block() }
            assertTrue(e is KoblasException, "expected a typed koblas failure, got ${e::class.simpleName}")
        }
    }
}
