package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import kotlin.test.*

class ExceptionsTest {

    @Test
    fun `a shape mismatch is a DimensionMismatch`() {
        val a = F64DenseMatrix.zero(2, 3)
        assertFailsWith<DimensionMismatch> { F64DenseVector.zero(2) dot F64DenseVector.zero(3) }
        assertFailsWith<DimensionMismatch> { koblas.gemm(a, F64DenseMatrix.zero(2, 2)) }
    }

    @Test
    fun `a factorization without a pivot position reports unknown singularity`() {
        val factorization = F64SingularSparseFactorization(2, SINGULAR_POSITION_UNKNOWN)

        val e = assertFailsWith<SingularMatrix> { factorization.solve(doubleArrayOf(1.0, 2.0)) }

        assertEquals(SINGULAR_POSITION_UNKNOWN, e.position)
        assertTrue("factorization is singular" in e.message!!, "the message should omit a made-up pivot")
    }
}
