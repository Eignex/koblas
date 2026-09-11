package com.eignex.koblas

import com.eignex.koblas.*
import kotlin.test.*

class ExceptionsTest {

    @Test
    fun `a shape mismatch is a DimensionMismatch`() {
        val a = DenseMatrix.zero(2, 3)
        assertFailsWith<DimensionMismatch> { DenseVector.zero(2) dot DenseVector.zero(3) }
        assertFailsWith<DimensionMismatch> { koblas.gemm(a, DenseMatrix.zero(2, 2)) }
    }
}
