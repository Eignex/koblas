package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.test.Test

class SymmetricInputTest {

    private val rhs = doubleArrayOf(3.0, 5.0)

    @Test
    fun `lower input ignores the upper triangle`() {
        val lower = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, Double.NaN),
                doubleArrayOf(1.0, 3.0),
            ),
        )

        assertClose(doubleArrayOf(0.8, 1.4), lower.cholesky().solve(rhs), "Cholesky")
        assertClose(doubleArrayOf(0.8, 1.4), lower.pivotedSymmetricIndefinite().solve(rhs), "LDL")
    }

    @Test
    fun `upper input ignores the lower triangle`() {
        val upper = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, 1.0),
                doubleArrayOf(Double.NaN, 3.0),
            ),
        )

        assertClose(doubleArrayOf(0.8, 1.4), upper.cholesky(lower = false).solve(rhs), "Cholesky")
        assertClose(doubleArrayOf(0.8, 1.4), upper.pivotedSymmetricIndefinite(lower = false).solve(rhs), "LDL")
    }
}
