package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.assertClose
import kotlin.test.Test
import kotlin.test.assertFailsWith

class SymmetricInputTest {

    private val rhs = doubleArrayOf(3.0, 5.0)

    @Test
    fun `full input rejects triangles that disagree`() {
        val asymmetric = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, 9.0),
                doubleArrayOf(1.0, 3.0),
            ),
        )

        assertFailsWith<IllegalArgumentException> { asymmetric.cholesky() }
        assertFailsWith<IllegalArgumentException> { asymmetric.ldl() }
    }

    @Test
    fun `lower input ignores the upper triangle`() {
        val lower = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, Double.NaN),
                doubleArrayOf(1.0, 3.0),
            ),
        )

        assertClose(doubleArrayOf(0.8, 1.4), lower.cholesky(uplo = Uplo.LOWER).solve(rhs), "Cholesky")
        assertClose(doubleArrayOf(0.8, 1.4), lower.ldl(uplo = Uplo.LOWER).solve(rhs), "LDL")
    }

    @Test
    fun `upper input ignores the lower triangle`() {
        val upper = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, 1.0),
                doubleArrayOf(Double.NaN, 3.0),
            ),
        )

        assertClose(doubleArrayOf(0.8, 1.4), upper.cholesky(uplo = Uplo.UPPER).solve(rhs), "Cholesky")
        assertClose(doubleArrayOf(0.8, 1.4), upper.ldl(uplo = Uplo.UPPER).solve(rhs), "LDL")
    }
}
