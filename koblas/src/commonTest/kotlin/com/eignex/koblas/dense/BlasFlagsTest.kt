package com.eignex.koblas.dense

import com.eignex.koblas.F64RouteQuery
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class BlasFlagsTest {

    private val dense = F64ReferenceBlas()

    @Test
    fun `typed matrix flags match Boolean migration overloads`() {
        val a = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)),
        )
        val b = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(2.0, 1.0, 0.0), doubleArrayOf(-1.0, 3.0, 2.0)),
        )
        val expected = F64DenseMatrix.zero(2, 2)
        val actual = F64DenseMatrix.zero(2, 2)

        dense.gemm(1.0, a, false, b, true, 0.0, expected)
        dense.gemm(1.0, a, Transpose.NO_TRANSPOSE, b, Transpose.TRANSPOSE, 0.0, actual)

        assertContentEquals(expected.data, actual.data)
    }

    @Test
    fun `typed triangular flags match Boolean migration overloads`() {
        val triangle = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(1.0, 3.0)),
        )
        val expected = F64DenseMatrix.ofColumns(
            arrayOf(doubleArrayOf(4.0, 7.0), doubleArrayOf(-2.0, 5.0)),
        )
        val actual = F64DenseMatrix.wrap(expected.rows, expected.cols, expected.data.copyOf())

        dense.trsm(triangle, expected, lower = true, transpose = false, unitDiag = false, right = false)
        dense.trsm(
            triangle,
            actual,
            uplo = Uplo.LOWER,
            transpose = Transpose.NO_TRANSPOSE,
            diag = Diag.NON_UNIT,
            side = Side.LEFT,
        )

        assertContentEquals(expected.data, actual.data)
    }

    @Test
    fun `typed sparse flags reach the same operation`() {
        val triangle = F64SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 2.0, 1 to 1.0), listOf(1 to 3.0)),
        )
        val expected = doubleArrayOf(4.0, 7.0)
        val actual = expected.copyOf()

        F64ReferenceSparseLinearAlgebra.trsv(triangle, expected, lower = true)
        F64ReferenceSparseLinearAlgebra.trsv(triangle, actual, Uplo.LOWER, diag = Diag.NON_UNIT)

        assertContentEquals(expected, actual)
    }

    @Test
    fun `full triangle is rejected before mutation`() {
        val triangle = F64DenseMatrix.diagonal(2)
        val rhs = doubleArrayOf(2.0, 3.0)

        assertFailsWith<IllegalArgumentException> { dense.trsv(triangle, rhs, Uplo.FULL) }

        assertContentEquals(doubleArrayOf(2.0, 3.0), rhs)
    }

    @Test
    fun `typed route flags preserve the inspected shape`() {
        val query = F64RouteQuery.SparseTriangularSolve(
            storedEntries = 40,
            rightHandSides = 3,
            uplo = Uplo.UPPER,
            side = Side.RIGHT,
            transpose = Transpose.TRANSPOSE,
            diag = Diag.UNIT,
        )

        assertEquals(false, query.lower)
        assertEquals(true, query.right)
        assertEquals(true, query.transpose)
        assertEquals(true, query.unitDiagonal)
    }
}
