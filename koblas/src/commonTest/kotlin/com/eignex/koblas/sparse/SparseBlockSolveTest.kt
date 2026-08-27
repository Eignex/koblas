package com.eignex.koblas.sparse

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.factorization.ldl.F64SparseUpLookingLdl
import kotlin.test.*

class SparseBlockSolveTest {

    @Test
    fun `portable block solves agree with vector solves in both directions`() {
        val factor = F64ReferenceSparseLinearAlgebra.factor(matrix())
        val b = rightHandSides()

        for (transpose in booleanArrayOf(false, true)) {
            val actual = factor.solve(b, transpose)
            for (column in 0 until b.cols) {
                val expected = factor.solve(DoubleArray(b.rows) { b[it, column] }, transpose)
                assertContentEquals(expected, DoubleArray(actual.rows) { actual[it, column] })
            }
        }
    }

    @Test
    fun `block destination may alias every right hand side`() {
        val factor = F64ReferenceSparseLinearAlgebra.factor(matrix())
        val aliased = rightHandSides()
        val expected = factor.solve(aliased)

        factor.solveInto(aliased, aliased)

        assertContentEquals(expected.data, aliased.data)
    }

    @Test
    fun `block solve validates both matrix shapes before mutation`() {
        val factor = F64ReferenceSparseLinearAlgebra.factor(matrix())
        val out = F64DenseMatrix(3, 2, DoubleArray(6) { 7.0 })

        assertFailsWith<DimensionMismatch> { factor.solveInto(F64DenseMatrix(2, 2), out) }

        assertContentEquals(DoubleArray(6) { 7.0 }, out.data)
    }

    @Test
    fun `portable reports expose permutations and distinguish missing statistics`() {
        val report = F64ReferenceSparseLinearAlgebra.factor(matrix()).report()

        assertEquals("reference", report.provider)
        assertEquals(3, report.order)
        assertNotNull(report.factorNonzeros)
        assertEquals(3, report.rowPermutation?.size)
        assertEquals(3, report.columnPermutation?.size)
        assertEquals("markowitz-threshold", report.ordering)
        assertNull(report.memoryBytes)
        assertNull(report.inertia)
    }

    @Test
    fun `ldl report exposes valid zero inertia separately from unavailable`() {
        val diagonal = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 2.0), listOf(1 to -3.0), listOf(2 to 4.0)),
        )
        val report = F64SparseUpLookingLdl.factorLower(diagonal).report()

        assertEquals(F64FactorizationInertia(positive = 2, negative = 1, zero = 0), report.inertia)
    }

    private fun matrix() = F64SparseMatrix.ofColumns(
        3,
        3,
        listOf(
            listOf(0 to 4.0, 1 to 1.0),
            listOf(0 to 1.0, 1 to 5.0, 2 to 1.0),
            listOf(1 to 1.0, 2 to 6.0),
        ),
    )

    private fun rightHandSides(): F64DenseMatrix = F64DenseMatrix(
        3,
        3,
        doubleArrayOf(1.0, 2.0, 3.0, -1.0, 4.0, 2.0, 0.5, -2.0, 1.0),
    )
}
