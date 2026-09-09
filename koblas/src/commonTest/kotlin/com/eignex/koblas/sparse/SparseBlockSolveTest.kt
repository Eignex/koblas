package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.sparse.factorization.ldl.QuasiDefiniteUpLookingLdl
import kotlin.test.*

class SparseBlockSolveTest {

    private class BlockTrackingFactor(private val delegate: SparseFactorization) :
        SparseFactorization by delegate {
        var calls: Int = 0
        var transpose: Boolean? = null
        var receivedWorkspace: Workspace? = null

        override fun solveInto(
            b: DenseMatrix,
            out: DenseMatrix,
            transpose: Boolean,
            workspace: Workspace?,
        ): DenseMatrix {
            calls++
            this.transpose = transpose
            receivedWorkspace = workspace
            return delegate.solveInto(b, out, transpose, workspace)
        }
    }

    @Test
    fun `portable block solves agree with vector solves in both directions`() {
        val factor = ReferenceSparseLinearAlgebra.factor(matrix())
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
    fun `the sparse decomposition seam delegates block solves directly`() {
        val tracking = BlockTrackingFactor(ReferenceSparseLinearAlgebra.factor(matrix()))
        val rhs = rightHandSides()
        val out = DenseMatrix(3, rhs.cols)
        val workspace = Workspace()

        assertSame(out, ReferenceSparseLinearAlgebra.solveInto(tracking, rhs, out, transpose = true, workspace))
        assertEquals(1, tracking.calls)
        assertEquals(true, tracking.transpose)
        assertSame(workspace, tracking.receivedWorkspace)
    }

    @Test
    fun `block destination may alias every right hand side`() {
        val factor = ReferenceSparseLinearAlgebra.factor(matrix())
        val aliased = rightHandSides()
        val expected = factor.solve(aliased)

        factor.solveInto(aliased, aliased)

        assertContentEquals(expected.data, aliased.data)
    }

    @Test
    fun `block solve validates both matrix shapes before mutation`() {
        val factor = ReferenceSparseLinearAlgebra.factor(matrix())
        val out = DenseMatrix(3, 2, DoubleArray(6) { 7.0 })

        assertFailsWith<DimensionMismatch> { factor.solveInto(DenseMatrix(2, 2), out) }

        assertContentEquals(DoubleArray(6) { 7.0 }, out.data)
    }

    @Test
    fun `ldl exposes pivot inertia`() {
        val diagonal = SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 2.0), listOf(1 to -3.0), listOf(2 to 4.0)),
        )
        val factor = QuasiDefiniteUpLookingLdl.factorLower(diagonal)

        assertEquals(FactorizationInertia(positive = 2, negative = 1, zero = 0), factor.inertia)
    }

    private fun matrix() = SparseMatrix.ofColumns(
        3,
        3,
        listOf(
            listOf(0 to 4.0, 1 to 1.0),
            listOf(0 to 1.0, 1 to 5.0, 2 to 1.0),
            listOf(1 to 1.0, 2 to 6.0),
        ),
    )

    private fun rightHandSides(): DenseMatrix = DenseMatrix(
        3,
        3,
        doubleArrayOf(1.0, 2.0, 3.0, -1.0, 4.0, 2.0, 0.5, -2.0, 1.0),
    )
}
