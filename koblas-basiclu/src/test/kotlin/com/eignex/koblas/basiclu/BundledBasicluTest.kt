package com.eignex.koblas.basiclu

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import kotlin.test.*

class BundledBasicluTest {
    @Test
    fun `the bundled BASICLU is available`() {
        assertTrue(BundledBasiclu(factorizeMin = 0).isAvailable)
    }

    @Test
    fun `the bundled BASICLU solves sparse systems in both directions`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))
        val factorization = BundledBasiclu(factorizeMin = 0).factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(9.0, 4.0)))
        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(6.0, 6.0), transpose = true))
    }

    @Test
    fun `the bundled BASICLU solves in place with a workspace`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))
        val values = doubleArrayOf(9.0, 4.0)

        BundledBasiclu(factorizeMin = 0).factor(matrix).solveInto(values, values, workspace = Workspace())

        assertContentEquals(doubleArrayOf(2.0, 3.0), values)
    }

    @Test
    fun `the bundled BASICLU replaces a basis column`() {
        val basis = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val entering = F64SparseVector.of(2, intArrayOf(0, 1), doubleArrayOf(1.0, 2.0))
        val expected = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 1.0, 1 to 2.0)))
        val factorization = BundledBasiclu(factorizeMin = 0).factorBasis(basis).replaceColumn(1, entering)

        assertContentEquals(
            F64ReferenceSparseLinearAlgebra.factor(expected).solve(doubleArrayOf(3.0, 4.0)),
            factorization.solve(doubleArrayOf(3.0, 4.0)),
        )
    }

    /**
     * `nnz` and `rcond` are the only readers of `koblas_basiclu_info`, so without them that downcall is
     * never made and a wrong descriptor for it would ship unnoticed. The operand carries off-diagonal fill,
     * since a permutation factors to no off-diagonal entries at all and would report nothing.
     */
    @Test
    fun `the bundled BASICLU reports its fill and pivot ratio`() {
        val matrix = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)),
        )
        val factorization = BundledBasiclu(factorizeMin = 0).factor(matrix)
        assertTrue(factorization.nnz > 0, "expected off-diagonal fill, got ${factorization.nnz}")
        assertTrue(
            factorization.rcond > 0.0 && factorization.rcond <= 1.0,
            "pivot ratio ${factorization.rcond} outside (0, 1]",
        )
    }
}
