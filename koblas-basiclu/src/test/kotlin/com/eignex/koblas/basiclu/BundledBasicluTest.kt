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

    @Test
    fun `a configured library path is preferred over the bundled build`() {
        val previous = System.getProperty("koblas.basiclu.path")
        System.setProperty("koblas.basiclu.path", "/nonexistent/libkoblas_basiclu.so.1")
        try {
            val backend = BundledBasiclu(factorizeMin = 0)
            assertEquals("basiclu", backend.name)
            assertFalse(backend.isAvailable, "the configured path does not exist, so nothing should resolve")
        } finally {
            if (previous == null) {
                System.clearProperty("koblas.basiclu.path")
            } else {
                System.setProperty("koblas.basiclu.path", previous)
            }
        }
    }

    @Test
    fun `the bundled build answers when no path is configured`() {
        assertEquals("basiclu-bundled", BundledBasiclu(factorizeMin = 0).name)
    }

    /**
     * Rows of 4 and 8 equilibrate by 2^-2 and 2^-3, so the scaling is active and every value stays a power
     * of two, which keeps the solve exact and the comparison with the portable one bit for bit.
     */
    @Test
    fun `an equilibrated factorization solves as the portable one does`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(0 to 2.0, 1 to 8.0)))
        val rhs = doubleArrayOf(8.0, 8.0)
        val factorization = BundledBasiclu(factorizeMin = 0).factor(matrix, equilibrate = true)

        assertContentEquals(
            F64ReferenceSparseLinearAlgebra.factor(matrix, equilibrate = true).solve(rhs),
            factorization.solve(rhs),
        )
    }

    @Test
    fun `an equilibrated transposed solve agrees with the portable one`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(0 to 2.0, 1 to 8.0)))
        val rhs = doubleArrayOf(8.0, 8.0)
        val factorization = BundledBasiclu(factorizeMin = 0).factor(matrix, equilibrate = true)

        assertContentEquals(
            F64ReferenceSparseLinearAlgebra.factor(matrix, equilibrate = true).solve(rhs, transpose = true),
            factorization.solve(rhs, transpose = true),
        )
    }

    @Test
    fun `an equilibrated solve leaves its right-hand side alone`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4e8), listOf(1 to 3e-7)))
        val rhs = doubleArrayOf(1.0, 2.0)

        BundledBasiclu(factorizeMin = 0).factor(matrix, equilibrate = true).solve(rhs)

        assertContentEquals(doubleArrayOf(1.0, 2.0), rhs)
    }
}
