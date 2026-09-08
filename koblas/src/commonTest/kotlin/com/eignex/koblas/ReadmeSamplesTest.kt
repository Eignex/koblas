package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.sparse.sparseConformanceSystem
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ReadmeSamplesTest {

    @Test
    fun `the sparse sample factorizes and solves both directions`() {
        val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
        val s = SparseMatrix.ofColumns(2, 2, cols)
        val lu = s.lu()
        val forward = lu.solve(doubleArrayOf(3.0, 5.0)) // B x = b
        val backward = lu.solve(doubleArrayOf(3.0, 5.0), transpose = true) // Bᵀ x = b
        assertClose(doubleArrayOf(0.8, 1.4), forward, "README ftran sample", tolerance = 1e-9)
        assertClose(koblas.gemv(s, backward, transpose = true), doubleArrayOf(3.0, 5.0), "README btran", 1e-9)
    }

    @Test
    fun `the triplet sample builds the same matrix the column sample does`() {
        val s = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 0, 1),
            colIdx = intArrayOf(0, 0, 1, 1),
            values = doubleArrayOf(2.0, 1.0, 1.0, 3.0),
        )
        val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
        assertEquals(
            s,
            SparseMatrix.ofColumns(2, 2, cols),
            "README triplet sample should match the column one",
        )
    }

    @Test
    fun `the strict sparse allocation sample reserves its declared scratch`() {
        val n = 4
        val sparseFactor = sparseConformanceSystem(n, Random(20260829)).lu()
        val sparseRhs = DoubleArray(n) { it + 1.0 }
        val sparseSolution = DoubleArray(n)
        val solveWorkspace = Workspace()
        sparseFactor.solveAllocation(aliasing = false).scratch.forEach(solveWorkspace::reserve)

        sparseFactor.solveInto(
            sparseRhs,
            sparseSolution,
            workspace = solveWorkspace,
            allocationPolicy = AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED,
        )

        assertClose(
            sparseRhs,
            koblas.gemv(sparseConformanceSystem(n, Random(20260829)), sparseSolution),
            "strict sample",
        )
    }

    @Test
    fun `koblasInfo has the shape the sample shows`() {
        assertTrue(koblasInfo.startsWith("backend="), koblasInfo)
        assertTrue(", kernels=" in koblasInfo, koblasInfo)
    }
}
