package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.gemv
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.sparse.sparseConformanceSystem
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class ReadmeSamplesTest {

    @Test
    fun `the LU sample solves the system it claims`() {
        val rows = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
        val a = F64DenseMatrix.of(rows)
        val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
        assertClose(doubleArrayOf(0.8, 1.4), x, "README LU sample", tolerance = 1e-12)
    }

    @Test
    fun `the Cholesky sample factors and solves`() {
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        val chol = a.cholesky() // A = L·Lᵀ
        val xs = chol.solve(doubleArrayOf(3.0, 5.0))
        assertClose(doubleArrayOf(0.8, 1.4), xs, "README Cholesky sample", tolerance = 1e-12)
    }

    @Test
    fun `the sparse sample factorizes and solves both directions`() {
        val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
        val s = F64SparseMatrix.ofColumns(2, 2, cols)
        val lu = s.lu()
        val forward = lu.solve(doubleArrayOf(3.0, 5.0)) // B x = b
        val backward = lu.solve(doubleArrayOf(3.0, 5.0), transpose = true) // Bᵀ x = b
        assertClose(doubleArrayOf(0.8, 1.4), forward, "README ftran sample", tolerance = 1e-9)
        assertClose(s.gemv(backward, transpose = true), doubleArrayOf(3.0, 5.0), "README btran", 1e-9)
    }

    @Test
    fun `the README's failure-recovery sample compiles and recovers`() {
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 1.0)))
        val chol = try {
            a.cholesky()
        } catch (e: NotPositiveDefinite) {
            assertTrue(e.position >= 0, "the sample relies on the reported position")
            a.cholesky(CholeskyPolicy.Regularize())
        }
        assertTrue(chol.n == 2)
    }

    @Test
    fun `the triplet sample builds the same matrix the column sample does`() {
        val s = F64SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 0, 1),
            colIdx = intArrayOf(0, 0, 1, 1),
            values = doubleArrayOf(2.0, 1.0, 1.0, 3.0),
        )
        val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
        assertTrue(s == F64SparseMatrix.ofColumns(2, 2, cols), "README triplet sample should match the column one")
    }

    @Test
    fun `the workspace sample calls exactly what it shows`() {
        val n = 4
        val rng = Random(20260803)
        val a = wellConditioned(n, rng)
        val b = randomVector(n, rng)
        val identity = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val basis = identity.lu()
        val lu = a.lu()
        val anorm = a.norm1()
        val threshold = 1e-12
        val iterations = 3

        val ws = Workspace().apply { reserve(n, count = 5) }
        val x = DoubleArray(n)
        repeat(iterations) {
            basis.solveInto(b, x, workspace = ws) // no allocation
            if (koblas.rcond(lu, anorm, ws) < threshold) koblas.factorInto(a, lu)
        }
        assertClose(b, x, "README workspace sample", tolerance = 1e-12)
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

        assertClose(sparseRhs, sparseConformanceSystem(n, Random(20260829)).gemv(sparseSolution), "strict sample")
    }

    @Test
    fun `koblasInfo has the shape the sample shows`() {
        assertTrue(koblasInfo.startsWith("backend="), koblasInfo)
        assertTrue(", kernels=" in koblasInfo, koblasInfo)
    }
}
