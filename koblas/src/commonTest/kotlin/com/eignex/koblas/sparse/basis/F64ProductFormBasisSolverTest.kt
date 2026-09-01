package com.eignex.koblas.sparse.basis

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.sparseConformanceSystem
import kotlin.random.Random
import kotlin.test.*

class F64ProductFormBasisSolverTest {
    private val reference = F64ReferenceSparseLinearAlgebra

    /** `[S | I]`, the shape a simplex hands over: structural columns and then the logical ones. */
    private fun simplexMatrix(n: Int, rng: Random): F64SparseMatrix {
        val structural = sparseConformanceSystem(n, rng)
        val columns = ArrayList<List<Pair<Int, Double>>>(2 * n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            structural.forEachInColumn(j) { i, v -> column.add(i to v) }
            columns.add(column)
        }
        for (i in 0 until n) columns.add(listOf(i to 1.0))
        return F64SparseMatrix.ofColumns(n, 2 * n, columns)
    }

    private fun logicalBasis(n: Int) = IntArray(n) { n + it }

    /** The basis of [basicIndex] as a standalone matrix, so a check never goes through the seam. */
    private fun basisOf(a: F64SparseMatrix, basicIndex: IntArray): F64SparseMatrix {
        val columns = basicIndex.map { j ->
            val column = ArrayList<Pair<Int, Double>>()
            a.forEachInColumn(j) { i, v -> column.add(i to v) }
            column
        }
        return F64SparseMatrix.ofColumns(a.rows, basicIndex.size, columns)
    }

    private fun solved(solver: F64BasisSolver, b: DoubleArray, transpose: Boolean): DoubleArray {
        val x = F64IndexedVector(b.size)
        x.scatter(b)
        if (transpose) solver.btran(x) else solver.ftran(x)
        return x.toDoubleArray()
    }

    @Test
    fun `a fresh solver solves as the basis it factorized`() {
        val rng = Random(20260901)
        val n = 9
        val a = simplexMatrix(n, rng)
        val basicIndex = IntArray(n) { it }
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val solver = reference.basisSolver(a)

        assertTrue(solver.refactorize(basicIndex))

        val expected = reference.factor(basisOf(a, basicIndex))
        for (transpose in booleanArrayOf(false, true)) {
            assertClose(expected.solve(b, transpose), solved(solver, b, transpose), "transpose=$transpose", 1e-12)
        }
    }

    @Test
    fun `basis quality measures a completed solve`() {
        val n = 8
        val a = simplexMatrix(n, Random(20261101))
        val basicIndex = IntArray(n) { it }
        val rhs = DoubleArray(n) { (it - 3).toDouble() }
        val solution = F64IndexedVector(n)
        val solver = reference.basisSolver(a)

        assertTrue(solver.refactorize(basicIndex))
        solution.scatter(rhs)
        solver.ftran(solution)

        assertTrue(solver.rcond > 0.0)
        assertTrue(solver.solveQuality(rhs, solution).relativeResidual <= 1e-12)
    }

    @Test
    fun `an update solves as the basis it names`() {
        val rng = Random(20260902)
        val n = 9
        val a = simplexMatrix(n, rng)
        val basicIndex = logicalBasis(n)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val solver = reference.basisSolver(a)
        solver.refactorize(basicIndex)

        val spike = F64IndexedVector(n)
        spike.scatterColumn(a, 4)
        solver.ftran(spike)
        assertEquals(BasisUpdate.APPLIED, solver.update(4, 4, spike))

        basicIndex[4] = 4
        val expected = reference.factor(basisOf(a, basicIndex))
        for (transpose in booleanArrayOf(false, true)) {
            assertClose(expected.solve(b, transpose), solved(solver, b, transpose), "transpose=$transpose", 1e-10)
        }
    }

    @Test
    fun `a chain of updates solves as the basis it names`() {
        val rng = Random(20260903)
        val n = 11
        val a = simplexMatrix(n, rng)
        val basicIndex = logicalBasis(n)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val solver = reference.basisSolver(a)
        solver.refactorize(basicIndex)

        val spike = F64IndexedVector(n)
        for (slot in 0 until n) {
            spike.scatterColumn(a, slot)
            solver.ftran(spike)
            solver.update(slot, slot, spike)
            basicIndex[slot] = slot
        }

        val expected = reference.factor(basisOf(a, basicIndex))
        for (transpose in booleanArrayOf(false, true)) {
            assertClose(expected.solve(b, transpose), solved(solver, b, transpose), "transpose=$transpose", 1e-9)
        }
    }

    @Test
    fun `refactorizing at the basis the updates reached solves the same way`() {
        val rng = Random(20260904)
        val n = 8
        val a = simplexMatrix(n, rng)
        val basicIndex = logicalBasis(n)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val solver = reference.basisSolver(a)
        solver.refactorize(basicIndex)
        val spike = F64IndexedVector(n)
        for (slot in 0 until n step 2) {
            spike.scatterColumn(a, slot)
            solver.ftran(spike)
            solver.update(slot, slot, spike)
            basicIndex[slot] = slot
        }
        val updated = solved(solver, b, transpose = false)

        solver.refactorize(basicIndex)

        assertEquals(0, solver.updateCount)
        assertClose(updated, solved(solver, b, transpose = false), "the rebuilt basis", 1e-9)
    }

    @Test
    fun `an update past the eta limit asks for a rebuild`() {
        val rng = Random(20260905)
        val n = 6
        val a = simplexMatrix(n, rng)
        val solver = F64ProductFormBasisSolver(a, reference, etaLimit = 2)
        solver.refactorize(logicalBasis(n))

        val outcomes = (0 until 3).map { slot ->
            val spike = F64IndexedVector(n)
            spike.scatterColumn(a, slot)
            solver.ftran(spike)
            solver.update(slot, slot, spike)
        }

        assertEquals(listOf(BasisUpdate.APPLIED, BasisUpdate.REFACTORIZE, BasisUpdate.REFACTORIZE), outcomes)
    }

    @Test
    fun `an update on a zero pivot is refused`() {
        val rng = Random(20260906)
        val n = 5
        val a = simplexMatrix(n, rng)
        val solver = reference.basisSolver(a)
        solver.refactorize(logicalBasis(n))
        val spike = F64IndexedVector(n)
        spike.store(2, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
        assertEquals(0, solver.updateCount)
    }

    @Test
    fun `a singular basis is reported rather than solved against`() {
        val n = 3
        val a = F64SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = reference.basisSolver(a)

        assertFalse(solver.refactorize(IntArray(n) { it }))
        assertTrue(solver.singular)
        assertFailsWith<SingularMatrix> { solved(solver, DoubleArray(n), transpose = false) }
    }

    @Test
    fun `solving before a basis is factorized is refused`() {
        val rng = Random(20260907)
        val n = 4
        val solver = reference.basisSolver(simplexMatrix(n, rng))

        assertFailsWith<SingularMatrix> { solved(solver, DoubleArray(n), transpose = false) }
    }

    @Test
    fun `a basicIndex of the wrong length is rejected`() {
        val rng = Random(20260908)
        val n = 4
        val solver = reference.basisSolver(simplexMatrix(n, rng))

        assertFailsWith<DimensionMismatch> { solver.refactorize(IntArray(n - 1)) }
    }

    @Test
    fun `an update before a basis is factorized is refused`() {
        val rng = Random(20260910)
        val n = 5
        val solver = reference.basisSolver(simplexMatrix(n, rng))
        val spike = F64IndexedVector(n)
        spike.store(0, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
        assertEquals(0, solver.updateCount)
    }

    @Test
    fun `an update on a singular basis is refused`() {
        val n = 3
        val a = F64SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = reference.basisSolver(a)
        assertFalse(solver.refactorize(IntArray(n) { it }))
        val spike = F64IndexedVector(n)
        spike.store(0, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
    }

    /** A backend that gives up outright rather than answering singular, which a rebuild must survive. */
    private class GivingUpSparseLu(private val delegate: F64SparseDecompositions) :
        F64SparseDecompositions by delegate {
        var refuse: Boolean = false

        override fun factor(a: F64SparseMatrix): F64SparseLuFactorization {
            check(!refuse) { "the library gave up" }
            return delegate.factor(a)
        }
    }

    @Test
    fun `a rebuild that throws leaves the basis the solver already had`() {
        val rng = Random(20260911)
        val n = 6
        val a = simplexMatrix(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val lu = GivingUpSparseLu(reference)
        val solver = F64ProductFormBasisSolver(a, lu)
        solver.refactorize(logicalBasis(n))
        val before = solved(solver, b, transpose = false)

        lu.refuse = true
        assertFailsWith<IllegalStateException> { solver.refactorize(IntArray(n) { it }) }

        assertContentEquals(before, solved(solver, b, transpose = false))
    }

    @Test
    fun `the fill it reports grows with the chain`() {
        val rng = Random(20260909)
        val n = 7
        val a = simplexMatrix(n, rng)
        val solver = reference.basisSolver(a)
        solver.refactorize(logicalBasis(n))
        val factored = solver.nnz

        val spike = F64IndexedVector(n)
        spike.scatterColumn(a, 0)
        solver.ftran(spike)
        solver.update(0, 0, spike)

        assertTrue(solver.nnz > factored, "an eta adds to the fill: $factored then ${solver.nnz}")
    }

    @Test
    fun `rebuild and close release each owned base factor once`() {
        val n = 5
        val tracking = TrackingSparseLu(reference)
        val solver = F64ProductFormBasisSolver(simplexMatrix(n, Random(20261011)), tracking)

        solver.refactorize(logicalBasis(n))
        solver.refactorize(logicalBasis(n))
        solver.close()
        solver.close()

        assertEquals(listOf(1, 1), tracking.factors.map { it.closes })
        assertFailsWith<IllegalStateException> { solver.refactorize(logicalBasis(n)) }
    }

    private class TrackingSparseLu(private val delegate: F64SparseDecompositions) :
        F64SparseDecompositions by delegate {
        val factors = ArrayList<TrackingFactor>()

        override fun factor(a: F64SparseMatrix): F64SparseLuFactorization =
            TrackingFactor(delegate.factor(a)).also(factors::add)
    }

    private class TrackingFactor(private val delegate: F64SparseLuFactorization) :
        F64SparseLuFactorization by delegate {
        var closes = 0
            private set

        override fun close() {
            closes++
            delegate.close()
        }
    }
}
