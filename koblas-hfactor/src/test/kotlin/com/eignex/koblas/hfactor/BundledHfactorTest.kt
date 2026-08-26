package com.eignex.koblas.hfactor

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.basis.BasisUpdate
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64IndexedVector
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

class BundledHfactorTest {
    private val backend = BundledHfactor(factorizeMin = 0)

    /** `[S | I]`, the shape a simplex hands over: structural columns and then the logical ones. */
    private fun simplexMatrix(n: Int, rng: Random): F64SparseMatrix {
        val columns = ArrayList<List<Pair<Int, Double>>>(2 * n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val v = when {
                    i == j -> n + 10.0
                    rng.nextDouble() < 0.25 -> rng.nextDouble(-1.0, 1.0)
                    else -> 0.0
                }
                if (v != 0.0) column.add(i to v)
            }
            columns.add(column)
        }
        for (i in 0 until n) columns.add(listOf(i to 1.0))
        return SparseMatrix.ofColumns(n, 2 * n, columns)
    }

    private fun logicalBasis(n: Int) = IntArray(n) { n + it }

    private fun portable(a: F64SparseMatrix) = F64ProductFormBasisSolver(a, F64ReferenceSparseLinearAlgebra)

    private fun solved(solver: F64BasisSolver, b: DoubleArray, transpose: Boolean): DoubleArray {
        val x = F64IndexedVector(b.size)
        x.scatter(b)
        if (transpose) solver.btran(x) else solver.ftran(x)
        return x.toDoubleArray()
    }

    /**
     * Walks [solver] through the pivots, each spike coming from that solver's own forward solve, which is
     * what a simplex hands back and what an update reads.
     */
    private fun pivot(solver: F64BasisSolver, a: F64SparseMatrix, slots: IntArray, rebuildAt: Int, basis: IntArray) {
        val spike = F64IndexedVector(solver.n)
        val eta = F64IndexedVector(solver.n)
        for ((taken, slot) in slots.withIndex()) {
            if (taken == rebuildAt) solver.refactorize(basis)
            spike.scatterColumn(a, slot)
            solver.ftran(spike)
            eta.unit(slot)
            solver.btran(eta)
            assertNotEquals(BasisUpdate.SINGULAR, solver.update(slot, slot, spike, eta), "pivot $slot")
            basis[slot] = slot
        }
    }

    private fun assertAgreesWithPortable(n: Int, seed: Int, pivots: Int, rebuildAt: Int = -1) {
        val rng = Random(seed)
        val a = simplexMatrix(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val slots = IntArray(pivots) { it }

        val host = backend.basisSolver(a)
        val reference = portable(a)
        assertTrue(host.refactorize(logicalBasis(n)), "the host called a logical basis singular")
        assertTrue(reference.refactorize(logicalBasis(n)), "the reference called a logical basis singular")
        pivot(host, a, slots, rebuildAt, logicalBasis(n))
        pivot(reference, a, slots, rebuildAt, logicalBasis(n))

        for (transpose in booleanArrayOf(false, true)) {
            val expected = solved(reference, b, transpose)
            val actual = solved(host, b, transpose)
            for (i in 0 until n) {
                assertTrue(
                    abs(expected[i] - actual[i]) <= 1e-9 * maxOf(1.0, abs(expected[i])),
                    "transpose=$transpose index $i: expected ${expected[i]} actual ${actual[i]}",
                )
            }
        }
    }

    @Test
    fun `the bundled HFactor is available`() {
        assertTrue(backend.isAvailable)
    }

    /**
     * Its updates are on [BundledHfactor.basisSolver]. `supportsBasisUpdates` is about what `factorBasis`
     * answers with, and that contract takes any entering column, which HFactor could update against but
     * could not rebuild from.
     */
    @Test
    fun `it does not claim the column replacement contract`() {
        assertFalse(backend.supportsBasisUpdates)
    }

    @Test
    fun `a freshly factorized basis solves as the portable one does`() {
        assertAgreesWithPortable(n = 12, seed = 20260910, pivots = 0)
    }

    @Test
    fun `a chain of updates solves as the portable one does`() {
        assertAgreesWithPortable(n = 12, seed = 20260911, pivots = 12)
    }

    /**
     * HFactor reorders the basis into its own pivot order as it factorizes, which a logical starting basis
     * hides because that permutation is the identity. Rebuilding partway leaves a basis that does not, so
     * this is where a binding taking HFactor's slots for the caller's would part company with the reference.
     */
    @Test
    fun `a rebuild partway through a chain solves as the portable one does`() {
        assertAgreesWithPortable(n = 14, seed = 20260912, pivots = 14, rebuildAt = 7)
    }

    @Test
    fun `an update on a pivot it cannot invert is refused`() {
        val rng = Random(20260913)
        val n = 6
        val a = simplexMatrix(n, rng)
        val solver = backend.basisSolver(a)
        solver.refactorize(logicalBasis(n))
        val spike = F64IndexedVector(n)
        spike.store(2, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
    }

    @Test
    fun `a singular basis is reported rather than solved against`() {
        val n = 3
        val a = SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = backend.basisSolver(a)

        assertFalse(solver.refactorize(IntArray(n) { it }))
        assertTrue(solver.singular)
    }

    @Test
    fun `the updates it folded in are counted`() {
        val rng = Random(20260914)
        val n = 10
        val a = simplexMatrix(n, rng)
        val solver = backend.basisSolver(a)
        solver.refactorize(logicalBasis(n))

        pivot(solver, a, IntArray(4) { it }, rebuildAt = -1, basis = logicalBasis(n))

        assertEquals(4, solver.updateCount)
    }

    @Test
    fun `the bundled HFactor solves sparse systems in both directions`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))
        val factorization = backend.factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(9.0, 4.0)))
        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(6.0, 6.0), transpose = true))
    }

    @Test
    fun `the bundled HFactor reports its fill and pivot ratio`() {
        val matrix = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)),
        )
        val factorization = backend.factor(matrix)

        assertTrue(factorization.nnz > 0, "expected fill, got ${factorization.nnz}")
        assertTrue(
            factorization.rcond > 0.0 && factorization.rcond <= 1.0,
            "pivot ratio ${factorization.rcond} outside (0, 1]",
        )
    }
}
