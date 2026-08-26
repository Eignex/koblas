package com.eignex.koblas.hfactor

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.basis.BasisUpdate
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64IndexedVector
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization
import com.eignex.koblas.sparse.host.hfactor.HfactorOptions
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
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
     *
     * With [reuse] off the update is handed vectors the solver did not fill, which is a caller that solved
     * some other way, and a backend reusing its own solves has to recompute them instead.
     */
    private fun pivot(
        solver: F64BasisSolver,
        a: F64SparseMatrix,
        slots: IntArray,
        rebuildAt: Int,
        basis: IntArray,
        reuse: Boolean = true,
    ) {
        val spike = F64IndexedVector(solver.n)
        val eta = F64IndexedVector(solver.n)
        for ((taken, slot) in slots.withIndex()) {
            if (taken == rebuildAt) solver.refactorize(basis)
            spike.scatterColumn(a, slot)
            solver.ftran(spike)
            val outcome = if (reuse) {
                eta.unit(slot)
                solver.btran(eta)
                solver.update(slot, slot, spike, eta)
            } else {
                solver.update(slot, slot, handedOver(spike))
            }
            assertNotEquals(BasisUpdate.SINGULAR, outcome, "pivot $slot")
            basis[slot] = slot
        }
    }

    /** The same vector by value and a different one by identity, which is all a reuse check goes on. */
    private fun handedOver(v: F64IndexedVector): F64IndexedVector {
        val copy = F64IndexedVector(v.size)
        copy.scatter(v.toDoubleArray())
        return copy
    }

    private fun assertAgreesWithPortable(n: Int, seed: Int, pivots: Int, rebuildAt: Int = -1, reuse: Boolean = true) {
        val rng = Random(seed)
        val a = simplexMatrix(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val slots = IntArray(pivots) { it }

        val host = backend.basisSolver(a)
        val reference = portable(a)
        assertTrue(host.refactorize(logicalBasis(n)), "the host called a logical basis singular")
        assertTrue(reference.refactorize(logicalBasis(n)), "the reference called a logical basis singular")
        pivot(host, a, slots, rebuildAt, logicalBasis(n), reuse)
        pivot(reference, a, slots, rebuildAt, logicalBasis(n), reuse)

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

    @Test
    fun `shared equilibration option reaches the bundled fallback policy`() {
        val equilibrated = BundledHfactor(HfactorOptions(factorizeMin = 0, equilibrate = true))
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 8.0)))

        val factorization = equilibrated.factor(matrix)

        assertIs<F64SparseLuFactorization>(factorization)
        assertEquals("true", equilibrated.backendMetadata.options["equilibrate"])
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

    /**
     * The bridge reuses the caller's own solves where it can, which is what a dual simplex hands it. A
     * caller that solved some other way leaves it to solve for the spike and the pivotal row itself, which
     * is the other path through the update and has to arrive at the same basis.
     */
    @Test
    fun `an update recomputing its own vectors solves as the portable one does`() {
        assertAgreesWithPortable(n = 12, seed = 20260915, pivots = 12, reuse = false)
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
    fun `an update before a basis is factorized is refused`() {
        val rng = Random(20260918)
        val n = 5
        val solver = backend.basisSolver(simplexMatrix(n, rng))
        val spike = F64IndexedVector(n)
        spike.store(0, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
    }

    @Test
    fun `an update on a singular basis is refused`() {
        val n = 3
        val a = SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = backend.basisSolver(a)
        assertFalse(solver.refactorize(IntArray(n) { it }))
        val spike = F64IndexedVector(n)
        spike.store(0, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
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

    /** Fill is tracked rather than read back from the factors, so what it counts is worth pinning. */
    @Test
    fun `the fill it reports grows with the chain`() {
        val rng = Random(20260916)
        val n = 10
        val a = simplexMatrix(n, rng)
        val solver = backend.basisSolver(a)
        solver.refactorize(logicalBasis(n))
        val factored = solver.nnz

        pivot(solver, a, IntArray(3) { it }, rebuildAt = -1, basis = logicalBasis(n))

        assertTrue(solver.nnz > factored, "an update adds to the fill: $factored then ${solver.nnz}")
    }

    /** A rebuild drops what the updates added, so the count starts again from the factors. */
    @Test
    fun `a rebuild reports the fill of the factors alone`() {
        val rng = Random(20260917)
        val n = 10
        val a = simplexMatrix(n, rng)
        val solver = backend.basisSolver(a)
        solver.refactorize(logicalBasis(n))
        val factored = solver.nnz

        pivot(solver, a, IntArray(3) { it }, rebuildAt = -1, basis = logicalBasis(n))
        solver.refactorize(logicalBasis(n))

        assertEquals(factored, solver.nnz)
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

    /** A caller reaching this library by name gets its own routines only if the two share a type. */
    @Test
    fun `the bundled HFactor is the binding rather than a wrapper around it`() {
        assertIs<HfactorSparseLu>(BundledHfactor(), "the bundled providers all answer as the type their binding is")
    }
}
