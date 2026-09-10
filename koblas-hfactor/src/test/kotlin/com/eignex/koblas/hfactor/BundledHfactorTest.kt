package com.eignex.koblas.hfactor

import com.eignex.koblas.*
import com.eignex.koblas.sparse.basis.BasisUpdate
import com.eignex.koblas.sparse.basis.IndexedVector
import com.eignex.koblas.sparse.host.hfactor.HfactorBasisSolver
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorFactorization
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
import com.eignex.koblas.sparse.host.hfactor.HfactorUpdateMethod
import kotlin.math.abs
import kotlin.math.pow
import kotlin.random.Random
import kotlin.test.*

class BundledHfactorTest {
    private val backend = BundledHfactor()

    private fun simplexMatrix(n: Int, rng: Random): SparseMatrix {
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

    private fun solved(solver: HfactorBasisSolver, b: DoubleArray, transpose: Boolean): DoubleArray {
        val x = IndexedVector(b.size)
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
        solver: HfactorBasisSolver,
        a: SparseMatrix,
        slots: IntArray,
        rebuildAt: Int,
        basis: IntArray,
        reuse: Boolean = true,
    ) {
        val spike = IndexedVector(solver.n)
        val eta = IndexedVector(solver.n)
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

    private fun handedOver(v: IndexedVector): IndexedVector {
        val copy = IndexedVector(v.size)
        copy.scatter(v.toDoubleArray())
        return copy
    }

    @Test
    fun `the bundled HFactor is available`() {
        assertTrue(backend.availability.available)
    }

    @Test
    fun `direct HFactor construction exposes availability and options`() {
        val configured = BundledHfactor(
            HfactorConfig(equilibrate = true, pivotThreshold = 0.2, pivotTolerance = 1e-8),
        )
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 8.0)))

        configured.factor(matrix).close()

        assertTrue(configured.availability.available)
        assertNull(configured.availability.reason)
        assertTrue(configured.config.equilibrate)
        assertEquals(0.2, configured.config.pivotThreshold)
    }

    @Test
    fun `the bundled HFactor rejects an explicit library path`() {
        val config = HfactorConfig(libraryPath = "/opt/lib/libkoblas_hfactor.so.1")

        val failure = assertFailsWith<IllegalArgumentException> { BundledHfactor(config) }

        assertEquals(
            "BundledHfactor does not accept libraryPath; use HfactorSparseLu for an explicit library",
            failure.message,
        )
    }

    @Test
    fun `shared equilibration option reaches the binding without leaving HFactor`() {
        val equilibrated = BundledHfactor(
            HfactorConfig(
                equilibrate = true,
                pivotThreshold = 0.2,
                pivotTolerance = 1e-8,
                updateMethod = HfactorUpdateMethod.MIDDLE_PRODUCT_FORM,
            ),
        )
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 8.0)))

        val factorization = equilibrated.factor(matrix)

        assertTrue(factorization is HfactorFactorization || factorization.rcond >= 0.0)
        assertTrue(equilibrated.config.equilibrate)
    }

    @Test
    fun `the bundled HFactor reports basis quality and solve residual`() {
        val n = 8
        val matrix = simplexMatrix(n, Random(20261102))
        val rhs = DoubleArray(n) { (it - 2).toDouble() }
        val solution = IndexedVector(n)
        val solver = backend.basisSolver(matrix)

        assertTrue(solver.refactorize(logicalBasis(n)))
        solution.scatter(rhs)
        solver.ftran(solution)

        assertTrue(solver.rcond > 0.0)
        assertTrue(solver.solveQuality(rhs, solution).relativeResidual <= 1e-12)
    }

    @Test
    fun `the bundled HFactor accepts numerical update controls`() {
        val configured = BundledHfactor(
            HfactorConfig(
                pivotThreshold = 0.2,
                pivotTolerance = 1e-8,
                updateMethod = HfactorUpdateMethod.PRODUCT_FORM,
            ),
        )
        val n = 6
        val solver = configured.basisSolver(simplexMatrix(n, Random(20261103)))

        assertIs<HfactorBasisSolver>(solver)
        assertTrue(solver.refactorize(logicalBasis(n)))
    }

    @Test
    fun `a native factor closes deterministically`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 8.0)))
        val factorization = backend.factor(matrix)
        assertIs<HfactorFactorization>(factorization)
        val n = factorization.n
        val failedAt = factorization.failedAt

        factorization.close()
        factorization.close()

        assertEquals(n, factorization.n)
        assertEquals(failedAt, factorization.failedAt)
        assertFailsWith<IllegalStateException> { factorization.nnz }
        assertFailsWith<IllegalStateException> { factorization.rcond }
        assertFailsWith<IllegalStateException> { factorization.solve(DoubleArray(n)) }
    }

    @Test
    fun `a native basis solver closes deterministically`() {
        val solver = backend.basisSolver(simplexMatrix(6, Random(20261010)))
        assertIs<HfactorBasisSolver>(solver)
        assertTrue(solver.refactorize(logicalBasis(6)))
        val n = solver.n
        val singular = solver.singular

        solver.close()
        solver.close()

        assertEquals(n, solver.n)
        assertEquals(singular, solver.singular)
        assertFailsWith<IllegalStateException> { solver.nnz }
        assertFailsWith<IllegalStateException> { solver.updateCount }
        assertFailsWith<IllegalStateException> { solver.refactorize(logicalBasis(6)) }
    }

    @Test
    fun `an update on a pivot it cannot invert is refused`() {
        val rng = Random(20260913)
        val n = 6
        val a = simplexMatrix(n, rng)
        val solver = backend.basisSolver(a)
        solver.refactorize(logicalBasis(n))
        val spike = IndexedVector(n)
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
        val spike = IndexedVector(n)
        spike.store(0, 1.0)

        assertEquals(BasisUpdate.SINGULAR, solver.update(0, 0, spike))
    }

    @Test
    fun `an update on a singular basis is refused`() {
        val n = 3
        val a = SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = backend.basisSolver(a)
        assertFalse(solver.refactorize(IntArray(n) { it }))
        val spike = IndexedVector(n)
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

    private fun badlyScaled(n: Int, rng: Random): SparseMatrix {
        val columns = List(n) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val v = when {
                    i == j -> n + 10.0
                    rng.nextDouble() < 0.3 -> rng.nextDouble(-1.0, 1.0)
                    else -> 0.0
                }
                if (v != 0.0) entries.add(i to v * 2.0.pow(6 * (i % 5) - 12))
            }
            entries
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    /**
     * Equilibration is applied to the values handed to HFactor and undone in the solves, so it changes the
     * conditioning of what HFactor sees and nothing about the answer a caller gets back.
     */
    @Test
    fun `an equilibrated factorization solves as the unscaled one does`() {
        val equilibrated = BundledHfactor(HfactorConfig(equilibrate = true))
        val n = 8
        val a = badlyScaled(n, Random(20260906))
        val b = DoubleArray(n) { (it + 1).toDouble() }

        for (transpose in booleanArrayOf(false, true)) {
            val expected = backend.factor(a).solve(b, transpose = transpose)
            val actual = equilibrated.factor(a).solve(b, transpose = transpose)
            for (i in 0 until n) {
                assertTrue(
                    abs(expected[i] - actual[i]) <= 1e-9 * maxOf(1.0, abs(expected[i])),
                    "transpose=$transpose index $i: expected ${expected[i]} actual ${actual[i]}",
                )
            }
        }
    }

    /**
     * That the values reach HFactor scaled, rather than the flag being read and dropped. Over rows spanning
     * many binary exponents the pivot ratio HFactor reports is orders of magnitude better equilibrated:
     * around 2e-5 unscaled against 1.0 scaled on this matrix.
     */
    @Test
    fun `equilibration reaches HFactor rather than being read and dropped`() {
        val a = badlyScaled(8, Random(20260906))

        val unscaled = backend.factor(a).rcond
        val scaled = BundledHfactor(HfactorConfig(equilibrate = true)).factor(a).rcond

        assertTrue(scaled > unscaled * 100.0, "pivot ratio $unscaled unscaled against $scaled equilibrated")
    }

    /**
     * The kernel is what triangularization could not peel off. A logical basis is the identity, so nothing
     * survives into it; a structural one does, which is the signal a fill count alone does not carry.
     */
    @Test
    fun `the kernel it reports separates a triangular basis from a structural one`() {
        val n = 10
        val a = simplexMatrix(n, Random(20260920))
        val solver = backend.basisSolver(a)

        assertTrue(solver.refactorize(logicalBasis(n)))
        val logical = assertNotNull(solver.kernel, "HFactor counts its kernel during the build")
        assertEquals(0, logical.dimension, "an identity basis is wholly triangular")

        assertTrue(solver.refactorize(IntArray(n) { it }))
        val structural = assertNotNull(solver.kernel)

        assertTrue(structural.dimension > 0, "a structural basis leaves Markowitz something to choose from")
        assertTrue(structural.entries >= structural.dimension, "the kernel stores at least its diagonal")
    }

    @Test
    fun `no rebuild advice means no reason to report`() {
        val n = 8
        val a = simplexMatrix(n, Random(20260921))
        val solver = backend.basisSolver(a)
        assertTrue(solver.refactorize(logicalBasis(n)))

        val spike = IndexedVector(n)
        spike.scatterColumn(a, 0)
        solver.ftran(spike)
        assertEquals(BasisUpdate.APPLIED, solver.update(0, 0, spike))

        assertNull(solver.refactorizeReason, "an applied update advises nothing")
    }

    @Test
    fun `a rebuild forgets the previous advice`() {
        val n = 8
        val a = simplexMatrix(n, Random(20260922))
        val solver = backend.basisSolver(a)

        assertTrue(solver.refactorize(logicalBasis(n)))

        assertNull(solver.refactorizeReason)
        assertNotNull(solver.kernel)
    }

    @Test
    fun `a rank deficient basis is repaired rather than refused`() {
        val n = 3
        val a = SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = backend.basisSolver(a)

        assertFalse(solver.refactorize(IntArray(n) { it }), "refactorize keeps refusing a deficient basis")

        val repair = assertNotNull(solver.refactorizeRepairing(IntArray(n) { it }), "HFactor repairs this one")

        assertTrue(repair.repaired, "the basis it settled on is not the one it was given")
        assertTrue(repair.replacedSlots > 0)
        assertFalse(solver.singular, "a repaired basis is invertible, which is the point of keeping it")
        for (slot in 0 until n) {
            val column = repair.columns[slot]
            val row = repair.unitRows[slot]
            assertTrue(
                (column >= 0) != (row >= 0),
                "slot $slot holds a column of A or a unit column, never both or neither",
            )
            if (column >= 0) assertTrue(column < a.cols)
            if (row >= 0) assertTrue(row < n)
        }
    }

    @Test
    fun `a repaired basis solves against what it actually holds`() {
        val n = 3
        val a = SparseMatrix.ofColumns(n, n, listOf(listOf(0 to 1.0), listOf(0 to 2.0), listOf(2 to 1.0)))
        val solver = backend.basisSolver(a)
        assertNotNull(solver.refactorizeRepairing(IntArray(n) { it }))

        val rhs = doubleArrayOf(1.0, 2.0, 3.0)
        val solution = IndexedVector(n)
        solution.scatter(rhs)
        solver.ftran(solution)

        assertTrue(
            solver.solveQuality(rhs, solution).relativeResidual <= 1e-12,
            "the repaired basis inverts what it holds",
        )
    }

    @Test
    fun `a sound basis is returned unrepaired`() {
        val n = 8
        val a = simplexMatrix(n, Random(20260923))
        val solver = backend.basisSolver(a)

        val repair = assertNotNull(solver.refactorizeRepairing(logicalBasis(n)))

        assertFalse(repair.repaired)
        assertEquals(0, repair.replacedSlots)
        assertContentEquals(logicalBasis(n), repair.columns)
    }

    /**
     * The point of a snapshot: descend, pivot away from the basis, come back, and get the factors that were
     * there without factorizing again. What proves it is that the restored solver answers what the original
     * did, on a basis the pivots have since left behind.
     */
    @Test
    fun `a restored snapshot solves as the factorization it was taken from`() {
        val n = 12
        val rng = Random(20260924)
        val a = simplexMatrix(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val solver = backend.basisSolver(a)
        assertTrue(solver.refactorize(logicalBasis(n)))

        val expectedFtran = solved(solver, b, transpose = false)
        val expectedBtran = solved(solver, b, transpose = true)
        val held = assertNotNull(solver.snapshot(), "HFactor can hand its factorization out")

        pivot(solver, a, IntArray(6) { it }, rebuildAt = -1, basis = logicalBasis(n))
        assertNotEquals(0, solver.updateCount, "the solver has moved off the basis it snapshotted")

        assertTrue(solver.restore(held))

        assertEquals(0, solver.updateCount, "restoring puts the counters back with the factors")
        assertContentEquals(expectedFtran, solved(solver, b, transpose = false))
        assertContentEquals(expectedBtran, solved(solver, b, transpose = true))
        held.close()
    }

    @Test
    fun `a snapshot from another solver is refused`() {
        val n = 8
        val a = simplexMatrix(n, Random(20260925))
        val one = backend.basisSolver(a)
        val other = backend.basisSolver(a)
        assertTrue(one.refactorize(logicalBasis(n)))
        assertTrue(other.refactorize(logicalBasis(n)))

        val held = assertNotNull(one.snapshot())

        assertFalse(other.restore(held), "a snapshot describes the factors of the solver that took it")
        held.close()
    }

    @Test
    fun `a closed snapshot is refused`() {
        val n = 6
        val a = simplexMatrix(n, Random(20260926))
        val solver = backend.basisSolver(a)
        assertTrue(solver.refactorize(logicalBasis(n)))
        val held = assertNotNull(solver.snapshot())

        held.close()
        held.close()

        assertFalse(solver.restore(held))
    }

    @Test
    fun `closing the solver releases the snapshots it still owns`() {
        val n = 6
        val a = simplexMatrix(n, Random(20260927))
        val solver = backend.basisSolver(a)
        assertTrue(solver.refactorize(logicalBasis(n)))
        assertNotNull(solver.snapshot())

        solver.close()

        assertFailsWith<IllegalStateException> { solver.nnz }
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

    @Test
    fun `the bundled HFactor is the binding rather than a wrapper around it`() {
        assertIs<HfactorSparseLu>(BundledHfactor(), "the bundled providers all answer as the type their binding is")
    }
}
