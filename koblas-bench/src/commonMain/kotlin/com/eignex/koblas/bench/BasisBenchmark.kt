package com.eignex.koblas.bench

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.basis.BasisUpdate
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64IndexedVector
import kotlinx.benchmark.*

/**
 * What a simplex asks a basis solver for: a factorization, and then pivots against it.
 *
 * Both rows do a whole determined piece of work, since a solver is mutable and a row measuring one pivot
 * would drift as the chain it leaves behind lengthened. The cost of a pivot is [PIVOTS] into the difference
 * between them.
 *
 * The shapes separate the two regimes the choice of solver turns on. Where the columns stay sparse the
 * solves are hypersparse and a solver that exploits that wins; where enough of them are spikes the vectors
 * fill and it has to stop.
 *
 * The HFactor bridge is koblas's own build rather than a library a host carries, so `auto` reaches it only
 * when a run points `koblas.hfactor.path` at one. This module does not depend on `koblas-hfactor`, which
 * would put a native build and a source download in front of every build of the benchmarks. Setup prints
 * the backend it resolved; read it before using a number.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class BasisBenchmark {
    @Param("256", "1024")
    var n: Int = 0

    @Param(SPARSE_COLUMNS, SPIKED_COLUMNS)
    var shape: String = SPARSE_COLUMNS

    @Param(REFERENCE_BACKEND, AUTO_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64SparseMatrix
    private lateinit var solver: F64BasisSolver
    private lateinit var logical: IntArray
    private lateinit var spike: F64IndexedVector
    private lateinit var eta: F64IndexedVector

    @Setup
    fun setup() {
        val portable = koblas.with(sparseLu = F64ReferenceSparseLinearAlgebra)
        installBackends(if (backend == REFERENCE_BACKEND) portable else null)
        val rng = benchRng()
        a = simplexProblem(n, rng, spikeFraction = if (shape == SPIKED_COLUMNS) 0.5 else 0.0)
        logical = IntArray(n) { n + it }
        solver = koblas.sparseLu.basisSolver(a)
        spike = F64IndexedVector(n)
        eta = F64IndexedVector(n)
        check(solver.refactorize(logical)) { "the logical basis did not factor" }
        println("resolved: sparseLu=${koblas.sparseLu.name} shape=$shape nnz(A)=${a.nnz} fill=${solver.nnz}")
    }

    @Benchmark
    fun refactorize(): Boolean = solver.refactorize(logical)

    @Benchmark
    fun pivots(): Int {
        solver.refactorize(logical)
        for (slot in 0 until PIVOTS) {
            spike.scatterColumn(a, slot)
            solver.ftran(spike, spike.density)
            eta.unit(slot)
            solver.btran(eta, eta.density)
            check(solver.update(slot, slot, spike, eta) != BasisUpdate.SINGULAR) { "pivot $slot was refused" }
        }
        return solver.updateCount
    }

    private companion object {
        /** Enough pivots that the row is dominated by them rather than by the factorization it starts with. */
        const val PIVOTS = 32
    }
}

/** Columns that stay sparse, so the solves are hypersparse throughout. */
internal const val SPARSE_COLUMNS = "sparse"

/** Half the columns are spikes, so the solves fill in and hypersparsity stops paying. */
internal const val SPIKED_COLUMNS = "spiked"
