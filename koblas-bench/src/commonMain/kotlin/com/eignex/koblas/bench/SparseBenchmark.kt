package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseLu
import kotlin.random.Random
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/**
 * Sparse factorization and solve at a fixed off-diagonal density.
 *
 * The sparse kernels are portable Kotlin with no backend seam behind them, so there is no `backend`
 * parameter: a native BLAS has no bearing on CSC traversal.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: SparseMatrix
    private lateinit var rhs: DoubleArray

    @Setup
    fun setup() {
        val rng = Random(BENCH_SEED)
        a = sparseDominantMatrix(n, density = 0.01, rng = rng)
        rhs = randomVector(n, rng)
    }

    /** Sparse LU factorize plus an FTRAN solve, the pair a simplex iteration pays on refactorization. */
    @Benchmark
    fun sparseLuSolve(): DoubleArray {
        val lu = SparseLu.factorize(a, equilibrate = true) ?: return rhs
        return lu.ftran(rhs)
    }

    @Benchmark
    fun sparseGemv(): DoubleArray = a.gemv(rhs)
}
