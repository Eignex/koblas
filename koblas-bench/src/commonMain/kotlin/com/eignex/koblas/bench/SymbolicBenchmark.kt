package com.eignex.koblas.bench

import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.sparse.SparseOrdering
import com.eignex.koblas.sparse.SparseSymbolic
import com.eignex.koblas.sparse.analyze
import kotlinx.benchmark.*

/**
 * The symbolic phase on its own, which the fill-reducing ordering dominates. Both shapes have `O(n)`
 * nonzeros, so anything here that grows faster than `n` is the ordering and not the matrix.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SymbolicBenchmark {
    @Param("500", "2000", "8000")
    var n: Int = 0

    private lateinit var band: F64SparseMatrix
    private lateinit var grid: F64SparseMatrix

    @Setup
    fun setup() {
        band = bandUpperTriangle(n)
        grid = gridUpperTriangle(n)
    }

    @Benchmark
    fun bandMinimumDegree(): SparseSymbolic = band.analyze()

    @Benchmark
    fun bandNatural(): SparseSymbolic = band.analyze(SparseOrdering.Natural)

    @Benchmark
    fun gridMinimumDegree(): SparseSymbolic = grid.analyze()
}
