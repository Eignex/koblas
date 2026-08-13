package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseLu
import com.eignex.koblas.sparse.gemv
import com.eignex.koblas.sparse.lu
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

    @Benchmark
    fun sparseLuSolve(): DoubleArray = a.lu(equilibrate = true).solve(rhs)

    @Benchmark
    fun sparseGemv(): DoubleArray = a.gemv(rhs)
}
