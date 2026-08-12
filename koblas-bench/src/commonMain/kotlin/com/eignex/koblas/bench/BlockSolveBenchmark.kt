package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.LuDecomposition
import com.eignex.koblas.dense.lu
import com.eignex.koblas.koblas
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
class BlockSolveBenchmark {
    @Param("64", "256")
    var n: Int = 0

    @Param("1", "2", "4", "8", "16", "32", "64")
    var nrhs: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var factored: LuDecomposition
    private lateinit var block: DenseMatrix

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        factored = dominantMatrix(n, rng).lu()
        block = randomMatrix(n, nrhs, rng)
    }

    @Benchmark
    fun blockSolve(): DenseMatrix = koblas.solve(factored, block)

    @Benchmark
    fun columnSolves(): DoubleArray {
        var last = DoubleArray(0)
        for (c in 0 until nrhs) {
            val col = DoubleArray(n) { block[it, c] }
            last = koblas.solve(factored, col)
        }
        return last
    }
}
