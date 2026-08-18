package com.eignex.koblas.bench

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.dense.F64LuDecomposition
import com.eignex.koblas.dense.lu
import com.eignex.koblas.koblas
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class BlockSolveBenchmark {
    @Param("64", "256")
    var n: Int = 0

    @Param("1", "2", "4", "8", "16", "32", "64")
    var nrhs: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var factored: F64LuDecomposition
    private lateinit var block: F64DenseMatrix

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        factored = dominantMatrix(n, rng).lu()
        block = randomMatrix(n, nrhs, rng)
    }

    @Benchmark
    fun blockSolve(): F64DenseMatrix = koblas.solve(factored, block)

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
