package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class CholeskyBenchmark {
    @Param("256", "1024", "2048")
    var n: Int = 0

    @Param(REFERENCE_BACKEND, HOST_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var a: F64DenseMatrix
    private lateinit var factor: F64CholeskyDecomposition
    private lateinit var b: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        a = spdMatrix(n, rng)
        factor = a.cholesky()
        b = randomVector(n, rng)
    }

    @Benchmark
    fun cholesky(): F64CholeskyDecomposition = a.cholesky()

    @Benchmark
    fun solveSpd(): DoubleArray = factor.solve(b)

    @Benchmark
    fun invertSpd(): F64DenseMatrix = factor.invert()
}
