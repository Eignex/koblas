package com.eignex.koblas.bench

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.CholeskyDecomposition
import com.eignex.koblas.dense.invert
import com.eignex.koblas.dense.solve
import com.eignex.koblas.koblas
import kotlinx.benchmark.*

/** koblas configures a host OpenBLAS single-threaded, so large sizes are not what a threaded build reports. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class CholeskyBenchmark {
    @Param("256", "1024", "2048")
    var n: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var a: F64DenseMatrix
    private lateinit var factor: CholeskyDecomposition
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
    fun cholesky(): CholeskyDecomposition = a.cholesky()

    @Benchmark
    fun solveSpd(): DoubleArray = factor.solve(b)

    @Benchmark
    fun invertSpd(): F64DenseMatrix = factor.invert()
}
