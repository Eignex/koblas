package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.CholeskyDecomposition
import com.eignex.koblas.dense.invert
import com.eignex.koblas.dense.solve
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

/** koblas configures a host OpenBLAS single-threaded, so large sizes are not what a threaded build reports. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class CholeskyBenchmark {
    @Param("256", "1024", "2048")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var a: DenseMatrix
    private lateinit var factor: CholeskyDecomposition
    private lateinit var b: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        a = spdMatrix(n, rng)
        factor = a.cholesky()
        b = randomVector(n, rng)
    }

    @Benchmark
    fun cholesky(): CholeskyDecomposition = a.cholesky()

    @Benchmark
    fun solveSpd(): DoubleArray = factor.solve(b)

    @Benchmark
    fun invertSpd(): DenseMatrix = factor.invert()
}
