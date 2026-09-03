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
    private lateinit var update: DoubleArray

    @Setup
    fun setup() {
        installDenseBackend(backend)
        val rng = benchRng()
        a = spdMatrix(n, rng)
        factor = a.cholesky()
        b = randomVector(n, rng)
        update = randomVector(n, rng)
    }

    @Benchmark
    fun cholesky(): F64CholeskyDecomposition = a.cholesky()

    @Benchmark
    fun solveSpd(): DoubleArray = factor.solve(b)

    @Benchmark
    fun invertSpd(): F64DenseMatrix = factor.invert()

    /**
     * The update mutates the factor it is given, so this drives it with a sigma small enough that a whole
     * run's worth of updates cannot move the factored matrix enough to affect the timing. Copying the factor
     * per invocation instead would measure an n by n copy alongside the update, which is the same order.
     */
    @Benchmark
    fun rank1UpdateSpd(): F64CholeskyDecomposition = factor.rank1Update(update, 1e-12)
}
