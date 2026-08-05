package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
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

/**
 * Matrix-vector products across a wide size range, out to sizes where memory bandwidth rather than call
 * dispatch decides.
 *
 * These are `O(n^2)` work over `O(n^2)` data, so a native backend cannot amortize its per-call
 * marshalling: this sweep is what established that `gemv` and `symv` belong on the portable kernels at
 * every size measured, and it guards that conclusion against changes on either side of the seam.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level2Benchmark {
    @Param("16", "64", "256", "1024", "2048")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var a: DenseMatrix
    private lateinit var sym: DenseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        a = randomMatrix(n, n, rng)
        sym = lowerSymmetricMatrix(n, rng)
        x = randomVector(n, rng)
        y = DoubleArray(n)
    }

    @Benchmark
    fun gemv() {
        koblas.gemv(1.0, a, x, 0.0, y)
    }

    @Benchmark
    fun gemvTransposed() {
        koblas.gemv(1.0, a, x, 0.0, y, transpose = true)
    }

    @Benchmark
    fun symv() {
        koblas.symv(1.0, sym, x, 0.0, y)
    }
}
