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

/**
 * The symmetric positive-definite family: factorization, the solve against a factor, and the explicit
 * inverse.
 *
 * Sized for Gaussian-process work, where the factorization is repeated per likelihood evaluation while
 * hyperparameters are optimized, and `invertSpd` is needed explicitly for the marginal-likelihood gradient.
 *
 * Measured on one machine, us/op: JVM SIMD against the host OpenBLAS (single-threaded, as koblas configures
 * it) and the native scalar loops.
 *
 *     routine        n   native host   native scalar   JVM SIMD
 *     cholesky     256        1524.4          4944.7      696.3
 *     cholesky    1024       51372.3        219900.5    49233.8
 *     cholesky    2048      325546.8        883317.2   635218.9
 *     invertSpd    256        2227.9          7083.9     1245.8
 *     invertSpd   1024       59938.0        343201.1    73564.0
 *     invertSpd   2048      632306.9       1932392.0   686422.5
 *     solveSpd     256         207.2            66.1       10.6
 *     solveSpd    1024        6579.0           739.6      181.2
 *     solveSpd    2048       64710.6          5319.6      993.1
 *
 * Two caveats. These were taken while koblas stored matrices row-major, so every native call also paid for
 * LAPACKE transposing into a column-major temporary — gone now, and much of why the `O(n²)` `solveSpd` lost so
 * heavily. And OpenBLAS ran single-threaded; its threading would move the large sizes the other way. The JVM
 * gates were re-measured under column-major storage and moved; see `HostLapack`.
 */
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
