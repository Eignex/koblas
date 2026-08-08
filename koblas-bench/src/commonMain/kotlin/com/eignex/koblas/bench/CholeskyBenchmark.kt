package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.invertSpd
import com.eignex.koblas.dense.solveSpd
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
 * These carry Gaussian-process work, where the factorization is the dominant cost and is repeated for
 * every likelihood evaluation while hyperparameters are optimized — so the O(n^3) term is multiplied by
 * the optimizer's step count rather than paid once. Sizes reach 2048 because that is the range such a
 * workload occupies before it switches to inducing-point approximations.
 *
 * `invertSpd` is here because the marginal-likelihood gradient needs an explicit inverse, not just a
 * solve.
 *
 * Measured on one machine, us/op. The JVM ships no native backend, so its column is the SIMD kernels;
 * the native columns are the host OpenBLAS (single-threaded, as koblas configures it) and the scalar
 * loops:
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
 * Three things worth carrying forward. The SIMD factorization matches single-threaded OpenBLAS up to
 * n=1024 and only falls behind by 2x at 2048, so the JVM having no native LAPACK costs far less here
 * than the LU ratios suggested — this loop vectorizes well, where LU's pivoting and row swaps do not.
 * `solveSpd` is `O(n^2)` over `O(n^2)` data and loses natively, so it delegates. Two caveats on these
 * numbers: they were taken while koblas stored matrices row-major, so every native call also paid for
 * LAPACKE transposing the matrix into a column-major temporary — a tax that column-major storage removed and
 * explains much of why the `O(n^2)` routine lost so heavily. And OpenBLAS ran single-threaded; enabling
 * its threading would move the large sizes the other way.
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
    private lateinit var factor: DenseMatrix
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
    fun cholesky(): DenseMatrix = a.cholesky()

    @Benchmark
    fun solveSpd(): DoubleArray = solveSpd(factor, b)

    @Benchmark
    fun invertSpd(): DenseMatrix = invertSpd(factor)
}
