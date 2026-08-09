package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.matMul
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
 * Matrix-matrix products, the routines where a native backend earns its keep: `O(n^3)` work over `O(n^2)`
 * data leaves per-call marshalling far behind, and a blocked native kernel beats the portable one outright.
 *
 * `syrk` is measured in both triangle modes because they take different paths. [Uplo.FULL] is this
 * library's extension and writes both triangles, which a native backend reaches by mirroring into
 * scratch; [Uplo.LOWER] is strict BLAS `dsyrk` and needs no mirror.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level3Benchmark {
    @Param("4", "16", "32", "64", "128", "256")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var a: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var c: DenseMatrix

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
        a = randomMatrix(n, n, rng)
        b = randomMatrix(n, n, rng)
        c = DenseMatrix.zero(n, n)
    }

    @Benchmark
    fun gemm(): DenseMatrix = a.matMul(b)

    @Benchmark
    fun syrkFull(): DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c)
        return c
    }

    @Benchmark
    fun syrkLower(): DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c, uplo = Uplo.LOWER)
        return c
    }
}
