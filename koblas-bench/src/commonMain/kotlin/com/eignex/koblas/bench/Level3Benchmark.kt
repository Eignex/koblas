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
