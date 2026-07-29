package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.installLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State
import kotlin.random.Random

/**
 * Level-2 products across a wide size range, isolated from the level-3 and factorization work so the
 * sweep can reach sizes where memory bandwidth rather than dispatch decides. These are the shapes
 * where a native backend's per-call cost competes against the portable SIMD kernels.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class GemvBenchmark {
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
        installLinearAlgebra(if (backend == "reference") ReferenceLinearAlgebra else null)
        println("resolved: $koblasInfo")
        val rng = Random(20260729)
        a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        sym = DenseMatrix(n, n)
        for (i in 0 until n) for (j in 0..i) sym[i, j] = rng.nextDouble(-1.0, 1.0)
        x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
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
