package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.*

/** Triangular-result general product, including transpose, triangle, depth, and tile-edge variants. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class GemmtBenchmark {
    @Param("4x3", "9x5", "127x31", "128x129", "129x257", "256x129", "512x127")
    var productShape: String = "4x3"

    @Param("false", "true") var transposeA: Boolean = false
    @Param("false", "true") var transposeB: Boolean = false
    @Param("true", "false") var lower: Boolean = true
    @Param(BUILTIN_BACKEND, OPENBLAS_BACKEND, ONEMKL_BACKEND) var denseArm: String = BUILTIN_BACKEND

    private lateinit var arm: DenseBenchmarkArm
    private lateinit var a: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var c: DenseMatrix

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val (n, k) = productShape.split('x').map(String::toInt)
        val rng = benchRng()
        a = if (transposeA) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
        b = if (transposeB) randomMatrix(n, k, rng) else randomMatrix(k, n, rng)
        c = DenseMatrix.zero(n)
        reportAllocatingWorkload("gemmt/$denseArm/$productShape", "built-in packed workspace or foreign boundary wrappers")
    }

    @Benchmark
    fun gemmt(): DenseMatrix {
        arm.external?.gemmt(1.0, a, transposeA, b, transposeB, 0.0, c, lower)
            ?: arm.context!!.gemmt(1.0, a, transposeA, b, transposeB, 0.0, c, lower)
        return c
    }
}
