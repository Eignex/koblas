package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.*

/** Dedicated rank-k matrix for triangle, transpose, tile-edge, and rectangular-depth comparisons. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SyrkBenchmark {
    @Param(
        "4x3", "7x5", "8x5", "9x5", "127x31", "128x129",
        "129x257", "255x31", "256x129", "257x259", "512x127",
    )
    var rankShape: String = "4x3"

    @Param("false", "true")
    var transpose: Boolean = false

    @Param("true", "false")
    var lower: Boolean = true

    @Param(BUILTIN_BACKEND, OPENBLAS_BACKEND, ONEMKL_BACKEND)
    var denseArm: String = BUILTIN_BACKEND

    private lateinit var arm: DenseBenchmarkArm
    private lateinit var a: DenseMatrix
    private lateinit var c: DenseMatrix

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val dimensions = rankShape.split('x')
        val n = dimensions[0].toInt()
        val k = dimensions[1].toInt()
        a = if (transpose) randomMatrix(k, n, benchRng()) else randomMatrix(n, k, benchRng())
        c = DenseMatrix.zero(n, n)
        val allocationReason = if (arm.external == null) {
            "built-in packed workspace"
        } else {
            "benchmark foreign-function boundary wrappers"
        }
        reportAllocatingWorkload("syrk/$denseArm/$rankShape/t=$transpose/l=$lower", allocationReason)
    }

    @Benchmark
    fun syrk(): DenseMatrix {
        arm.external?.syrk(1.0, a, transpose, 0.0, c, lower) ?: arm.context!!.syrk(
            1.0,
            a,
            transpose,
            0.0,
            c,
            lower,
        )
        return c
    }
}
