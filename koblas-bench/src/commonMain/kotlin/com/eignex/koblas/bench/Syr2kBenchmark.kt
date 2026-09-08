package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.*

/** Dedicated rank-2k matrix for triangle, transpose, tile-edge, and rectangular-depth comparisons. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Syr2kBenchmark {
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
    private lateinit var b: DenseMatrix
    private lateinit var c: DenseMatrix

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val dimensions = rankShape.split('x')
        val n = dimensions[0].toInt()
        val k = dimensions[1].toInt()
        val rng = benchRng()
        a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
        b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
        c = DenseMatrix.zero(n, n)
        val allocationReason = if (arm.external == null) {
            "built-in packed workspace"
        } else {
            "benchmark foreign-function boundary wrappers"
        }
        reportAllocatingWorkload("syr2k/$denseArm/$rankShape/t=$transpose/l=$lower", allocationReason)
    }

    @Benchmark
    fun syr2k(): DenseMatrix {
        arm.external?.syr2k(1.0, a, b, transpose, 0.0, c, lower) ?: arm.context!!.syr2k(
            1.0,
            a,
            b,
            transpose,
            0.0,
            c,
            lower,
        )
        return c
    }
}
