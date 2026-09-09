package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/** End-to-end triangular multiplication across dispatch, panel and packed-block boundaries. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class TrmmBenchmark {
    @Param(
        "4x4",
        "15x64",
        "16x31",
        "16x32",
        "16x33",
        "17x64",
        "31x8",
        "32x128",
        "33x16",
        "63x129",
        "64x32",
        "65x127",
        "127x33",
        "128x256",
        "129x31",
        "255x8",
        "256x32",
        "257x16",
    )
    var shape: String = "16x32"

    @Param(
        "left-lower",
        "left-upper-transposed",
        "left-lower-unit",
        "right-lower",
        "right-upper-transposed",
        "right-lower-transposed-unit",
    )
    var variant: String = "left-lower"

    @Param(BUILTIN_BACKEND, OPENBLAS_BACKEND, ONEMKL_BACKEND)
    var denseArm: String = BUILTIN_BACKEND

    private lateinit var arm: DenseBenchmarkArm
    private lateinit var triangle: DenseMatrix
    private lateinit var source: DenseMatrix
    private lateinit var result: DenseMatrix
    private lateinit var workspace: Workspace
    private var lower: Boolean = true
    private var transpose: Boolean = false
    private var unitDiagonal: Boolean = false
    private var right: Boolean = false

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val dimensions = shape.split('x')
        val order = dimensions[0].toInt()
        val panel = dimensions[1].toInt()
        right = variant.startsWith("right")
        lower = "-lower" in variant
        transpose = "transposed" in variant
        unitDiagonal = "unit" in variant
        val rng = benchRng()
        triangle = dominantMatrix(order, rng)
        source = if (right) randomMatrix(panel, order, rng) else randomMatrix(order, panel, rng)
        result = DenseMatrix.zero(source.rows, source.cols)
        workspace = Workspace()
        reportAllocatingWorkload(
            "trmm/$denseArm/$shape/$variant",
            "input reset plus built-in source snapshot and packed panels",
        )
    }

    @Benchmark
    fun denseTrmm(): DenseMatrix {
        source.data.copyInto(result.data)
        arm.external?.trmm(
            triangle, result, lower, transpose, unitDiagonal, right, 1.0,
        ) ?: arm.context!!.trmm(
            triangle,
            result,
            lower,
            transpose,
            unitDiagonal,
            right,
            workspace = workspace,
        )
        return result
    }
}
