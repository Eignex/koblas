package com.eignex.koblas.bench

import com.eignex.koblas.Workspace
import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/** End-to-end triangular solve coverage for panel widths, packed block edges and BLAS orientations. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class TrsmBenchmark {
    @Param(
        "4x64",
        "15x64",
        "16x64",
        "17x64",
        "31x8",
        "32x31",
        "32x64",
        "32x32",
        "32x33",
        "33x64",
        "63x128",
        "64x16",
        "65x128",
        "128x256",
    )
    var shape: String = "16x64"

    @Param(
        "right-lower-transposed",
        "right-lower",
        "right-upper",
        "left-lower",
        "left-upper-transposed",
        "right-lower-unit",
    )
    var variant: String = "right-lower-transposed"

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
            "trsm/$denseArm/$shape/$variant",
            "input reset plus built-in packing workspace",
        )
    }

    @Benchmark
    fun denseTrsm(): DenseMatrix {
        source.data.copyInto(result.data)
        arm.external?.trsm(
            triangle, result, lower, transpose, unitDiagonal, right, 1.0,
        ) ?: arm.context!!.trsm(
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
