package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.KoblasContext
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/** Repeated end-to-end right-transposed TRSM with forced scalar/C kernels and eligibility outcomes. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class ExplicitPackedTrsmBenchmark {
    @Param("16x32", "32x32", "64x128")
    var shape: String = "32x32"

    @Param("eligible", "structural-zero", "overflow-bound")
    var scenario: String = "eligible"

    @Param(SCALAR_KERNELS, C_KERNELS)
    var kernels: String = SCALAR_KERNELS

    private lateinit var triangle: DenseMatrix
    private lateinit var source: DenseMatrix
    private lateinit var result: DenseMatrix
    private lateinit var workspace: Workspace
    private lateinit var engine: KoblasContext

    @Setup
    fun setup() {
        engine = kernelEngine(kernels)
        val (order, panel) = shape.split('x').map(String::toInt)
        val rng = benchRng()
        triangle = dominantMatrix(order, rng)
        source = randomMatrix(panel, order, rng)
        when (scenario) {
            "eligible" -> Unit
            "structural-zero" -> triangle[order - 1, order - 1] = 0.0
            "overflow-bound" -> source[panel - 1, order - 1] = 1e308
            else -> error("unknown eligibility scenario $scenario")
        }
        result = DenseMatrix.zero(panel, order)
        workspace = Workspace()
        source.data.copyInto(result.data)
        engine.trsm(
            triangle, result, lower = true, transpose = true, right = true, workspace = workspace,
        )
        reportAllocatingWorkload(
            "explicit-packed-trsm/$kernels/$shape/$scenario",
            "input reset plus retained packing workspace when eligible",
        )
        println("resolved: explicit-packed-trsm kernels=${engine.vectorKernels.name} shape=$shape scenario=$scenario")
    }

    @Benchmark
    fun denseTrsm(): DenseMatrix {
        source.data.copyInto(result.data)
        engine.trsm(
            triangle, result, lower = true, transpose = true, right = true, workspace = workspace,
        )
        return result
    }
}
