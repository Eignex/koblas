package com.eignex.koblas.bench

import com.eignex.koblas.dense.Kernels
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/** Explicit scalar/C-provider packed leaves, including partial tiles and fused-versus-composed update paths. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class ExplicitPackedKernelBenchmark {
    @Param("3", "31", "128")
    var depth: Int = 3

    @Param("full", "partial")
    var edge: String = "full"

    @Param(SCALAR_KERNELS, C_KERNELS)
    var kernels: String = SCALAR_KERNELS

    private lateinit var selected: Kernels
    private lateinit var packedA: DoubleArray
    private lateinit var packedNegativeA: DoubleArray
    private lateinit var packedB: DoubleArray
    private lateinit var packedTriangle: DoubleArray
    private lateinit var c: DoubleArray
    private lateinit var x: DoubleArray
    private var rows = 0
    private var columns = 0

    @Setup
    fun setup() {
        selected = kernelEngine(kernels).kernels
        check(selected.gemmTileRows == 4 && selected.gemmTileCols == 4) {
            "explicit scalar/C packed comparison requires the shared four by four tile"
        }
        rows = if (edge == "full") 4 else 3
        columns = if (edge == "full") 4 else 3
        val rng = benchRng()
        packedA = DoubleArray(depth * 4) { rng.nextDouble(-0.25, 0.25) }
        packedB = DoubleArray(depth * 4) { rng.nextDouble(-0.25, 0.25) }
        if (edge == "partial") {
            for (step in 0 until depth) {
                packedA[step * 4 + 3] = 0.0
                packedB[step * 4 + 3] = 0.0
            }
        }
        packedNegativeA = DoubleArray(depth * 4) { -packedA[it] }
        packedTriangle = DoubleArray(16)
        for (column in 0 until columns) {
            for (row in column until columns) {
                packedTriangle[row * 4 + column] = if (row == column) 1.000_000_001 else 1e-12 * (row + column + 1)
            }
        }
        c = DoubleArray(16) { rng.nextDouble(-0.5, 0.5) }
        x = c.copyOf()
        verifyNearZeroManagedAllocation("explicit-packed/$kernels/gemm/$depth/$edge") {
            selected.gemmTile(depth, packedA, 0, packedB, 0, c, 0, 4)
        }
        verifyNearZeroManagedAllocation("explicit-packed/$kernels/trsm/$edge") {
            selected.trsmTile(rows, columns, packedTriangle, 0, lower = true, unitDiag = false, x, 0)
        }
        verifyNearZeroManagedAllocation("explicit-packed/$kernels/gemm-trsm/$depth/$edge") {
            selected.gemmTrsmTile(
                depth, rows, columns, packedA, 0, packedB, 0,
                packedTriangle, 0, lower = true, unitDiag = false, x, 0,
            )
        }
        println("resolved: explicit-packed kernels=${selected.name} depth=$depth edge=$edge")
    }

    @Benchmark
    fun trsmTile(): DoubleArray {
        selected.trsmTile(rows, columns, packedTriangle, 0, lower = true, unitDiag = false, x, 0)
        return x
    }

    @Benchmark
    fun gemmTile(): DoubleArray {
        selected.gemmTile(depth, packedA, 0, packedB, 0, c, 0, 4)
        return c
    }

    /** The C full-tile path is true register-resident fusion; the portable fallback is a composition. */
    @Benchmark
    fun gemmTrsmTile(): DoubleArray {
        selected.gemmTrsmTile(
            depth, rows, columns, packedA, 0, packedB, 0,
            packedTriangle, 0, lower = true, unitDiag = false, x, 0,
        )
        return x
    }

    @Benchmark
    fun separateGemmAndTrsm(): DoubleArray {
        selected.gemmTile(depth, packedNegativeA, 0, packedB, 0, x, 0, 4)
        selected.trsmTile(rows, columns, packedTriangle, 0, lower = true, unitDiag = false, x, 0)
        return x
    }
}
