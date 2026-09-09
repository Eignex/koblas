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

/** Explicit packed leaves, including platform tile shapes, logical edges and fused-versus-composed updates. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class ExplicitPackedKernelBenchmark {
    @Param("3", "31", "128")
    var depth: Int = 3

    @Param("full", "partial")
    var edge: String = "full"

    @Param("lower-nonunit", "upper-unit")
    var variant: String = "lower-nonunit"

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
    private var tileRows = 0
    private var tileColumns = 0
    private var lower = true
    private var unitDiagonal = false

    @Setup
    fun setup() {
        selected = kernelEngine(kernels).kernels
        tileRows = selected.gemmTileRows
        tileColumns = selected.gemmTileCols
        rows = if (edge == "full") tileRows else tileRows - 1
        columns = if (edge == "full") tileColumns else tileColumns - 1
        when (variant) {
            "lower-nonunit" -> {
                lower = true
                unitDiagonal = false
            }
            "upper-unit" -> {
                lower = false
                unitDiagonal = true
            }
            else -> error("unknown packed triangular variant $variant")
        }
        val rng = benchRng()
        packedA = DoubleArray(depth * tileRows) { rng.nextDouble(-0.25, 0.25) }
        packedB = DoubleArray(depth * tileColumns) { rng.nextDouble(-0.25, 0.25) }
        if (edge == "partial") {
            for (step in 0 until depth) {
                packedA[step * tileRows + rows] = 0.0
                packedB[step * tileColumns + columns] = 0.0
            }
        }
        packedNegativeA = DoubleArray(depth * tileRows) { -packedA[it] }
        packedTriangle = DoubleArray(tileColumns * tileColumns)
        for (column in 0 until columns) {
            val rowRange = if (lower) column until columns else 0..column
            for (row in rowRange) {
                packedTriangle[row * tileColumns + column] =
                    if (row == column) 1.25 else 1e-12 * (row + column + 1)
            }
        }
        c = DoubleArray(tileRows * tileColumns) { rng.nextDouble(-0.5, 0.5) }
        x = c.copyOf()
        verifyNearZeroManagedAllocation("explicit-packed/$kernels/gemm/$depth/$edge") {
            selected.gemmTile(depth, packedA, 0, packedB, 0, c, 0, tileRows)
        }
        verifyNearZeroManagedAllocation("explicit-packed/$kernels/trsm/$edge") {
            selected.trsmTile(rows, columns, packedTriangle, 0, lower, unitDiagonal, x, 0)
        }
        verifyNearZeroManagedAllocation("explicit-packed/$kernels/gemm-trsm/$depth/$edge") {
            selected.gemmTrsmTile(
                depth, rows, columns, packedA, 0, packedB, 0,
                packedTriangle, 0, lower, unitDiagonal, x, 0,
            )
        }
        println("resolved: explicit-packed kernels=${selected.name} depth=$depth edge=$edge variant=$variant")
    }

    @Benchmark
    fun trsmTile(): DoubleArray {
        selected.trsmTile(rows, columns, packedTriangle, 0, lower, unitDiagonal, x, 0)
        return x
    }

    @Benchmark
    fun gemmTile(): DoubleArray {
        selected.gemmTile(depth, packedA, 0, packedB, 0, c, 0, tileRows)
        return c
    }

    /** The C full-tile path is true register-resident fusion; JVM SIMD composes its update and solve. */
    @Benchmark
    fun gemmTrsmTile(): DoubleArray {
        selected.gemmTrsmTile(
            depth, rows, columns, packedA, 0, packedB, 0,
            packedTriangle, 0, lower, unitDiagonal, x, 0,
        )
        return x
    }

    @Benchmark
    fun separateGemmAndTrsm(): DoubleArray {
        selected.gemmTile(depth, packedNegativeA, 0, packedB, 0, x, 0, tileRows)
        selected.trsmTile(rows, columns, packedTriangle, 0, lower, unitDiagonal, x, 0)
        return x
    }
}
