package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.dense.PackedPanels
import kotlinx.benchmark.*

/** Packing cost beside execution over retained microkernel panels. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class PackedPanelBenchmark {
    @Param("3", "31", "128")
    var depth: Int = 3

    @Param("full", "partial")
    var edge: String = "full"

    private lateinit var a: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var restoredA: DenseMatrix
    private lateinit var restoredB: DenseMatrix
    private lateinit var packedA: DoubleArray
    private lateinit var packedB: DoubleArray
    private var rows: Int = 0
    private var columns: Int = 0

    @Setup
    fun setup() {
        rows = if (edge == "full") PackedPanels.tileRows else PackedPanels.tileRows - 1
        columns = if (edge == "full") PackedPanels.tileColumns else PackedPanels.tileColumns - 1
        val rng = benchRng()
        a = randomMatrix(rows, depth, rng)
        b = randomMatrix(depth, columns, rng)
        restoredA = DenseMatrix.zero(rows, depth)
        restoredB = DenseMatrix.zero(depth, columns)
        packedA = DoubleArray(PackedPanels.leftSize(rows, depth))
        packedB = DoubleArray(PackedPanels.rightSize(depth, columns))
        PackedPanels.packLeft(a, packedA, rows, depth)
        PackedPanels.packRight(b, packedB, depth, columns)
        println()
        verifyNearZeroManagedAllocation("packed-panels/pack-left/$depth/$edge") {
            PackedPanels.packLeft(a, packedA, rows, depth)
        }
        verifyNearZeroManagedAllocation("packed-panels/pack-right/$depth/$edge") {
            PackedPanels.packRight(b, packedB, depth, columns)
        }
        verifyNearZeroManagedAllocation("packed-panels/write-left/$depth/$edge") {
            PackedPanels.writeLeft(packedA, restoredA, rows, depth)
        }
        verifyNearZeroManagedAllocation("packed-panels/write-right/$depth/$edge") {
            PackedPanels.writeRight(packedB, restoredB, depth, columns)
        }
        println(
            "resolved: packed-panels rows=${PackedPanels.tileRows} columns=${PackedPanels.tileColumns}",
        )
    }

    @Benchmark
    fun packLeft(): DoubleArray {
        PackedPanels.packLeft(a, packedA, rows, depth)
        return packedA
    }

    @Benchmark
    fun packRight(): DoubleArray {
        PackedPanels.packRight(b, packedB, depth, columns)
        return packedB
    }

    @Benchmark
    fun writeLeft(): DenseMatrix {
        PackedPanels.writeLeft(packedA, restoredA, rows, depth)
        return restoredA
    }

    @Benchmark
    fun writeRight(): DenseMatrix {
        PackedPanels.writeRight(packedB, restoredB, depth, columns)
        return restoredB
    }
}

/** Allocation-checked tile execution over panels retained outside the measured operation. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class PackedPanelExecutionBenchmark {
    @Param("3", "31", "128")
    var depth: Int = 3

    @Param("full", "partial")
    var edge: String = "full"

    private lateinit var kernels: Kernels
    private lateinit var packedA: DoubleArray
    private lateinit var packedB: DoubleArray
    private lateinit var c: DoubleArray

    @Setup
    fun setup() {
        kernels = explicitBuiltInContext().kernels
        check(kernels.gemmTileRows == PackedPanels.tileRows)
        check(kernels.gemmTileCols == PackedPanels.tileColumns)
        val rows = if (edge == "full") PackedPanels.tileRows else PackedPanels.tileRows - 1
        val columns = if (edge == "full") PackedPanels.tileColumns else PackedPanels.tileColumns - 1
        val rng = benchRng()
        val a = randomMatrix(rows, depth, rng)
        val b = randomMatrix(depth, columns, rng)
        packedA = DoubleArray(PackedPanels.leftSize(rows, depth))
        packedB = DoubleArray(PackedPanels.rightSize(depth, columns))
        c = DoubleArray(PackedPanels.tileRows * PackedPanels.tileColumns)
        PackedPanels.packLeft(a, packedA, rows, depth)
        PackedPanels.packRight(b, packedB, depth, columns)
        println()
        verifyNearZeroManagedAllocation("packed-panels/reused-product/$depth/$edge") {
            kernels.gemmTile(depth, packedA, 0, packedB, 0, c, 0, PackedPanels.tileRows)
        }
        println(
            "resolved: packed-panels kernels=${kernels.name} rows=${PackedPanels.tileRows} " +
                "columns=${PackedPanels.tileColumns}",
        )
    }

    @Benchmark
    fun reusedPanelProduct(): DoubleArray {
        kernels.gemmTile(depth, packedA, 0, packedB, 0, c, 0, PackedPanels.tileRows)
        return c
    }
}
