package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64PackedPanels
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

    private lateinit var kernels: F64Kernels
    private lateinit var a: F64DenseMatrix
    private lateinit var b: F64DenseMatrix
    private lateinit var restoredA: F64DenseMatrix
    private lateinit var restoredB: F64DenseMatrix
    private lateinit var packedA: DoubleArray
    private lateinit var packedB: DoubleArray
    private lateinit var c: DoubleArray
    private var rows: Int = 0
    private var columns: Int = 0

    @Setup
    fun setup() {
        kernels = explicitBuiltInContext().kernels
        check(kernels.gemmTileRows == F64PackedPanels.tileRows)
        check(kernels.gemmTileCols == F64PackedPanels.tileColumns)
        rows = if (edge == "full") F64PackedPanels.tileRows else F64PackedPanels.tileRows - 1
        columns = if (edge == "full") F64PackedPanels.tileColumns else F64PackedPanels.tileColumns - 1
        val rng = benchRng()
        a = randomMatrix(rows, depth, rng)
        b = randomMatrix(depth, columns, rng)
        restoredA = F64DenseMatrix.zero(rows, depth)
        restoredB = F64DenseMatrix.zero(depth, columns)
        packedA = DoubleArray(F64PackedPanels.leftSize(rows, depth))
        packedB = DoubleArray(F64PackedPanels.rightSize(depth, columns))
        c = DoubleArray(F64PackedPanels.tileRows * F64PackedPanels.tileColumns)
        F64PackedPanels.packLeft(a, packedA, rows, depth)
        F64PackedPanels.packRight(b, packedB, depth, columns)
        verifyNearZeroManagedAllocation("packed-panels/pack-left/$depth/$edge") {
            F64PackedPanels.packLeft(a, packedA, rows, depth)
        }
        verifyNearZeroManagedAllocation("packed-panels/pack-right/$depth/$edge") {
            F64PackedPanels.packRight(b, packedB, depth, columns)
        }
        verifyNearZeroManagedAllocation("packed-panels/reused-product/$depth/$edge") {
            kernels.gemmTile(depth, packedA, 0, packedB, 0, c, 0, F64PackedPanels.tileRows)
        }
        println(
            "resolved: packed-panels kernels=${kernels.name} rows=${F64PackedPanels.tileRows} " +
                "columns=${F64PackedPanels.tileColumns}",
        )
    }

    @Benchmark
    fun packLeft(): DoubleArray {
        F64PackedPanels.packLeft(a, packedA, rows, depth)
        return packedA
    }

    @Benchmark
    fun packRight(): DoubleArray {
        F64PackedPanels.packRight(b, packedB, depth, columns)
        return packedB
    }

    @Benchmark
    fun writeLeft(): F64DenseMatrix {
        F64PackedPanels.writeLeft(packedA, restoredA, rows, depth)
        return restoredA
    }

    @Benchmark
    fun writeRight(): F64DenseMatrix {
        F64PackedPanels.writeRight(packedB, restoredB, depth, columns)
        return restoredB
    }

    @Benchmark
    fun reusedPanelProduct(): DoubleArray {
        kernels.gemmTile(depth, packedA, 0, packedB, 0, c, 0, F64PackedPanels.tileRows)
        return c
    }
}
