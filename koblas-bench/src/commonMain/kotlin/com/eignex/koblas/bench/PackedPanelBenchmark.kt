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
    private lateinit var packedNegativeA: DoubleArray
    private lateinit var packedB: DoubleArray
    private lateinit var packedTriangle: DoubleArray
    private lateinit var packedX: DoubleArray
    private lateinit var c: DoubleArray
    private var rows: Int = 0
    private var columns: Int = 0

    @Setup
    fun setup() {
        kernels = explicitBuiltInContext().kernels
        check(kernels.gemmTileRows == PackedPanels.tileRows)
        check(kernels.gemmTileCols == PackedPanels.tileColumns)
        rows = if (edge == "full") PackedPanels.tileRows else PackedPanels.tileRows - 1
        columns = if (edge == "full") PackedPanels.tileColumns else PackedPanels.tileColumns - 1
        val rng = benchRng()
        val a = randomMatrix(rows, depth, rng)
        val b = randomMatrix(depth, columns, rng)
        packedA = DoubleArray(PackedPanels.leftSize(rows, depth))
        packedNegativeA = DoubleArray(PackedPanels.leftSize(rows, depth))
        packedB = DoubleArray(PackedPanels.rightSize(depth, columns))
        packedTriangle = DoubleArray(PackedPanels.rightSize(columns, columns))
        packedX = DoubleArray(PackedPanels.tileRows * PackedPanels.tileColumns)
        c = DoubleArray(PackedPanels.tileRows * PackedPanels.tileColumns)
        PackedPanels.packLeft(a, packedA, rows, depth)
        PackedPanels.packLeft(a, packedNegativeA, rows, depth, alpha = -1.0)
        PackedPanels.packRight(b, packedB, depth, columns)
        val triangle = DenseMatrix.zero(columns)
        for (j in 0 until columns) {
            for (i in j until columns) triangle[i, j] = if (i == j) 2.0 else 0.01 * (i + j + 1.0)
        }
        PackedPanels.packTriangularRight(
            triangle, packedTriangle, columns, columns, lower = true,
        )
        val rightHandSide = randomMatrix(rows, columns, rng)
        PackedPanels.packLeft(rightHandSide, packedX, rows, columns)
        println()
        verifyNearZeroManagedAllocation("packed-panels/reused-product/$depth/$edge") {
            kernels.gemmTile(depth, packedA, 0, packedB, 0, c, 0, PackedPanels.tileRows)
        }
        verifyNearZeroManagedAllocation("packed-panels/trsm/$depth/$edge") {
            PackedPanels.trsm(packedTriangle, packedX, rows, columns, lower = true)
        }
        verifyNearZeroManagedAllocation("packed-panels/gemm-trsm/$depth/$edge") {
            PackedPanels.gemmTrsm(
                packedA, packedB, packedTriangle, packedX, rows, columns, depth, lower = true,
            )
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

    @Benchmark
    fun packedTrsm(): DoubleArray {
        PackedPanels.trsm(packedTriangle, packedX, rows, columns, lower = true)
        return packedX
    }

    @Benchmark
    fun packedGemmTrsm(): DoubleArray {
        PackedPanels.gemmTrsm(
            packedA, packedB, packedTriangle, packedX, rows, columns, depth, lower = true,
        )
        return packedX
    }

    @Benchmark
    fun separatePackedGemmAndTrsm(): DoubleArray {
        kernels.gemmTile(depth, packedNegativeA, 0, packedB, 0, packedX, 0, PackedPanels.tileRows)
        kernels.trsmTile(rows, columns, packedTriangle, 0, lower = true, unitDiag = false, packedX, 0)
        return packedX
    }
}
