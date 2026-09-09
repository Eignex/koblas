package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.sparse.SparseWorkspace
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/** Direct costs of caller-owned sparse support operations; none has a numerically equivalent BLAS call. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseWorkspaceBenchmark {
    @Param("8", "64", "512", "4096")
    var count: Int = 8

    private var dimension: Int = 0
    private lateinit var indices: IntArray
    private lateinit var values: DoubleArray
    private lateinit var accumulator: DoubleArray
    private lateinit var firstTouchAccumulator: DoubleArray
    private lateinit var marks: IntArray
    private lateinit var clearAccumulator: DoubleArray
    private lateinit var clearMarks: IntArray
    private lateinit var clearTouched: IntArray
    private lateinit var firstTouchMarks: IntArray
    private lateinit var touched: IntArray
    private lateinit var firstTouchTouched: IntArray
    private lateinit var outIndices: IntArray
    private lateinit var outValues: DoubleArray
    private lateinit var activeRows: BooleanArray
    private var maximum: Double = 0.0
    private var epoch: Int = 1

    @Setup
    fun setup() {
        dimension = count * 4
        indices = IntArray(count) { (it * 17 + 3) % dimension }
        values = DoubleArray(count) { (it % 19 - 9) * 0.125 }
        accumulator = DoubleArray(dimension)
        firstTouchAccumulator = DoubleArray(dimension)
        marks = IntArray(dimension)
        firstTouchMarks = IntArray(dimension)
        touched = IntArray(count * 2)
        firstTouchTouched = IntArray(count)
        outIndices = IntArray(count)
        outValues = DoubleArray(count)
        activeRows = BooleanArray(dimension) { it % 3 != 0 }
        SparseWorkspace.scatterAxpy(
            1.0, indices, 0, values, 0, count,
            accumulator, marks, 1, touched, 0, 0,
        )
        clearAccumulator = accumulator.copyOf()
        clearMarks = marks.copyOf()
        clearTouched = touched.copyOf()
        maximum = SparseWorkspace.activeColumnMaxAbs(indices, 0, values, 0, count, activeRows)
        println()
        verifyNearZeroManagedAllocation("sparse-workspace/scatter-first/$count") {
            epoch++
            SparseWorkspace.scatterAxpy(
                1.0, indices, 0, values, 0, count,
                firstTouchAccumulator, firstTouchMarks, epoch, firstTouchTouched, 0, 0,
            )
        }
        verifyNearZeroManagedAllocation("sparse-workspace/scatter-existing/$count") {
            SparseWorkspace.scatterAxpy(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 1, touched, 0, count,
            )
        }
        verifyNearZeroManagedAllocation("sparse-workspace/gather/$count") {
            SparseWorkspace.gatherTouched(touched, 0, count, accumulator, outIndices, 0, outValues, 0)
        }
        verifyNearZeroManagedAllocation("sparse-workspace/gather-clear/$count") {
            SparseWorkspace.gatherClearTouched(
                clearTouched, 0, count, clearAccumulator, clearMarks, outIndices, 0, outValues, 0,
            )
        }
        verifyNearZeroManagedAllocation("sparse-workspace/max/$count") {
            SparseWorkspace.activeColumnMaxAbs(indices, 0, values, 0, count, activeRows)
        }
        verifyNearZeroManagedAllocation("sparse-workspace/filter/$count") {
            SparseWorkspace.pivotCandidatePositions(
                indices, 0, values, 0, count, activeRows, maximum, 0.01, 0.1, outIndices, 0,
            )
        }
        println("resolved: sparse-workspace=portable count=$count dimension=$dimension")
    }

    /** A one-shot column update where every supplied index enters the support. */
    @Benchmark
    fun scatterFirstTouch(): Int {
        epoch++
        return SparseWorkspace.scatterAxpy(
            1.0, indices, 0, values, 0, count,
            firstTouchAccumulator, firstTouchMarks, epoch, firstTouchTouched, 0, 0,
        )
    }

    /** A repeated update into a retained support and accumulator. */
    @Benchmark
    fun scatterExistingSupport(): Int = SparseWorkspace.scatterAxpy(
        1e-12, indices, 0, values, 0, count,
        accumulator, marks, 1, touched, 0, count,
    )

    @Benchmark
    fun gatherTouched(): Int = SparseWorkspace.gatherTouched(
        touched, 0, count, accumulator, outIndices, 0, outValues, 0,
    )

    @Benchmark
    fun gatherClearTouched(): Int = SparseWorkspace.gatherClearTouched(
        clearTouched, 0, count, clearAccumulator, clearMarks, outIndices, 0, outValues, 0,
    )

    @Benchmark
    fun activeMaxAbs(): Double = SparseWorkspace.activeColumnMaxAbs(
        indices, 0, values, 0, count, activeRows,
    )

    @Benchmark
    fun filterCandidatePositions(): Int = SparseWorkspace.pivotCandidatePositions(
        indices, 0, values, 0, count, activeRows, maximum, 0.01, 0.1, outIndices, 0,
    )
}

/** Repeated short scatters whose retained support grows throughout each benchmark invocation. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseWorkspaceGrowingScatterBenchmark {
    @Param("64", "512", "4096")
    var supportSize: Int = 64

    @Param("4", "8")
    var scatterSize: Int = 4

    private lateinit var indices: IntArray
    private lateinit var values: DoubleArray
    private lateinit var accumulator: DoubleArray
    private lateinit var marks: IntArray
    private lateinit var touched: IntArray
    private lateinit var checkedAccumulator: DoubleArray
    private lateinit var checkedMarks: IntArray
    private lateinit var checkedTouched: IntArray
    private lateinit var arithmeticStatus: IntArray
    private var epoch: Int = 1

    @Setup
    fun setup() {
        indices = IntArray(supportSize) { it }
        values = DoubleArray(supportSize) { (it % 11 - 5) * 0.125 }
        accumulator = DoubleArray(supportSize)
        marks = IntArray(supportSize)
        touched = IntArray(supportSize)
        checkedAccumulator = DoubleArray(supportSize)
        checkedMarks = IntArray(supportSize)
        checkedTouched = IntArray(supportSize)
        arithmeticStatus = IntArray(1)
        println()
        verifyNearZeroManagedAllocation("sparse-workspace/scatter-growing/$supportSize/$scatterSize") {
            scatterGrowing()
        }
        verifyNearZeroManagedAllocation("sparse-workspace/scatter-growing-checked/$supportSize/$scatterSize") {
            scatterGrowingChecked()
        }
        println("resolved: sparse-workspace-growing=portable support=$supportSize scatter=$scatterSize")
    }

    @Benchmark
    fun manyShortScattersIntoGrowingSupport(): Int = scatterGrowing()

    @Benchmark
    fun manyShortCheckedScattersIntoGrowingSupport(): Int = scatterGrowingChecked()

    private fun scatterGrowing(): Int {
        epoch++
        var touchedCount = 0
        var offset = 0
        while (offset < supportSize) {
            val count = minOf(scatterSize, supportSize - offset)
            touchedCount = SparseWorkspace.scatterAxpy(
                1.0, indices, offset, values, offset, count,
                accumulator, marks, epoch, touched, 0, touchedCount,
            )
            offset += count
        }
        return touchedCount
    }

    private fun scatterGrowingChecked(): Int {
        epoch++
        arithmeticStatus[0] = 0
        var touchedCount = 0
        var offset = 0
        while (offset < supportSize) {
            val count = minOf(scatterSize, supportSize - offset)
            touchedCount = SparseWorkspace.scatterAxpyChecked(
                1.0, indices, offset, values, offset, count,
                checkedAccumulator, checkedMarks, epoch, checkedTouched, 0, touchedCount,
                arithmeticStatus, 0,
            )
            offset += count
        }
        return touchedCount + arithmeticStatus[0]
    }
}
