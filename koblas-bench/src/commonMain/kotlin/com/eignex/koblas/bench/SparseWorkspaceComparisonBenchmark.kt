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

/** Equivalent-work comparisons for workspace operations that compose with oneMKL Sparse BLAS Level 1. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseWorkspaceOneMklComparisonBenchmark {
    @Param("4", "64", "512")
    var count: Int = 4

    @Param("0", "50", "100")
    var firstTouchPercent: Int = 0

    @Param("false", "true")
    var compactExactZeros: Boolean = false

    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND)
    var sparseArm: String = BUILTIN_BACKEND

    private var dimension: Int = 0
    private var existingCount: Int = 0
    private var epoch: Int = 1
    private var external: SparseComparator? = null
    private lateinit var indices: IntArray
    private lateinit var values: DoubleArray
    private lateinit var accumulator: DoubleArray
    private lateinit var marks: IntArray
    private lateinit var touched: IntArray
    private lateinit var gatherAccumulator: DoubleArray
    private lateinit var clearAccumulator: DoubleArray
    private lateinit var clearMarks: IntArray
    private lateinit var gatherScratch: DoubleArray
    private lateinit var outIndices: IntArray
    private lateinit var outValues: DoubleArray

    @Setup
    fun setup() {
        require(firstTouchPercent in listOf(0, 50, 100))
        dimension = count * 8 + 1
        existingCount = count - count * firstTouchPercent / 100
        indices = IntArray(count + 2) { -1 }
        values = DoubleArray(count + 2)
        for (k in 0 until count) {
            indices[k + 1] = (k * 37 + 3) % dimension
            values[k + 1] = (k % 17 - 8) * 0.125 + 0.03125
        }
        accumulator = DoubleArray(dimension)
        marks = IntArray(dimension)
        touched = IntArray(count + 4)
        gatherAccumulator = DoubleArray(dimension) { -71.0 }
        for (k in 0 until count) {
            gatherAccumulator[indices[k + 1]] = if (k % 7 == 0) 0.0 else values[k + 1]
        }
        clearAccumulator = gatherAccumulator.copyOf()
        clearMarks = IntArray(dimension)
        gatherScratch = DoubleArray(count + 2)
        outIndices = IntArray(count + 4)
        outValues = DoubleArray(count + 4)
        external = when (sparseArm) {
            BUILTIN_BACKEND -> null
            ONEMKL_BACKEND -> checkNotNull(oneMklSparseComparator()) {
                "the benchmark-only oneMKL sparse comparator is unavailable"
            }
            else -> error("unknown sparse workspace arm: $sparseArm")
        }
        resetScatter()
        resetClear()
        verifyEquivalentState()
        if (count == 64 && firstTouchPercent == 50 && !compactExactZeros) {
            verifyNearZeroManagedAllocation("sparse-workspace-equivalent/$sparseArm/scatter") {
                scatterAxpyEquivalent()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-equivalent/$sparseArm/gather") {
                gatherTouchedEquivalent()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-equivalent/$sparseArm/gather-clear") {
                gatherClearTouchedEquivalent()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-partial/$sparseArm/indexed-axpy") {
                indexedAxpyPartialWork()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-partial/$sparseArm/indexed-gather") {
                indexedGatherPartialWork()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-partial/$sparseArm/indexed-gather-zero") {
                indexedGatherZeroPartialWork()
            }
        }
        println()
        println(
            "resolved: arm=$sparseArm sparse-workspace=" +
                (external?.identity ?: "built-in/portable") +
                " threading=${external?.threading ?: "single caller thread"} coverage=composed " +
                "count=$count dimension=$dimension first-touch=$firstTouchPercent% compact=$compactExactZeros",
        )
    }

    /** Full scatter contract; fixture restoration is included and reported separately by [scatterFixtureReset]. */
    @Benchmark
    fun scatterAxpyEquivalent(): Int {
        resetScatter()
        return scatter()
    }

    /** Fixture-only cost paired with [scatterAxpyEquivalent]. */
    @Benchmark
    fun scatterFixtureReset(): Int {
        resetScatter()
        return existingCount
    }

    /** Full ordered gather contract, including optional exact-zero compaction. */
    @Benchmark
    fun gatherTouchedEquivalent(): Int = gather()

    /** Full destructive gather contract; restoration is included and reported separately by [gatherClearFixtureReset]. */
    @Benchmark
    fun gatherClearTouchedEquivalent(): Int {
        resetClear()
        return gatherClear()
    }

    /** Fixture-only cost paired with [gatherClearTouchedEquivalent]. */
    @Benchmark
    fun gatherClearFixtureReset(): Int {
        resetClear()
        return count
    }

    /** Indexed AXPY alone: partial work that excludes all workspace support bookkeeping. */
    @Benchmark
    fun indexedAxpyPartialWork(): Double {
        resetScatter()
        val comparator = external
        if (comparator == null) {
            for (k in 0 until count) accumulator[indices[k + 1]] += 0.75 * values[k + 1]
        } else {
            comparator.indexedAxpy(0.75, values, 1, indices, 1, count, accumulator)
        }
        return accumulator[indices[1]]
    }

    /** Indexed gather alone: partial work that excludes ordered index emission and compaction. */
    @Benchmark
    fun indexedGatherPartialWork(): Double {
        val comparator = external
        if (comparator == null) {
            for (k in 0 until count) outValues[k + 2] = gatherAccumulator[indices[k + 1]]
        } else {
            comparator.indexedGather(indices, 1, count, gatherAccumulator, outValues, 2)
        }
        return outValues[2]
    }

    /** Indexed destructive gather alone; populated-state restoration is included and reported separately. */
    @Benchmark
    fun indexedGatherZeroPartialWork(): Double {
        resetClear()
        val comparator = external
        if (comparator == null) {
            for (k in 0 until count) {
                val index = indices[k + 1]
                outValues[k + 2] = clearAccumulator[index]
                clearAccumulator[index] = 0.0
            }
        } else {
            comparator.indexedGatherZero(indices, 1, count, clearAccumulator, outValues, 2)
        }
        return outValues[2]
    }

    private fun scatter(): Int {
        val comparator = external
        return if (comparator == null) {
            SparseWorkspace.scatterAxpy(
                0.75, indices, 1, values, 1, count,
                accumulator, marks, epoch, touched, 2, existingCount,
            )
        } else {
            SparseWorkspaceComparators.scatterAxpyOneMkl(
                comparator, 0.75, indices, 1, values, 1, count,
                accumulator, marks, epoch, touched, 2, existingCount,
            )
        }
    }

    private fun gather(): Int {
        val comparator = external
        return if (comparator == null) {
            SparseWorkspace.gatherTouched(
                indices, 1, count, gatherAccumulator, outIndices, 2, outValues, 2, compactExactZeros,
            )
        } else {
            SparseWorkspaceComparators.gatherTouchedOneMkl(
                comparator, indices, 1, count, gatherAccumulator, gatherScratch, 1,
                outIndices, 2, outValues, 2, compactExactZeros,
            )
        }
    }

    private fun gatherClear(): Int {
        val comparator = external
        return if (comparator == null) {
            SparseWorkspace.gatherClearTouched(
                indices, 1, count, clearAccumulator, clearMarks,
                outIndices, 2, outValues, 2, compactExactZeros,
            )
        } else {
            SparseWorkspaceComparators.gatherClearTouchedOneMkl(
                comparator, indices, 1, count, clearAccumulator, clearMarks, gatherScratch, 1,
                outIndices, 2, outValues, 2, compactExactZeros,
            )
        }
    }

    private fun resetScatter() {
        epoch++
        accumulator.fill(-123.0)
        marks.fill(0)
        touched.fill(-1)
        for (k in 0 until existingCount) {
            val index = indices[k + 1]
            accumulator[index] = k * 0.25 - 2.0
            marks[index] = epoch
            touched[k + 2] = index
        }
    }

    private fun resetClear() {
        gatherAccumulator.copyInto(clearAccumulator)
        clearMarks.fill(0)
        for (k in 0 until count) clearMarks[indices[k + 1]] = epoch
    }

    private fun verifyEquivalentState() {
        val comparator = external ?: return
        resetScatter()
        val expectedAccumulator = accumulator.copyOf()
        val expectedMarks = marks.copyOf()
        val expectedTouched = touched.copyOf()
        val expectedCount = SparseWorkspace.scatterAxpy(
            0.75, indices, 1, values, 1, count,
            expectedAccumulator, expectedMarks, epoch, expectedTouched, 2, existingCount,
        )
        val actualCount = SparseWorkspaceComparators.scatterAxpyOneMkl(
            comparator, 0.75, indices, 1, values, 1, count,
            accumulator, marks, epoch, touched, 2, existingCount,
        )
        check(expectedCount == actualCount)
        check(expectedAccumulator.contentEquals(accumulator))
        check(expectedMarks.contentEquals(marks))
        check(expectedTouched.contentEquals(touched))

        for (compact in listOf(false, true)) {
            val expectedIndices = IntArray(outIndices.size)
            val expectedValues = DoubleArray(outValues.size)
            outIndices.fill(0)
            outValues.fill(0.0)
            val expectedWritten = SparseWorkspace.gatherTouched(
                indices, 1, count, gatherAccumulator, expectedIndices, 2, expectedValues, 2, compact,
            )
            val actualWritten = SparseWorkspaceComparators.gatherTouchedOneMkl(
                comparator, indices, 1, count, gatherAccumulator, gatherScratch, 1,
                outIndices, 2, outValues, 2, compact,
            )
            check(expectedWritten == actualWritten)
            check(expectedIndices.contentEquals(outIndices))
            check(expectedValues.contentEquals(outValues))

            val expectedAccumulator = gatherAccumulator.copyOf()
            val expectedMarks = IntArray(dimension)
            for (k in 0 until count) expectedMarks[indices[k + 1]] = epoch
            resetClear()
            expectedIndices.fill(0)
            expectedValues.fill(0.0)
            outIndices.fill(0)
            outValues.fill(0.0)
            val expectedClearWritten = SparseWorkspace.gatherClearTouched(
                indices, 1, count, expectedAccumulator, expectedMarks,
                expectedIndices, 2, expectedValues, 2, compact,
            )
            val actualClearWritten = SparseWorkspaceComparators.gatherClearTouchedOneMkl(
                comparator, indices, 1, count, clearAccumulator, clearMarks, gatherScratch, 1,
                outIndices, 2, outValues, 2, compact,
            )
            check(expectedClearWritten == actualClearWritten)
            check(expectedAccumulator.contentEquals(clearAccumulator))
            check(expectedMarks.contentEquals(clearMarks))
            check(expectedIndices.contentEquals(outIndices))
            check(expectedValues.contentEquals(outValues))
        }
        resetScatter()
        resetClear()
    }
}

/** Repeated short mixed-overlap scatters that grow support within every invocation. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseWorkspaceGrowingComparisonBenchmark {
    @Param("64", "512", "4096")
    var supportSize: Int = 64

    @Param("4", "8")
    var scatterSize: Int = 4

    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND)
    var sparseArm: String = BUILTIN_BACKEND

    private var epoch: Int = 1
    private var callCount: Int = 0
    private var external: SparseComparator? = null
    private lateinit var indices: IntArray
    private lateinit var values: DoubleArray
    private lateinit var accumulator: DoubleArray
    private lateinit var marks: IntArray
    private lateinit var touched: IntArray

    @Setup
    fun setup() {
        val newPerCall = scatterSize / 2
        callCount = 1 + (supportSize - scatterSize + newPerCall - 1) / newPerCall
        indices = IntArray(callCount * scatterSize)
        values = DoubleArray(indices.size)
        var known = 0
        for (call in 0 until callCount) {
            val base = call * scatterSize
            for (k in 0 until scatterSize) {
                val index = if (call == 0 || k >= newPerCall) {
                    minOf(known++, supportSize - 1)
                } else {
                    (call * 13 + k * 7) % known
                }
                indices[base + k] = index
                values[base + k] = (base + k) % 23 * 0.03125 - 0.25
            }
        }
        accumulator = DoubleArray(supportSize)
        marks = IntArray(supportSize)
        touched = IntArray(supportSize)
        external = when (sparseArm) {
            BUILTIN_BACKEND -> null
            ONEMKL_BACKEND -> checkNotNull(oneMklSparseComparator()) {
                "the benchmark-only oneMKL sparse comparator is unavailable"
            }
            else -> error("unknown sparse workspace arm: $sparseArm")
        }
        if (supportSize == 512 && scatterSize == 4) {
            verifyNearZeroManagedAllocation("sparse-workspace-equivalent/$sparseArm/growing") {
                manyShortScattersEquivalent()
            }
        }
        println()
        println(
            "resolved: arm=$sparseArm sparse-workspace-growing=" +
                (external?.identity ?: "built-in/portable") +
                " threading=${external?.threading ?: "single caller thread"} coverage=composed " +
                "support=$supportSize scatter=$scatterSize calls=$callCount overlap=50% seed=$BENCH_SEED",
        )
    }

    @Benchmark
    fun manyShortScattersEquivalent(): Int {
        epoch++
        var touchedCount = 0
        for (call in 0 until callCount) {
            val offset = call * scatterSize
            val comparator = external
            touchedCount = if (comparator == null) {
                SparseWorkspace.scatterAxpy(
                    1.0, indices, offset, values, offset, scatterSize,
                    accumulator, marks, epoch, touched, 0, touchedCount,
                )
            } else {
                SparseWorkspaceComparators.scatterAxpyOneMkl(
                    comparator, 1.0, indices, offset, values, offset, scatterSize,
                    accumulator, marks, epoch, touched, 0, touchedCount,
                )
            }
        }
        return touchedCount
    }
}

/** Independent Kotlin implementation comparisons for helpers with no vendor counterpart. */
@OptIn(ExperimentalKoblasApi::class)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseWorkspaceBaselineComparisonBenchmark {
    @Param("8", "64", "512", "4096")
    var count: Int = 8

    @Param(BUILTIN_BACKEND, "baseline")
    var baselineArm: String = BUILTIN_BACKEND

    private var dimension: Int = 0
    private var epoch: Int = 1
    private var maximum: Double = 0.0
    private lateinit var indices: IntArray
    private lateinit var values: DoubleArray
    private lateinit var activeRows: BooleanArray
    private lateinit var accumulator: DoubleArray
    private lateinit var marks: IntArray
    private lateinit var touched: IntArray
    private lateinit var status: IntArray
    private lateinit var outPositions: IntArray

    @Setup
    fun setup() {
        dimension = count * 8 + 1
        indices = IntArray(count + 2) { -1 }
        values = DoubleArray(count + 2)
        for (k in 0 until count) {
            indices[k + 1] = (k * 37 + 3) % dimension
            values[k + 1] = if (k % 29 == 0) 0.0 else (k % 19 - 9) * 0.125
        }
        activeRows = BooleanArray(dimension) { (it * 11 + 5) % 7 != 0 }
        accumulator = DoubleArray(dimension)
        marks = IntArray(dimension)
        touched = IntArray(count + 2)
        status = IntArray(3)
        outPositions = IntArray(count + 2)
        maximum = SparseWorkspace.activeColumnMaxAbs(indices, 1, values, 1, count, activeRows)
        if (count == 64) {
            verifyNearZeroManagedAllocation("sparse-workspace-baseline/$baselineArm/max") {
                activeColumnMaxAbsEquivalent()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-baseline/$baselineArm/candidates") {
                pivotCandidatePositionsEquivalent()
            }
            verifyNearZeroManagedAllocation("sparse-workspace-baseline/$baselineArm/checked") {
                manyShortCheckedScattersEquivalent()
            }
        }
        println()
        println(
            "resolved: sparse-workspace-baseline=$baselineArm coverage=no-vendor-counterpart " +
                "count=$count dimension=$dimension seed=$BENCH_SEED",
        )
    }

    @Benchmark
    fun activeColumnMaxAbsEquivalent(): Double = if (baselineArm == BUILTIN_BACKEND) {
        SparseWorkspace.activeColumnMaxAbs(indices, 1, values, 1, count, activeRows)
    } else {
        SparseWorkspaceComparators.activeColumnMaxAbsBaseline(indices, 1, values, 1, count, activeRows)
    }

    @Benchmark
    fun pivotCandidatePositionsEquivalent(): Int = if (baselineArm == BUILTIN_BACKEND) {
        SparseWorkspace.pivotCandidatePositions(
            indices, 1, values, 1, count, activeRows, maximum, 0.01, 0.1, outPositions, 1,
        )
    } else {
        SparseWorkspaceComparators.pivotCandidatePositionsBaseline(
            indices, 1, values, 1, count, activeRows, maximum, 0.01, 0.1, outPositions, 1,
        )
    }

    @Benchmark
    fun manyShortCheckedScattersEquivalent(): Int {
        epoch++
        status[1] = 0
        var touchedCount = 0
        var offset = 1
        while (offset < count + 1) {
            val length = minOf(4, count + 1 - offset)
            touchedCount = if (baselineArm == BUILTIN_BACKEND) {
                SparseWorkspace.scatterAxpyChecked(
                    1.0, indices, offset, values, offset, length,
                    accumulator, marks, epoch, touched, 1, touchedCount, status, 1,
                )
            } else {
                SparseWorkspaceComparators.scatterAxpyCheckedBaseline(
                    1.0, indices, offset, values, offset, length,
                    accumulator, marks, epoch, touched, 1, touchedCount, status, 1,
                )
            }
            offset += length
        }
        return touchedCount + status[1]
    }
}
