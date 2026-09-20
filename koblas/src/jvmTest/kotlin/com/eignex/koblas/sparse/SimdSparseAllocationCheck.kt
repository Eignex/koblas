package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Matrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.gemmInto
import com.eignex.koblas.testutil.allocation.AllocationProbe
import com.eignex.koblas.testutil.allocation.bytesPerCall

/**
 * Runs allocation checks for the Vector API indexed sparse paths in an uninstrumented JVM.
 *
 * Kover's test instrumentation prevents HotSpot from scalar-replacing Vector API carriers, so this
 * intentionally runs through the `simdSparseAllocationCheck` Gradle task instead of a test task.
 *
 * Three kinds of probe: the indexed Level 1 leaves, the two indexed panel leaves, and the whole sparse
 * operations around them, which a panel measured on its own cannot speak for. Each whole operation is handed
 * a warmed workspace, which is the contract a repeated call is held to.
 *
 * Whether a body allocates depends on the species width and on whether a multiply-add is one instruction,
 * both fixed at virtual machine start, so the Gradle tasks run this in three configurations.
 */
internal object SimdSparseAllocationCheck {
    internal const val ENTRY_COUNT = 512
    private const val DIMENSION = ENTRY_COUNT * 4
    private const val MAX_BYTES_PER_CALL = 8.0
    private const val WARMUP_ITERATIONS = 20_000
    private const val MEASUREMENT_ITERATIONS = 10_000
    private const val MEASUREMENT_WINDOWS = 3

    /** Right-hand sides the panel probes hand over, which is several lane blocks at any species. */
    private const val PANEL_SIDES = 64

    /** Stored entries one probed run holds, which is more than any backend's own grouping. */
    private const val PANEL_ENTRIES = 32

    /** A square operand large enough to stage and small enough to repeat thousands of times. */
    private const val ORDER = 96

    /** Right-hand sides a whole-operation probe carries, wider than one group and not a multiple of it. */
    private const val SIDES = 13

    /** One whole sparse call is the arithmetic of many panels, so it repeats fewer times. */
    private const val OPERATION_WARMUP = 2_000
    private const val OPERATION_ITERATIONS = 500

    @JvmStatic
    fun main(args: Array<String>) {
        val engine = requireNotNull(BuiltinEngines.simd) {
            "SIMD sparse allocation check requires the Vector API module"
        }
        require(engine.sparseKernels.name == "simd-sparse") {
            "SIMD sparse allocation check selected ${engine.sparseKernels.name}"
        }
        require(SparseSimd.autoIndexedLoadEligible) {
            "SIMD sparse allocation check requires a preferred species with more than one lane"
        }
        require(crossesSimdCrossover(SparseTuning.simdIndexedCrossover)) {
            "SIMD sparse allocation check uses $ENTRY_COUNT entries below " +
                "the ${SparseTuning.simdIndexedCrossover}-entry crossover"
        }
        val panels = engine.panelKernels
        require(panels.implementationFor(PanelWork.SparseRightHandSides, PANEL_SIDES, PANEL_ENTRIES) == panels.name) {
            "the panel probes use $PANEL_SIDES right-hand sides, which do not reach the vector body"
        }

        val indices = IntArray(ENTRY_COUNT) { 1 + it * 4 }
        val values = DoubleArray(ENTRY_COUNT) { it * 0.125 - 16.0 }
        val dense = DoubleArray(DIMENSION) { 1.0 + (it % 17) * 0.03125 }

        assertAllocationFree("indexed dot") {
            // Exercise the Vector API leaf even where production prefers scalar indexed loads.
            SparseSimd.dot(indices, 0, values, 0, ENTRY_COUNT, dense)
        }
        assertAllocationFree("indexed gather") {
            // Exercise the Vector API leaf even where production prefers scalar indexed loads.
            SparseSimd.gather(indices, 0, values, 0, ENTRY_COUNT, dense)
            values[ENTRY_COUNT / 2]
        }
        assertAllocationFree("indexed norm") {
            // Exercise the Vector API leaf even where production prefers scalar indexed loads.
            SparseSimd.nrm2(indices, 0, ENTRY_COUNT, dense)
        }
        checkPanels(engine)
        checkOperations(engine)
    }

    /** The two indexed panel leaves, over an adjacent group of right-hand sides and over a strided one. */
    private fun checkPanels(engine: KoblasEngine) {
        val panels = engine.panelKernels
        val entries = PANEL_ENTRIES
        val positions = 4 * entries
        val indices = IntArray(entries) { it * 4 }
        val values = DoubleArray(entries) { 0.5 + it * 0.125 }
        val work = DoubleArray(PANEL_SIDES) { 1.0 + it * 0.25 }
        for (adjacent in booleanArrayOf(true, false)) {
            val rowStride = if (adjacent) 1 else positions
            val indexStride = if (adjacent) PANEL_SIDES else 1
            val block = DoubleArray(PANEL_SIDES * rowStride + positions * indexStride) { 1.0 + (it % 13) * 0.0625 }
            val layout = if (adjacent) "adjacent" else "strided"
            assertAllocationFree("indexed column update over $layout right-hand sides") {
                panels.indexedColumnUpdate(
                    0.875, block, 0, rowStride, indexStride, indices, values, 0, entries, PANEL_SIDES, work, 0,
                )
                work[0]
            }
            assertAllocationFree("indexed rank update over $layout right-hand sides") {
                panels.indexedRankUpdate(
                    0.875, block, 0, rowStride, indexStride, indices, values, 0, entries, PANEL_SIDES, work, 0,
                )
                block[0]
            }
            // The coupled pass holds more live vectors than either of the two it fuses, so it spills first.
            val source = DoubleArray(block.size) { 0.5 + (it % 11) * 0.125 }
            // A row of the panel, where a symmetric column's mirrored half lands.
            val sums = DoubleArray(PANEL_SIDES * rowStride)
            assertAllocationFree("indexed coupled update over $layout right-hand sides") {
                panels.indexedCoupledUpdate(
                    0.875, block, source, 0, rowStride, indexStride, indices, values, 0, entries,
                    PANEL_SIDES, 0, sums, 0, -1,
                )
                sums[0]
            }
            assertAllocationFree("indexed coupled update with an excluded position over $layout sides") {
                panels.indexedCoupledUpdate(
                    0.875, block, source, 0, rowStride, indexStride, indices, values, 0, entries,
                    PANEL_SIDES, 0, sums, 0, entries / 2,
                )
                sums[0]
            }
        }
    }

    /**
     * The whole sparse operations a caller writes, each with a warmed workspace, in both orientations since
     * those are different schedules. A fresh CSC result is not here: its allocation is its answer.
     */
    private fun checkOperations(engine: KoblasEngine) {
        val a = banded(ORDER, ORDER, 12)
        val triangle = lowerTriangle(ORDER, 12)
        val b = DenseMatrix.wrap(ORDER, SIDES, DoubleArray(ORDER * SIDES) { 1.0 + (it % 11) * 0.125 })
        val wide = DenseMatrix.wrap(SIDES, ORDER, DoubleArray(ORDER * SIDES) { 1.0 + (it % 7) * 0.25 })
        val c = DenseMatrix.wrap(ORDER, SIDES, DoubleArray(ORDER * SIDES))
        val cWide = DenseMatrix.wrap(SIDES, ORDER, DoubleArray(ORDER * SIDES))
        val x = DoubleArray(ORDER) { 1.0 + (it % 5) * 0.5 }
        val y = DoubleArray(ORDER)
        val workspace = Workspace()

        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val operand = if (transposeB) DenseMatrix.wrap(SIDES, ORDER, b.values) else b
                assertAllocationFree(
                    "sparse product transposeA=$transposeA transposeB=$transposeB",
                    OPERATION_WARMUP,
                    OPERATION_ITERATIONS,
                ) {
                    engine.gemm(0.875, a, transposeA, operand, transposeB, -0.25, c, false, workspace)
                    c.values[0]
                }
            }
        }
        assertAllocationFree("sparse product from the right", OPERATION_WARMUP, OPERATION_ITERATIONS) {
            engine.gemm(0.875, a, false, wide, false, -0.25, cWide, true, workspace)
            cWide.values[0]
        }
        for (right in booleanArrayOf(false, true)) {
            val operand = if (right) wide else b
            val destination = if (right) cWide else c
            assertAllocationFree("symmetric product right=$right", OPERATION_WARMUP, OPERATION_ITERATIONS) {
                engine.symm(0.875, triangle, operand, -0.25, destination, true, right, workspace)
                destination.values[0]
            }
            for (transpose in booleanArrayOf(false, true)) {
                assertAllocationFree(
                    "triangular solve right=$right transpose=$transpose",
                    OPERATION_WARMUP,
                    OPERATION_ITERATIONS,
                ) {
                    engine.trsm(triangle, operand, true, transpose, false, right, 0.875, workspace)
                    operand.values[0]
                }
                assertAllocationFree(
                    "triangular multiply right=$right transpose=$transpose",
                    OPERATION_WARMUP,
                    OPERATION_ITERATIONS,
                ) {
                    engine.trmm(triangle, operand, true, transpose, false, right, 0.875, workspace)
                    operand.values[0]
                }
            }
        }
        for (transpose in booleanArrayOf(false, true)) {
            assertAllocationFree("sparse gemv transpose=$transpose", OPERATION_WARMUP, OPERATION_ITERATIONS) {
                engine.gemv(0.875, a, x, -0.25, y, transpose)
                y[0]
            }
        }
        assertAllocationFree("sparse symv", OPERATION_WARMUP, OPERATION_ITERATIONS) {
            engine.symv(0.875, triangle, x, -0.25, y, true)
            y[0]
        }
        assertAllocationFree("sparse rank update into a dense triangle", OPERATION_WARMUP, OPERATION_ITERATIONS) {
            engine.syrk(0.875, a, false, -0.25, square, true, workspace)
            square.values[0]
        }
        checkGenericDispatch(a, b, wide, c, cWide, workspace)
    }

    /**
     * The generic product, which decides its storage pairing per call rather than being told it.
     *
     * A caller holding a [Matrix] names no engine and no pairing, so this is where a descriptor built to
     * make that decision, or an operand adapted to reach a kernel, would show up. Both sides are probed
     * because a sparse operand keeps its own traversal on either, and the side it is on is what the dense
     * one is read through.
     */
    @Suppress("LongParameterList") // the two operands, the two destinations they need, and the scratch
    private fun checkGenericDispatch(
        a: Matrix,
        b: Matrix,
        wide: Matrix,
        c: DenseMatrix,
        cWide: DenseMatrix,
        workspace: Workspace,
    ) {
        assertAllocationFree("generic product with a sparse left operand", OPERATION_WARMUP, OPERATION_ITERATIONS) {
            a.gemmInto(0.875, false, b, false, -0.25, c, workspace)
            c.values[0]
        }
        assertAllocationFree("generic product with a sparse right operand", OPERATION_WARMUP, OPERATION_ITERATIONS) {
            wide.gemmInto(0.875, false, a, false, -0.25, cWide, workspace)
            cWide.values[0]
        }
    }

    /** A square destination for the rank update, which is the one probe whose shape is the order twice. */
    private val square = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER))

    /** A banded operand, which is where a real sparse product's destination stays resident. */
    private fun banded(rows: Int, cols: Int, width: Int): SparseMatrix {
        val pointers = IntArray(cols + 1)
        val indices = ArrayList<Int>()
        val values = ArrayList<Double>()
        for (j in 0 until cols) {
            val centre = j * rows / cols
            for (offset in -width..width) {
                val row = centre + offset
                if (row in 0 until rows) {
                    indices.add(row)
                    values.add(0.5 + (indices.size % 9) * 0.125)
                }
            }
            pointers[j + 1] = indices.size
        }
        return SparseMatrix.wrap(rows, cols, pointers, indices.toIntArray(), values.toDoubleArray())
    }

    /** A lower triangle with a dominant diagonal, so the solve probes stay finite. */
    private fun lowerTriangle(order: Int, width: Int): SparseMatrix {
        val pointers = IntArray(order + 1)
        val indices = ArrayList<Int>()
        val values = ArrayList<Double>()
        for (j in 0 until order) {
            for (row in j until minOf(order, j + width)) {
                indices.add(row)
                values.add(if (row == j) 4.0 * width else 0.25)
            }
            pointers[j + 1] = indices.size
        }
        return SparseMatrix.wrap(order, order, pointers, indices.toIntArray(), values.toDoubleArray())
    }

    internal fun crossesSimdCrossover(crossover: Int): Boolean = ENTRY_COUNT >= crossover

    private fun assertAllocationFree(
        name: String,
        warmup: Int = WARMUP_ITERATIONS,
        iterations: Int = MEASUREMENT_ITERATIONS,
        block: AllocationProbe,
    ) {
        val bytes = bytesPerCall(block, warmup, iterations, MEASUREMENT_WINDOWS)
        check(bytes <= MAX_BYTES_PER_CALL) { "$name allocated $bytes B per call" }
        println("$name: $bytes B per call")
    }
}
