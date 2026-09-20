@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// The panel contract, over any implementation. Every backend has to satisfy it at every extent, including
// extents narrower than its own grouping and shorter than a vector, so the assertions live here rather than
// beside one of them.

/** A window inside a larger buffer, so an implementation that ignores an offset or a leading dimension fails. */
private const val PAD = 3

/** Extents that straddle a lane width, a grouping and the boundary between them. */
private val ROWS = intArrayOf(0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 64, 65, 200)

/** Logical panel widths, including one, a tail beside every grouping, and one wider than any of them. */
private val COLUMNS = intArrayOf(1, 2, 3, 4, 5, 8)

private fun panelBuffer(columns: Int, lda: Int, rng: Random): DoubleArray =
    DoubleArray(PAD + lda * columns + PAD) { rng.nextDouble(-1.0, 1.0) }

/** `y(c) = alpha · Σ a(i, c) · x(i) + beta · y(c)`, written out. */
private fun referenceMultiDot(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    rows: Int,
    columns: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
): DoubleArray {
    val result = y.copyOf()
    for (c in 0 until columns) {
        var sum = 0.0
        for (i in 0 until rows) sum += a[aOffset + i + c * lda] * x[xOffset + i * xStride]
        val at = yOffset + c * yStride
        result[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * y[at]
    }
    return result
}

/** `y(i) += Σ (alpha · x(c)) · a(i, c)`, written out. */
private fun referenceColumnUpdate(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    rows: Int,
    columns: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
): DoubleArray {
    val result = y.copyOf()
    for (i in 0 until rows) {
        var sum = result[yOffset + i * yStride]
        for (c in 0 until columns) sum += (alpha * x[xOffset + c * xStride]) * a[aOffset + i + c * lda]
        result[yOffset + i * yStride] = sum
    }
    return result
}

/** `a(i, c) += (alpha · coefficients(c)) · x(i)`, written out. */
private fun referenceRankUpdate(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    rows: Int,
    columns: Int,
    coefficients: DoubleArray,
    coefficientOffset: Int,
    coefficientStride: Int,
): DoubleArray {
    val result = a.copyOf()
    for (c in 0 until columns) {
        val t = alpha * coefficients[coefficientOffset + c * coefficientStride]
        for (i in 0 until rows) result[aOffset + i + c * lda] += t * x[xOffset + i * xStride]
    }
    return result
}

/** `y(i) += Σ (alpha · values(c)) · a(i, indices(c))`, written out. */
private fun referenceIndexedColumnUpdate(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    rowStride: Int,
    indexStride: Int,
    indices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    columns: Int,
    rows: Int,
    y: DoubleArray,
    yOffset: Int,
): DoubleArray {
    val result = y.copyOf()
    for (i in 0 until rows) {
        var sum = result[yOffset + i]
        for (c in 0 until columns) {
            sum += (alpha * values[fromIndex + c]) * a[aOffset + i * rowStride + indices[fromIndex + c] * indexStride]
        }
        result[yOffset + i] = sum
    }
    return result
}

/** `a(i, indices(c)) += (alpha · values(c)) · x(i)`, written out. */
private fun referenceIndexedRankUpdate(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    rowStride: Int,
    indexStride: Int,
    indices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    columns: Int,
    rows: Int,
    x: DoubleArray,
    xOffset: Int,
): DoubleArray {
    val result = a.copyOf()
    for (c in 0 until columns) {
        val t = alpha * values[fromIndex + c]
        for (i in 0 until rows) {
            result[aOffset + i * rowStride + indices[fromIndex + c] * indexStride] += t * x[xOffset + i]
        }
    }
    return result
}

/**
 * Every panel of [kernels] against the definitions written out above, over extents that straddle any
 * grouping and any lane width, at a nonzero offset and a leading dimension wider than the window. The
 * strides rotate with the extent rather than multiplying out, which keeps the sweep bounded.
 */
internal fun assertPanelKernelsAgreeWithReference(kernels: DensePanelKernels) {
    val rng = Random(20260919)
    var variant = 0
    for (rows in ROWS) {
        for (columns in COLUMNS) {
            variant++
            val lda = rows + PAD
            val xStride = 1 + variant % 2
            val yStride = 1 + variant % 3
            val a = panelBuffer(columns, lda, rng)
            val x = DoubleArray(PAD + rows * xStride + PAD) { rng.nextDouble(-1.0, 1.0) }
            val coefficients = DoubleArray(PAD + columns * xStride + PAD) { rng.nextDouble(-1.0, 1.0) }
            val wide = DoubleArray(PAD + maxOf(rows, columns) * yStride + PAD) { rng.nextDouble(-1.0, 1.0) }
            val alpha = 0.875
            val beta = -0.25
            val context = "rows=$rows columns=$columns"

            val expectedDot =
                referenceMultiDot(alpha, a, PAD, lda, x, PAD, xStride, rows, columns, beta, wide, PAD, yStride)
            val actualDot = wide.copyOf()
            kernels.multiDot(alpha, a, PAD, lda, x, PAD, xStride, rows, columns, beta, actualDot, PAD, yStride)
            assertClose(expectedDot, actualDot, "multiDot $context")

            val expectedUpdate =
                referenceColumnUpdate(alpha, a, PAD, lda, coefficients, PAD, xStride, rows, columns, wide, PAD, yStride)
            val actualUpdate = wide.copyOf()
            kernels.columnUpdate(
                alpha, a, PAD, lda, coefficients, PAD, xStride, rows, columns, actualUpdate, PAD, yStride,
            )
            assertClose(expectedUpdate, actualUpdate, "columnUpdate $context")

            val expectedRank =
                referenceRankUpdate(alpha, a, PAD, lda, x, PAD, xStride, rows, columns, coefficients, PAD, xStride)
            val actualRank = a.copyOf()
            kernels.rankUpdate(
                alpha, actualRank, PAD, lda, x, PAD, xStride, rows, columns, coefficients, PAD, xStride,
            )
            assertClose(expectedRank, actualRank, "rankUpdate $context")

            assertCoupledAgrees(kernels, alpha, a, lda, rows, columns, context, rng)
        }
    }
    assertIndexedPanelsAgreeWithReference(kernels)
}

/** The coupled indexed pass, written out as the two halves it fuses. */
private fun referenceIndexedCoupled(
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray,
    offset: Int,
    rowStride: Int,
    indexStride: Int,
    indices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    columns: Int,
    rows: Int,
    pivot: Int,
    sums: DoubleArray,
    sumOffset: Int,
    excluded: Int,
): Pair<DoubleArray, DoubleArray> {
    val panel = a.copyOf()
    val reduced = if (sums === a) panel else sums.copyOf()
    for (c in 0 until columns) {
        val t = alpha * values[fromIndex + c]
        for (i in 0 until rows) {
            val at = offset + i * rowStride + indices[fromIndex + c] * indexStride
            panel[at] += t * b[pivot + i * rowStride]
            if (c != excluded) reduced[sumOffset + i * rowStride] += t * b[at]
        }
    }
    return panel to reduced
}

/**
 * The two indexed panels against their written-out definitions, in both layouts a sparse product hands them.
 *
 * The rows are a group of right-hand sides and the columns the stored entries of one sparse column, so the
 * sweep is over both: a group narrower than a lane block, one that leaves a tail, and a column holding
 * anything from nothing upwards. Adjacent and strided right-hand sides are different bodies on a vector
 * backend and the same answer either way. The selected positions are spread rather than consecutive.
 */
private fun assertIndexedPanelsAgreeWithReference(kernels: DensePanelKernels) {
    val rng = Random(20260930)
    val alpha = 0.875
    for (rows in intArrayOf(0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 64)) {
        for (columns in intArrayOf(0, 1, 2, 3, 5, 8)) {
            val positions = 11
            // Indices and values are one slice of a CSC column, so both are read from the same offset.
            val indices = IntArray(PAD + columns + PAD) { if (it < PAD) 0 else ((it - PAD) * 3 + 1) % positions }
            val values = DoubleArray(PAD + columns + PAD) { rng.nextDouble(-1.0, 1.0) }
            val x = DoubleArray(PAD + maxOf(rows, 1) + PAD) { rng.nextDouble(-1.0, 1.0) }
            for (adjacent in booleanArrayOf(true, false)) {
                val rowStride = if (adjacent) 1 else positions
                val indexStride = if (adjacent) maxOf(rows, 1) else 1
                val span = maxOf(rows, 1) * rowStride + positions * indexStride
                val block = DoubleArray(PAD + span + PAD) { rng.nextDouble(-1.0, 1.0) }
                val context = "rows=$rows columns=$columns adjacent=$adjacent"

                val expectedUpdate = referenceIndexedColumnUpdate(
                    alpha, block, PAD, rowStride, indexStride, indices, values, PAD, columns, rows, x, PAD,
                )
                val actualUpdate = x.copyOf()
                kernels.indexedColumnUpdate(
                    alpha, block, PAD, rowStride, indexStride, indices, values, PAD, columns, rows,
                    actualUpdate, PAD,
                )
                assertClose(expectedUpdate, actualUpdate, "indexedColumnUpdate $context")

                val expectedScatter = referenceIndexedRankUpdate(
                    alpha, block, PAD, rowStride, indexStride, indices, values, PAD, columns, rows, x, PAD,
                )
                val actualScatter = block.copyOf()
                kernels.indexedRankUpdate(
                    alpha, actualScatter, PAD, rowStride, indexStride, indices, values, PAD, columns, rows,
                    x, PAD,
                )
                assertClose(expectedScatter, actualScatter, "indexedRankUpdate $context")

                // With and without an excluded position, which is where a symmetric column meets its diagonal.
                for (excluded in intArrayOf(-1, 0, maxOf(columns - 1, 0))) {
                    val source = DoubleArray(block.size) { block[it] * 0.5 - 0.25 }
                    // A pivot in a window of its own, and then one inside the panel: both layouts a
                    // symmetric column's pivot row arrives in.
                    val sums = DoubleArray(PAD + maxOf(rows, 1) * rowStride + PAD) {
                        rng.nextDouble(-1.0, 1.0)
                    }
                    val (expectedPanel, expectedSums) = referenceIndexedCoupled(
                        alpha, block, source, PAD, rowStride, indexStride, indices, values, PAD, columns,
                        rows, PAD, sums, PAD, excluded,
                    )
                    val actualPanel = block.copyOf()
                    val actualSums = sums.copyOf()
                    kernels.indexedCoupledUpdate(
                        alpha, actualPanel, source, PAD, rowStride, indexStride, indices, values, PAD,
                        columns, rows, PAD, actualSums, PAD, excluded,
                    )
                    assertClose(expectedPanel, actualPanel, "indexedCoupledUpdate panel $context $excluded")
                    assertClose(expectedSums, actualSums, "indexedCoupledUpdate sums $context $excluded")

                    val inPlaceExpected = referenceIndexedCoupled(
                        alpha, block, source, PAD, rowStride, indexStride, indices, values, PAD, columns,
                        rows, PAD, block, PAD, excluded,
                    ).first
                    val inPlace = block.copyOf()
                    kernels.indexedCoupledUpdate(
                        alpha, inPlace, source, PAD, rowStride, indexStride, indices, values, PAD, columns,
                        rows, PAD, inPlace, PAD, excluded,
                    )
                    assertClose(inPlaceExpected, inPlace, "indexedCoupledUpdate in place $context $excluded")
                }
            }
        }
    }
}

/**
 * The coupled pass against the two halves it fuses, which is a column update and a multi-dot over the same
 * window; contiguous throughout, which is what its contract takes.
 */
private fun assertCoupledAgrees(
    kernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    rows: Int,
    columns: Int,
    context: String,
    rng: Random,
) {
    val x = DoubleArray(PAD + rows + PAD) { rng.nextDouble(-1.0, 1.0) }
    val coefficients = DoubleArray(PAD + columns + PAD) { rng.nextDouble(-1.0, 1.0) }
    val y = DoubleArray(PAD + rows + PAD) { rng.nextDouble(-1.0, 1.0) }
    val sums = DoubleArray(PAD + columns + PAD) { rng.nextDouble(-1.0, 1.0) }
    val expectedY = referenceColumnUpdate(alpha, a, PAD, lda, coefficients, PAD, 1, rows, columns, y, PAD, 1)
    val expectedSums = sums.copyOf()
    // A window with no rows writes neither half, which is the contract's rule rather than an empty sum.
    for (c in 0 until columns) {
        if (rows == 0) continue
        var sum = 0.0
        for (i in 0 until rows) sum += a[PAD + i + c * lda] * x[PAD + i]
        expectedSums[PAD + c] += alpha * sum
    }
    val actualY = y.copyOf()
    val actualSums = sums.copyOf()
    kernels.coupledUpdateDot(
        alpha, a, PAD, lda, x, PAD, rows, columns, actualY, PAD, coefficients, PAD, actualSums, PAD,
    )
    assertClose(expectedY, actualY, "coupledUpdateDot window $context")
    assertClose(expectedSums, actualSums, "coupledUpdateDot sums $context")
}

/**
 * That an address past what a word of twenty-one bits holds is addressed as itself.
 *
 * The offset and the two strides are ordinary array indices that a legal window of a large dense operand can
 * push past two million, so each is taken past that bound in turn, in both panel orientations.
 */
internal fun assertIndexedPanelsAddressPastAPackedBound(kernels: DensePanelKernels) {
    val bound = 1 shl 21
    val block = DoubleArray(bound + 16)
    val indices = intArrayOf(0, 1)
    val values = doubleArrayOf(2.0, 3.0)
    val x = doubleArrayOf(1.0, 10.0)

    // An offset past the bound, with both strides small: two right-hand sides at two selected positions.
    val offset = bound + 4
    kernels.indexedRankUpdate(1.0, block, offset, 1, 2, indices, values, 0, 2, 2, x, 0)
    assertEquals(2.0, block[offset], "an offset past a packed bound wrote elsewhere")
    assertEquals(20.0, block[offset + 1])
    assertEquals(3.0, block[offset + 2])
    assertEquals(30.0, block[offset + 3])

    // A row stride past the bound, which is a strided panel of an operand with that many rows.
    block.fill(0.0)
    kernels.indexedRankUpdate(1.0, block, 0, bound + 1, 1, indices, values, 0, 2, 2, x, 0)
    assertEquals(2.0, block[0], "a row stride past a packed bound wrote elsewhere")
    assertEquals(20.0, block[bound + 1])
    assertEquals(3.0, block[1])
    assertEquals(30.0, block[bound + 2])

    // An index stride past the bound, which is a staged panel that wide.
    block.fill(0.0)
    block[0] = 1.0
    block[bound + 1] = 100.0
    val sums = doubleArrayOf(0.0, 0.0)
    kernels.indexedColumnUpdate(1.0, block, 0, 1, bound + 1, indices, values, 0, 2, 2, sums, 0)
    assertEquals(2.0 * 1.0 + 3.0 * 100.0, sums[0], "an index stride past a packed bound read elsewhere")
}

/**
 * The rules a panel keeps that a random sweep cannot see: a zero beta overwrites a destination it never
 * reads, and a zero coefficient is still multiplied, which is the opposite of the Level 1 `axpy` rule and
 * must not leak in here. A backwards source is what a vector with a negative step reaches these with.
 */
internal fun assertPanelContractHolds(kernels: DensePanelKernels) {
    val rows = 17
    val a = DoubleArray(rows * 2) { 1.0 + it }
    a[rows] = Double.POSITIVE_INFINITY
    val x = DoubleArray(rows) { 1.0 }

    val poisoned = DoubleArray(2) { Double.NaN }
    kernels.multiDot(1.0, a, 0, rows, x, 0, 1, rows, 1, 0.0, poisoned, 0, 1)
    assertTrue(poisoned[0].isFinite(), "a zero beta read the destination it was told to overwrite")

    val zeroCoefficients = DoubleArray(2)
    val target = DoubleArray(rows)
    kernels.columnUpdate(1.0, a, rows, rows, zeroCoefficients, 0, 1, rows, 1, target, 0, 1)
    assertTrue(target[0].isNaN(), "a zero coefficient skipped its product against an infinity")

    val rankTarget = DoubleArray(rows)
    kernels.rankUpdate(1.0, rankTarget, 0, rows, a, rows, 1, rows, 1, zeroCoefficients, 0, 1)
    assertTrue(rankTarget[0].isNaN(), "a zero rank-update coefficient skipped its product against an infinity")

    assertIndexedPanelsEvaluateZeroCoefficients(kernels)
    assertExcludedPositionIsNotReduced(kernels)

    val forwards = DoubleArray(rows) { it * 0.5 }
    val backwards = DoubleArray(rows) { (rows - 1 - it) * 0.5 }
    val fromForwards = DoubleArray(rows)
    val fromBackwards = DoubleArray(rows)
    val ones = doubleArrayOf(1.0)
    kernels.rankUpdate(1.0, fromForwards, 0, rows, forwards, 0, 1, rows, 1, ones, 0, 1)
    kernels.rankUpdate(1.0, fromBackwards, 0, rows, backwards, rows - 1, -1, rows, 1, ones, 0, 1)
    assertClose(fromForwards, fromBackwards, "a backwards source did not read its window in reverse")
}

/** The grouping a backend recommends, which has to be usable and must not claim more columns than exist. */
internal fun assertExecutionGroupIsUsable(kernels: DensePanelKernels) {
    for (work in PanelWork.entries) {
        for (columns in intArrayOf(0, 1, 2, 3, 7, 1024)) {
            for (contiguous in booleanArrayOf(true, false)) {
                val group = kernels.executionGroup(work, 512, columns, contiguous)
                assertTrue(group >= 1, "$work recommended $group columns at $columns")
                assertTrue(columns == 0 || group <= columns, "$work recommended $group of $columns columns")
            }
            // A backend that recommends a copy must name a different body for the copied layout, or a route
            // would report the body the copy was not made for. The converse is not required.
            for (rows in intArrayOf(1, 2, 4, 8, 16, 512)) {
                val prefers = kernels.prefersContiguous(work, rows, maxOf(columns, 1))
                val differs = kernels.implementationFor(work, rows, maxOf(columns, 1), contiguous = true) !=
                    kernels.implementationFor(work, rows, maxOf(columns, 1), contiguous = false)
                assertTrue(
                    !prefers || differs,
                    "$work at $rows by $columns recommends a copy to the same body",
                )
            }
        }
    }
    assertEquals(
        kernels.executionGroup(PanelWork.MultiDot, 512, 1024),
        kernels.executionGroup(PanelWork.MultiDot, 512, 1024),
        "a recommendation changed between two identical questions",
    )
}

/**
 * The empty extents. Every window is passed as an array too short to index, so an implementation that read
 * an entry before noticing the extent fails with an index error rather than quietly passing. A multi-dot
 * still writes its outputs, since those are selected by the columns and not by the rows.
 */
internal fun assertEmptyExtentsReadNothing(kernels: DensePanelKernels) {
    val nothing = DoubleArray(0)
    val destination = DoubleArray(2) { Double.NaN }

    kernels.multiDot(0.875, nothing, 0, 0, nothing, 0, 1, 0, 2, 0.0, destination, 0, 1)
    assertEquals(0.0, destination[0], "an empty multi-dot did not write its first output")
    assertEquals(0.0, destination[1], "an empty multi-dot did not write its second output")

    kernels.multiDot(0.875, nothing, 0, 0, nothing, 0, 1, 4, 0, 0.0, nothing, 0, 1)
    kernels.columnUpdate(0.875, nothing, 0, 0, nothing, 0, 1, 0, 2, nothing, 0, 1)
    kernels.columnUpdate(0.875, nothing, 0, 0, nothing, 0, 1, 4, 0, nothing, 0, 1)
    kernels.coupledUpdateDot(0.875, nothing, 0, 0, nothing, 0, 0, 2, nothing, 0, nothing, 0, nothing, 0)
    kernels.coupledUpdateDot(0.875, nothing, 0, 0, nothing, 0, 4, 0, nothing, 0, nothing, 0, nothing, 0)
    kernels.rankUpdate(0.875, nothing, 0, 0, nothing, 0, 1, 0, 2, nothing, 0, 1)
    kernels.rankUpdate(0.875, nothing, 0, 0, nothing, 0, 1, 4, 0, nothing, 0, 1)
    val noIndices = IntArray(0)
    kernels.indexedColumnUpdate(0.875, nothing, 0, 1, 1, noIndices, nothing, 0, 0, 4, nothing, 0)
    kernels.indexedColumnUpdate(0.875, nothing, 0, 1, 1, noIndices, nothing, 0, 2, 0, nothing, 0)
    kernels.indexedRankUpdate(0.875, nothing, 0, 1, 1, noIndices, nothing, 0, 0, 4, nothing, 0)
    kernels.indexedRankUpdate(0.875, nothing, 0, 1, 1, noIndices, nothing, 0, 2, 0, nothing, 0)
    kernels.indexedCoupledUpdate(
        0.875, nothing, nothing, 0, 1, 1, noIndices, nothing, 0, 0, 4, 0, nothing, 0, -1,
    )
    kernels.indexedCoupledUpdate(
        0.875, nothing, nothing, 0, 1, 1, noIndices, nothing, 0, 2, 0, 0, nothing, 0, -1,
    )
}

/**
 * That a panel writes inside its window and nowhere else. The buffers are wider than the window on both
 * sides and filled with a value no arithmetic here produces, so a body that ran one lane past its bound, or
 * wrote a masked tail through, changes a guard entry.
 */
internal fun assertPanelsStayInsideTheirWindows(kernels: DensePanelKernels) {
    val guard = -12345.0
    for (rows in intArrayOf(1, 3, 5, 8, 13)) {
        for (columns in intArrayOf(1, 2, 3, 5)) {
            val lda = rows + 2
            val a = DoubleArray(PAD + lda * columns + PAD) { guard }
            val x = DoubleArray(PAD + rows + PAD) { guard }
            val coefficients = DoubleArray(PAD + columns + PAD) { guard }
            val y = DoubleArray(PAD + rows + PAD) { guard }
            val sums = DoubleArray(PAD + columns + PAD) { guard }
            kernels.columnUpdate(0.5, a, PAD, lda, coefficients, PAD, 1, rows, columns, y, PAD, 1)
            kernels.coupledUpdateDot(0.5, a, PAD, lda, x, PAD, rows, columns, y, PAD, coefficients, PAD, sums, PAD)
            kernels.rankUpdate(0.5, a, PAD, lda, x, PAD, 1, rows, columns, coefficients, PAD, 1)
            val indices = IntArray(PAD + columns + PAD) { if (it < PAD) 0 else it - PAD }
            val indexed = DoubleArray(PAD + rows * columns + PAD) { guard }
            val gathered = DoubleArray(PAD + rows + PAD) { guard }
            kernels.indexedColumnUpdate(
                0.5, indexed, PAD, 1, rows, indices, coefficients, PAD, columns, rows, gathered, PAD,
            )
            kernels.indexedRankUpdate(
                0.5, indexed, PAD, 1, rows, indices, coefficients, PAD, columns, rows, gathered, PAD,
            )
            val context = "rows=$rows columns=$columns"
            assertGuards(gathered, rows, "indexedColumnUpdate destination $context", guard)
            assertGuards(indexed, rows * columns, "indexedRankUpdate panel $context", guard)
            // A source of its own: the leaf reads b and writes a, and handing it one array for both would
            // be asking it to read a window it is in the middle of writing.
            val source = DoubleArray(PAD + rows * columns + PAD) { guard }
            val coupledSums = DoubleArray(PAD + rows + PAD) { guard }
            kernels.indexedCoupledUpdate(
                0.5, indexed, source, PAD, 1, rows, indices, coefficients, PAD, columns, rows,
                PAD, coupledSums, PAD, -1,
            )
            assertGuards(coupledSums, rows, "indexedCoupledUpdate sums $context", guard)
            assertGuards(indexed, rows * columns, "indexedCoupledUpdate panel $context", guard)
            assertGuards(source, rows * columns, "indexedCoupledUpdate source $context", guard)
            assertGuards(y, rows, "columnUpdate destination $context", guard)
            assertGuards(sums, columns, "coupledUpdateDot sums $context", guard)
            assertGuards(a, lda * columns, "rankUpdate panel $context", guard)
        }
    }
}

/**
 * The excluded position, which is scattered and not reduced, as a symmetric column's diagonal needs. The
 * source there is a NaN, so a sum that touched it could not come back finite. The pivot is a column of its
 * own because the scatter reads it at every position, the excluded one included.
 */
private fun assertExcludedPositionIsNotReduced(kernels: DensePanelKernels) {
    val width = vectorWidth(kernels) + 1
    val positions = 3
    for (excluded in 0 until positions) {
        val indices = IntArray(positions) { it }
        val values = DoubleArray(positions) { 1.0 + it }
        val panel = DoubleArray(width * positions)
        val source = DoubleArray(width * (positions + 1)) { 1.0 }
        for (i in 0 until width) source[excluded * width + i] = Double.NaN
        val sums = DoubleArray(width)

        kernels.indexedCoupledUpdate(
            1.0, panel, source, 0, 1, width, indices, values, 0, positions, width,
            positions * width, sums, 0, excluded,
        )

        for (i in 0 until width) {
            assertTrue(
                sums[i].isFinite(),
                "the excluded position was reduced at width $width lane $i on ${kernels.name}",
            )
            assertEquals(
                values[excluded],
                panel[excluded * width + i],
                "the excluded position was not scattered at width $width lane $i",
            )
        }
    }
}

/**
 * The zero-evaluation rule for the indexed panels, at the narrowest width the backend answers for with its
 * own body and again one wider, where the last right-hand side is a scalar tail. A fixed narrow width would
 * only ever exercise the portable fallback.
 */
private fun assertIndexedPanelsEvaluateZeroCoefficients(kernels: DensePanelKernels) {
    val full = vectorWidth(kernels)
    for (width in intArrayOf(2, full, full + 1)) {
        val positions = 2
        val zeroValues = DoubleArray(positions)
        val indices = IntArray(positions) { it }
        val infinite = DoubleArray(width * positions) { Double.POSITIVE_INFINITY }
        val gathered = DoubleArray(width)
        // Adjacent right-hand sides, which is the layout the vector body is reached over.
        kernels.indexedColumnUpdate(1.0, infinite, 0, 1, width, indices, zeroValues, 0, positions, width, gathered, 0)
        for (i in 0 until width) {
            assertTrue(
                gathered[i].isNaN(),
                "a zero indexed coefficient skipped its product at width $width lane $i on ${kernels.name}",
            )
        }

        val scattered = DoubleArray(width * positions)
        val ones = DoubleArray(width) { Double.POSITIVE_INFINITY }
        kernels.indexedRankUpdate(1.0, scattered, 0, 1, width, indices, zeroValues, 0, positions, width, ones, 0)
        for (i in 0 until width * positions) {
            assertTrue(
                scattered[i].isNaN(),
                "a zero indexed scatter coefficient skipped its product at width $width entry $i",
            )
        }

        val coupledPanel = DoubleArray(width * positions)
        val coupledSums = DoubleArray(width)
        kernels.indexedCoupledUpdate(
            1.0, coupledPanel, infinite, 0, 1, width, indices, zeroValues, 0, positions, width,
            0, coupledSums, 0, -1,
        )
        for (i in 0 until width) {
            assertTrue(coupledSums[i].isNaN(), "a zero coupled coefficient skipped its reduction at lane $i")
            assertTrue(coupledPanel[i].isNaN(), "a zero coupled coefficient skipped its scatter at lane $i")
        }
    }
    // And the widths really were the two sides of this backend's own boundary.
    assertEquals(
        kernels.implementationFor(PanelWork.SparseRightHandSides, full, 2),
        kernels.name,
        "the full width did not reach ${kernels.name}'s own body",
    )
}

/** The narrowest group of adjacent right-hand sides this backend answers for with its own body. */
internal fun vectorWidth(kernels: DensePanelKernels): Int {
    for (width in 1..64) {
        if (kernels.implementationFor(PanelWork.SparseRightHandSides, width, 2) == kernels.name) return width
    }
    return 1
}

/** The [PAD] entries on each side of a window, which nothing may touch. */
private fun assertGuards(buffer: DoubleArray, window: Int, context: String, guard: Double) {
    for (i in 0 until PAD) {
        assertEquals(guard, buffer[i], "$context: entry $i before the window changed")
        val after = PAD + window + i
        assertEquals(guard, buffer[after], "$context: entry $after after the window changed")
    }
}
