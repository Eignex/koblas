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

/**
 * Every panel of [kernels] against the definitions written out above, over extents that straddle any
 * grouping and any lane width, at a nonzero offset and a leading dimension wider than the window.
 *
 * The strides rotate with the extent rather than multiplying out, which keeps the sweep bounded while still
 * reaching a strided shared vector, a strided destination and a backwards source at several widths.
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
 * The rules a panel keeps that a random sweep cannot see.
 *
 * A zero beta overwrites a destination it never reads, which is what lets a caller pass an uninitialised one.
 * A zero coefficient is still multiplied, because this is matrix arithmetic and every position the traversal
 * reaches contributes what it evaluates to; the Level 1 `axpy` rule that returns without touching the
 * destination is the opposite one and must not leak in here. A backwards source is what a vector with a
 * negative step reaches these with.
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
            val group = kernels.executionGroup(work, 512, columns)
            assertTrue(group >= 1, "$work recommended $group columns at $columns")
            assertTrue(columns == 0 || group <= columns, "$work recommended $group of $columns columns")
        }
    }
    assertEquals(
        kernels.executionGroup(PanelWork.MultiDot, 512, 1024),
        kernels.executionGroup(PanelWork.MultiDot, 512, 1024),
        "a recommendation changed between two identical questions",
    )
}

/**
 * The empty extents, which are settled by the contract rather than by where a loop sits.
 *
 * Every window here is passed as an array too short to index, so an implementation that read a coefficient
 * or a matrix entry before noticing the extent fails with an index error rather than quietly passing. A
 * multi-dot still writes its outputs, since those are selected by the columns and not by the rows.
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
}

/**
 * That a panel writes inside its window and nowhere else.
 *
 * The buffers are wider than the window on both sides and filled with a value no arithmetic here produces, so
 * a body that ran one lane past its bound, or wrote a masked tail through, changes a guard entry.
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
            val context = "rows=$rows columns=$columns"
            assertGuards(y, rows, "columnUpdate destination $context", guard)
            assertGuards(sums, columns, "coupledUpdateDot sums $context", guard)
            assertGuards(a, lda * columns, "rankUpdate panel $context", guard)
        }
    }
}

/** The [PAD] entries on each side of a window, which nothing may touch. */
private fun assertGuards(buffer: DoubleArray, window: Int, context: String, guard: Double) {
    for (i in 0 until PAD) {
        assertEquals(guard, buffer[i], "$context: entry $i before the window changed")
        val after = PAD + window + i
        assertEquals(guard, buffer[after], "$context: entry $after after the window changed")
    }
}
