package com.eignex.koblas.dense

import com.eignex.koblas.DenseVector
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import com.eignex.koblas.vendor.RouteKind
import kotlin.random.Random
import kotlin.test.assertEquals

// A route is checked against the windows a call really cut, not against the extents they were cut from: a
// length the schedule never produces tells nothing about which bodies ran.

/**
 * A backend that records the body each window it is handed reaches, and delegates the arithmetic. [group]
 * and [threshold] override the delegate's grouping and its choice of body, which is how a sweep reaches
 * shapes a real backend's own answers never produce.
 */
internal class RecordingPanels(
    private val delegate: DensePanelKernels,
    private val group: Int? = null,
    private val threshold: Int? = null,
) : DensePanelKernels {
    /** Every window length handed over, in call order, including the empty ones a traversal skips. */
    val windows: MutableList<Int> = ArrayList()

    /** The body each nonempty window reached, in call order. */
    val bodies: MutableList<String> = ArrayList()

    override val name: String get() = "recording(${delegate.name})"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int = when {
        group == null -> delegate.executionGroup(work, rows, columns, contiguous)
        columns <= 0 -> 1
        else -> minOf(group, columns)
    }

    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean = when (threshold) {
        null -> delegate.prefersContiguous(work, rows, columns)
        else -> rows >= threshold
    }

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String = when {
        threshold == null -> delegate.implementationFor(work, rows, columns, contiguous)
        rows >= threshold -> "wide"
        else -> "narrow"
    }

    private fun record(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean = true) {
        windows.add(rows)
        if (rows > 0) bodies.add(implementationFor(work, rows, columns, contiguous))
    }

    override fun multiDot(
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
    ) {
        record(PanelWork.MultiDot, rows, columns)
        delegate.multiDot(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride)
    }

    override fun columnUpdate(
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
    ) {
        record(PanelWork.ColumnUpdate, rows, columns)
        delegate.columnUpdate(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride)
    }

    override fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
    ) {
        record(PanelWork.CoupledDotUpdate, rows, columns)
        delegate.coupledUpdateDot(
            alpha, a, aOffset, lda, x, xOffset, rows, columns, y, yOffset,
            coefficients, coefficientOffset, sums, sumOffset,
        )
    }

    override fun rankUpdate(
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
    ) {
        record(PanelWork.RankUpdate, rows, columns)
        delegate.rankUpdate(
            alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns,
            coefficients, coefficientOffset, coefficientStride,
        )
    }

    override fun indexedColumnUpdate(
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
    ) {
        if (columns > 0) record(PanelWork.SparseRightHandSides, rows, columns, rowStride == 1)
        delegate.indexedColumnUpdate(
            alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, y, yOffset,
        )
    }

    override fun indexedRankUpdate(
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
    ) {
        if (columns > 0) record(PanelWork.SparseRightHandSides, rows, columns, rowStride == 1)
        delegate.indexedRankUpdate(
            alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, x, xOffset,
        )
    }

    override fun indexedCoupledUpdate(
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
    ) {
        if (columns > 0) record(PanelWork.SparseRightHandSides, rows, columns, rowStride == 1)
        delegate.indexedCoupledUpdate(
            alpha, a, b, offset, rowStride, indexStride, indices, values, fromIndex, columns, rows,
            pivot, sums, sumOffset, excluded,
        )
    }
}

/** Every operation that schedules a panel, which is every one a route can name a body for. */
internal val PANEL_OPERATIONS: List<DenseMatrixOperation> = listOf(
    DenseMatrixOperation.Gemv,
    DenseMatrixOperation.GemvTransposed,
    DenseMatrixOperation.Symv,
    DenseMatrixOperation.Ger,
    DenseMatrixOperation.Syr,
    DenseMatrixOperation.Trmv,
    DenseMatrixOperation.TrmvTransposed,
    DenseMatrixOperation.Trsv,
    DenseMatrixOperation.TrsvTransposed,
)

/**
 * Orders that reach the boundaries a window schedule has: odd and even, since a symmetric traversal grouped
 * by two cuts only even windows at an even order; short ones, where a triangle is narrower than a lane
 * block; and one long enough that a body change in the middle of the schedule is visible.
 */
internal val ROUTE_ORDERS: IntArray = intArrayOf(1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 512)

/** The same boundaries without the long one, for a sweep that multiplies out over groups and thresholds. */
internal val SHORT_ROUTE_ORDERS: IntArray = intArrayOf(1, 2, 3, 4, 5, 7, 8, 9, 17)

/**
 * That [panels] routes every panel operation to exactly the bodies its own traversal reaches, over both
 * triangles and both transpose flags, since the windows a triangular traversal cuts depend on both.
 */
internal fun assertRouteNamesExecutedBodies(
    panels: DensePanelKernels,
    group: Int? = null,
    threshold: Int? = null,
    orders: IntArray = ROUTE_ORDERS,
) {
    val rng = Random(20260926)
    for (n in orders) {
        for (lower in booleanArrayOf(true, false)) {
            for (operation in PANEL_OPERATIONS) {
                assertRouteNamesExecutedBodies(operation, panels, n, lower, group, threshold, rng)
            }
        }
    }
}

@Suppress("LongParameterList") // the operation, the backend under test, its shape and the two overrides
private fun assertRouteNamesExecutedBodies(
    operation: DenseMatrixOperation,
    panels: DensePanelKernels,
    n: Int,
    lower: Boolean,
    group: Int?,
    threshold: Int?,
    rng: Random,
) {
    val recorder = RecordingPanels(panels, group, threshold)
    val blas = PortableDenseBlas(ScalarVectorKernels, recorder)
    val a = randomMatrix(n, n, rng)
    for (i in 0 until n) a.values[i + i * n] = 2.0 + i % 3
    val x = randomVector(n, rng)
    val y = randomVector(n, rng)
    when (operation) {
        DenseMatrixOperation.Gemv -> blas.gemv(0.875, a, x, -0.25, y)
        DenseMatrixOperation.GemvTransposed -> blas.gemv(0.875, a, x, -0.25, y, transpose = true)
        DenseMatrixOperation.Symv -> blas.symv(0.875, a, x, -0.25, y, lower)
        DenseMatrixOperation.Ger -> blas.ger(0.875, x, y, a)
        DenseMatrixOperation.Syr -> blas.syr(0.875, DenseVector.wrap(x), a, lower)
        DenseMatrixOperation.Trmv -> blas.trmv(a, x, lower)
        DenseMatrixOperation.TrmvTransposed -> blas.trmv(a, x, lower, transpose = true)
        DenseMatrixOperation.Trsv -> blas.trsv(a, x, lower)
        DenseMatrixOperation.TrsvTransposed -> blas.trsv(a, x, lower, transpose = true)
        else -> error("$operation schedules no panel")
    }
    val executed = recorder.bodies.distinct()
    val route = blas.routeOf(operation, DenseCall(n, n, 0.875, -0.25, lower = lower))
    val named = route.components.filterNot { it.endsWith("/scale") }.map { it.substringBefore('/') }
    val context = "$operation n=$n lower=$lower group=$group threshold=$threshold on ${panels.name}"

    assertEquals(executed, named, "$context: windows ${recorder.windows}")
    assertEquals(
        if (executed.size > 1) RouteKind.Composed else RouteKind.Direct,
        route.kind,
        "$context: windows ${recorder.windows}",
    )
}
