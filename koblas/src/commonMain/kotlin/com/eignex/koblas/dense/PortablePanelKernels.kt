@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.dense

/**
 * The portable panel arithmetic every platform has, and the floor a vector backend falls back to.
 *
 * Written as ordinary Kotlin loops with two private specializations, four columns and two, behind a
 * one-column remainder. Four is where the old `dot4` and `axpy4` bodies land: a group that wide reads the
 * shared vector once for four columns instead of once each. The widths are this file's own and appear in no
 * algorithm above it.
 */
internal object PortablePanelKernels : DensePanelKernels {
    override val name: String get() = "scalar-panel"

    /**
     * Four columns where one walk of a shared vector feeds four accumulators, and two where each column
     * carries its own destination.
     *
     * A multi-dot and a column update read the same vector for every column of a group, so each of its loads
     * pays for four columns of arithmetic rather than one, and grouping them measured ahead of one at a time
     * at every extent. A coupled pass and a rank update are two columns wide in the bodies below, so a wider
     * group would only change how many calls the traversal makes, and for a triangular caller it also grows
     * the scalar corner with the square of the width.
     *
     * The comparison is in the stage evidence rather than here, because a figure in this file would be one
     * machine's and the benchmark suites are where a quotable number comes from.
     */
    override fun executionGroup(work: PanelWork, rows: Int, columns: Int): Int {
        val preferred = when (work) {
            PanelWork.CoupledDotUpdate, PanelWork.RankUpdate -> NARROW_GROUP
            else -> WIDE_GROUP
        }
        return if (columns <= 0) 1 else minOf(preferred, columns)
    }

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String = name

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
        var c = 0
        while (c + WIDE_GROUP <= columns) {
            val to = yOffset + c * yStride
            dotFour(a, aOffset + c * lda, lda, x, xOffset, xStride, rows, alpha, beta, y, to, yStride)
            c += WIDE_GROUP
        }
        while (c < columns) {
            val sum = dotOne(a, aOffset + c * lda, x, xOffset, xStride, rows)
            store(y, yOffset + c * yStride, alpha, sum, beta)
            c++
        }
    }

    /** The recovered four-dot body: one walk of the shared vector feeding four independent sums. */
    private fun dotFour(
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        alpha: Double,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        var s0 = 0.0
        var s1 = 0.0
        var s2 = 0.0
        var s3 = 0.0
        var at = xOffset
        for (i in 0 until rows) {
            val v = x[at]
            s0 += a[aOffset + i] * v
            s1 += a[aOffset + i + lda] * v
            s2 += a[aOffset + i + 2 * lda] * v
            s3 += a[aOffset + i + 3 * lda] * v
            at += xStride
        }
        store(y, yOffset, alpha, s0, beta)
        store(y, yOffset + yStride, alpha, s1, beta)
        store(y, yOffset + 2 * yStride, alpha, s2, beta)
        store(y, yOffset + 3 * yStride, alpha, s3, beta)
    }

    private fun dotOne(a: DoubleArray, aOffset: Int, x: DoubleArray, xOffset: Int, xStride: Int, rows: Int): Double {
        var sum = 0.0
        var at = xOffset
        for (i in 0 until rows) {
            sum += a[aOffset + i] * x[at]
            at += xStride
        }
        return sum
    }

    /** `y = alpha · sum + beta · y`, where a zero beta overwrites without reading the destination. */
    private fun store(y: DoubleArray, at: Int, alpha: Double, sum: Double, beta: Double) {
        y[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * y[at]
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
        if (rows <= 0 || columns <= 0) return
        var c = 0
        while (c + WIDE_GROUP <= columns) {
            val t0 = alpha * x[xOffset + c * xStride]
            val t1 = alpha * x[xOffset + (c + 1) * xStride]
            val t2 = alpha * x[xOffset + (c + 2) * xStride]
            val t3 = alpha * x[xOffset + (c + 3) * xStride]
            val at = aOffset + c * lda
            var to = yOffset
            for (i in 0 until rows) {
                y[to] += t0 * a[at + i] + t1 * a[at + i + lda] + t2 * a[at + i + 2 * lda] + t3 * a[at + i + 3 * lda]
                to += yStride
            }
            c += WIDE_GROUP
        }
        while (c < columns) {
            val t = alpha * x[xOffset + c * xStride]
            val at = aOffset + c * lda
            var to = yOffset
            for (i in 0 until rows) {
                y[to] += t * a[at + i]
                to += yStride
            }
            c++
        }
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
        if (rows <= 0 || columns <= 0) return
        var c = 0
        while (c + NARROW_GROUP <= columns) {
            val t0 = alpha * coefficients[coefficientOffset + c]
            val t1 = alpha * coefficients[coefficientOffset + c + 1]
            val at = aOffset + c * lda
            var s0 = 0.0
            var s1 = 0.0
            for (i in 0 until rows) {
                val v0 = a[at + i]
                val v1 = a[at + i + lda]
                y[yOffset + i] += t0 * v0 + t1 * v1
                val xi = x[xOffset + i]
                s0 += v0 * xi
                s1 += v1 * xi
            }
            sums[sumOffset + c] += alpha * s0
            sums[sumOffset + c + 1] += alpha * s1
            c += NARROW_GROUP
        }
        while (c < columns) {
            val t = alpha * coefficients[coefficientOffset + c]
            val at = aOffset + c * lda
            var sum = 0.0
            for (i in 0 until rows) {
                val v = a[at + i]
                y[yOffset + i] += t * v
                sum += v * x[xOffset + i]
            }
            sums[sumOffset + c] += alpha * sum
            c++
        }
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
        if (rows <= 0 || columns <= 0) return
        var c = 0
        while (c + NARROW_GROUP <= columns) {
            val t0 = alpha * coefficients[coefficientOffset + c * coefficientStride]
            val t1 = alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride]
            val at = aOffset + c * lda
            var from = xOffset
            for (i in 0 until rows) {
                val v = x[from]
                a[at + i] += t0 * v
                a[at + i + lda] += t1 * v
                from += xStride
            }
            c += NARROW_GROUP
        }
        while (c < columns) {
            val t = alpha * coefficients[coefficientOffset + c * coefficientStride]
            val at = aOffset + c * lda
            var from = xOffset
            for (i in 0 until rows) {
                a[at + i] += t * x[from]
                from += xStride
            }
            c++
        }
    }

    /** Columns grouped where one walk of a shared vector serves all of them. */
    private const val WIDE_GROUP = 4

    /** Columns grouped where the body itself is that wide. */
    private const val NARROW_GROUP = 2
}
