@file:Suppress("TooManyFunctions", "LongParameterList")

package com.eignex.koblas

import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.internal.sparse.indexedSquareSum
import kotlin.math.sqrt

/** Complete portable arithmetic, independent of every accelerated provider. */
internal object PortableEngine : KoblasEngine {
    override val name: String = "scalar"
    override val id: String = "portable"
    override val implementations = EngineImplementation.entries.filterNot { it.matrix }.toSet()

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = scalarDot(a, aOff, aStride, b, bOff, bStride, len)

    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ): Unit = scalarAxpy(y, yOff, yStride, alpha, x, xOff, xStride, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int): Unit =
        scalarScale(v, vOff, vStride, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = euclideanNorm(v, vOff, vStride, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = absoluteSum(
        v,
        vOff,
        vStride,
        len,
    )

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int = scalarIamax(
        v,
        vOff,
        vStride,
        len,
    )

    override fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int,
        yStride: Int,
    ) {
        if (xStride == 1 && yStride == 1) {
            scalarRot(x, xOff, y, yOff, len, c, s)
        } else {
            for (i in 0 until len) {
                val a = x[xOff + i * xStride]
                val b = y[yOff + i * yStride]
                x[xOff + i * xStride] = c * a + s * b
                y[yOff + i * yStride] = c * b - s * a
            }
        }
    }

    override fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Unit = scalarSwap(a, aOff, aStride, b, bOff, bStride, len)

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (aStride == 1 && bStride == 1) {
            a.copyInto(b, bOff, aOff, aOff + len)
        } else {
            for (i in 0 until len) b[bOff + i * bStride] = a[aOff + i * aStride]
        }
    }

    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = scalarSum(
        v,
        vOff,
        vStride,
        len,
    )

    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        var sum = 0.0
        for (k in 0 until count) sum += values[valueOffset + k] * dense[indices[indexOffset + k]]
        return sum
    }

    override fun dotSparse(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
    ): Double {
        var sum = 0.0
        var a = 0
        var b = 0
        while (a < xCount && b < yCount) {
            val ia = xIndices[xIndexOffset + a]
            val ib = yIndices[yIndexOffset + b]
            when {
                ia < ib -> a++

                ia > ib -> b++

                else -> {
                    sum += xValues[xValueOffset + a] * yValues[yValueOffset + b]
                    a++
                    b++
                }
            }
        }
        return sum
    }

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        for (k in 0 until count) destination[indices[indexOffset + k]] += alpha * values[valueOffset + k]
    }

    override fun computeIndexedScatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        for (k in 0 until count) destination[indices[indexOffset + k]] = values[valueOffset + k]
    }

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        for (k in 0 until count) values[valueOffset + k] = source[indices[indexOffset + k]]
    }

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            values[valueOffset + k] = source[index]
            source[index] = 0.0
        }
    }

    override fun computeIndexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        var squares = 0.0
        if (count >= 64) {
            squares = indexedSquareSum(indices, indexOffset, count, values)
        } else {
            for (k in 0 until count) {
                val value = values[indices[indexOffset + k]]
                squares += value * value
            }
        }
        if (squares.isFinite() && squares >= MIN_NORMAL) return sqrt(squares)
        return rescaledEuclideanNorm(count, squares) { values[indices[indexOffset + it]] }
    }

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
     * The comparison is in the local evidence rather than here, because a figure in this file would be one
     * machine's and the benchmark suites are where a quotable number comes from.
     */
    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int {
        val preferred = when {
            work == PanelWork.CoupledDotUpdate || work == PanelWork.RankUpdate -> PANEL_NARROW_GROUP
            work == PanelWork.SparseRightHandSides -> PANEL_SPARSE_GROUP
            work == PanelWork.SparseRightHandSideReduction && !contiguous -> PANEL_SPARSE_REDUCTION_GROUP
            work == PanelWork.SparseRightHandSideReduction -> PANEL_SPARSE_GROUP
            else -> PANEL_WIDE_GROUP
        }
        return if (columns <= 0) 1 else minOf(preferred, columns)
    }

    /**
     * Never, because these bodies read one entry at a time whatever the spacing between them is.
     *
     * A caller offering to copy a strided window into adjacent storage is offering this backend a pass over
     * the data for nothing, so the answer is the same at every extent.
     */
    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean = false

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
        while (c + PANEL_WIDE_GROUP <= columns) {
            val to = yOffset + c * yStride
            panelDotFour(a, aOffset + c * lda, lda, x, xOffset, xStride, rows, alpha, beta, y, to, yStride)
            c += PANEL_WIDE_GROUP
        }
        while (c < columns) {
            val sum = panelDotOne(a, aOffset + c * lda, x, xOffset, xStride, rows)
            panelStore(y, yOffset + c * yStride, alpha, sum, beta)
            c++
        }
    }

    /** The recovered four-dot body: one walk of the shared vector feeding four independent sums. */
    private fun panelDotFour(
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
        panelStore(y, yOffset, alpha, s0, beta)
        panelStore(y, yOffset + yStride, alpha, s1, beta)
        panelStore(y, yOffset + 2 * yStride, alpha, s2, beta)
        panelStore(y, yOffset + 3 * yStride, alpha, s3, beta)
    }

    private fun panelDotOne(
        a: DoubleArray,
        aOffset: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
    ): Double {
        var sum = 0.0
        var at = xOffset
        for (i in 0 until rows) {
            sum += a[aOffset + i] * x[at]
            at += xStride
        }
        return sum
    }

    /** `y = alpha · sum + beta · y`, where a zero beta overwrites without reading the destination. */
    private fun panelStore(y: DoubleArray, at: Int, alpha: Double, sum: Double, beta: Double) {
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
        while (c + PANEL_WIDE_GROUP <= columns) {
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
            c += PANEL_WIDE_GROUP
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
        xStride: Int,
        yStride: Int,
        coefficientStride: Int,
        sumStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        var c = 0
        while (c + PANEL_NARROW_GROUP <= columns) {
            val t0 = alpha * coefficients[coefficientOffset + c * coefficientStride]
            val t1 = alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride]
            val at = aOffset + c * lda
            var s0 = 0.0
            var s1 = 0.0
            for (i in 0 until rows) {
                val v0 = a[at + i]
                val v1 = a[at + i + lda]
                y[yOffset + i * yStride] += t0 * v0 + t1 * v1
                val xi = x[xOffset + i * xStride]
                s0 += v0 * xi
                s1 += v1 * xi
            }
            sums[sumOffset + c * sumStride] += alpha * s0
            sums[sumOffset + (c + 1) * sumStride] += alpha * s1
            c += PANEL_NARROW_GROUP
        }
        while (c < columns) {
            val t = alpha * coefficients[coefficientOffset + c * coefficientStride]
            val at = aOffset + c * lda
            var sum = 0.0
            for (i in 0 until rows) {
                val v = a[at + i]
                y[yOffset + i * yStride] += t * v
                sum += v * x[xOffset + i * xStride]
            }
            sums[sumOffset + c * sumStride] += alpha * sum
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
        while (c + PANEL_NARROW_GROUP <= columns) {
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
            c += PANEL_NARROW_GROUP
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

    override fun rankTwoUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        rows: Int,
        columns: Int,
        u: DoubleArray,
        uOffset: Int,
        uStride: Int,
        v: DoubleArray,
        vOffset: Int,
        vStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        for (c in 0 until columns) {
            val s = alpha * u[uOffset + c * uStride]
            val t = alpha * v[vOffset + c * vStride]
            val at = aOffset + c * lda
            var fromX = xOffset
            var fromY = yOffset
            for (i in 0 until rows) {
                a[at + i] += s * x[fromX] + t * y[fromY]
                fromX += xStride
                fromY += yStride
            }
        }
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
        if (rows <= 0 || columns <= 0) return
        var c = 0
        while (c + PANEL_NARROW_GROUP <= columns) {
            val t0 = alpha * values[fromIndex + c]
            val t1 = alpha * values[fromIndex + c + 1]
            val at0 = aOffset + indices[fromIndex + c] * indexStride
            val at1 = aOffset + indices[fromIndex + c + 1] * indexStride
            var from = 0
            for (i in 0 until rows) {
                y[yOffset + i] += t0 * a[at0 + from] + t1 * a[at1 + from]
                from += rowStride
            }
            c += PANEL_NARROW_GROUP
        }
        while (c < columns) {
            val t = alpha * values[fromIndex + c]
            val at = aOffset + indices[fromIndex + c] * indexStride
            var from = 0
            for (i in 0 until rows) {
                y[yOffset + i] += t * a[at + from]
                from += rowStride
            }
            c++
        }
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
        if (rows <= 0 || columns <= 0) return
        var c = 0
        while (c + PANEL_NARROW_GROUP <= columns) {
            val t0 = alpha * values[fromIndex + c]
            val t1 = alpha * values[fromIndex + c + 1]
            val at0 = aOffset + indices[fromIndex + c] * indexStride
            val at1 = aOffset + indices[fromIndex + c + 1] * indexStride
            var to = 0
            for (i in 0 until rows) {
                val v = x[xOffset + i]
                a[at0 + to] += t0 * v
                a[at1 + to] += t1 * v
                to += rowStride
            }
            c += PANEL_NARROW_GROUP
        }
        while (c < columns) {
            val t = alpha * values[fromIndex + c]
            val at = aOffset + indices[fromIndex + c] * indexStride
            var to = 0
            for (i in 0 until rows) {
                a[at + to] += t * x[xOffset + i]
                to += rowStride
            }
            c++
        }
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
        destinationOffset: Int,
        destinationRowStride: Int,
        destinationIndexStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        for (c in 0 until columns) {
            val t = alpha * values[fromIndex + c]
            val at = offset + indices[fromIndex + c] * indexStride
            var position = 0
            if (c == excluded) {
                for (i in 0 until rows) {
                    a[destinationOffset + indices[fromIndex + c] * destinationIndexStride + i * destinationRowStride] +=
                        t * b[pivot + position]
                    position += rowStride
                }
            } else {
                for (i in 0 until rows) {
                    a[destinationOffset + indices[fromIndex + c] * destinationIndexStride + i * destinationRowStride] +=
                        t * b[pivot + position]
                    sums[sumOffset + i * destinationRowStride] += t * b[at + position]
                    position += rowStride
                }
            }
        }
    }

    /** Columns grouped where one walk of a shared vector serves all of them. */
    private const val PANEL_WIDE_GROUP = 4

    /** Columns grouped where the body itself is that wide. */
    private const val PANEL_NARROW_GROUP = 2

    /**
     * Right-hand sides a sparse column walk serves at once.
     *
     * Wider than the dense groups because nothing here is held in registers: the group sizes a scratch the
     * caller owns, and what it buys is that a column's indices and values are read once for that many
     * right-hand sides. A sweep of the widths on one machine put the turn at this one, with the narrower
     * groups paying for the extra walks and the wider ones losing the destination's locality.
     */
    private const val PANEL_SPARSE_GROUP = 8

    /**
     * Right-hand sides a reduction over a strided block serves at once, which is one.
     *
     * A reduction carries an accumulator per right-hand side. One of them is a value this body keeps in a
     * register for the whole walk of a column; a group of them is the caller's array, read and written per
     * stored entry. Where the block is strided there is nothing else for the group to save, and the sweep
     * behind this number measured every wider group behind it. Where it is adjacent the accumulators are a
     * vector and the group is worth having again, which is why this answer is only the strided one.
     */
    private const val PANEL_SPARSE_REDUCTION_GROUP = 1

    override val tileRows: Int get() = PORTABLE_PRODUCT_TILE

    override val tileColumns: Int get() = PORTABLE_PRODUCT_TILE

    override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean =
        packsProductByWork(rows, columns, depth, tileRows, tileColumns)

    override fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        scaleProductWindow(beta, c, cOffset, ldc, rows, columns)
        if (depth <= 0) return
        var column = 0
        while (column < columns) {
            val presentColumns = if (PORTABLE_PRODUCT_TILE <
                columns - column
            ) {
                PORTABLE_PRODUCT_TILE
            } else {
                columns - column
            }
            val bPanel = bOffset + column / PORTABLE_PRODUCT_TILE * bGroupStride
            var row = 0
            while (row < rows) {
                val presentRows = if (PORTABLE_PRODUCT_TILE < rows - row) PORTABLE_PRODUCT_TILE else rows - row
                val aPanel = aOffset + row / PORTABLE_PRODUCT_TILE * aGroupStride
                productTile(
                    depth, packedA, aPanel, packedB, bPanel, alpha,
                    c, cOffset + row + column * ldc, ldc, presentRows, presentColumns,
                )
                row += PORTABLE_PRODUCT_TILE
            }
            column += PORTABLE_PRODUCT_TILE
        }
    }

    /**
     * One four by four productTile, accumulated over the whole depth and written back once.
     *
     * Padding lanes of a short edge accumulate zeros, because a packed group pads with positive zero, and
     * are then not stored: [presentRows] and [presentColumns] are what reaches the destination.
     */
    private fun productTile(
        depth: Int,
        a: DoubleArray,
        aOffset: Int,
        b: DoubleArray,
        bOffset: Int,
        alpha: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
        presentRows: Int,
        presentColumns: Int,
    ) {
        var c00 = 0.0
        var c10 = 0.0
        var c20 = 0.0
        var c30 = 0.0
        var c01 = 0.0
        var c11 = 0.0
        var c21 = 0.0
        var c31 = 0.0
        var c02 = 0.0
        var c12 = 0.0
        var c22 = 0.0
        var c32 = 0.0
        var c03 = 0.0
        var c13 = 0.0
        var c23 = 0.0
        var c33 = 0.0
        var ap = aOffset
        var bp = bOffset
        var step = 0
        while (step < depth) {
            val a0 = a[ap]
            val a1 = a[ap + 1]
            val a2 = a[ap + 2]
            val a3 = a[ap + 3]
            var coefficient = b[bp]
            c00 += a0 * coefficient
            c10 += a1 * coefficient
            c20 += a2 * coefficient
            c30 += a3 * coefficient
            coefficient = b[bp + 1]
            c01 += a0 * coefficient
            c11 += a1 * coefficient
            c21 += a2 * coefficient
            c31 += a3 * coefficient
            coefficient = b[bp + 2]
            c02 += a0 * coefficient
            c12 += a1 * coefficient
            c22 += a2 * coefficient
            c32 += a3 * coefficient
            coefficient = b[bp + 3]
            c03 += a0 * coefficient
            c13 += a1 * coefficient
            c23 += a2 * coefficient
            c33 += a3 * coefficient
            ap += PORTABLE_PRODUCT_TILE
            bp += PORTABLE_PRODUCT_TILE
            step++
        }
        productStoreColumn(c, cOffset, presentRows, alpha, c00, c10, c20, c30)
        if (presentColumns > 1) productStoreColumn(c, cOffset + ldc, presentRows, alpha, c01, c11, c21, c31)
        if (presentColumns > 2) productStoreColumn(c, cOffset + 2 * ldc, presentRows, alpha, c02, c12, c22, c32)
        if (presentColumns > 3) productStoreColumn(c, cOffset + 3 * ldc, presentRows, alpha, c03, c13, c23, c33)
    }

    private fun productStoreColumn(
        c: DoubleArray,
        at: Int,
        presentRows: Int,
        alpha: Double,
        v0: Double,
        v1: Double,
        v2: Double,
        v3: Double,
    ) {
        c[at] += alpha * v0
        if (presentRows > 1) c[at + 1] += alpha * v1
        if (presentRows > 2) c[at + 2] += alpha * v2
        if (presentRows > 3) c[at + 3] += alpha * v3
    }

    override fun rightHandSideGroup(order: Int, sides: Int): Int = if (sides > 0) sides else 1

    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean = false

    override fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
    ) {
        if (size <= 0 || sides <= 0) return
        if (sides == 1) {
            singleSolve(t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, tOffset)
            return
        }
        for (r in 0 until sides) {
            val column = xOffset + r * rhsStride
            for (step in 0 until size) {
                // Substitute in dependency order so each dependent value is already solved.
                val p = if (lower) step else size - 1 - step
                var value = x[column + p * orderStride]
                val inner = if (lower) 0 until p else p + 1 until size
                for (q in inner) {
                    value -= triangleEntry(t, n, start, p, q, transposed, tOffset) * x[column + q * orderStride]
                }
                if (!unitDiag) value /= triangleEntry(t, n, start, p, p, transposed, tOffset)
                x[column + p * orderStride] = value
            }
        }
    }

    override fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
    ) {
        if (size <= 0 || sides <= 0) return
        if (sides == 1) {
            singleMultiply(t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, tOffset)
            return
        }
        for (r in 0 until sides) {
            val column = xOffset + r * rhsStride
            for (step in 0 until size) {
                // Multiply in reverse dependency order to preserve input values until their last use.
                val p = if (lower) size - 1 - step else step
                val own = x[column + p * orderStride]
                var value = if (unitDiag) own else triangleEntry(t, n, start, p, p, transposed, tOffset) * own
                val inner = if (lower) 0 until p else p + 1 until size
                for (q in inner) {
                    value += triangleEntry(t, n, start, p, q, transposed, tOffset) * x[column + q * orderStride]
                }
                x[column + p * orderStride] = value
            }
        }
    }

    /**
     * One right-hand side, walked in whichever direction the block is contiguous in. Untransposed, a column
     * of the block is adjacent in memory, so each finished step is removed from the steps after it; transposed,
     * a row is, so each step reduces the steps before it. The general loop reads every coefficient through
     * [triangleEntry] in dot order, which strides across columns on an untransposed block.
     */
    internal fun singleSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        stride: Int,
        tOffset: Int,
    ) {
        val base = tOffset + start + start * n
        // M(p, q) is at base + p * rowStep + q * columnStep.
        val rowStep = if (transposed) n else 1
        val columnStep = if (transposed) 1 else n
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            if (transposed) {
                var value = x[xOffset + p * stride]
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                for (q in from until until) value -= t[base + p * rowStep + q * columnStep] * x[xOffset + q * stride]
                if (!unitDiag) value /= t[base + p * (rowStep + columnStep)]
                x[xOffset + p * stride] = value
            } else {
                var value = x[xOffset + p * stride]
                if (!unitDiag) value /= t[base + p * (rowStep + columnStep)]
                x[xOffset + p * stride] = value
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                val column = base + p * columnStep
                for (q in from until until) x[xOffset + q * stride] -= t[column + q * rowStep] * value
            }
        }
    }

    /** The multiply counterpart of [singleSolve], in the order that keeps each input until its last use. */
    internal fun singleMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        stride: Int,
        tOffset: Int,
    ) {
        val base = tOffset + start + start * n
        val rowStep = if (transposed) n else 1
        val columnStep = if (transposed) 1 else n
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val own = x[xOffset + p * stride]
            if (transposed) {
                var value = if (unitDiag) own else t[base + p * (rowStep + columnStep)] * own
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                for (q in from until until) value += t[base + p * rowStep + q * columnStep] * x[xOffset + q * stride]
                x[xOffset + p * stride] = value
            } else {
                x[xOffset + p * stride] = if (unitDiag) own else t[base + p * (rowStep + columnStep)] * own
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                val column = base + p * columnStep
                for (q in from until until) x[xOffset + q * stride] += t[column + q * rowStep] * own
            }
        }
    }
}
