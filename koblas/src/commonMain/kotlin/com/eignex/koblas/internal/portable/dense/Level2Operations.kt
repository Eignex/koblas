@file:Suppress("LongParameterList") // Validated BLAS operands and portable matrix schedules.

package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*

internal fun KoblasEngine.portableGemv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    transpose: Boolean,
) {
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
        applyBeta(this, y, yOffset, if (transpose) a.cols else a.rows, beta, yStride)
        return
    }
    val panels = panelProvider
    val rows = a.rows
    if (transpose) {
        forEachPanel(
            a.cols,
            panels.executionGroup(PanelWork.MultiDot, rows, a.cols, xStride == 1),
        ) { start, width ->
            panels.multiDot(
                alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                xOffset, xStride, rows, width, beta, y,
                yOffset + start * yStride, yStride,
            )
        }
    } else {
        applyBeta(this, y, yOffset, rows, beta, yStride)
        forEachPanel(
            a.cols,
            panels.executionGroup(PanelWork.ColumnUpdate, rows, a.cols, yStride == 1),
        ) { start, width ->
            panels.columnUpdate(
                alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                xOffset + start * xStride, xStride, rows, width, y, yOffset, yStride,
            )
        }
    }
}

@Suppress("LongParameterList") // the operands, their snapshots and the BLAS dsymv scalars
internal fun KoblasEngine.portableSymv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    lower: Boolean,
) {
    if (alpha == 0.0 || a.rows == 0) {
        applyBeta(this, y, yOffset, a.rows, beta, yStride)
        return
    }
    val panels = panelProvider
    val n = a.rows
    applyBeta(this, y, yOffset, a.rows, beta, yStride)
    val group = panels.executionGroup(PanelWork.CoupledDotUpdate, n, n, xStride == 1 && yStride == 1)
    forEachTrianglePanel(n, group, lower, fromDiagonal = false) { start, width, window, rows ->
        symmetricCorner(alpha, a, x, xOffset, xStride, y, yOffset, yStride, start, width, lower)
        if (rows > 0) {
            panels.coupledUpdateDot(
                alpha, a.values, a.offset + window + start * a.leadingDimension, a.leadingDimension, x,
                xOffset + window * xStride, rows, width, y, yOffset + window * yStride, x,
                xOffset + start * xStride, y, yOffset + start * yStride, xStride, yStride,
                xStride, yStride,
            )
        }
    }
}

/**
 * The square block where a group's own columns and rows meet, which the shared window cannot reach.
 *
 * Every stored position in it contributes twice, once down its column and once along the mirrored row,
 * except the diagonal which is its own mirror. The arithmetic is the panel's, written out: a coefficient
 * scaled by alpha multiplies down, and the reduction is scaled by alpha on its way into the destination.
 */
private fun symmetricCorner(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    start: Int,
    width: Int,
    lower: Boolean,
) {
    val lda = a.leadingDimension
    for (c in start until start + width) {
        val coefficient = alpha * x[xOffset + c * xStride]
        val diagonal = a.values[a.offset + c + c * lda]
        var mirrored = 0.0
        // Only the stored positions of this column, so no test per entry; the mirrored row sums once.
        val first = if (lower) c + 1 else start
        val last = if (lower) start + width else c
        for (i in first until last) {
            val value = a.values[a.offset + i + c * lda]
            y[yOffset + i * yStride] += coefficient * value
            mirrored += value * x[xOffset + i * xStride]
        }
        y[yOffset + c * yStride] += coefficient * diagonal + alpha * mirrored
    }
}

internal fun KoblasEngine.portableGer(
    alpha: Double,
    sx: DoubleArray,
    sxOffset: Int,
    sxStride: Int,
    sy: DoubleArray,
    syOffset: Int,
    syStride: Int,
    a: DenseMatrixRegion,
    aOffset: Int,
    rows: Int,
    columns: Int,
) {
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    val panels = panelProvider
    forEachPanel(
        columns,
        panels.executionGroup(PanelWork.RankUpdate, rows, columns, sxStride == 1),
    ) { start, width ->
        panels.rankUpdate(
            alpha, a.values, aOffset + start * a.leadingDimension, a.leadingDimension,
            sx, sxOffset, sxStride, rows, width,
            sy, syOffset + start * syStride, syStride,
        )
    }
}

internal fun KoblasEngine.portableSyr2(
    alpha: Double,
    xv: DoubleArray,
    xOrigin: Int,
    xStep: Int,
    yv: DoubleArray,
    yOrigin: Int,
    yStep: Int,
    a: DenseMatrixRegion,
    lower: Boolean,
) {
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    val panels = panelProvider
    val n = a.rows
    val lda = a.leadingDimension
    val group = panels.executionGroup(PanelWork.RankUpdate, n, n, xStep == 1 && yStep == 1)
    forEachTrianglePanel(n, group, lower, fromDiagonal = true) { start, width, window, rows ->
        // Only the diagonal corner has differing stored rows; the remaining window is shared by the group.
        val cornerFirst = if (lower) start else start + 1
        val cornerLast = if (lower) start + width - 1 else start + width
        for (c in start until start + width) {
            val xc = xv[xOrigin + c * xStep]
            val yc = yv[yOrigin + c * yStep]
            for (i in cornerFirst until cornerLast) {
                if (if (lower) i < c else i > c) continue
                a.values[a.offset + i + c * lda] +=
                    alpha * (xv[xOrigin + i * xStep] * yc + yv[yOrigin + i * yStep] * xc)
            }
        }
        panels.rankTwoUpdate(
            alpha, a.values, a.offset + window + start * lda, lda,
            xv, xOrigin + window * xStep, xStep, yv, yOrigin + window * yStep, yStep, rows, width,
            yv, yOrigin + start * yStep, yStep, xv, xOrigin + start * xStep, xStep,
        )
    }
}

/**
 * Solve `op(T) · x = b` in place, one stored column at a time.
 *
 * Untransposed, a finished entry is divided by its diagonal and then removed from every entry that
 * depends on it, which is one column update. Transposed, an entry is finished by reducing the column
 * against the entries already done, which is one multi-dot. Either way the dependency order is the
 * substitution's and cannot be grouped away; what a panel covers is the independent arithmetic within
 * one step.
 */
internal fun KoblasEngine.portableTrsv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val panels = panelProvider
    val triangles = triangularProvider
    val n = a.rows
    val lda = a.leadingDimension
    val ascending = lower != transpose
    forEachTriangularBlock(n, ascending) { start, end ->
        // The rest of the stored triangle beside this block: below it when lower, above it when upper.
        val outside = if (lower) end else 0
        val outsideRows = if (lower) n - end else start
        val block = a.offset + outside + start * lda
        if (transpose && outsideRows > 0) {
            panels.multiDot(
                -1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                1.0, x, xOffset + start * xStride, xStride,
            )
        }
        triangles.diagonalSolve(
            a.values, lda, start, end - start, transpose, lower != transpose, unitDiag,
            x, xOffset + start * xStride, xStride, 1, 1, a.offset,
        )
        if (!transpose && outsideRows > 0) {
            panels.columnUpdate(
                -1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                x, xOffset + outside * xStride, xStride,
            )
        }
    }
}

internal fun KoblasEngine.portableTrmv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val panels = panelProvider
    val triangles = triangularProvider
    val n = a.rows
    val lda = a.leadingDimension
    // Traverse in opposite directions for transpose variants to preserve values until their last use.
    val ascending = transpose == lower
    forEachTriangularBlock(n, ascending) { start, end ->
        // The rest of the stored triangle beside this block: below it when lower, above it when upper.
        val outside = if (lower) end else 0
        val outsideRows = if (lower) n - end else start
        val block = a.offset + outside + start * lda
        if (!transpose && outsideRows > 0) {
            panels.columnUpdate(
                1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                x, xOffset + outside * xStride, xStride,
            )
        }
        triangles.diagonalMultiply(
            a.values, lda, start, end - start, transpose, lower != transpose, unitDiag,
            x, xOffset + start * xStride, xStride, 1, 1, a.offset,
        )
        if (transpose && outsideRows > 0) {
            panels.multiDot(
                1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                1.0, x, xOffset + start * xStride, xStride,
            )
        }
    }
}

internal fun KoblasEngine.portableSyr(
    alpha: Double,
    xv: DoubleArray,
    origin: Int,
    step: Int,
    a: DenseMatrixRegion,
    lower: Boolean,
) {
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    val n = a.rows
    val panels = panelProvider
    val group = panels.executionGroup(PanelWork.RankUpdate, n, n, step == 1)
    forEachTrianglePanel(n, group, lower, fromDiagonal = true) { start, width, window, rows ->
        // Only the diagonal corner has differing stored rows; the remaining window is shared by the group.
        val cornerFirst = if (lower) start else start + 1
        val cornerLast = if (lower) start + width - 1 else start + width
        for (c in start until start + width) {
            val coefficient = alpha * xv[origin + c * step]
            for (i in cornerFirst until cornerLast) {
                if (if (lower) i < c else i > c) continue
                a.values[a.offset + i + c * a.leadingDimension] += coefficient * xv[origin + i * step]
            }
        }
        panels.rankUpdate(
            alpha, a.values, a.offset + window + start * a.leadingDimension, a.leadingDimension, xv,
            origin + window * step, step, rows, width,
            xv, origin + start * step, step,
        )
    }
}
