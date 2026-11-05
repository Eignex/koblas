@file:Suppress("LongParameterList") // BLAS operands, flags and reusable scratch.

package com.eignex.koblas.internal.operations.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.internal.engine.routedWholeCall
import com.eignex.koblas.internal.operations.gerWindow
import com.eignex.koblas.internal.operations.prepareMatrixVector
import com.eignex.koblas.internal.operations.prepareRankUpdate
import com.eignex.koblas.internal.operations.prepareTriangularVector
import com.eignex.koblas.internal.operations.requireSquare
import com.eignex.koblas.internal.operations.updateRank
import com.eignex.koblas.internal.scheduling.dense.scheduleGemv
import com.eignex.koblas.internal.scheduling.dense.scheduleSymv
import com.eignex.koblas.internal.scheduling.dense.scheduleSyr
import com.eignex.koblas.internal.scheduling.dense.scheduleSyr2
import com.eignex.koblas.internal.scheduling.dense.scheduleTrmv
import com.eignex.koblas.internal.scheduling.dense.scheduleTrsv

internal fun KoblasEngine.executeGemv(
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
    scratch: Scratch?,
) {
    val operation = if (transpose) EngineImplementation.GemvTransposed else EngineImplementation.Gemv
    val routed = routedWholeCall(operation, a.rows, a.cols, 1, a !is DenseMatrix)
    if (routed !=
        null
    ) {
        return routed.gemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch)
    }
    prepareMatrixVector(this, alpha, a, transpose, x, xOffset, xStride, beta, y, yOffset, yStride, null, scratch) {
            sa,
            sx,
            so,
            ss,
        ->
        scheduleGemv(alpha, sa, sx, so, ss, beta, y, yOffset, yStride, transpose)
    }
}

internal fun KoblasEngine.executeSymv(
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
    scratch: Scratch?,
) {
    val routed = routedWholeCall(EngineImplementation.Symv, a.rows, a.cols, 1, a !is DenseMatrix)
    if (routed != null) return routed.symv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch)
    requireSquare(a, "symv")
    prepareMatrixVector(this, alpha, a, false, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch) {
            sa,
            sx,
            so,
            ss,
        ->
        scheduleSymv(alpha, sa, sx, so, ss, beta, y, yOffset, yStride, lower)
    }
}

internal fun KoblasEngine.executeGer(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    scratch: Scratch?,
) {
    val routed = routedWholeCall(EngineImplementation.Ger, a.rows, a.cols, 1, a !is DenseMatrix)
    if (routed != null) return routed.ger(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch)
    updateRank(alpha, x, xOffset, xStride, y, yOffset, yStride, a, 0, 0, a.rows, a.cols, scratch)
}

internal fun KoblasEngine.executeSyr(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    a: DenseMatrixRegion,
    lower: Boolean,
    scratch: Scratch?,
) {
    val routed = routedWholeCall(EngineImplementation.Syr, a.rows, a.rows, 1, a !is DenseMatrix)
    if (routed != null) return routed.syr(alpha, x, xOffset, xStride, a, lower, scratch)
    requireSquare(a, "syr")
    prepareRankUpdate(alpha, x, xOffset, xStride, x, xOffset, xStride, a, scratch) { sx, xo, xs, _, _, _ ->
        scheduleSyr(alpha, sx, xo, xs, a, lower)
    }
}

internal fun KoblasEngine.executeSyr2(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    lower: Boolean,
    scratch: Scratch?,
) {
    val routed = routedWholeCall(EngineImplementation.Syr2, a.rows, a.rows, 1, a !is DenseMatrix)
    if (routed != null) return routed.syr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch)
    requireSquare(a, "syr2")
    prepareRankUpdate(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch) {
            sx,
            xo,
            xs,
            sy,
            yo,
            ys,
        ->
        scheduleSyr2(alpha, sx, xo, xs, sy, yo, ys, a, lower)
    }
}

internal fun KoblasEngine.executeTrsv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    scratch: Scratch?,
) {
    val operation = if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv
    val routed = routedWholeCall(operation, a.rows, a.rows, 1, a !is DenseMatrix)
    if (routed != null) return routed.trsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
    prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", scratch) { sa ->
        scheduleTrsv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
    }
}

internal fun KoblasEngine.executeTrmv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    scratch: Scratch?,
) {
    val operation = if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv
    val routed = routedWholeCall(operation, a.rows, a.rows, 1, a !is DenseMatrix)
    if (routed != null) return routed.trmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
    prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", scratch) { sa ->
        scheduleTrmv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
    }
}

internal fun KoblasEngine.executeGer(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrixRegion,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    xOffset: Int,
    yOffset: Int,
    scratch: Scratch?,
) {
    gerWindow(alpha, x, y, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset, scratch)
}
