@file:Suppress("LongParameterList", "TooManyFunctions") // one routine per BLAS entry, with its operands

package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.triangle
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*

/*
 * The dense routines as the shared schedule runs them, on whichever engine they are called on: its own panel,
 * product and substitution kernels do the arithmetic. These are the default bodies of the routines on
 * [KoblasEngine], so the portable engine and a composed one run the same code over different kernels.
 */

internal fun KoblasEngine.scheduledGemv(
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
    arrays: ArraySupplier?,
): Unit = prepareGemv(this, alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, arrays) {
        sa,
        sx,
        sxOffset,
        sxStride,
    ->
    portableGemv(alpha, sa, sx, sxOffset, sxStride, beta, y, yOffset, yStride, transpose)
}

internal fun KoblasEngine.scheduledSymv(
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
    arrays: ArraySupplier?,
): Unit = prepareSymv(this, alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, arrays) {
        sa,
        sx,
        sxOffset,
        sxStride,
    ->
    portableSymv(alpha, sa, sx, sxOffset, sxStride, beta, y, yOffset, yStride, lower)
}

internal fun KoblasEngine.scheduledGer(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    arrays: ArraySupplier?,
): Unit = prepareGer(alpha, x, xOffset, xStride, y, yOffset, yStride, a, arrays) { sx, sxo, sxs, sy, syo, sys ->
    portableGer(alpha, sx, sxo, sxs, sy, syo, sys, a, a.offset, a.rows, a.cols)
}

internal fun KoblasEngine.scheduledSyr(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    a: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareSyr(alpha, x, xOffset, xStride, a, arrays) { sx, sxo, sxs ->
    portableSyr(alpha, sx, sxo, sxs, a, lower)
}

internal fun KoblasEngine.scheduledSyr2(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareSyr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, arrays) { sx, sxo, sxs, sy, syo, sys ->
    portableSyr2(alpha, sx, sxo, sxs, sy, syo, sys, a, lower)
}

internal fun KoblasEngine.scheduledTrsv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", arrays) { sa ->
    portableTrsv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
}

internal fun KoblasEngine.scheduledTrmv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", arrays) { sa ->
    portableTrmv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
}

internal fun KoblasEngine.scheduledGemm(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    arrays: ArraySupplier?,
): Unit = prepareGemm(alpha, a, transposeA, b, transposeB, beta, c, arrays) { sa, sb, depth ->
    portableGemm(alpha, sa, transposeA, sb, transposeB, beta, c, depth, arrays)
}

internal fun KoblasEngine.scheduledGemmt(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareGemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, arrays) { sa, sb ->
    portableGemmt(alpha, sa, transposeA, sb, transposeB, beta, c, lower, arrays)
}

internal fun KoblasEngine.scheduledSymm(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    right: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareSymm(this, alpha, a, b, beta, c, lower, right, arrays) { sa, sb ->
    portableSymm(alpha, sa, sb, beta, c, lower, right, arrays)
}

internal fun KoblasEngine.scheduledSyrk(
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareSyrk(alpha, a, transpose, beta, c, lower, arrays) { sa ->
    portableSyrk(alpha, sa, transpose, beta, c, lower, arrays)
}

/**
 * A rank-2k update is the sum of the two products its definition names, and both are ordinary windows of the
 * shared product schedule, so each is packed, blocked and tiled like any other product. A fused traversal would
 * need a second implementation of all of that to beat them together; the local evidence compares the two.
 */
internal fun KoblasEngine.scheduledSyr2k(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareSyr2k(alpha, a, b, transpose, beta, c, lower, arrays) { sa, sb ->
    portableSyr2k(alpha, sa, sb, transpose, beta, c, lower, arrays)
}

internal fun KoblasEngine.scheduledTriangularMatrix(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
    arrays: ArraySupplier?,
): Unit = prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, if (solve) "trsm" else "trmm", arrays) { sa ->
    portableTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, solve, arrays)
}
