@file:Suppress("LongParameterList") // BLAS operands, flags and workspace

package com.eignex.koblas.dense

import com.eignex.koblas.*

/** Packs only logical entries, borrowing scratch rather than retaining a parent-sized snapshot. */
internal inline fun <T> packedRegion(
    workspace: Workspace?,
    a: DenseMatrixRegion,
    read: Boolean = true,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrix) -> T,
): T {
    if (a is DenseMatrix) return block(a)
    return workspace.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { copy ->
        if (read) {
            for (j in 0 until a.cols) {
                for (i in 0 until a.rows) {
                    if (lower == null || (if (lower) i >= j else i <= j)) {
                        copy[i + j * a.rows] = if (unitDiag && i == j) 1.0 else a[i, j]
                    }
                }
            }
        }
        block(DenseMatrix.wrap(a.rows, a.cols, copy))
    }
}

/** Snapshots a region only when it shares the destination buffer. */
internal inline fun <T> stagedRegion(
    workspace: Workspace?,
    a: DenseMatrixRegion,
    destination: DoubleArray,
    stable: DenseMatrixRegion? = null,
    original: DenseMatrixRegion? = null,
    block: (DenseMatrixRegion) -> T,
): T {
    if (a.values !== destination) return block(a)
    if (original != null && stable != null && a.values === original.values && a.offset == original.offset &&
        a.rows == original.rows && a.cols == original.cols && a.leadingDimension == original.leadingDimension
    ) {
        return block(stable)
    }
    return workspace.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { copy ->
        for (j in 0 until a.cols) for (i in 0 until a.rows) copy[i + j * a.rows] = a[i, j]
        block(DenseMatrix.wrap(a.rows, a.cols, copy))
    }
}

/** Whether two logical rectangles touch, excluding the gaps between their columns. */
internal fun regionsOverlap(a: DenseMatrixRegion, b: DenseMatrixRegion): Boolean {
    if (a.values !== b.values || a.rows == 0 || a.cols == 0 || b.rows == 0 || b.cols == 0) return false
    var left = 0
    var right = 0
    while (left < a.cols && right < b.cols) {
        val aStart = a.offset + left * a.leadingDimension
        val bStart = b.offset + right * b.leadingDimension
        if (aStart < bStart + b.rows && bStart < aStart + a.rows) return true
        if (aStart < bStart) left++ else right++
    }
    return false
}

/** Disjoint submatrices of the same parent need no snapshot. */
internal inline fun <T> stagedRegion(
    workspace: Workspace?,
    a: DenseMatrixRegion,
    destination: DenseMatrixRegion,
    stable: DenseMatrixRegion? = null,
    original: DenseMatrixRegion? = null,
    block: (DenseMatrixRegion) -> T,
): T {
    if (!regionsOverlap(a, destination)) return block(a)
    return stagedRegion(workspace, a, destination.values, stable, original, block)
}

/** Scales selected entries, never reading a zero-scaled destination or visiting its padding. */
internal fun scaleRegion(a: DenseMatrixRegion, beta: Double, lower: Boolean? = null) {
    if (beta == 1.0) return
    for (j in 0 until a.cols) {
        val first = if (lower == true) j else 0
        val last = if (lower == false) j + 1 else a.rows
        for (i in first until last) a[i, j] = if (beta == 0.0) 0.0 else beta * a[i, j]
    }
}

/** Only the destination's selected entries are copied back after a packed backend call. */
private fun copyRegion(source: DenseMatrix, destination: DenseMatrixRegion, lower: Boolean? = null) {
    if (source === destination) return
    for (j in 0 until destination.cols) {
        for (i in 0 until destination.rows) {
            if (lower == null || (if (lower) i >= j else i <= j)) destination[i, j] = source[i, j]
        }
    }
}

internal fun regionGemv(
    blas: DenseBlas,
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean,
    workspace: Workspace?,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    if (y.isEmpty()) return
    if (alpha == 0.0 || (if (transpose) a.rows else a.cols) == 0) {
        if (beta != 1.0) for (i in y.indices) y[i] = if (beta == 0.0) 0.0 else beta * y[i]
        return
    }
    packedRegion(workspace, a) { sa -> blas.gemv(alpha, sa, x, beta, y, transpose, workspace) }
}

internal fun regionGemm(
    blas: DenseBlas,
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    workspace: Workspace?,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta)
        return
    }
    packedRegion(workspace, a) { sa ->
        packedRegion(workspace, b) { sb ->
            packedRegion(workspace, c, read = beta != 0.0) { sc ->
                blas.gemm(alpha, sa, transposeA, sb, transposeB, beta, sc, workspace)
                copyRegion(sc, c)
            }
        }
    }
}

internal fun regionSyrk(
    blas: DenseBlas,
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    workspace: Workspace?,
) {
    requireSyrkOperands(a, transpose, c)
    if (c.rows == 0) return
    if (alpha == 0.0 || (if (transpose) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    packedRegion(workspace, a) { sa ->
        packedRegion(workspace, c, read = beta != 0.0, lower = lower) { sc ->
            blas.syrk(alpha, sa, transpose, beta, sc, lower, workspace)
            copyRegion(sc, c, lower)
        }
    }
}

internal fun regionTrsv(
    blas: DenseBlas,
    a: DenseMatrixRegion,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    workspace: Workspace?,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    if (x.isEmpty()) return
    packedRegion(workspace, a, lower = lower, unitDiag = unitDiag) { sa ->
        blas.trsv(sa, x, lower, transpose, unitDiag, workspace)
    }
}

internal fun regionTrsm(
    blas: DenseBlas,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    workspace: Workspace?,
) {
    requireTriangularMatrixOperands(a, b, right, "trsm")
    if (b.rows == 0 || b.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    packedRegion(workspace, a, lower = lower, unitDiag = unitDiag) { sa ->
        packedRegion(workspace, b) { sb ->
            blas.trsm(sa, sb, lower, transpose, unitDiag, right, alpha, workspace)
            copyRegion(sb, b)
        }
    }
}
