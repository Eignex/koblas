@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare
import com.eignex.koblas.requireTriangularMatrixShape
import kotlin.math.max
import kotlin.math.min

/*
 * The portable triangular kernels, netlib dtrsv, dtrsm, dtrmv and dtrmm over a flat column-major buffer.
 * These are the semantic definition a native triangular routine is validated against, and what
 * The scalar engine and the host adapters' fallbacks call these. `Triangular.kt` is the public facade that routes
 * through the installed context instead.
 */

/** Diagonal block width for the blocked triangular routines, bounded by the mask that indexes it. */
internal val TRIANGULAR_BLOCK: Int = DenseTuning.triangularBlock

/**
 * Fails when [TRIANGULAR_BLOCK] outgrows the zero-pivot mask that indexes it.
 *
 * A wrapped shift would not throw or read out of bounds. It would quietly retain the products of one row
 * against the mask bit of another, which only shows up as a wrong answer on a matrix whose quotient
 * underflows, so the check has to be explicit rather than left to a test to notice.
 */
internal fun requireTriangularBlockFitsMask() {
    require(TRIANGULAR_BLOCK <= Long.SIZE_BITS) {
        "TRIANGULAR_BLOCK is $TRIANGULAR_BLOCK, above the ${Long.SIZE_BITS} rows a " +
            "Long zero-pivot mask can index"
    }
}

/**
 * The body [Blas.trsv] and [Blas.trmv] share. The two BLAS routines take the same arguments and differ only
 * in which core runs, which [solve] selects.
 *
 * The selection is a flag rather than a passed-in core so the call stays direct.
 */
@Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
internal fun triangularVector(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    a: DenseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
) {
    val what = if (solve) "trsv" else "trmv"
    requireSquare(a, what)
    requireShape(x.size == a.rows) { "$what: x length ${x.size} != ${a.rows}" }
    if (solve) {
        triangularSolveSubstitution(
            vectorKernels,
            panelKernels,
            a.data,
            a.rows,
            x,
            lower = lower,
            transpose = transpose,
            unitDiag = unitDiag,
        )
    } else {
        triangularMultiplySubstitution(
            vectorKernels,
            panelKernels,
            a.data,
            a.rows,
            x,
            lower = lower,
            transpose = transpose,
            unitDiag = unitDiag,
        )
    }
}

/**
 * The body [Blas.trsm] and [Blas.trmm] share, with [solve] selecting the core as in [triangularVector].
 *
 * Matrix operations solve or multiply diagonal blocks with the vector cores and send off-diagonal updates
 * through shared blocked level-3 kernels. A transposed triangle is read in its original storage orientation.
 */
@Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
internal fun triangularMatrix(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    packedKernels: PackedKernels,
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
    workspace: Workspace? = null,
) {
    val what = if (solve) "trsm" else "trmm"
    requireTriangularMatrixShape(a, b, right, what)
    if (alpha == 0.0) {
        b.data.fill(0.0)
        return
    }
    if (alpha != 1.0) vectorKernels.scale(b.data, 0, alpha, b.data.size)
    if (a.rows == 0) return
    val normalizedRows = if (right) b.rows else b.cols
    if (
        !solve &&
        a.rows >= DenseTuning.trmmPackedMinOrder &&
        normalizedRows >= DenseTuning.trmmPackedMinRows &&
        packedTrmmSupports(a, b, lower, unitDiag)
    ) {
        packedTrmmCore(packedKernels, a, b, lower, transpose, unitDiag, right, workspace)
        return
    }
    if (
        solve &&
        a.rows >= DenseTuning.trsmPackedMinOrder &&
        normalizedRows >= DenseTuning.trsmPackedMinRows &&
        packedTrsmSupports(a, b, lower, unitDiag)
    ) {
        packedTrsmCore(packedKernels, a, b, lower, transpose, unitDiag, right, workspace)
        return
    }
    /*
     * The four blocked walks, chosen in one place. [scratch] is the row buffer a right-side walk reads
     * through, the transposed update's sums buffer otherwise, and nothing at all where the left,
     * non-transposed walk needs none; the borrow below is what decides which of those it is.
     */
    fun dispatch(scratch: DoubleArray?) {
        when {
            right && solve ->
                blockedRightSolve(
                    vectorKernels, panelKernels, a.data, a.rows, b, lower, transpose, unitDiag,
                    requireNotNull(scratch),
                )

            right ->
                blockedRightMultiply(
                    vectorKernels, panelKernels, a.data, a.rows, b, lower, transpose, unitDiag,
                    requireNotNull(scratch),
                )

            solve ->
                blockedLeftSolve(
                    vectorKernels, panelKernels, a.data, a.rows, b, lower, transpose, unitDiag, scratch,
                )

            else ->
                blockedLeftMultiply(
                    vectorKernels, panelKernels, a.data, a.rows, b, lower, transpose, unitDiag, scratch,
                )
        }
    }
    when {
        right -> workspace.borrow(a.rows) { row -> dispatch(row) }
        transpose -> workspace.borrow(4) { sums -> dispatch(sums) }
        else -> dispatch(null)
    }
}

private fun blockedLeftSolve(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    triangle: DoubleArray,
    n: Int,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    sums: DoubleArray?,
) {
    val bd = b.data
    val nrhs = b.cols
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) 0 else n
    while (if (effectiveLower) boundary < n else boundary > 0) {
        val start = if (effectiveLower) boundary else max(0, boundary - TRIANGULAR_BLOCK)
        val end = if (effectiveLower) min(boundary + TRIANGULAR_BLOCK, n) else boundary
        val size = end - start
        var zeroCoefficientMasks: LongArray? = null
        for (column in 0 until nrhs) {
            val mask = triangularSolveSubstitution(
                vectorKernels, panelKernels, triangle, size, bd, start + start * n, column * n + start, n,
                lower, transpose, unitDiag,
            )
            if (mask != 0L) {
                val masks = zeroCoefficientMasks ?: LongArray(nrhs).also { zeroCoefficientMasks = it }
                masks[column] = mask
            }
        }
        if (effectiveLower && end < n) {
            blockedLeftTriangularUpdate(
                vectorKernels, panelKernels, -1.0, triangle, n, transpose, bd, end, n - end, start, size, nrhs,
                zeroCoefficientMasks, sums,
            )
        } else if (!effectiveLower && start > 0) {
            blockedLeftTriangularUpdate(
                vectorKernels, panelKernels, -1.0, triangle, n, transpose, bd, 0, start, start, size, nrhs,
                zeroCoefficientMasks, sums,
            )
        }
        boundary = if (effectiveLower) end else start
    }
}

@Suppress("LongParameterList")
private fun blockedLeftTriangularUpdate(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    alpha: Double,
    triangle: DoubleArray,
    n: Int,
    transpose: Boolean,
    b: DoubleArray,
    rowStart: Int,
    rowCount: Int,
    innerStart: Int,
    innerCount: Int,
    columns: Int,
    zeroCoefficientMasks: LongArray? = null,
    sums: DoubleArray?,
) {
    if (transpose) {
        blockedTransposedLeftUpdate(
            panelKernels, vectorKernels, alpha, triangle, innerStart + rowStart * n, n,
            b, innerStart, n, b, rowStart, n, rowCount, columns, innerCount,
            requireNotNull(sums),
        )
    } else {
        blockedUpdate(
            panelKernels, alpha, triangle, rowStart + innerStart * n, n,
            b, innerStart, n, b, rowStart, n, rowCount, columns, innerCount,
            zeroCoefficientMasks = zeroCoefficientMasks,
        )
    }
}

private fun blockedRightSolve(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    triangle: DoubleArray,
    n: Int,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    row: DoubleArray,
) {
    val rows = b.rows
    val bd = b.data
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) n else 0
    while (if (effectiveLower) boundary > 0 else boundary < n) {
        val start = if (effectiveLower) max(0, boundary - TRIANGULAR_BLOCK) else boundary
        val end = if (effectiveLower) boundary else min(boundary + TRIANGULAR_BLOCK, n)
        val size = end - start
        val guardZeros = triangleHasZero(triangle, start + start * n, size, n, lower)
        forEachRow(size, b, row, start) { row ->
            triangularSolveSubstitution(
                vectorKernels, panelKernels, triangle, size, row, start + start * n, 0, n, lower, !transpose, unitDiag,
                guardZeroPivot = false,
                guardZeroMatrix = guardZeros,
            )
        }
        if (effectiveLower && start > 0) {
            blockedRightTriangularUpdate(
                panelKernels, -1.0, bd, rows, triangle, n, transpose, start, size, 0, start,
            )
        } else if (!effectiveLower && end < n) {
            blockedRightTriangularUpdate(
                panelKernels, -1.0, bd, rows, triangle, n, transpose, start, size, end, n - end,
            )
        }
        boundary = if (effectiveLower) start else end
    }
}

private fun blockedLeftMultiply(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    triangle: DoubleArray,
    n: Int,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    sums: DoubleArray?,
) {
    val bd = b.data
    val nrhs = b.cols
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) n else 0
    while (if (effectiveLower) boundary > 0 else boundary < n) {
        val start = if (effectiveLower) max(0, boundary - TRIANGULAR_BLOCK) else boundary
        val end = if (effectiveLower) boundary else min(boundary + TRIANGULAR_BLOCK, n)
        val size = end - start
        for (column in 0 until nrhs) {
            triangularMultiplySubstitution(
                vectorKernels, panelKernels, triangle, size, bd, start + start * n, column * n + start, n,
                lower, transpose, unitDiag,
            )
        }
        if (effectiveLower && start > 0) {
            blockedLeftTriangularUpdate(
                vectorKernels, panelKernels, 1.0, triangle, n, transpose, bd,
                start, size, 0, start, nrhs, sums = sums,
            )
        } else if (!effectiveLower && end < n) {
            blockedLeftTriangularUpdate(
                vectorKernels, panelKernels, 1.0, triangle, n, transpose, bd,
                start, size, end, n - end, nrhs, sums = sums,
            )
        }
        boundary = if (effectiveLower) start else end
    }
}

private fun blockedRightMultiply(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    triangle: DoubleArray,
    n: Int,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    row: DoubleArray,
) {
    val bd = b.data
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) 0 else n
    while (if (effectiveLower) boundary < n else boundary > 0) {
        val start = if (effectiveLower) boundary else max(0, boundary - TRIANGULAR_BLOCK)
        val end = if (effectiveLower) min(boundary + TRIANGULAR_BLOCK, n) else boundary
        val size = end - start
        val guardZeros = triangleHasZero(triangle, start + start * n, size, n, lower)
        forEachRow(size, b, row, start) { row ->
            triangularMultiplySubstitution(
                vectorKernels, panelKernels, triangle, size, row, start + start * n, 0, n, lower, !transpose, unitDiag,
                guardZeroInput = false,
                guardZeroMatrix = guardZeros,
            )
        }
        if (effectiveLower && end < n) {
            blockedRightTriangularUpdate(
                panelKernels, 1.0, bd, b.rows, triangle, n, transpose, end, n - end, start, size,
            )
        } else if (!effectiveLower && start > 0) {
            blockedRightTriangularUpdate(
                panelKernels, 1.0, bd, b.rows, triangle, n, transpose, 0, start, start, size,
            )
        }
        boundary = if (effectiveLower) end else start
    }
}
