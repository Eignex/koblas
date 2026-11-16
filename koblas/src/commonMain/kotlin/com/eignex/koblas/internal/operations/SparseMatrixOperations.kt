@file:Suppress("LongParameterList") // BLAS operands, flags and optional scratch.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.kernels
import com.eignex.koblas.internal.scheduling.sparse.*

/** Sparse symmetric products snapshot aliased values before scaling the destination. */
internal fun KoblasEngine.multiplySymmetricMatrices(
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    right: Boolean,
    scratch: Scratch?,
) {
    requireSymmOperands(a, b, c, right)
    // Return before staging or loans: an empty axis may be paired with an enormous one.
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
    val backend = kernels
    staged(scratch, a.values, a.values === c.values) { stableA ->
        stagedRegion(scratch, b, c) { bv, bo, bld ->
            scaleRegion(c.values, c.offset, c.rows, c.cols, c.leadingDimension, beta, vectors = backend)
            if (right) {
                multiplySymmetricFromTheRight(
                    backend,
                    alpha,
                    stableA,
                    a.colPointers,
                    a.rowIndices,
                    a.cols,
                    bv,
                    bo,
                    b.rows,
                    bld,
                    c.values,
                    c.offset,
                    c.leadingDimension,
                    lower,
                )
            } else {
                multiplySymmetricFromTheLeft(
                    backend,
                    alpha,
                    stableA,
                    a.colPointers,
                    a.rowIndices,
                    a.rows,
                    a.cols,
                    bv,
                    bo,
                    b.cols,
                    bld,
                    c.values,
                    c.offset,
                    c.leadingDimension,
                    lower,
                    scratch,
                )
            }
        }
    }
}

/** Sparse products validate the chosen side and snapshot aliases before destination scaling. */
internal fun KoblasEngine.multiplySparseMatrix(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    right: Boolean,
    scratch: Scratch?,
) {
    // Right-side multiplication uses the same derivation with operands reversed.
    if (right) {
        requireGemmOperands(b, transposeB, a, transposeA, c)
    } else {
        requireGemmOperands(a, transposeA, b, transposeB, c)
    }
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
    val depth = if (right) (if (transposeB) b.rows else b.cols) else (if (transposeA) a.rows else a.cols)
    if (depth == 0) return scaleRegion(c, beta, vectors = this)
    val backend = kernels
    staged(scratch, a.values, a.values === c.values) { stableA ->
        stagedRegion(scratch, b, c) { bv, bo, bld ->
            scaleRegion(c.values, c.offset, c.rows, c.cols, c.leadingDimension, beta, vectors = backend)
            if (right) {
                multiplyFromTheRight(
                    backend,
                    alpha,
                    stableA,
                    a.colPointers,
                    a.rowIndices,
                    a.cols,
                    transposeA,
                    bv,
                    bo,
                    b.rows,
                    b.cols,
                    bld,
                    transposeB,
                    c.values,
                    c.offset,
                    c.leadingDimension,
                    c.rows,
                    scratch,
                )
            } else {
                multiplyFromTheLeft(
                    backend,
                    alpha,
                    stableA,
                    a.colPointers,
                    a.rowIndices,
                    a.cols,
                    transposeA,
                    bv,
                    bo,
                    bld,
                    transposeB,
                    c.values,
                    c.offset,
                    c.leadingDimension,
                    c.rows,
                    c.cols,
                    depth,
                    scratch,
                )
            }
        }
    }
}

/** Sparse rank-k updates preserve stored support while staging destination aliases. */
internal fun KoblasEngine.updateSymmetricRankK(
    alpha: Double,
    a: SparseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val n = if (transpose) a.cols else a.rows
    requireSyrkOperands(a, transpose, c)
    // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
    if (alpha == 0.0 || a.nnz == 0 || n == 0) return scaleRegion(c, beta, lower)
    staged(scratch, a.values, a.values === c.values) { stableA ->
        scaleRegion(c, beta, lower)
        withSymmetricRankScratch(scratch, n, a.rows, a.nnz) {
                sums,
                touchedAt,
                touched,
                rowPointers,
                adjacentColumns,
                adjacentPositions,
                rowCursor,
            ->
            symmetricRankInto(
                alpha,
                stableA,
                a.colPointers,
                a.rowIndices,
                a.rows,
                a.cols,
                transpose,
                c.values,
                c.offset,
                c.rows,
                c.leadingDimension,
                lower,
                sums,
                touchedAt,
                touched,
                rowPointers,
                adjacentColumns,
                adjacentPositions,
                rowCursor,
            )
        }
    }
}
