@file:Suppress("LongParameterList") // BLAS operands, flags and reusable scratch.

package com.eignex.koblas.internal.operations.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.operations.requireGemmOperands
import com.eignex.koblas.internal.operations.requireSymmOperands
import com.eignex.koblas.internal.operations.requireSyrkOperands
import com.eignex.koblas.internal.operations.scaleRegion
import com.eignex.koblas.internal.operations.staged
import com.eignex.koblas.internal.operations.stagedRegion
import com.eignex.koblas.internal.scheduling.sparse.multiplyFromTheLeft
import com.eignex.koblas.internal.scheduling.sparse.multiplyFromTheRight
import com.eignex.koblas.internal.scheduling.sparse.multiplySymmetricFromTheLeft
import com.eignex.koblas.internal.scheduling.sparse.multiplySymmetricFromTheRight
import com.eignex.koblas.internal.scheduling.sparse.sparseTriangularMatrix
import com.eignex.koblas.internal.scheduling.sparse.symmetricRankInto

internal fun KoblasEngine.executeSymm(
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    right: Boolean,
    scratch: Scratch?,
) {
    requireSymmOperands(a, b, c, right)
    // Return before staging or loans: an empty axis may be paired with an enormous one.
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
    staged(scratch, a, a.values === c.values) { stableA ->
        stagedRegion(scratch, b, c) { stableB ->
            scaleRegion(c, beta, vectors = this)
            if (right) {
                multiplySymmetricFromTheRight(this, alpha, stableA, stableB, c, lower)
            } else {
                multiplySymmetricFromTheLeft(this, alpha, stableA, stableB, c, lower, scratch)
            }
        }
    }
}

internal fun KoblasEngine.executeGemm(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    right: Boolean,
    scratch: Scratch?,
) {
    // Right-side multiplication uses the same derivation with operands reversed.
    if (right) {
        requireGemmOperands(
            b,
            transposeB,
            a,
            transposeA,
            c,
        )
    } else {
        requireGemmOperands(a, transposeA, b, transposeB, c)
    }
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
    val depth = if (right) (if (transposeB) b.rows else b.cols) else (if (transposeA) a.rows else a.cols)
    staged(scratch, a, a.values === c.values) { stableA ->
        stagedRegion(scratch, b, c) { stableB ->
            scaleRegion(c, beta, vectors = this)
            if (right) {
                multiplyFromTheRight(this, alpha, stableA, transposeA, stableB, transposeB, c, c.rows, scratch)
            } else {
                multiplyFromTheLeft(
                    this, alpha, stableA, transposeA, stableB, transposeB, c, c.rows, c.cols, depth, scratch,
                )
            }
        }
    }
}

internal fun KoblasEngine.executeSyrk(
    alpha: Double,
    a: SparseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    scratch: Scratch?,
) {
    val n = if (transpose) a.cols else a.rows
    requireSyrkOperands(a, transpose, c)
    // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
    if (alpha == 0.0 || a.nnz == 0 || n == 0) return scaleRegion(c, beta, lower)
    staged(scratch, a, a.values === c.values) { stableA ->
        scaleRegion(c, beta, lower)
        withSymmetricRankScratch(scratch, n, stableA.rows, stableA.nnz) {
                sums,
                touchedAt,
                touched,
                rowPointers,
                adjacentColumns,
                adjacentPositions,
                rowCursor,
            ->
            symmetricRankInto(
                alpha, stableA, transpose, c, lower, sums, touchedAt, touched,
                rowPointers, adjacentColumns, adjacentPositions, rowCursor,
            )
        }
    }
}

internal fun KoblasEngine.executeTrsm(
    a: SparseMatrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    scratch: Scratch?,
): Unit = sparseTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, scratch)

internal fun KoblasEngine.executeTrmm(
    a: SparseMatrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    scratch: Scratch?,
): Unit = sparseTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, scratch)
