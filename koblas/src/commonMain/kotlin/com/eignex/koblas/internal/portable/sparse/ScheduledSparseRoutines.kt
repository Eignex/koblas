@file:Suppress("LongParameterList", "TooManyFunctions") // one routine per sparse BLAS entry, with its operands

package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.dense.applyBeta
import com.eignex.koblas.internal.sparse.*

/*
 * The sparse routines on [KoblasEngine], which are always the portable CSC traversal over the engine's own
 * kernels. Each checks its operands, settles the calls with nothing to compute, and snapshots coefficients
 * that share the destination's storage before the traversal writes. A structural array is never a
 * destination, so an aliased sparse operand shares it and only its coefficients are copied.
 */

internal fun KoblasEngine.scheduledSparseGemv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean,
    arrays: ArraySupplier?,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    // Zero alpha or zero extent reads neither operand; only destination scaling remains.
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
        applyBeta(this, y, 0, y.size, beta)
        return
    }
    // Snapshot aliased operands before destination scaling overwrites their coefficients.
    staged(arrays, x, x === y) { stableX ->
        staged(arrays, a, a.values === y) { stableA ->
            portableSparseGemv(alpha, stableA, stableX, beta, y, transpose)
        }
    }
}

internal fun KoblasEngine.scheduledSparseSymv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean,
    arrays: ArraySupplier?,
) {
    requireSymvOperands(a, x.size, y.size)
    if (alpha == 0.0) {
        applyBeta(this, y, 0, y.size, beta)
        return
    }
    staged(arrays, x, x === y) { stableX ->
        staged(arrays, a, a.values === y) { stableA ->
            portableSparseSymv(alpha, stableA, stableX, beta, y, lower)
        }
    }
}

internal fun KoblasEngine.scheduledSparseSymm(
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    right: Boolean,
    arrays: ArraySupplier?,
) {
    requireSymmOperands(a, b, c, right)
    // Return before staging or loans: an empty axis may be paired with an enormous one.
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    staged(arrays, a, a.values === c.values) { stableA ->
        stagedRegion(arrays, b, c) { stableB ->
            scaleRegion(c, beta, vectors = this)
            if (right) {
                multiplySymmetricFromTheRight(this, alpha, stableA, stableB, c, lower)
            } else {
                multiplySymmetricFromTheLeft(this, alpha, stableA, stableB, c, lower, arrays)
            }
        }
    }
}

internal fun KoblasEngine.scheduledSparseTrsv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    arrays: ArraySupplier?,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    // Snapshot an aliased triangle before substitution overwrites coefficients needed by later columns.
    staged(arrays, a, a.values === x) { stable ->
        trsvCore(stable, x, lower, transpose, unitDiag)
    }
}

internal fun KoblasEngine.scheduledSparseTrmv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    arrays: ArraySupplier?,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    staged(arrays, a, a.values === x) { stable ->
        trmvCore(stable, x, lower, transpose, unitDiag)
    }
}

internal fun KoblasEngine.scheduledSparseDenseGemm(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    right: Boolean,
    arrays: ArraySupplier?,
) {
    // Right-side multiplication uses the same derivation with operands reversed.
    if (right) {
        requireGemmOperands(b, transposeB, a, transposeA, c)
    } else {
        requireGemmOperands(a, transposeA, b, transposeB, c)
    }
    val m = c.rows
    val n = c.cols
    val k = if (right) {
        if (transposeB) b.rows else b.cols
    } else {
        if (transposeA) a.rows else a.cols
    }
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    staged(arrays, a, a.values === c.values) { stableA ->
        stagedRegion(arrays, b, c) { stableB ->
            scaleRegion(c, beta, vectors = this)
            if (right) {
                multiplyFromTheRight(this, alpha, stableA, transposeA, stableB, transposeB, c, m, arrays)
            } else {
                multiplyFromTheLeft(this, alpha, stableA, transposeA, stableB, transposeB, c, m, n, k, arrays)
            }
        }
    }
}

internal fun scheduledSparseProduct(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: SparseMatrix,
    transposeB: Boolean,
): SparseMatrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    val aRows = if (transposeA) a.cols else a.rows
    val bCols = if (transposeB) b.rows else b.cols
    // Return before orienting: transpose pointers are row-indexed and may exceed array limits
    // even when the product is empty.
    if (a.nnz == 0 || b.nnz == 0 || aRows == 0 || bCols == 0) return emptyResult(aRows, bCols, "gemm")
    val left = oriented(a, transposeA, alpha != 0.0)
    val right = oriented(b, transposeB, alpha != 0.0)
    return multiplySparse(left, right, alpha)
}

internal fun KoblasEngine.scheduledSparseProductInto(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: SparseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    arrays: ArraySupplier?,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (alpha == 0.0 || a.nnz == 0 || b.nnz == 0) {
        // Only destination scaling remains; avoid row-indexed orientation scratch.
        scaleRegion(c, beta, vectors = this)
        return
    }
    staged(arrays, a, a.values === c.values) { stableA ->
        staged(arrays, b, b.values === c.values) { stableB ->
            val left = oriented(stableA, transposeA, true)
            val right = oriented(stableB, transposeB, true)
            scaleRegion(c, beta, vectors = this)
            multiplySparseInto(alpha, left, right, c)
        }
    }
}

internal fun scheduledSparseSyrk(
    alpha: Double,
    a: SparseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
) {
    val n = if (transpose) a.cols else a.rows
    requireSyrkOperands(a, transpose, c)
    if (alpha == 0.0 || a.nnz == 0 || n == 0) {
        // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
        scaleRegion(c, beta, lower)
        return
    }
    staged(arrays, a, a.values === c.values) { stableA ->
        scaleRegion(c, beta, lower)
        withSymmetricRankScratch(arrays, n, stableA.rows, stableA.nnz) {
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

internal fun scheduledSparseAddScaled(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: SparseMatrix,
): SparseMatrix {
    val rows = if (transposeA) a.cols else a.rows
    val cols = if (transposeA) a.rows else a.cols
    requireSameShape(rows, cols, b, "addScaled")
    if (a.nnz == 0 && b.nnz == 0) return emptyResult(rows, cols, "addScaled")
    val left = oriented(a, transposeA, alpha != 0.0)
    // An identical pattern needs neither a structural merge nor room for a second copy of every entry.
    // Different patterns usually disagree at an early row, before every column pointer needs comparing.
    if (left.rowIndices.contentEquals(b.rowIndices) && left.colPointers.contentEquals(b.colPointers)) {
        val values = if (alpha == 0.0) {
            b.values.copyOf()
        } else {
            DoubleArray(left.nnz) { alpha * left.values[it] + b.values[it] }
        }
        return SparseMatrix.wrapTrusted(rows, cols, left.copyColumnPointers(), left.copyRowIndices(), values)
    }
    val union = left.nnz.toLong() + b.nnz
    requireShape(
        union <= Int.MAX_VALUE,
    ) { "scaled addition needs $union stored entries, more than one array can hold" }
    val pointers = IntArray(left.cols + 1)
    val rowIndices = IntArray(union.toInt())
    val values = DoubleArray(rowIndices.size)
    var count = 0
    for (column in 0 until left.cols) {
        count += SparseAccumulationKernels.mergeScaledColumns(
            alpha,
            left.rowIndices,
            left.values,
            left.colPointers[column],
            left.colPointers[column + 1],
            b.rowIndices,
            b.values,
            b.colPointers[column],
            b.colPointers[column + 1],
            rowIndices,
            values,
            count,
        )
        pointers[column + 1] = count
    }
    return SparseMatrix.wrapTrusted(
        left.rows,
        left.cols,
        pointers,
        if (count == rowIndices.size) rowIndices else rowIndices.copyOf(count),
        if (count == values.size) values else values.copyOf(count),
    )
}

internal fun KoblasEngine.scheduledSparseTrsm(
    a: SparseMatrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    arrays: ArraySupplier?,
) {
    requireTriangularMatrixOperands(a, b, right, "trsm")
    val n = a.rows
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    val rightHandSides = if (right) b.rows else b.cols
    if (n == 0 || rightHandSides == 0) return
    // Snapshot an aliased triangle before alpha scales the right-hand sides and changes its coefficients.
    staged(arrays, a, a.values === b.values) { triangle ->
        scaleRegion(b, alpha, vectors = this)
        withExplicitDiagonal(triangle, n, unitDiag, arrays) { diagonal ->
            if (right) {
                trsmRightCore(this, triangle, b, lower, !transpose, diagonal)
            } else {
                triangularLeftCore(this, true, triangle, b, lower, transpose, unitDiag, diagonal, arrays)
            }
        }
    }
}

internal fun KoblasEngine.scheduledSparseTrmm(
    a: SparseMatrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    arrays: ArraySupplier?,
) {
    requireTriangularMatrixOperands(a, b, right, "trmm")
    val n = a.rows
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    val rightHandSides = if (right) b.rows else b.cols
    if (n == 0 || rightHandSides == 0) return
    staged(arrays, a, a.values === b.values) { triangle ->
        scaleRegion(b, alpha, vectors = this)
        // Unlike trsm, trmm reads the diagonal once per right-hand side.
        withExplicitDiagonal(triangle, n, unitDiag, arrays) { diagonal ->
            if (right) {
                trmmRightCore(this, triangle, b, lower, transpose, unitDiag, diagonal)
            } else {
                triangularLeftCore(this, false, triangle, b, lower, transpose, unitDiag, diagonal, arrays)
            }
        }
    }
}

/**
 * A CSC result of this shape with nothing stored, for a call whose operands reach no position. [what]
 * names the operation in the one error this raises, a column count no array of pointers can hold.
 */
private fun emptyResult(rows: Int, cols: Int, what: String): SparseMatrix = SparseMatrix.wrapTrusted(
    rows,
    cols,
    IntArray(pointerLength(cols, what)),
    IntArray(0),
    DoubleArray(0),
)

/**
 * [a] in the requested orientation, materializing a transpose only where one is asked for.
 *
 * A zero alpha discovers the same structure without reading coefficients, so the transposed pattern is
 * built without copying coefficients: the contract says alpha of zero reads no operand value, and a
 * transpose that copied them would break it before the product ever ran.
 */
private fun oriented(a: SparseMatrix, transpose: Boolean, readValues: Boolean): SparseMatrix {
    if (!transpose) return a
    return transposeCsc(a, readValues)
}
