package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*

/* Alias snapshots use arrays loans; callers decide which operands need staging.
 * Sparse snapshots share read-only structure and copy only values.
 */

/** [block] over [values], or over a copy of it when [aliased]. */
internal inline fun <T> staged(arrays: Scratch?, values: DoubleArray, aliased: Boolean, block: (DoubleArray) -> T): T {
    if (!aliased) return block(values)
    return arrays.borrow(values.size) { copy ->
        values.copyInto(copy)
        block(copy)
    }
}

/**
 * [staged] for a sparse operand, where only the coefficients can alias.
 *
 * The structural arrays are never a destination, so the snapshot shares them and copies the coefficients
 * alone.
 */
@OptIn(UnsafeKoblasApi::class)
internal inline fun <T> staged(arrays: Scratch?, a: SparseMatrix, aliased: Boolean, block: (SparseMatrix) -> T): T {
    if (!aliased) return block(a)
    return arrays.borrow(a.nnz) { copy ->
        a.values.copyInto(copy)
        block(SparseMatrix.wrapTrusted(a.rows, a.cols, a.colPointers, a.rowIndices, copy))
    }
}

/**
 * Adapts a non-sparse operand into dense storage borrowed for [block]. Dense operands pass through unchanged.
 * Both extents are retained even when empty; sparse operands must be dispatched before reaching this helper.
 */
internal inline fun <T> denseOperand(arrays: Scratch?, matrix: Matrix, block: (DenseMatrixRegion) -> T): T {
    if (matrix is DenseMatrixRegion) return block(matrix)
    val rows = matrix.rows
    val cols = matrix.cols
    return arrays.borrow(DenseMatrix.entryCount(rows, cols)) { values ->
        for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) values[base + i] = matrix[i, j]
        }
        block(DenseMatrix.wrap(rows, cols, values))
    }
}

/** Identical logical vector addressing can share one immutable snapshot. */
internal fun sameVectorMapping(a: DenseVector, b: DenseVector): Boolean =
    a.values === b.values && a.offset == b.offset && a.size == b.size && (a.size <= 1 || a.stride == b.stride)

/** Equal mappings have no cross-index dependency; partial aliases need a snapshot. */
internal fun DenseVector.stableFor(destination: DenseVector): DenseVector =
    if (!sameVectorMapping(this, destination) && overlaps(destination)) DenseVector.wrap(toDoubleArray()) else this

/** Opaque storage may read the destination without exposing its buffer. */
internal fun Vector.stableFor(destination: DenseVector): Vector = when (this) {
    is SparseVector -> if (values === destination.values) SparseVector.wrap(size, indices, values.copyOf()) else this
    is DenseVector -> stableFor(destination)
    else -> DenseVector.wrap(toDoubleArray())
}

/** Whether this vector spans its entire backing array in order. */
internal val DenseVector.isWholeArray: Boolean
    get() = offset == 0 && stride == 1 && values.size == size
