package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.overlaps
import com.eignex.koblas.internal.storage.sameVectorMapping

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
