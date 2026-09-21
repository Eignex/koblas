package com.eignex.koblas

/* Alias snapshots use workspace loans; callers decide which operands need staging.
 * Sparse snapshots share read-only structure and copy only values.
 */

/** [block] over [values], or over a copy of it when [aliased]. */
internal inline fun <T> staged(
    workspace: Workspace?,
    values: DoubleArray,
    aliased: Boolean,
    block: (DoubleArray) -> T,
): T {
    if (!aliased) return block(values)
    return workspace.borrow(values.size) { copy ->
        values.copyInto(copy)
        block(copy)
    }
}

/** [staged] for a dense matrix operand, whose shape the copy keeps. */
internal inline fun <T> staged(workspace: Workspace?, a: DenseMatrix, aliased: Boolean, block: (DenseMatrix) -> T): T {
    if (!aliased) return block(a)
    return workspace.borrow(a.values.size) { copy ->
        a.values.copyInto(copy)
        block(DenseMatrix.wrap(a.rows, a.cols, copy))
    }
}

/** Copies only the logical vector entries, so a short strided view does not stage its entire backing array. */
internal inline fun <T> staged(workspace: Workspace?, x: DenseVector, aliased: Boolean, block: (DenseVector) -> T): T {
    if (!aliased) return block(x)
    return workspace.borrow(x.size) { copy ->
        x.gatherInto(copy)
        block(ContiguousVector(copy))
    }
}

/**
 * [staged] for a sparse operand, where only the coefficients can alias.
 *
 * The structural arrays are never a destination, so the snapshot shares them and copies the coefficients
 * alone.
 */
@OptIn(UnsafeKoblasApi::class)
internal inline fun <T> staged(
    workspace: Workspace?,
    a: SparseMatrix,
    aliased: Boolean,
    block: (SparseMatrix) -> T,
): T {
    if (!aliased) return block(a)
    return workspace.borrow(a.nnz) { copy ->
        a.values.copyInto(copy)
        block(SparseMatrix.wrapTrusted(a.rows, a.cols, a.colPointers, a.rowIndices, copy))
    }
}

/** This vector's entries in logical order at the front of [destination], which holds at least [Vector.size]. */
internal fun DenseVector.gatherInto(destination: DoubleArray) {
    if (stride == 1) {
        values.copyInto(destination, 0, offset, offset + size)
        return
    }
    for (i in 0 until size) destination[i] = values[offset + i * stride]
}

/**
 * Adapts a non-sparse operand into dense storage borrowed for [block]. Dense operands pass through unchanged.
 * Both extents are retained even when empty; sparse operands must be dispatched before reaching this helper.
 */
internal inline fun <T> denseOperand(workspace: Workspace?, matrix: Matrix, block: (DenseMatrix) -> T): T {
    if (matrix is DenseMatrix) return block(matrix)
    val rows = matrix.rows
    val cols = matrix.cols
    requireNonNegativeShape(rows, cols)
    val entries = rows.toLong() * cols
    requireShape(entries <= Int.MAX_VALUE) {
        "operand ${rows}x$cols needs $entries entries, more than one array can hold"
    }
    return workspace.borrow(entries.toInt()) { values ->
        for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) values[base + i] = matrix[i, j]
        }
        block(DenseMatrix.wrap(rows, cols, values))
    }
}

/**
 * [block] over this vector's entries as one contiguous array, which a seam taking a plain array needs.
 *
 * A vector spanning its own array in order is passed through; a window or a step is gathered into a
 * workspace loan for the call rather than into a fresh array, so a repeated call over a strided operand
 * allocates nothing. The gather also snapshots, so an operand reached this way needs no staging of its own
 * against a destination it overlaps.
 */
internal inline fun <T> contiguous(workspace: Workspace?, x: DenseVector, block: (DoubleArray) -> T): T {
    if (x.isWholeArray) return block(x.values)
    return workspace.borrow(x.size) { gathered ->
        x.gatherInto(gathered)
        block(gathered)
    }
}
