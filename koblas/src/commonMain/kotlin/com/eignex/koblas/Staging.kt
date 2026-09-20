package com.eignex.koblas

/*
 * Staging an operand that shares a destination's storage.
 *
 * BLAS leaves a destination overlapping an input undefined, and this library answers instead: the operand is
 * copied before the first write and the call reads the values it was given. The copy is a workspace loan for
 * the duration of the call, which is exactly how long it is read for, so a caller repeating one shape reuses
 * the same buffer rather than allocating one each time. Where nothing is aliased there is no loan at all,
 * which is what keeps an ordinary call from touching the workspace.
 *
 * The caller decides what aliasing means for its own operands rather than passing a destination here, because
 * a routine that hands the same matrix over twice stages it once and the second operand then stands for the
 * first. One shape per storage kind, so that a dense schedule, a whole-call binding and a sparse traversal all
 * say the same word for the same thing.
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

/** [staged] for a vector operand, whose origin and spacing the copy keeps. */
internal inline fun <T> staged(workspace: Workspace?, x: DenseVector, aliased: Boolean, block: (DenseVector) -> T): T {
    if (!aliased) return block(x)
    return workspace.borrow(x.values.size) { copy ->
        x.values.copyInto(copy)
        block(StridedVector(copy, x.offset, x.size, x.stride))
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
