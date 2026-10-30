package com.eignex.koblas

/** Reads a custom triangle once, excluding the unselected half and an implicit diagonal. */
internal inline fun <T> triangleOperand(
    workspace: Workspace?,
    a: Matrix,
    lower: Boolean,
    unitDiag: Boolean,
    block: (DenseMatrixRegion) -> T,
): T {
    if (a is DenseMatrixRegion) return block(a)
    return workspace.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { values ->
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) {
                if (if (lower) i >= j else i <= j) values[i + j * a.rows] = if (unitDiag && i == j) 1.0 else a[i, j]
            }
        }
        block(DenseMatrix.wrap(a.rows, a.cols, values))
    }
}
