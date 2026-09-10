package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.PortableSparsePanelKernels

/** Adds `alpha · A · x` to [y], mirroring only entries stored in [lower]'s selected triangle. */
internal fun symmetricMultiplyVector(
    kernels: PortableSparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
) {
    for (j in 0 until a.cols) {
        kernels.symmetricVectorColumn(alpha, j, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], x, y, lower)
    }
}

/** Adds a selected-triangle symmetric sparse product on the requested side of dense [b]. */
internal fun symmetricMultiplyMatrix(
    kernels: PortableSparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    lower: Boolean,
    right: Boolean,
) {
    if (right) {
        for (j in 0 until a.cols) {
            kernels.symmetricRightColumn(
                alpha, j, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], b.data, c.data, b.rows, lower,
            )
        }
    } else {
        for (j in 0 until a.cols) {
            kernels.symmetricLeftColumn(
                alpha, j, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], b.data, c.data, b.rows, b.cols, lower,
            )
        }
    }
}
