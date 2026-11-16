@file:Suppress("LongParameterList") // BLAS operands, flags and scratch.

@file:OptIn(UnsafeKoblasApi::class)

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.scheduling.sparse.*

/** Sparse `trsm` when [solve] and `trmm` otherwise, from either side. */
internal fun KoblasEngine.sparseTriangularMatrix(
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
    arrays: Scratch?,
) {
    requireTriangularMatrixOperands(a, b, right, if (solve) "trsm" else "trmm")
    if (alpha == 0.0) return scaleRegion(b, 0.0)
    val n = a.rows
    if (n == 0 || (if (right) b.rows else b.cols) == 0) return
    val backend = kernels
    // Snapshot an aliased triangle before alpha scales the right-hand sides and changes its coefficients.
    staged(arrays, a.values, a.values === b.values) { triangle ->
        scaleRegion(b.values, b.offset, b.rows, b.cols, b.leadingDimension, alpha, vectors = backend)
        withExplicitDiagonal(triangle, a.colPointers, a.rowIndices, n, unitDiag, arrays) { diagonal ->
            if (right) {
                triangularRightCore(
                    backend,
                    solve,
                    triangle,
                    a.colPointers,
                    a.rowIndices,
                    a.rows,
                    b.values,
                    b.offset,
                    b.rows,
                    b.leadingDimension,
                    lower,
                    solve == transpose,
                    diagonal,
                )
            } else {
                triangularLeftCore(
                    backend,
                    solve,
                    triangle,
                    a.colPointers,
                    a.rowIndices,
                    a.rows,
                    a.cols,
                    b.values,
                    b.offset,
                    b.cols,
                    b.leadingDimension,
                    lower,
                    transpose,
                    diagonal,
                    arrays,
                )
            }
        }
    }
}
