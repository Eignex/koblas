@file:Suppress("LongParameterList") // Sparse triangular traversal and BLAS flags.

package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.scalar.sparse.sparseTriangularGatherSingle
import com.eignex.koblas.internal.kernels.scalar.sparse.sparseTriangularScatterSingle

/**
 * Sparse `dtrsv` when [solve] and `dtrmv` otherwise, in place over [x]. The direction preserves each source
 * before its destination is written, so no work buffer is needed for the ordinary in-place case.
 *
 * The diagonal is probed as `A(j, j)`, a binary search per column, and a unit diagonal is taken as 1, which
 * multiplies and divides exactly. That is the vector routines' to pay: the matrix ones walk several
 * right-hand sides against one triangle and precompute the diagonal for themselves instead.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the vector, and the four BLAS triangle flags
internal fun triangularVectorCore(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
) {
    forEachTriangleColumn(a.rows, triangularForward(lower, transpose, solve)) { j ->
        // A zero scattered pivot is skipped before its diagonal is searched for.
        if (!transpose && x[j] == 0.0) return@forEachTriangleColumn
        val diagonal = if (unitDiag) 1.0 else a[j, j]
        val from = a.colPointers[j]
        val to = a.colPointers[j + 1]
        if (transpose) {
            sparseTriangularGatherSingle(solve, j, lower, diagonal, a.rowIndices, a.values, from, to, x, 0, 1)
        } else {
            sparseTriangularScatterSingle(solve, j, lower, diagonal, a.rowIndices, a.values, from, to, x, 0, 1)
        }
    }
}
