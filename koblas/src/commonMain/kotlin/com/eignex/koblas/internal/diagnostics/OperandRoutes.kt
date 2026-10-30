@file:Suppress("LongParameterList") // generic product facts

package com.eignex.koblas.internal.diagnostics

import com.eignex.koblas.*
import com.eignex.koblas.RouteKind
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseMatrixRoute
import com.eignex.koblas.internal.dense.DENSE_SCHEDULING
import com.eignex.koblas.internal.dense.scaleComponent
import com.eignex.koblas.internal.operands.scalingLength
import com.eignex.koblas.internal.operands.stagedLeadingDimension
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixOperation
import com.eignex.koblas.sparse.SparseMatrixRoute

internal fun KoblasEngine.productRoute(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    rows: Int,
    columns: Int,
    elements: Int?,
    aliased: Boolean,
    destination: DenseMatrixRegion? = null,
): MatrixRoute {
    if (alpha == 0.0 && elements != null) {
        val scaling = if (elements > 0 && beta != 0.0 && beta != 1.0) {
            scaleComponent(
                vectorKernels,
                scalingLength(rows, columns, destination?.leadingDimension ?: rows),
            )
        } else {
            emptyList()
        }
        return DenseMatrixRoute(
            DenseMatrixOperation.Gemm,
            RouteKind.NoWork,
            DENSE_SCHEDULING,
            "gemm",
            scaling,
            0,
        )
    }
    val depth = if (transposeA) a.rows else a.cols
    return dispatchProduct(
        a,
        b,
        sparsePair = { left, right ->
            val operation = if (elements == null) {
                SparseMatrixOperation.GemmSparse
            } else {
                SparseMatrixOperation.GemmSparseDense
            }
            val route = routeOf(
                operation,
                SparseCall(
                    left,
                    alpha,
                    beta,
                    destinationElements = elements,
                    depth = depth,
                    transposeSparse = transposeA,
                    destinationRun = destination?.let { scalingLength(it.rows, it.cols, it.leadingDimension) },
                ),
            )
            val empty = left.nnz == 0 || right.nnz == 0
            // Sparse-sparse arithmetic has no per-column SIMD leaf, so routing needs no derived pattern.
            SparseMatrixRoute(
                route.operation,
                if (empty) RouteKind.NoWork else route.kind,
                route.scheduling,
                route.entryPoint,
                route.components,
                route.executionGroup,
                route.executionTail,
                route.resolved,
            )
        },

        sparseLeft = { left ->
            routeOf(
                SparseMatrixOperation.GemmDense,
                SparseCall(
                    left,
                    alpha,
                    beta,
                    destinationElements = elements,
                    depth = depth,
                    rightHandSides = columns,
                    transposeSparse = transposeA,
                    transposeDense = transposeB,
                    destinationRun = destination?.let { scalingLength(it.rows, it.cols, it.leadingDimension) },
                ),
            )
        },

        sparseRight = { right ->
            routeOf(
                SparseMatrixOperation.GemmDenseRight,
                SparseCall(
                    right, alpha, beta, destinationElements = elements, depth = depth, updateRun = rows,
                    rightHandSides = rows, transposeSparse = transposeB, transposeDense = transposeA,
                    destinationRun = destination?.let { scalingLength(it.rows, it.cols, it.leadingDimension) },
                ),
            )
        },

        densePair = {
            routeOf(
                DenseMatrixOperation.Gemm,
                DenseCall(
                    rows,
                    columns,
                    alpha,
                    beta,
                    depth = depth,
                    transposeA = transposeA,
                    transposeB = transposeB,
                    aliased = aliased,
                    borrowed = a is DenseMatrixView || b is DenseMatrixView || destination is DenseMatrixView,
                    leadingDimension = destination?.leadingDimension,
                    leftLeadingDimension = (a as? DenseMatrixRegion)?.let { stagedLeadingDimension(it, destination) },
                    rightLeadingDimension = (b as? DenseMatrixRegion)?.let { stagedLeadingDimension(it, destination) },
                ),
            )
        },
    )
}

internal fun sharesBuffer(matrix: Matrix, destination: DenseMatrixRegion): Boolean = when (matrix) {
    is DenseMatrixRegion -> com.eignex.koblas.internal.operands.regionsOverlap(matrix, destination)
    else -> storedCsc(matrix)?.values === destination.values
}
