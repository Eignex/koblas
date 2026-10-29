@file:Suppress("LongParameterList") // the matrix product signature
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DENSE_SCHEDULING
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseMatrixRoute
import com.eignex.koblas.dense.scaleComponent
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixOperation
import com.eignex.koblas.sparse.SparseMatrixRoute
import com.eignex.koblas.RouteKind

/**
 * Describes a [Matrix.gemmInto] product from its operands, without reading values or preparing storage.
 * Execution and inspection both use this receiver engine, including prepared operands.
 */
public fun KoblasEngine.routeOf(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
): MatrixRoute {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    return productRoute(
        alpha, a, transposeA, b, transposeB, beta, c.rows, c.cols, DenseMatrix.entryCount(c.rows, c.cols),
        aliased = sharesBuffer(a, c) || sharesBuffer(b, c),
    )
}

/**
 * Describes [Matrix.gemm] without allocating its result. Two sparse operands, prepared or ordinary,
 * produce CSC storage; other pairings produce dense storage. No cached orientation is constructed.
 */
public fun KoblasEngine.routeOf(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
): MatrixRoute {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    val rows = if (transposeA) a.cols else a.rows
    val columns = if (transposeB) b.rows else b.cols
    val sparsePair = storedCsc(a) != null && storedCsc(b) != null
    return productRoute(
        alpha, a, transposeA, b, transposeB, 0.0, rows, columns,
        elements = if (sparsePair) null else DenseMatrix.entryCount(rows, columns), aliased = false,
    )
}

private fun KoblasEngine.productRoute(
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
): MatrixRoute {
    if (alpha == 0.0 && elements != null) {
        val scaling = if (elements > 0 && beta != 0.0 && beta != 1.0) {
            scaleComponent(vectorKernels, elements)
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
    return dispatchProduct(a, b,
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

        sparseLeft = { left -> routeOf(
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
            ),
        ) },

        sparseRight = { right -> routeOf(
            SparseMatrixOperation.GemmDenseRight,
            SparseCall(
                right, alpha, beta, destinationElements = elements, depth = depth, updateRun = rows,
                rightHandSides = rows, transposeSparse = transposeB, transposeDense = transposeA,
            ),
        ) },

        densePair = { routeOf(
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
                borrowed = a is DenseMatrixView || b is DenseMatrixView,
                rightLeadingDimension = (b as? DenseMatrixRegion)?.leadingDimension,
            ),
        ) },
    )
}

private fun sharesBuffer(matrix: Matrix, destination: DenseMatrixRegion): Boolean = when (matrix) {
    is DenseMatrixRegion -> com.eignex.koblas.dense.regionsOverlap(matrix, destination)
    else -> storedCsc(matrix)?.values === destination.values
}
