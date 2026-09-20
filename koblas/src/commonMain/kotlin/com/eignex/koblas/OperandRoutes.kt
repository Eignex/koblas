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
import com.eignex.koblas.vendor.RouteKind

/**
 * Describes a [Matrix.gemmInto] product from its operands, without reading values or preparing storage.
 * Prepared operands retain their bound engine; other products are described using this engine.
 * The generic zero-alpha shortcut uses the default engine's scaling kernels.
 */
public fun KoblasEngine.routeOf(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
): MatrixRoute {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    return productRoute(
        alpha, a, transposeA, b, transposeB, beta, c.rows, c.cols, c.values.size,
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
            scaleComponent(koblas.vectorKernels, elements)
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
            "the generic product only scales its destination",
        )
    }
    val left = storedCsc(a)
    val right = storedCsc(b)
    val depth = if (transposeA) a.rows else a.cols
    val engine = if (a is PreparedSparseMatrix || b is PreparedSparseMatrix) sparseEngine(a, b) else this
    return when {
        left != null && right != null -> {
            val operation = if (elements == null) {
                SparseMatrixOperation.GemmSparse
            } else {
                SparseMatrixOperation.GemmSparseDense
            }
            val route = engine.routeOf(
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
            val orientation = if (reachesAPosition(alpha, left, right) &&
                ((a is PreparedSparseMatrix && transposeA) || (b is PreparedSparseMatrix && transposeB))
            ) {
                "a prepared transpose is derived on first use and traversed untransposed"
            } else {
                "no prepared orientation is derived"
            }
            // Sparse-sparse products perform their own arithmetic, so no per-column SIMD leaf depends on
            // the derived pattern. The schedule can be named without building that pattern to inspect it.
            SparseMatrixRoute(
                route.operation, if (empty) RouteKind.NoWork else route.kind, route.scheduling,
                route.entryPoint, route.components, "$orientation; ${route.reason.orEmpty()}",
                route.executionGroup, route.executionTail, route.resolved,
            )
        }

        left != null -> engine.routeOf(
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
        )

        right != null -> engine.routeOf(
            SparseMatrixOperation.GemmDenseRight,
            SparseCall(
                right, alpha, beta, destinationElements = elements, depth = depth, updateRun = rows,
                rightHandSides = rows, transposeSparse = transposeB, transposeDense = transposeA,
            ),
        )

        else -> routeOf(
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
            ),
        )
    }
}

private fun sharesBuffer(matrix: Matrix, destination: DenseMatrix): Boolean =
    matrix is DenseMatrix && matrix.values === destination.values
