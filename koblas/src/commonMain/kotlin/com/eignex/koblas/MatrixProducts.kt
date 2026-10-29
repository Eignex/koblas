@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.scaleRegion

/** Allocates through [koblas], preserving CSC storage for a pair of sparse or prepared operands. */
public fun Matrix.gemm(other: Matrix): Matrix = koblas.gemm(this, other)

/** Allocates a scaled, independently transposed product through the default [koblas] engine. */
public fun Matrix.gemm(alpha: Double, transpose: Boolean, other: Matrix, transposeOther: Boolean): Matrix =
    koblas.gemm(alpha, this, transpose, other, transposeOther)

/** Common allocating product with the runtime storage result rule of [KoblasEngine.gemm]. */
@kotlin.jvm.JvmName("multiplyMatrices")
public operator fun Matrix.times(other: Matrix): Matrix = koblas.gemm(this, other)

/**
 * Writes a generic product through [koblas]. Shapes are validated before mutation, zero alpha reads no
 * input, and zero beta overwrites without reading the destination. Aliased built-in operands are staged
 * over their logical entries. Custom operands are normalized once after no-read quick returns.
 * Call [KoblasEngine.gemm] to select an explicit engine, including for prepared operands.
 */
public fun Matrix.gemmInto(
    alpha: Double, transpose: Boolean, other: Matrix, transposeOther: Boolean,
    beta: Double, destination: DenseMatrixRegion, workspace: Workspace? = null,
): Unit = koblas.gemm(alpha, this, transpose, other, transposeOther, beta, destination, workspace)

/** Storage classification shared by execution and inspection; callbacks inline on the arithmetic path. */
internal inline fun <T> dispatchProduct(
    a: Matrix, b: Matrix,
    sparsePair: (SparseMatrix, SparseMatrix) -> T,
    sparseLeft: (SparseMatrix) -> T,
    sparseRight: (SparseMatrix) -> T,
    densePair: () -> T,
): T {
    val left = storedCsc(a)
    val right = storedCsc(b)
    return when {
        left != null && right != null -> sparsePair(left, right)
        left != null -> sparseLeft(left)
        right != null -> sparseRight(right)
        else -> densePair()
    }
}

/** Engine-owned allocating dispatch, with no policy retained by either operand. */
internal fun KoblasEngine.product(
    alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean,
): Matrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    return dispatchProduct(a, b,
        sparsePair = { left, right ->
            val reached = reachesAPosition(alpha, left, right)
            val la = orientationFor(a, transposeA, reached)
            val rb = orientationFor(b, transposeB, reached)
            gemm(alpha, la ?: left, transposeA && la == null, rb ?: right, transposeB && rb == null)
        },
        sparseLeft = { denseProduct(alpha, a, transposeA, b, transposeB) },
        sparseRight = { denseProduct(alpha, a, transposeA, b, transposeB) },
        densePair = { denseProduct(alpha, a, transposeA, b, transposeB) },
    )
}

private fun KoblasEngine.denseProduct(
    alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean,
): DenseMatrix {
    val result = DenseMatrix.zero(if (transposeA) a.cols else a.rows, if (transposeB) b.rows else b.cols)
    productInto(alpha, a, transposeA, b, transposeB, 0.0, result, null)
    return result
}

/** Canonical generic destination dispatch. Sparse structure never passes through custom materialization. */
internal fun KoblasEngine.productInto(
    alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean,
    beta: Double, c: DenseMatrixRegion, workspace: Workspace?,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, vectors = vectorKernels)
        return
    }
    dispatchProduct(a, b,
        sparsePair = { left, right ->
            val reached = reachesAPosition(alpha, left, right)
            val la = orientationFor(a, transposeA, reached)
            val rb = orientationFor(b, transposeB, reached)
            gemm(alpha, la ?: left, transposeA && la == null, rb ?: right, transposeB && rb == null, beta, c, workspace)
        },
        sparseLeft = { left -> denseOperand(workspace, b) { right ->
            gemm(alpha, left, transposeA, right, transposeB, beta, c, right = false, workspace = workspace)
        } },
        sparseRight = { right -> denseOperand(workspace, a) { left ->
            gemm(alpha, right, transposeB, left, transposeA, beta, c, right = true, workspace = workspace)
        } },
        densePair = { denseOperand(workspace, a) { left ->
            if (a === b) gemm(alpha, left, transposeA, left, transposeB, beta, c, workspace)
            else denseOperand(workspace, b) { right -> gemm(alpha, left, transposeA, right, transposeB, beta, c, workspace) }
        } },
    )
}

/**
 * The CSC an operand traverses, or null when it has none.
 *
 * A [PreparedSparseMatrix] is the snapshot it copied: it takes part in a product as the sparse matrix it
 * holds, and those arrays stay behind this seam rather than being handed to a caller.
 */
internal fun storedCsc(matrix: Matrix): SparseMatrix? = when (matrix) {
    is SparseMatrix -> matrix
    is PreparedSparseMatrix -> matrix.snapshot
    else -> null
}

/**
 * The orientation a prepared operand lends a transposed product, or null when the call keeps its own flag.
 *
 * Deriving one is what preparing buys a product against a second sparse operand: the untransposed schedule
 * over a transpose derived once measured ahead of the transposed traversal there, and behind it for a
 * product against a dense block, which is why only this pairing asks. A call reaching no position orients
 * nothing, because an operand a product never reads must stay unread and a snapshot with more rows than an
 * array can index has no transpose to build.
 */
private fun orientationFor(matrix: Matrix, transpose: Boolean, reached: Boolean): SparseMatrix? =
    if (transpose && reached && matrix is PreparedSparseMatrix) matrix.transposedSnapshot else null

/** Whether a product between two sparse operands evaluates any position, which decides what is read. */
internal fun reachesAPosition(alpha: Double, left: SparseMatrix, right: SparseMatrix): Boolean =
    alpha != 0.0 && left.nnz > 0 && right.nnz > 0

