@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.operands.denseOperand
import com.eignex.koblas.internal.operands.scaleRegion

/** Allocates through [koblas], preserving CSC storage for a pair of sparse operands. */
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
 * Call [KoblasEngine.gemm] to select an explicit engine.
 */
public fun Matrix.gemmInto(
    alpha: Double,
    transpose: Boolean,
    other: Matrix,
    transposeOther: Boolean,
    beta: Double,
    destination: DenseMatrixRegion,
    workspace: Workspace? = null,
): Unit = koblas.gemm(alpha, this, transpose, other, transposeOther, beta, destination, workspace)

/** Storage classification; callbacks inline on the arithmetic path. */
internal inline fun <T> dispatchProduct(
    a: Matrix,
    b: Matrix,
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
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
): Matrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    return dispatchProduct(
        a,
        b,
        sparsePair = { left, right -> gemm(alpha, left, transposeA, right, transposeB) },
        sparseLeft = { denseProduct(alpha, a, transposeA, b, transposeB) },
        sparseRight = { denseProduct(alpha, a, transposeA, b, transposeB) },
        densePair = { denseProduct(alpha, a, transposeA, b, transposeB) },
    )
}

private fun KoblasEngine.denseProduct(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
): DenseMatrix {
    val result = DenseMatrix.zero(if (transposeA) a.cols else a.rows, if (transposeB) b.rows else b.cols)
    productInto(alpha, a, transposeA, b, transposeB, 0.0, result, null)
    return result
}

/** Canonical generic destination dispatch. Sparse structure never passes through custom materialization. */
internal fun KoblasEngine.productInto(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    workspace: Workspace?,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, vectors = vectorKernels)
        return
    }
    dispatchProduct(
        a,
        b,
        sparsePair = { left, right -> gemm(alpha, left, transposeA, right, transposeB, beta, c, workspace) },
        sparseLeft = { left ->
            denseOperand(workspace, b) { right ->
                gemm(alpha, left, transposeA, right, transposeB, beta, c, right = false, workspace = workspace)
            }
        },
        sparseRight = { right ->
            denseOperand(workspace, a) { left ->
                gemm(alpha, right, transposeB, left, transposeA, beta, c, right = true, workspace = workspace)
            }
        },
        densePair = {
            denseOperand(workspace, a) { left ->
                if (a === b) {
                    gemm(alpha, left, transposeA, left, transposeB, beta, c, workspace)
                } else {
                    denseOperand(
                        workspace,
                        b,
                    ) { right -> gemm(alpha, left, transposeA, right, transposeB, beta, c, workspace) }
                }
            }
        },
    )
}

/** The CSC an operand traverses, or null when it has none. */
internal fun storedCsc(matrix: Matrix): SparseMatrix? = matrix as? SparseMatrix
