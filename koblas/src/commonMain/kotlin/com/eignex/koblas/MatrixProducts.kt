@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.sparse.SparseBlas

/**
 * Allocates `this * other` through the common [Matrix] contract.
 *
 * Result storage follows the operands' runtime storage, not their static types: two sparse operands, prepared
 * or not, give a CSC [SparseMatrix] whose structure is what the two patterns meet at, and every other
 * built-in pairing gives a [DenseMatrix]. A caller wanting uniform dense storage from two sparse operands
 * uses [gemmInto] against a dense destination rather than converting a sparse result afterwards.
 *
 * No operand is densified to reach a kernel: a sparse operand keeps its structural traversal whichever side
 * it is on. An implementation of [Matrix] from outside this library is read through [Matrix.get] and staged
 * once into dense column-major storage, and a sparse operand beside it still keeps its own traversal.
 */
public fun Matrix.gemm(other: Matrix): Matrix = gemm(1.0, false, other, false)

/**
 * Allocates `alpha · op(this) · op(other)`, with the storage rule and the traversals [gemm] describes.
 *
 * The scaled and transposed form of the same product, for the operands a caller holds rather than the ones
 * a signature names. A [PreparedSparseMatrix] on either side contributes its snapshot, and a transposed one
 * against a second sparse operand contributes the opposite orientation it derives once instead.
 */
public fun Matrix.gemm(alpha: Double, transpose: Boolean, other: Matrix, transposeOther: Boolean): Matrix {
    requireProductOperands(this, transpose, other, transposeOther, "gemm")
    val left = storedCsc(this)
    val right = storedCsc(other)
    if (left != null && right != null) {
        val reached = reachesAPosition(alpha, left, right)
        val leftOriented = orientationFor(this, transpose, reached)
        val rightOriented = orientationFor(other, transposeOther, reached)
        return sparseEngine(this, other).gemm(
            alpha,
            leftOriented ?: left,
            transpose && leftOriented == null,
            rightOriented ?: right,
            transposeOther && rightOriented == null,
        )
    }
    val result = DenseMatrix.zero(
        if (transpose) cols else rows,
        if (transposeOther) other.rows else other.cols,
    )
    gemmInto(alpha, transpose, other, transposeOther, 0.0, result)
    return result
}

/** Common allocating matrix product; the runtime storage pair determines the result storage. */
@kotlin.jvm.JvmName("multiplyMatrices")
public operator fun Matrix.times(other: Matrix): Matrix = gemm(other)

/**
 * Computes `destination = alpha * op(this) * op(other) + beta * destination`.
 *
 * Every dense, sparse and prepared pairing reaches the implementation its storage calls for, with independent
 * transpose flags on both operands and without an intermediate sparse result. Shapes and flags are validated
 * before anything is written, and a zero [alpha] scales [destination] without reading either operand, so a
 * matrix whose entries are expensive or absent is never touched for a product that contributes nothing. A
 * zero [beta] overwrites [destination] without reading it.
 *
 * Built-in operands may share [destination]'s backing array and are snapshotted before it is written.
 * [workspace] reuses alias snapshots, sparse scratch and temporary dense storage for custom operands.
 */
public fun Matrix.gemmInto(
    alpha: Double,
    transpose: Boolean,
    other: Matrix,
    transposeOther: Boolean,
    beta: Double,
    destination: DenseMatrix,
    workspace: Workspace? = null,
) {
    requireGemmOperands(this, transpose, other, transposeOther, destination, "gemmInto")
    // Before either operand is looked at, so a zero multiplier reads nothing at all. A custom Matrix can
    // compute its entries, and staging one to discover that alpha contributes nothing would be a call the
    // contract says does not happen.
    if (alpha == 0.0) {
        applyBeta(koblas.vectorKernels, destination.values, 0, destination.values.size, beta)
        return
    }
    val left = storedCsc(this)
    val right = storedCsc(other)
    when {
        left != null && right != null -> {
            val reached = reachesAPosition(alpha, left, right)
            val leftOriented = orientationFor(this, transpose, reached)
            val rightOriented = orientationFor(other, transposeOther, reached)
            sparseEngine(this, other).gemm(
                alpha,
                leftOriented ?: left,
                transpose && leftOriented == null,
                rightOriented ?: right,
                transposeOther && rightOriented == null,
                beta,
                destination,
                workspace,
            )
        }

        left != null -> denseOperand(workspace, other) { dense ->
            sparseEngine(this, other).gemm(
                alpha, left, transpose, dense, transposeOther, beta, destination,
                right = false, workspace = workspace,
            )
        }

        right != null -> denseOperand(workspace, this) { dense ->
            sparseEngine(this, other).gemm(
                alpha, right, transposeOther, dense, transpose, beta, destination,
                right = true, workspace = workspace,
            )
        }

        else -> denseOperand(workspace, this) { denseLeft ->
            denseOperand(workspace, other) { denseRight ->
                koblas.gemm(alpha, denseLeft, transpose, denseRight, transposeOther, beta, destination, workspace)
            }
        }
    }
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

/**
 * The sparse implementation a product runs on, which is the engine that prepared an operand where one is.
 *
 * A snapshot built by an exact engine keeps that engine's Level 1 selection when it is multiplied through
 * the common surface, so naming an engine to prepare with is not quietly undone by the product. Two prepared
 * operands from different engines run on the left one's, and a product with none runs on the default engine.
 */
internal fun sparseEngine(left: Matrix, right: Matrix): SparseBlas =
    (left as? PreparedSparseMatrix ?: right as? PreparedSparseMatrix)?.blas ?: koblas
