@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.applyBeta

/**
 * Allocates `this * other` through the common [Matrix] contract.
 *
 * Result storage follows the operands' runtime storage, not their static types: two [SparseMatrix] operands
 * give a CSC [SparseMatrix] whose structure is what the two patterns meet at, and every other built-in pairing
 * gives a [DenseMatrix]. A caller wanting uniform dense storage from two sparse operands uses [gemmInto]
 * against a dense destination rather than converting a sparse result afterwards.
 *
 * No operand is densified to reach a kernel: a sparse operand keeps its structural traversal whichever side
 * it is on. An implementation of [Matrix] from outside this library is read through [Matrix.get] and staged
 * once into dense column-major storage, and a sparse operand beside it still keeps its own traversal.
 */
public fun Matrix.gemm(other: Matrix): Matrix {
    requireShape(cols == other.rows) { "gemm: inner dimensions differ, $cols vs ${other.rows}" }
    if (this is SparseMatrix && other is SparseMatrix) return koblas.gemm(this, other)
    val result = DenseMatrix.zero(rows, other.cols)
    gemmInto(1.0, false, other, false, 0.0, result)
    return result
}

/** Common allocating matrix product; the runtime storage pair determines the result storage. */
@kotlin.jvm.JvmName("multiplyMatrices")
public operator fun Matrix.times(other: Matrix): Matrix = gemm(other)

/**
 * Computes `destination = alpha * op(this) * op(other) + beta * destination`.
 *
 * Every dense and sparse pairing reaches the implementation its storage calls for, with independent transpose
 * flags on both operands and without an intermediate sparse result. Shapes and flags are validated before
 * anything is written, and a zero [alpha] scales [destination] without reading either operand, so a matrix
 * whose entries are expensive or absent is never touched for a product that contributes nothing. A zero [beta]
 * overwrites [destination] without reading it.
 *
 * Built-in operands may share [destination]'s backing array and are snapshotted before it is written.
 * [workspace] lends that staging and the sparse scratch, so a repeated call over the same shapes reuses it.
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
    val m = if (transpose) cols else rows
    val depth = if (transpose) rows else cols
    val otherDepth = if (transposeOther) other.cols else other.rows
    val n = if (transposeOther) other.rows else other.cols
    requireShape(depth == otherDepth) { "gemmInto: inner dimensions differ, $depth vs $otherDepth" }
    requireShape(destination.rows == m && destination.cols == n) {
        "gemmInto: destination must be ${m}x$n, got ${destination.rows}x${destination.cols}"
    }
    // Before either operand is looked at, so a zero multiplier reads nothing at all. A custom Matrix can
    // compute its entries, and staging one to discover that alpha contributes nothing would be a call the
    // contract says does not happen.
    if (alpha == 0.0) {
        applyBeta(koblas.vectorKernels, destination.values, 0, destination.values.size, beta)
        return
    }
    val left = this
    val right = other
    when {
        left is SparseMatrix && right is SparseMatrix ->
            koblas.gemm(alpha, left, transpose, right, transposeOther, beta, destination, workspace)

        left is SparseMatrix ->
            koblas.gemm(
                alpha, left, transpose, denseOperand(right), transposeOther, beta, destination,
                right = false, workspace = workspace,
            )

        right is SparseMatrix ->
            // The sparse operand names itself in the sparse signature, so the side flag is what puts it on
            // the right of the dense one rather than transposing the whole product to get it there.
            koblas.gemm(
                alpha, right, transposeOther, denseOperand(left), transpose, beta, destination,
                right = true, workspace = workspace,
            )

        else -> denseProduct(
            alpha,
            denseOperand(left),
            transpose,
            denseOperand(right),
            transposeOther,
            beta,
            destination,
            workspace,
        )
    }
}

/**
 * The dense route, which is the dense product itself.
 *
 * Staging an operand that shares the destination is that product's own rule and its own loan, so this hands
 * the workspace on rather than copying first and passing a matrix that no longer needs one.
 */
private fun denseProduct(
    alpha: Double,
    left: DenseMatrix,
    transpose: Boolean,
    right: DenseMatrix,
    transposeOther: Boolean,
    beta: Double,
    destination: DenseMatrix,
    workspace: Workspace?,
) = koblas.gemm(alpha, left, transpose, right, transposeOther, beta, destination, workspace)

/**
 * Dense column-major storage for an operand that is not already in it.
 *
 * A [DenseMatrix] is used where it lies. Anything else is read once through [Matrix.get], which is the only
 * access the common contract offers; a [SparseMatrix] never arrives here, because every pairing routes it to
 * the sparse implementation that walks its stored entries instead.
 */
private fun denseOperand(matrix: Matrix): DenseMatrix = when (matrix) {
    is DenseMatrix -> matrix
    else -> DenseMatrix.ofColumns(Array(matrix.cols) { j -> DoubleArray(matrix.rows) { i -> matrix[i, j] } })
}
