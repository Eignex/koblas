@file:Suppress("LongParameterList", "MatchingDeclarationName") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/**
 * Reusable scratch for alias-safe matrix products.
 *
 * A workspace belongs to one invocation at a time. Independent calls may safely use distinct workspaces; calls
 * without one own their temporary storage. Its contents are implementation details and never retain an operand.
 */
public class MatrixWorkspace {
    private var left: DoubleArray = DoubleArray(0)
    private var right: DoubleArray = DoubleArray(0)

    internal fun leftCopy(values: DoubleArray): DoubleArray {
        if (left.size != values.size) left = DoubleArray(values.size)
        values.copyInto(left)
        return left
    }

    internal fun rightCopy(values: DoubleArray): DoubleArray {
        if (right.size != values.size) right = DoubleArray(values.size)
        values.copyInto(right)
        return right
    }
}

/**
 * Allocates `this * other` through the common [Matrix] contract.
 *
 * Dense-by-dense returns [DenseMatrix]. Sparse and mixed storage pairings are reserved by this API and are
 * restored in S2; until then they fail explicitly rather than densifying an operand behind the caller's back.
 */
public fun Matrix.gemm(other: Matrix): Matrix {
    requireShape(cols == other.rows) { "gemm: inner dimensions differ, $cols vs ${other.rows}" }
    return when {
        this is DenseMatrix && other is DenseMatrix -> koblas.gemm(this, other)

        this is SparseMatrix || other is SparseMatrix -> unsupportedSparseProduct()

        else -> {
            val result = DenseMatrix.zero(rows, other.cols)
            gemmInto(1.0, false, other, false, 0.0, result)
            result
        }
    }
}

/** Common allocating matrix product; the runtime storage pair determines the result storage. */
@kotlin.jvm.JvmName("multiplyMatrices")
public operator fun Matrix.times(other: Matrix): Matrix = gemm(other)

/**
 * Computes `destination = alpha * op(this) * op(other) + beta * destination`.
 *
 * The dense route is complete and alias-safe. Mixed and sparse routes are part of the same stable API but are
 * implemented in S2; they currently fail before mutating [destination]. Other [Matrix] implementations use
 * logical access and are staged once into dense column-major storage.
 */
public fun Matrix.gemmInto(
    alpha: Double,
    transpose: Boolean,
    other: Matrix,
    transposeOther: Boolean,
    beta: Double,
    destination: DenseMatrix,
    workspace: MatrixWorkspace? = null,
) {
    val m = if (transpose) cols else rows
    val depth = if (transpose) rows else cols
    val otherDepth = if (transposeOther) other.cols else other.rows
    val n = if (transposeOther) other.rows else other.cols
    requireShape(depth == otherDepth) { "gemmInto: inner dimensions differ, $depth vs $otherDepth" }
    requireShape(destination.rows == m && destination.cols == n) {
        "gemmInto: destination must be ${m}x$n, got ${destination.rows}x${destination.cols}"
    }
    if (this is SparseMatrix || other is SparseMatrix) unsupportedSparseProduct()

    val left = denseOperand(this)
    val right = denseOperand(other)
    val stableLeft = if (left.values === destination.values) {
        DenseMatrix.wrap(left.rows, left.cols, workspace?.leftCopy(left.values) ?: left.values.copyOf())
    } else {
        left
    }
    val stableRight = if (right.values === destination.values) {
        DenseMatrix.wrap(right.rows, right.cols, workspace?.rightCopy(right.values) ?: right.values.copyOf())
    } else {
        right
    }
    koblas.gemm(alpha, stableLeft, transpose, stableRight, transposeOther, beta, destination)
}

private fun denseOperand(matrix: Matrix): DenseMatrix = when (matrix) {
    is DenseMatrix -> matrix
    else -> DenseMatrix.ofColumns(Array(matrix.cols) { j -> DoubleArray(matrix.rows) { i -> matrix[i, j] } })
}

private fun unsupportedSparseProduct(): Nothing =
    throw UnsupportedOperationException("mixed and sparse matrix products are restored in S2")
