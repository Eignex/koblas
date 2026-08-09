package com.eignex.koblas

import com.eignex.koblas.dense.LinearAlgebra

// Operator spellings of routines that already exist.
//
// Every operator here is an alias: it forwards to a routine defined elsewhere, or composes two of them
// over a contiguous backing. None introduces a kernel of its own, which is what keeps the set small and
// decides the omissions -- `SparseMatrix + SparseMatrix` would need a pattern union, and there is no
// sparse `scale`, so neither has an operator.
//
// All of them allocate. That is the same trade `gemm(a, b)` and `matMul` already make, and the reason
// each KDoc names the in-place routine to reach for instead: an operator is for the call site where the
// expression reads better than the loop, not for the inner loop.

/** `A · B` (BLAS `dgemm`); allocates. [LinearAlgebra.gemm] accumulates into an existing `C` instead. */
operator fun DenseMatrix.times(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)

/** `A · x` (BLAS `dgemv`); allocates. [LinearAlgebra.gemv] writes into an existing `y` instead. */
operator fun DenseMatrix.times(x: DenseVector): DenseVector = DenseVector.wrap(koblas.gemv(this, x.data))

/** `A · x` for a CSC matrix; allocates. [com.eignex.koblas.sparse.SparseBlas.gemv] takes a destination. */
operator fun SparseMatrix.times(x: DenseVector): DenseVector = DenseVector.wrap(koblas.gemv(this, x.data))

/** `alpha · A`; allocates. [scale] multiplies in place. */
operator fun DenseMatrix.times(alpha: Double): DenseMatrix {
    val out = data.copyOf()
    koblas.vectorKernels.scale(out, 0, alpha, out.size)
    return DenseMatrix.wrap(rows, cols, out)
}

/** `alpha · A`; allocates. [scale] multiplies in place. */
operator fun Double.times(a: DenseMatrix): DenseMatrix = a * this

/** `-A`; allocates. */
operator fun DenseMatrix.unaryMinus(): DenseMatrix = this * -1.0

/** `A + B`; allocates. [axpy] accumulates into an existing operand. */
operator fun DenseMatrix.plus(other: DenseMatrix): DenseMatrix = combine(other, 1.0, "plus")

/** `A - B`; allocates. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
operator fun DenseMatrix.minus(other: DenseMatrix): DenseMatrix = combine(other, -1.0, "minus")

/** `a + b`; allocates. [axpy] accumulates into an existing operand. */
operator fun DenseVector.plus(other: DenseVector): DenseVector = combine(other, 1.0, "plus")

/** `a - b`; allocates. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
operator fun DenseVector.minus(other: DenseVector): DenseVector = combine(other, -1.0, "minus")

/** `alpha · x`; allocates. [scale] multiplies in place. */
operator fun DenseVector.times(alpha: Double): DenseVector {
    val out = data.copyOf()
    koblas.vectorKernels.scale(out, 0, alpha, out.size)
    return DenseVector.wrap(out)
}

/** `alpha · x`; allocates. [scale] multiplies in place. */
operator fun Double.times(x: DenseVector): DenseVector = x * this

/** `-x`; allocates. */
operator fun DenseVector.unaryMinus(): DenseVector = this * -1.0

/**
 * `A + alpha · B` over the flat backings, which are contiguous and identically laid out, so the whole
 * matrix is one `axpy` rather than one per column.
 */
private fun DenseMatrix.combine(other: DenseMatrix, alpha: Double, op: String): DenseMatrix {
    require(rows == other.rows && cols == other.cols) {
        "$op shape mismatch: ${rows}x$cols and ${other.rows}x${other.cols}"
    }
    val out = data.copyOf()
    koblas.vectorKernels.axpy(out, 0, alpha, other.data, 0, out.size)
    return DenseMatrix.wrap(rows, cols, out)
}

/** `x + alpha · y`. */
private fun DenseVector.combine(other: DenseVector, alpha: Double, op: String): DenseVector {
    require(size == other.size) { "$op size mismatch: $size vs ${other.size}" }
    val out = data.copyOf()
    koblas.vectorKernels.axpy(out, 0, alpha, other.data, 0, out.size)
    return DenseVector.wrap(out)
}
