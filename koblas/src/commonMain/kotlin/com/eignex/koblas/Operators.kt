package com.eignex.koblas

import com.eignex.koblas.dense.LinearAlgebra

/** `A * B` (BLAS `dgemm`), allocating. [LinearAlgebra.gemm] accumulates into an existing C instead. */
public operator fun DenseMatrix.times(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)

/** `A * x` (BLAS `dgemv`), allocating. [LinearAlgebra.gemv] writes into an existing y instead. */
public operator fun DenseMatrix.times(x: DenseVector): DenseVector = DenseVector.wrap(koblas.gemv(this, x.data))

/** `A * x` for a CSC matrix, allocating. [com.eignex.koblas.sparse.SparseBlas.gemv] takes a destination. */
public operator fun SparseMatrix.times(x: DenseVector): DenseVector = DenseVector.wrap(koblas.gemv(this, x.data))

/** `alpha * A`, allocating. [scale] multiplies in place. */
public operator fun DenseMatrix.times(alpha: Double): DenseMatrix = DenseMatrix.wrap(
    rows,
    cols,
    scaledCopy(data, alpha),
)

/** `alpha * A`, allocating. [scale] multiplies in place. */
public operator fun Double.times(a: DenseMatrix): DenseMatrix = a * this

/** `-A`, allocating. */
public operator fun DenseMatrix.unaryMinus(): DenseMatrix = this * -1.0

/** `A + B`, allocating. [axpy] accumulates into an existing operand. */
public operator fun DenseMatrix.plus(other: DenseMatrix): DenseMatrix = combine(other, 1.0, "plus")

/** `A - B`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
public operator fun DenseMatrix.minus(other: DenseMatrix): DenseMatrix = combine(other, -1.0, "minus")

/** `a + b`, allocating. [axpy] accumulates into an existing operand. */
public operator fun DenseVector.plus(other: DenseVector): DenseVector = combine(other, 1.0, "plus")

/** `a - b`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
public operator fun DenseVector.minus(other: DenseVector): DenseVector = combine(other, -1.0, "minus")

/** `alpha * x`, allocating. [scale] multiplies in place. */
public operator fun DenseVector.times(alpha: Double): DenseVector = DenseVector.wrap(scaledCopy(data, alpha))

/** `alpha * x`, allocating. [scale] multiplies in place. */
public operator fun Double.times(x: DenseVector): DenseVector = x * this

/** `-x`, allocating. */
public operator fun DenseVector.unaryMinus(): DenseVector = this * -1.0

/** `A + alpha * B` as a single `axpy` over the flat backings. */
private fun DenseMatrix.combine(other: DenseMatrix, alpha: Double, op: String): DenseMatrix {
    requireShape(rows == other.rows && cols == other.cols) {
        "$op shape mismatch: ${rows}x$cols and ${other.rows}x${other.cols}"
    }
    return DenseMatrix.wrap(rows, cols, axpyCopy(data, alpha, other.data))
}

/** `x + alpha * y`. */
private fun DenseVector.combine(other: DenseVector, alpha: Double, op: String): DenseVector {
    requireShape(size == other.size) { "$op size mismatch: $size vs ${other.size}" }
    return DenseVector.wrap(axpyCopy(data, alpha, other.data))
}

/** A fresh copy of [data] scaled by [alpha], which is what the allocating scalar products all return. */
private fun scaledCopy(data: DoubleArray, alpha: Double): DoubleArray {
    val out = data.copyOf()
    koblas.vectorKernels.scale(out, 0, alpha, out.size)
    return out
}

/** A fresh copy of [a] with `alpha * b` added, which is what the allocating sums and differences return. */
private fun axpyCopy(a: DoubleArray, alpha: Double, b: DoubleArray): DoubleArray {
    val out = a.copyOf()
    koblas.vectorKernels.axpy(out, 0, alpha, b, 0, out.size)
    return out
}
