package com.eignex.koblas

import com.eignex.koblas.*

/** `A * B` (BLAS `dgemm`), allocating. gemm accumulates into an existing C instead. */
public operator fun DenseMatrix.times(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)

/** `A * B` for a sparse left operand and dense right operand, allocating dense storage through the active
 *  sparse backend. */
public operator fun SparseMatrix.times(other: DenseMatrix): DenseMatrix = koblas.sparseBlas.gemm(this, other)

/** `A * B` for a dense left operand and sparse right operand, allocating dense storage through the active
 *  sparse backend. */
public operator fun DenseMatrix.times(other: SparseMatrix): DenseMatrix {
    val out = DenseMatrix.zero(rows, other.cols)
    koblas.sparseBlas.gemm(1.0, other, false, this, false, 0.0, out, right = true)
    return out
}

/** `A * B` for sparse matrices, allocating the discovered sparse structure through the active backend. */
public operator fun SparseMatrix.times(other: SparseMatrix): SparseMatrix = koblas.sparseBlas.gemm(this, other)

/**
 * Matrix-vector product into a fresh dense result for any [MatrixLike] against any [VectorLike].
 * [gemvInto] writes into a destination the caller owns, and provides the alpha and beta scalars.
 */
public operator fun MatrixLike.times(x: VectorLike): DenseVector {
    requireShape(cols == x.size) { "times shape mismatch: A is ${rows}x$cols, x size ${x.size}" }
    val out = DenseVector(rows)
    gemvInto(x, out.data)
    return out
}

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
    koblas.kernels.scale(out, 0, alpha, out.size)
    return out
}

/** A fresh copy of [a] with `alpha * b` added, which is what the allocating sums and differences return. */
private fun axpyCopy(a: DoubleArray, alpha: Double, b: DoubleArray): DoubleArray {
    val out = a.copyOf()
    koblas.kernels.axpy(out, 0, alpha, b, 0, out.size)
    return out
}
