package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.F64LinearAlgebra

/** `A * B` (BLAS `dgemm`), allocating. [F64LinearAlgebra.gemm] accumulates into an existing C instead. */
public operator fun F64DenseMatrix.times(other: F64DenseMatrix): F64DenseMatrix = koblas.gemm(this, other)

/** `A * x` (BLAS `dgemv`), allocating. [F64LinearAlgebra.gemv] writes into an existing y instead. */
public operator fun F64DenseMatrix.times(x: F64DenseVector): F64DenseVector = F64DenseVector.wrap(
    koblas.gemv(this, x.data),
)

/** `A * x` for a CSC matrix, allocating. [com.eignex.koblas.sparse.F64SparseBlas.gemv] takes a destination. */
public operator fun F64SparseMatrix.times(x: F64DenseVector): F64DenseVector = F64DenseVector.wrap(
    koblas.gemv(this, x.data),
)

/** `alpha * A`, allocating. [scale] multiplies in place. */
public operator fun F64DenseMatrix.times(alpha: Double): F64DenseMatrix = F64DenseMatrix.wrap(
    rows,
    cols,
    scaledCopy(data, alpha),
)

/** `alpha * A`, allocating. [scale] multiplies in place. */
public operator fun Double.times(a: F64DenseMatrix): F64DenseMatrix = a * this

/** `-A`, allocating. */
public operator fun F64DenseMatrix.unaryMinus(): F64DenseMatrix = this * -1.0

/** `A + B`, allocating. [axpy] accumulates into an existing operand. */
public operator fun F64DenseMatrix.plus(other: F64DenseMatrix): F64DenseMatrix = combine(other, 1.0, "plus")

/** `A - B`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
public operator fun F64DenseMatrix.minus(other: F64DenseMatrix): F64DenseMatrix = combine(other, -1.0, "minus")

/** `a + b`, allocating. [axpy] accumulates into an existing operand. */
public operator fun F64DenseVector.plus(other: F64DenseVector): F64DenseVector = combine(other, 1.0, "plus")

/** `a - b`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
public operator fun F64DenseVector.minus(other: F64DenseVector): F64DenseVector = combine(other, -1.0, "minus")

/** `alpha * x`, allocating. [scale] multiplies in place. */
public operator fun F64DenseVector.times(alpha: Double): F64DenseVector = F64DenseVector.wrap(scaledCopy(data, alpha))

/** `alpha * x`, allocating. [scale] multiplies in place. */
public operator fun Double.times(x: F64DenseVector): F64DenseVector = x * this

/** `-x`, allocating. */
public operator fun F64DenseVector.unaryMinus(): F64DenseVector = this * -1.0

/** `A + alpha * B` as a single `axpy` over the flat backings. */
private fun F64DenseMatrix.combine(other: F64DenseMatrix, alpha: Double, op: String): F64DenseMatrix {
    requireShape(rows == other.rows && cols == other.cols) {
        "$op shape mismatch: ${rows}x$cols and ${other.rows}x${other.cols}"
    }
    return F64DenseMatrix.wrap(rows, cols, axpyCopy(data, alpha, other.data))
}

/** `x + alpha * y`. */
private fun F64DenseVector.combine(other: F64DenseVector, alpha: Double, op: String): F64DenseVector {
    requireShape(size == other.size) { "$op size mismatch: $size vs ${other.size}" }
    return F64DenseVector.wrap(axpyCopy(data, alpha, other.data))
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
