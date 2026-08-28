package com.eignex.koblas

import com.eignex.koblas.core.*

/** `A * B` (BLAS `dgemm`), allocating. gemm accumulates into an existing C instead. */
public operator fun F64DenseMatrix.times(other: F64DenseMatrix): F64DenseMatrix = koblas.gemm(this, other)

/** `A * B` for a sparse left operand and dense right operand, allocating dense storage through the active
 *  sparse backend. */
public operator fun F64SparseMatrix.times(other: F64DenseMatrix): F64DenseMatrix = koblas.sparseBlas.gemm(this, other)

/** `A * B` for a dense left operand and sparse right operand, allocating dense storage through the active
 *  sparse backend. */
public operator fun F64DenseMatrix.times(other: F64SparseMatrix): F64DenseMatrix {
    val out = F64DenseMatrix.zero(rows, other.cols)
    koblas.sparseBlas.gemm(1.0, other, false, this, false, 0.0, out, right = true)
    return out
}

/** `A * B` for sparse matrices, allocating the discovered sparse structure through the active backend. */
public operator fun F64SparseMatrix.times(other: F64SparseMatrix): F64SparseMatrix = koblas.sparseBlas.gemm(this, other)

/**
 * Matrix-vector product into a fresh dense result for any [F64MatrixLike] against any [F64VectorLike].
 * gemv provide transpose and destination-buffer variants.
 */
public operator fun F64MatrixLike.times(x: F64VectorLike): F64DenseVector {
    val a = this
    requireShape(a.cols == x.size) { "times shape mismatch: A is ${a.rows}x${a.cols}, x size ${x.size}" }
    if (a is F64DenseMatrix && x is F64DenseVector) return F64DenseVector.wrap(koblas.gemv(a, x.data))
    if (a is F64SparseMatrix && x is F64DenseVector) return F64DenseVector.wrap(koblas.gemv(a, x.data))
    val out = F64DenseVector(a.rows)
    val od = out.data
    if (a is F64DenseMatrix) {
        val ad = a.data
        val rows = a.rows
        x.forEachStored { j, v ->
            if (v != 0.0) koblas.kernels.axpy(od, 0, v, ad, j * rows, rows)
        }
    } else if (a is F64SparseMatrix) {
        x.forEachStored { j, v ->
            if (v != 0.0) a.forEachInColumn(j) { i, aij -> od[i] += aij * v }
        }
    } else {
        for (i in 0 until a.rows) {
            var sum = 0.0
            x.forEachStored { j, v -> sum += a[i, j] * v }
            od[i] = sum
        }
    }
    return out
}

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
    koblas.kernels.scale(out, 0, alpha, out.size)
    return out
}

/** A fresh copy of [a] with `alpha * b` added, which is what the allocating sums and differences return. */
private fun axpyCopy(a: DoubleArray, alpha: Double, b: DoubleArray): DoubleArray {
    val out = a.copyOf()
    koblas.kernels.axpy(out, 0, alpha, b, 0, out.size)
    return out
}
