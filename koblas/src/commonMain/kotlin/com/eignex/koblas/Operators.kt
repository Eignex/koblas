@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/** `A * B` (BLAS `dgemm`), allocating. gemm accumulates into an existing C instead. */
@kotlin.jvm.JvmName("multiply")
public operator fun DenseMatrix.times(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)

/**
 * Matrix-vector product into a fresh dense result, against any [Vector].
 * [gemvInto] writes into a destination the caller owns, and provides the alpha and beta scalars.
 *
 * Dense storage only, for the reason [gemvInto] gives.
 */
@kotlin.jvm.JvmName("multiply")
public operator fun DenseMatrix.times(x: Vector): DenseVector {
    requireShape(cols == x.size) { "times shape mismatch: A is ${rows}x$cols, x size ${x.size}" }
    val out = DenseVector(rows)
    gemvInto(x, out.data)
    return out
}

/** `alpha * A`, allocating. [scale] multiplies in place. */
@kotlin.jvm.JvmName("multiply")
public operator fun DenseMatrix.times(alpha: Double): DenseMatrix = DenseMatrix.wrap(
    rows,
    cols,
    scaledCopy(data, alpha),
)

/** `alpha * A`, allocating. [scale] multiplies in place. */
@kotlin.jvm.JvmName("multiply")
public operator fun Double.times(a: DenseMatrix): DenseMatrix = a * this

/** `-A`, allocating. */
@kotlin.jvm.JvmName("negate")
public operator fun DenseMatrix.unaryMinus(): DenseMatrix = this * -1.0

/** `A + B`, allocating. [axpy] accumulates into an existing operand. */
@kotlin.jvm.JvmName("add")
public operator fun DenseMatrix.plus(other: DenseMatrix): DenseMatrix = combine(other, 1.0, "plus")

/** `A - B`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
@kotlin.jvm.JvmName("subtract")
public operator fun DenseMatrix.minus(other: DenseMatrix): DenseMatrix = combine(other, -1.0, "minus")

/** `a + b`, allocating. [axpy] accumulates into an existing operand. */
@kotlin.jvm.JvmName("add")
public operator fun DenseVector.plus(other: DenseVector): DenseVector = combine(other, 1.0, "plus")

/** `a - b`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
@kotlin.jvm.JvmName("subtract")
public operator fun DenseVector.minus(other: DenseVector): DenseVector = combine(other, -1.0, "minus")

/** `alpha * x`, allocating. [scale] multiplies in place. */
@kotlin.jvm.JvmName("multiply")
public operator fun DenseVector.times(alpha: Double): DenseVector = DenseVector.wrap(scaledCopy(data, alpha))

/** `alpha * x`, allocating. [scale] multiplies in place. */
@kotlin.jvm.JvmName("multiply")
public operator fun Double.times(x: DenseVector): DenseVector = x * this

/** `-x`, allocating. */
@kotlin.jvm.JvmName("negate")
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
