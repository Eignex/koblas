@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.core.F64StridedVectorView
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.requireSameSize
import com.eignex.koblas.sparse.F64SparseKernels
import kotlin.math.abs

/**
 * Visit each stored entry as (index, value), in ascending index order for any storage. A [F64SparseVector]
 * may present numerical zeros as stored, and any other [F64VectorLike] has every index visited.
 */
public inline fun F64VectorLike.forEachStored(block: (i: Int, v: Double) -> Unit) {
    when (this) {
        is F64DenseVector -> {
            val d = data
            for (i in 0 until d.size) block(i, d[i])
        }

        is F64SparseVector -> {
            val idx = indices
            val vals = values
            for (k in idx.indices) block(idx[k], vals[k])
        }

        else -> for (i in 0 until size) block(i, this[i])
    }
}

/** `aT * b`. Any sparse operand goes through [F64SparseKernels], walking the stored entries only. */
public infix fun F64VectorLike.dot(other: F64VectorLike): Double {
    requireSameSize(size, other.size)
    if (this is F64StridedVectorView && other is F64StridedVectorView) {
        var sum = 0.0
        for (i in 0 until size) sum += data[offset + i * stride] * other.data[other.offset + i * other.stride]
        return sum
    }
    if (this is F64DenseVector && other is F64DenseVector) {
        return koblas.kernels.dot(data, 0, other.data, 0, size)
    }
    if (this is F64SparseVector && other is F64SparseVector) return koblas.sparseKernels.dot(this, other)
    if (this is F64SparseVector && other is F64DenseVector) return koblas.sparseKernels.dot(this, other.data)
    if (this is F64DenseVector && other is F64SparseVector) return koblas.sparseKernels.dot(other, data)
    var s = 0.0
    for (i in 0 until size) s += this[i] * other[i]
    return s
}

/**
 * Euclidean norm (BLAS `dnrm2`). Rescales when the sum of squares would overflow or underflow, so any
 * finite input gives the correct norm.
 */
public fun F64VectorLike.norm2(): Double = when (this) {
    is F64DenseVector -> koblas.kernels.nrm2(data, 0, size)
    is F64SparseVector -> koblas.sparseKernels.nrm2(this)
    is F64StridedVectorView -> stridedNorm2(this)
    else -> euclideanNorm(toDoubleArray(), 0, size)
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun F64VectorLike.asum(): Double = when (this) {
    is F64DenseVector -> koblas.kernels.asum(data, 0, size)

    is F64SparseVector -> koblas.sparseKernels.asum(this)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += abs(x) }
        s
    }
}

/**
 * Index of the entry with maximal absolute value (BLAS `idamax`), -1 for a zero-length vector.
 * Ties resolve to the lowest index, and a vector with no stored entries returns 0.
 */
public fun F64VectorLike.iamax(): Int {
    if (size == 0) return -1
    var best = -1
    var bestAbs = 0.0
    forEachStored { i, x ->
        val a = abs(x)
        if (a > bestAbs) {
            bestAbs = a
            best = i
        }
    }
    return if (best == -1) 0 else best
}

/** `dst = src` (BLAS `dcopy`). A sparse source zero-fills the destination first, so nothing survives. */
public fun copy(src: F64VectorLike, dst: F64DenseVector) {
    requireSameSize(src.size, dst.size)
    when (src) {
        is F64DenseVector -> src.data.copyInto(dst.data)

        is F64SparseVector -> {
            dst.data.fill(0.0)
            koblas.sparseKernels.scatter(src, dst.data)
        }

        else -> {
            dst.data.fill(0.0)
            src.forEachStored { i, v -> dst.data[i] = v }
        }
    }
}

/** `dst = src` into a borrowed strided destination, without materializing either operand. */
public fun copy(src: F64VectorLike, dst: F64StridedVectorView) {
    requireSameSize(src.size, dst.size)
    for (i in 0 until src.size) dst[i] = src[i]
}

/**
 * Read [from] at [x]'s stored positions into [x] (Sparse BLAS `usga`), the inverse of [copy] from a sparse
 * source. [x] keeps its pattern, so a nonzero of [from] at an unstored position is not read.
 */
public fun gather(x: F64SparseVector, from: F64DenseVector) {
    requireSameSize(x.size, from.size)
    koblas.sparseKernels.gather(x, from.data)
}

/** [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`). */
public fun gatherZero(x: F64SparseVector, from: F64DenseVector) {
    requireSameSize(x.size, from.size)
    koblas.sparseKernels.gatherZero(x, from.data)
}

/** Exchange the contents of [a] and [b] (BLAS `dswap`). */
public fun swap(a: F64DenseVector, b: F64DenseVector) {
    requireSameSize(a.size, b.size)
    koblas.kernels.swap(a.data, 0, b.data, 0, a.size)
}

/** Exchanges two borrowed slices, including rows and columns of dense matrix views. */
public fun swap(a: F64StridedVectorView, b: F64StridedVectorView) {
    requireSameSize(a.size, b.size)
    for (i in 0 until a.size) {
        val value = a[i]
        a[i] = b[i]
        b[i] = value
    }
}

/** `y = y + alpha * x`. A sparse `x` touches only the positions it stores. */
public fun F64DenseVector.axpy(alpha: Double, x: F64VectorLike) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    when (x) {
        is F64DenseVector -> koblas.kernels.axpy(data, 0, alpha, x.data, 0, size)
        is F64SparseVector -> koblas.sparseKernels.axpy(data, alpha, x)
        else -> x.forEachStored { i, v -> data[i] += alpha * v }
    }
}

/** `this = this + alpha * x` over a borrowed strided destination. */
public fun F64StridedVectorView.axpy(alpha: Double, x: F64VectorLike) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    for (i in 0 until size) this[i] += alpha * x[i]
}

/** `v = alpha * v`. */
public fun F64DenseVector.scale(alpha: Double) {
    if (alpha == 1.0) return
    koblas.kernels.scale(data, 0, alpha, size)
}

/** `this = alpha * this` over a borrowed strided slice. */
public fun F64StridedVectorView.scale(alpha: Double) {
    if (alpha == 1.0) return
    for (i in 0 until size) this[i] *= alpha
}

/** Scaled sum-of-squares over a strided vector, retaining `dnrm2` overflow and underflow behavior. */
private fun stridedNorm2(vector: F64StridedVectorView): Double {
    var scale = 0.0
    var sumSquares = 1.0
    for (i in 0 until vector.size) {
        val value = abs(vector[i])
        if (value != 0.0) {
            if (scale < value) {
                val ratio = scale / value
                sumSquares = 1.0 + sumSquares * ratio * ratio
                scale = value
            } else {
                val ratio = value / scale
                sumSquares += ratio * ratio
            }
        }
    }
    return if (scale == 0.0) 0.0 else scale * kotlin.math.sqrt(sumSquares)
}
