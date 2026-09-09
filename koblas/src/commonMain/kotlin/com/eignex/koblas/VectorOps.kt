@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import com.eignex.koblas.StridedVectorView
import com.eignex.koblas.VectorLike
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.internal.numeric.neumaierSum
import com.eignex.koblas.sparse.SparseKernels
import kotlin.math.abs

/**
 * Visit each stored entry as (index, value), in ascending index order for any storage. A [SparseVector]
 * may present numerical zeros as stored, and any other [VectorLike] has every index visited.
 */
public inline fun VectorLike.forEachStored(block: (i: Int, v: Double) -> Unit) {
    when (this) {
        is DenseVector -> {
            val d = data
            for (i in 0 until d.size) block(i, d[i])
        }

        is SparseVector -> {
            val idx = indices
            val vals = values
            for (k in idx.indices) block(idx[k], vals[k])
        }

        else -> for (i in 0 until size) block(i, this[i])
    }
}

/** `aT * b`. Any sparse operand goes through [SparseKernels], walking the stored entries only. */
public infix fun VectorLike.dot(other: VectorLike): Double {
    requireSameSize(size, other.size)
    if (this is StridedVectorView && other is StridedVectorView) {
        var sum = 0.0
        for (i in 0 until size) sum += data[offset + i * stride] * other.data[other.offset + i * other.stride]
        return sum
    }
    if (this is DenseVector && other is DenseVector) {
        return koblas.kernels.dot(data, 0, other.data, 0, size)
    }
    if (this is SparseVector && other is SparseVector) return koblas.sparseKernels.dot(this, other)
    if (this is SparseVector && other is DenseVector) return koblas.sparseKernels.dot(this, other.data)
    if (this is DenseVector && other is SparseVector) return koblas.sparseKernels.dot(other, data)
    var s = 0.0
    for (i in 0 until size) s += this[i] * other[i]
    return s
}

/**
 * Euclidean norm (BLAS `dnrm2`). Rescales when the sum of squares would overflow or underflow, so any
 * finite input gives the correct norm.
 */
public fun VectorLike.norm2(): Double = when (this) {
    is DenseVector -> koblas.kernels.nrm2(data, 0, size)
    is SparseVector -> koblas.sparseKernels.nrm2(this)
    is StridedVectorView -> stridedNorm2(this)
    else -> euclideanNorm(toDoubleArray(), 0, size)
}

/**
 * Plain sum of the entries. Distinct from [asum], which sums their absolute values.
 *
 * A dense operand reaches [Kernels.sum]; any other storage sums its stored entries, since the ones it
 * does not store are zero and contribute nothing. Use [compensatedSum] where the length is large enough
 * that the rounding error of a naive sum matters.
 */
public fun VectorLike.sum(): Double = when (this) {
    is DenseVector -> koblas.kernels.sum(data, 0, size)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += x }
        s
    }
}

/**
 * Sum of the entries with Neumaier's compensation, holding to within one rounding of the exact sum whatever
 * the length or the ordering.
 *
 * A naive sum of `n` terms carries an error that grows with `n`, so over a long run the total says less than
 * its digits suggest. This tracks the low-order bits each addition drops and adds them back at the end, for
 * about twice the arithmetic of [sum].
 *
 * Deliberately not on the kernel seam and deliberately not vectorised: the error bound is the whole point,
 * and a compensator split across vector lanes has a different bound from one carried in order. A lane-wise
 * form is a different routine and would need its own analysis before it could take this name.
 *
 * This is the batch form, over entries that already exist. An accumulator fed one value at a time needs its
 * compensator in its own state, which no vector routine can supply.
 */
public fun VectorLike.compensatedSum(): Double = when (this) {
    is DenseVector -> neumaierSum(data, 0, size)

    else -> {
        var s = 0.0
        var compensation = 0.0
        forEachStored { _, x ->
            val t = s + x
            compensation += if (abs(s) >= abs(x)) (s - t) + x else (x - t) + s
            s = t
        }
        s + compensation
    }
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun VectorLike.asum(): Double = when (this) {
    is DenseVector -> koblas.kernels.asum(data, 0, size)

    is SparseVector -> koblas.sparseKernels.asum(this)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += abs(x) }
        s
    }
}

/**
 * Zero-based logical index of the first maximum absolute value (BLAS `idamax`), or `-1` when empty.
 * NaNs are ignored by the strict comparison; a later finite or infinite magnitude can therefore win after
 * a leading NaN. All-zero and all-NaN nonempty inputs return `0`, as do sparse inputs whose maximum is an
 * implicit zero. Strided views report their logical index rather than a backing-array offset.
 */
public fun VectorLike.iamax(): Int {
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
public fun copy(src: VectorLike, dst: DenseVector) {
    requireSameSize(src.size, dst.size)
    when (src) {
        is DenseVector -> src.data.copyInto(dst.data)

        is SparseVector -> {
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
public fun copy(src: VectorLike, dst: StridedVectorView) {
    requireSameSize(src.size, dst.size)
    for (i in 0 until src.size) dst[i] = src[i]
}

/**
 * Read [from] at [x]'s stored positions into [x] (Sparse BLAS `usga`), the inverse of [copy] from a sparse
 * source. [x] keeps its pattern, so a nonzero of [from] at an unstored position is not read.
 */
public fun gather(x: SparseVector, from: DenseVector) {
    requireSameSize(x.size, from.size)
    koblas.sparseKernels.gather(x, from.data)
}

/** [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`). */
public fun gatherZero(x: SparseVector, from: DenseVector) {
    requireSameSize(x.size, from.size)
    koblas.sparseKernels.gatherZero(x, from.data)
}

/** Exchange the contents of [a] and [b] (BLAS `dswap`). */
public fun swap(a: DenseVector, b: DenseVector) {
    requireSameSize(a.size, b.size)
    koblas.kernels.swap(a.data, 0, b.data, 0, a.size)
}

/** Exchanges two borrowed slices, including rows and columns of dense matrix views. */
public fun swap(a: StridedVectorView, b: StridedVectorView) {
    requireSameSize(a.size, b.size)
    for (i in 0 until a.size) {
        val value = a[i]
        a[i] = b[i]
        b[i] = value
    }
}

/** `y = y + alpha * x`. A sparse `x` touches only the positions it stores. */
public fun DenseVector.axpy(alpha: Double, x: VectorLike) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    when (x) {
        is DenseVector -> koblas.kernels.axpy(data, 0, alpha, x.data, 0, size)
        is SparseVector -> koblas.sparseKernels.axpy(data, alpha, x)
        else -> x.forEachStored { i, v -> data[i] += alpha * v }
    }
}

/** `this = this + alpha * x` over a borrowed strided destination. */
public fun StridedVectorView.axpy(alpha: Double, x: VectorLike) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    for (i in 0 until size) this[i] += alpha * x[i]
}

/** `v = alpha * v`. */
public fun DenseVector.scale(alpha: Double) {
    if (alpha == 1.0) return
    koblas.kernels.scale(data, 0, alpha, size)
}

/** `this = alpha * this` over a borrowed strided slice. */
public fun StridedVectorView.scale(alpha: Double) {
    if (alpha == 1.0) return
    for (i in 0 until size) this[i] *= alpha
}

/** Scaled sum-of-squares over a strided vector, retaining `dnrm2` overflow and underflow behavior. */
private fun stridedNorm2(vector: StridedVectorView): Double {
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
    // A NaN entry never raises `scale`, since `0.0 < NaN` is false, but it does reach `sumSquares`. Reading
    // the zero case off `scale` alone would then discard it and report a clean norm for a corrupt vector,
    // where dnrm2 and the dense path both propagate the NaN.
    if (scale == 0.0) return if (sumSquares.isNaN()) Double.NaN else 0.0
    return scale * kotlin.math.sqrt(sumSquares)
}
