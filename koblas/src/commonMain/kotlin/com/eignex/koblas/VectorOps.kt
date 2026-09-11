@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import com.eignex.koblas.StridedVectorView
import com.eignex.koblas.Vector
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.internal.numeric.neumaierSum
import kotlin.math.abs

/**
 * Visit each stored entry as (index, value), in ascending index order for any storage. A [SparseVector]
 * may present numerical zeros as stored, and any other [Vector] has every index visited.
 */
@kotlin.jvm.JvmSynthetic
public inline fun Vector.forEachStored(block: (i: Int, v: Double) -> Unit) {
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

/** `aT * b`. Sparse operands walk their stored entries only. */
public infix fun Vector.dot(other: Vector): Double {
    requireSameSize(size, other.size)
    if (this is StridedVectorView && other is StridedVectorView) {
        var sum = 0.0
        for (i in 0 until size) sum += data[offset + i * stride] * other.data[other.offset + i * other.stride]
        return sum
    }
    if (this is DenseVector && other is DenseVector) {
        return koblas.vectorKernels.dot(data, 0, other.data, 0, size)
    }
    if (this is SparseVector && other is SparseVector) return koblas.sparseKernels.dot(this, other)
    if (this is SparseVector && other is DenseVector) return koblas.sparseKernels.dot(this, other.data)
    if (this is DenseVector && other is SparseVector) return koblas.sparseKernels.dot(other, data)
    if (this is SparseVector) {
        var sum = 0.0
        for (k in indices.indices) sum += values[k] * other[indices[k]]
        return sum
    }
    if (other is SparseVector) return other dot this
    var s = 0.0
    for (i in 0 until size) s += this[i] * other[i]
    return s
}

/**
 * Euclidean norm (BLAS `dnrm2`). Rescales when the sum of squares would overflow or underflow, so any
 * finite input gives the correct norm.
 */
public fun Vector.norm2(): Double = when (this) {
    is DenseVector -> koblas.vectorKernels.nrm2(data, 0, size)
    is SparseVector -> koblas.sparseKernels.nrm2(this)
    is StridedVectorView -> stridedNorm2(this)
    else -> euclideanNorm(toDoubleArray(), 0, size)
}

/**
 * Plain sum of the entries. Distinct from [asum], which sums their absolute values.
 *
 * A dense operand reaches [DenseVectorKernels.sum]; any other storage sums its stored entries, since the ones it
 * does not store are zero and contribute nothing. Use [compensatedSum] where the length is large enough
 * that the rounding error of a naive sum matters.
 */
public fun Vector.sum(): Double = when (this) {
    is DenseVector -> koblas.vectorKernels.sum(data, 0, size)

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
public fun Vector.compensatedSum(): Double = when (this) {
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
public fun Vector.asum(): Double = when (this) {
    is DenseVector -> koblas.vectorKernels.asum(data, 0, size)

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
public fun Vector.iamax(): Int {
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

/**
 * `dst = src` (BLAS `dcopy`). A sparse source zero-fills the destination first, so nothing survives.
 * A borrowed or sparse source sharing [dst]'s buffer is snapshotted before writing.
 */
public fun copy(src: Vector, dst: DenseVector) {
    requireSameSize(src.size, dst.size)
    if ((src is StridedVectorView && src.data === dst.data) || (src is SparseVector && src.values === dst.data)) {
        src.toDoubleArray().copyInto(dst.data)
        return
    }
    when (src) {
        is DenseVector -> src.data.copyInto(dst.data)

        is SparseVector -> {
            dst.data.fill(0.0)
            koblas.sparseKernels.scatter(src, dst.data)
        }

        else -> {
            src.forEachStored { i, v -> dst.data[i] = v }
        }
    }
}

/**
 * `dst = src` into a borrowed strided destination. Overlapping built-in sources are snapshotted before
 * writing; disjoint operands are copied without materializing either one.
 */
public fun copy(src: Vector, dst: StridedVectorView) {
    requireSameSize(src.size, dst.size)
    val source = src.stableFor(dst)
    for (i in 0 until source.size) dst[i] = source[i]
}

/** Retains sparse support when snapshotting an input whose values may be overwritten through [destination]. */
private fun Vector.stableFor(destination: StridedVectorView): Vector = when (this) {
    is DenseVector -> if (data === destination.data) DenseVector.of(data) else this
    is SparseVector -> if (values === destination.data) SparseVector.wrap(size, indices, values.copyOf()) else this
    is StridedVectorView -> if (overlaps(destination)) DenseVector.wrap(toDoubleArray()) else this
    else -> this
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
    koblas.vectorKernels.swap(a.data, 0, b.data, 0, a.size)
}

/**
 * Exchanges two borrowed slices, including rows and columns of dense matrix views. Overlapping inputs are
 * snapshotted; at a shared physical entry the final write is from [b].
 */
public fun swap(a: StridedVectorView, b: StridedVectorView) {
    requireSameSize(a.size, b.size)
    if (a.overlaps(b)) {
        val snapshotA = a.toDoubleArray()
        val snapshotB = b.toDoubleArray()
        for (i in 0 until a.size) a[i] = snapshotB[i]
        for (i in 0 until b.size) b[i] = snapshotA[i]
        return
    }
    for (i in 0 until a.size) {
        val value = a[i]
        a[i] = b[i]
        b[i] = value
    }
}

/**
 * `y = y + alpha * x`. A sparse `x` touches only the positions it stores. A borrowed [x] sharing the
 * destination buffer is snapshotted before writing.
 */
public fun DenseVector.axpy(alpha: Double, x: Vector) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    val source = if (x is StridedVectorView && x.data === data) DenseVector.wrap(x.toDoubleArray()) else x
    when (source) {
        is DenseVector -> koblas.vectorKernels.axpy(data, 0, alpha, source.data, 0, size)
        is SparseVector -> koblas.sparseKernels.axpy(data, alpha, source)
        else -> source.forEachStored { i, v -> data[i] += alpha * v }
    }
}

/**
 * `this = this + alpha * x` over a borrowed strided destination. Sparse inputs touch only stored positions;
 * overlapping built-in inputs are snapshotted before writing.
 */
public fun StridedVectorView.axpy(alpha: Double, x: Vector) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    x.stableFor(this).forEachStored { i, value -> this[i] += alpha * value }
}

/** `v = alpha * v`. */
public fun DenseVector.scale(alpha: Double) {
    if (alpha == 1.0) return
    koblas.vectorKernels.scale(data, 0, alpha, size)
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
        if (value.isNaN()) return Double.NaN
        if (value.isInfinite()) {
            // Repeated infinities must not form infinity / infinity in the rescaling recurrence.
            scale = value
            sumSquares = 1.0
        } else if (value != 0.0) {
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
    if (scale == 0.0) return 0.0
    return scale * kotlin.math.sqrt(sumSquares)
}
