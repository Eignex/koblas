@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import com.eignex.koblas.StridedVector
import com.eignex.koblas.Vector
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.numeric.euclideanNorm
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
            var p = offset
            for (i in 0 until size) {
                block(i, d[p])
                p += stride
            }
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
    if (this is DenseVector && other is DenseVector) {
        return koblas.vectorKernels.dot(data, offset, other.data, other.offset, size, stride, other.stride)
    }
    if (this is SparseVector && other is SparseVector) return koblas.sparseKernels.dot(this, other)
    if (this is SparseVector && other is DenseVector) return koblas.sparseKernels.dot(this, other.densified())
    if (this is DenseVector && other is SparseVector) return koblas.sparseKernels.dot(other, densified())
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
    is DenseVector -> koblas.vectorKernels.nrm2(data, offset, size, stride)
    is SparseVector -> koblas.sparseKernels.nrm2(this)
    else -> euclideanNorm(toDoubleArray(), 0, 1, size)
}

/**
 * Plain sum of the entries. Distinct from [asum], which sums their absolute values.
 *
 * A dense operand reaches [DenseVectorKernels.sum]; any other storage sums its stored entries, since the ones it
 * does not store are zero and contribute nothing.
 */
public fun Vector.sum(): Double = when (this) {
    is DenseVector -> koblas.vectorKernels.sum(data, offset, size, stride)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += x }
        s
    }
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun Vector.asum(): Double = when (this) {
    is DenseVector -> koblas.vectorKernels.asum(data, offset, size, stride)

    is SparseVector -> koblas.sparseKernels.asum(this)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += abs(x) }
        s
    }
}

/**
 * Zero-based logical index of the maximum absolute value (BLAS `idamax`), or `-1` when empty.
 * Dense vectors use [DenseVectorKernels.iamax]; sparse vectors visit only their stored entries.
 * All-zero nonempty inputs return `0`, as do sparse inputs whose maximum is an implicit zero. Strided views
 * report their logical index rather than a backing-array offset.
 *
 * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is a
 * NaN, is the selected implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
 * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
 * instead. A caller that needs one of those answers specifically has to test for it rather than infer it from
 * this routine.
 */
public fun Vector.iamax(): Int {
    if (this is DenseVector) return koblas.vectorKernels.iamax(data, offset, size, stride)
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
 * A source sharing [dst]'s buffer is snapshotted before writing.
 *
 * One entry point for both dense spacings: [dst] is written through its own origin and step, so a contiguous
 * destination and a borrowed slice of a longer buffer are the same call.
 */
public fun copy(src: Vector, dst: DenseVector) {
    requireSameSize(src.size, dst.size)
    val source = src.stableFor(dst)
    if (source is SparseVector) {
        for (i in 0 until dst.size) dst[i] = 0.0
        source.forEachStored { i, v -> dst[i] = v }
        return
    }
    // Adjacent on both sides is a block move, which the platform does far better than a loop that bounds
    // checks every entry. Any other spacing has to be walked, and that walk is the general case below.
    if (source is DenseVector && source.stride == 1 && dst.stride == 1) {
        source.data.copyInto(dst.data, dst.offset, source.offset, source.offset + dst.size)
        return
    }
    source.forEachStored { i, v -> dst[i] = v }
}

/** Snapshots an input whose values would be overwritten through [destination] before the first write. */
private fun Vector.stableFor(destination: DenseVector): Vector = when (this) {
    is SparseVector -> if (values === destination.data) SparseVector.wrap(size, indices, values.copyOf()) else this
    is DenseVector -> if (data === destination.data) DenseVector.wrap(toDoubleArray()) else this
    else -> this
}

/** This vector's entries as a plain array, borrowing the backing one where its spacing already is that. */
private fun DenseVector.densified(): DoubleArray =
    if (offset == 0 && stride == 1 && data.size == size) data else toDoubleArray()

/**
 * Read [from] at [x]'s stored positions into [x] (Sparse BLAS `usga`), the inverse of [copy] from a sparse
 * source. [x] keeps its pattern, so a nonzero of [from] at an unstored position is not read.
 *
 * [from] is the adjacent shape rather than any dense one. A pattern indexes its operand directly, which is
 * what makes these routines worth having, and there is no increment to hand a stored position; the type says
 * so instead of a check that would refuse a caller at run time.
 */
public fun gather(x: SparseVector, from: ContiguousVector) {
    requireSameSize(x.size, from.size)
    koblas.sparseKernels.gather(x, from.data)
}

/** [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`). */
public fun gatherZero(x: SparseVector, from: ContiguousVector) {
    requireSameSize(x.size, from.size)
    koblas.sparseKernels.gatherZero(x, from.data)
}

/**
 * Exchange the contents of [a] and [b] (BLAS `dswap`), including rows and columns of a dense matrix.
 *
 * Overlapping operands are snapshotted; at a shared physical entry the final write is from [b].
 */
public fun swap(a: DenseVector, b: DenseVector) {
    requireSameSize(a.size, b.size)
    if (a.data === b.data) {
        val snapshotA = a.toDoubleArray()
        val snapshotB = b.toDoubleArray()
        for (i in 0 until a.size) a[i] = snapshotB[i]
        for (i in 0 until b.size) b[i] = snapshotA[i]
        return
    }
    koblas.vectorKernels.swap(a.data, a.offset, b.data, b.offset, a.size, a.stride, b.stride)
}

/**
 * `y = y + alpha * x`. A sparse `x` touches only the positions it stores. A borrowed [x] sharing the
 * destination buffer is snapshotted before writing.
 */
public fun DenseVector.axpy(alpha: Double, x: Vector) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    when (val source = x.stableFor(this)) {
        is DenseVector ->
            koblas.vectorKernels.axpy(data, offset, alpha, source.data, source.offset, size, stride, source.stride)

        else -> source.forEachStored { i, v -> this[i] += alpha * v }
    }
}

/** `v = alpha * v`. */
public fun DenseVector.scale(alpha: Double) {
    if (alpha == 1.0) return
    koblas.vectorKernels.scale(data, offset, alpha, size, stride)
}
