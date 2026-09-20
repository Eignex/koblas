@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

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
            val d = values
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
    requireSameSize(size, other.size, "dot")
    if (this is DenseVector && other is DenseVector) {
        return koblas.vectorKernels.dot(values, offset, other.values, other.offset, size, stride, other.stride)
    }
    if (this is SparseVector && other is SparseVector) return koblas.sparseKernels.dot(this, other)
    if (this is SparseVector && other is DenseVector) return koblas.sparseKernels.dot(this, other.asContiguousArray())
    if (this is DenseVector && other is SparseVector) return koblas.sparseKernels.dot(other, asContiguousArray())
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
    is DenseVector -> koblas.vectorKernels.nrm2(values, offset, size, stride)
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
    is DenseVector -> koblas.vectorKernels.sum(values, offset, size, stride)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += x }
        s
    }
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun Vector.asum(): Double = when (this) {
    is DenseVector -> koblas.vectorKernels.asum(values, offset, size, stride)

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
    if (this is DenseVector) return koblas.vectorKernels.iamax(values, offset, size, stride)
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
    requireSameSize(src.size, dst.size, "copy")
    val source = src.stableFor(dst)
    if (source is SparseVector) {
        // A contiguous destination is one fill and one scatter through the indexed kernel; any other spacing
        // has no kernel to reach, because the sparse seam addresses a pattern and carries no increment.
        if (dst.isWholeArray) {
            dst.values.fill(0.0)
            koblas.sparseKernels.scatter(source, dst.values)
        } else {
            for (i in 0 until dst.size) dst[i] = 0.0
            source.forEachStored { i, v -> dst[i] = v }
        }
        return
    }
    // Adjacent on both sides is a block move, which the platform does far better than a loop that bounds
    // checks every entry. Any other spacing has to be walked, and that walk is the general case below.
    if (source is DenseVector && source.stride == 1 && dst.stride == 1) {
        source.values.copyInto(dst.values, dst.offset, source.offset, source.offset + dst.size)
        return
    }
    source.forEachStored { i, v -> dst[i] = v }
}

/** Snapshots an input whose values would be overwritten through [destination] before the first write. */
private fun Vector.stableFor(destination: DenseVector): Vector = when (this) {
    is SparseVector -> if (values === destination.values) SparseVector.wrap(size, indices, values.copyOf()) else this
    is DenseVector -> if (values === destination.values) DenseVector.wrap(toDoubleArray()) else this
    else -> this
}

/** Whether this vector spans its entire backing array in order. */
internal val DenseVector.isWholeArray: Boolean
    get() = offset == 0 && stride == 1 && values.size == size

/**
 * This vector's own array where it [isWholeArray], and a gathered copy where it is not.
 *
 * Keeps an ordinary call through these convenience paths from allocating; a window or a step still pays for
 * its gather, which is the cost of addressing it that way.
 */
internal fun DenseVector.asContiguousArray(): DoubleArray = if (isWholeArray) values else toDoubleArray()

/**
 * Read [from] at [x]'s stored positions into [x] (Sparse BLAS `usga`), the inverse of [copy] from a sparse
 * source. [x] keeps its pattern, so a nonzero of [from] at an unstored position is not read.
 *
 * [from] is the adjacent shape rather than any dense one. A pattern indexes its operand directly, which is
 * what makes these routines worth having, and there is no increment to hand a stored position; the type says
 * so instead of a check that would refuse a caller at run time.
 */
public fun gather(x: SparseVector, from: ContiguousVector) {
    requireSameSize(x.size, from.size, "gather")
    koblas.sparseKernels.gather(x, from.values)
}

/** [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`). */
public fun gatherZero(x: SparseVector, from: ContiguousVector) {
    requireSameSize(x.size, from.size, "gatherZero")
    koblas.sparseKernels.gatherZero(x, from.values)
}

/**
 * Exchange the contents of [a] and [b] (BLAS `dswap`), including rows and columns of a dense matrix.
 *
 * Overlapping operands are snapshotted; at a shared physical entry the final write is from [b].
 */
public fun swap(a: DenseVector, b: DenseVector) {
    requireSameSize(a.size, b.size, "swap")
    // Sharing a buffer is not the same as covering an entry of it. Two rows or columns of one matrix share
    // their array and overlap nowhere, which is the case this overload exists for, so it reaches the kernel.
    if (a.overlaps(b)) {
        val snapshotA = a.toDoubleArray()
        val snapshotB = b.toDoubleArray()
        for (i in 0 until a.size) a[i] = snapshotB[i]
        for (i in 0 until b.size) b[i] = snapshotA[i]
        return
    }
    koblas.vectorKernels.swap(a.values, a.offset, b.values, b.offset, a.size, a.stride, b.stride)
}

/**
 * `y = y + alpha * x`. A sparse `x` touches only the positions it stores. A borrowed [x] sharing the
 * destination buffer is snapshotted before writing.
 */
public fun DenseVector.axpy(alpha: Double, x: Vector) {
    requireSameSize(size, x.size, "axpy")
    if (alpha == 0.0) return
    when (val source = x.stableFor(this)) {
        is DenseVector ->
            koblas.vectorKernels.axpy(values, offset, alpha, source.values, source.offset, size, stride, source.stride)

        // The indexed sparse kernels walk the pattern, and have a vectorised form; the generic loop has
        // neither, so it is what a foreign Vector implementation gets rather than what a SparseVector does.
        is SparseVector -> if (isWholeArray) {
            koblas.sparseKernels.axpy(values, alpha, source)
        } else {
            source.forEachStored { i, v -> this[i] += alpha * v }
        }

        else -> source.forEachStored { i, v -> this[i] += alpha * v }
    }
}

/**
 * `v = alpha * v`.
 *
 * A zero multiplier zeroes the vector rather than necessarily multiplying through it. BLAS does not require
 * the operand to be read at all in that case, and a library that writes zeros directly turns an entry holding
 * a NaN, an infinity, or a negative zero into a positive zero instead of into the product. Which of the two a
 * caller sees is the selected implementation's, as it is for [iamax].
 */
public fun DenseVector.scale(alpha: Double) {
    if (alpha == 1.0) return
    koblas.vectorKernels.scale(values, offset, alpha, size, stride)
}
