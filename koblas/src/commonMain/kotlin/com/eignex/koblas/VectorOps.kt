@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.dense.euclideanNorm
import com.eignex.koblas.sparse.*
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
        return koblas.dot(values, offset, other.values, other.offset, size, stride, other.stride)
    }
    if (this is SparseVector && other is SparseVector) return koblas.dot(this, other)
    if (this is SparseVector && other is DenseVector && other.isWholeArray) {
        return koblas.dot(this, other.values)
    }
    if (this is DenseVector && other is SparseVector && isWholeArray) return koblas.dot(other, values)
    // A view needs only the entries the sparse operand stores; gathering its whole logical extent would
    // make both the scratch and the reads scale with size rather than with the stored support.
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
    is DenseVector -> koblas.nrm2(values, offset, size, stride)
    is SparseVector -> koblas.nrm2(this)
    else -> euclideanNorm(toDoubleArray(), 0, 1, size)
}

/**
 * Plain sum of the entries. Distinct from [asum], which sums their absolute values.
 *
 * A dense operand reaches [KoblasEngine.sum]; any other storage sums its stored entries, since the ones it
 * does not store are zero and contribute nothing.
 */
public fun Vector.sum(): Double = when (this) {
    is DenseVector -> koblas.sum(values, offset, size, stride)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += x }
        s
    }
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun Vector.asum(): Double = when (this) {
    is DenseVector -> koblas.asum(values, offset, size, stride)

    is SparseVector -> koblas.asum(this)

    else -> {
        var s = 0.0
        forEachStored { _, x -> s += abs(x) }
        s
    }
}

/**
 * Zero-based logical index of the maximum absolute value (BLAS `idamax`), or `-1` when empty.
 * Dense vectors use [KoblasEngine.iamax]; sparse vectors visit only their stored entries.
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
    if (this is DenseVector) return koblas.iamax(values, offset, size, stride)
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
 * `this = src` (BLAS `dcopy`). A sparse source zero-fills this vector first, so nothing survives.
 * Identical dense mappings are a no-op. Portable contiguous copies use an overlap-safe block move;
 * other overlapping sources are snapshotted before writing.
 * Custom [Vector] implementations are also snapshotted, since their backing storage is opaque.
 *
 * One entry point for both dense spacings: this vector is written through its own origin and step, so a
 * contiguous destination and a borrowed slice of a longer buffer are the same call.
 */
public fun DenseVector.copyFrom(src: Vector) {
    if (src is DenseVector) return koblas.copy(src, this)
    requireSameSize(src.size, size, "copy")
    val source = src.stableFor(this)
    if (source is SparseVector) {
        // Only contiguous destinations can use the indexed kernel, whose seam carries no increment.
        if (isWholeArray) {
            values.fill(0.0)
            koblas.scatter(source, values)
        } else {
            for (i in 0 until size) this[i] = 0.0
            source.forEachStored { i, v -> this[i] = v }
        }
        return
    }
    koblas.copy(source as DenseVector, this)
}

/** Snapshots partial aliases and opaque storage, which may read [destination] without exposing its buffer. */
private fun Vector.stableFor(destination: DenseVector): Vector = when (this) {
    is SparseVector -> if (values === destination.values) SparseVector.wrap(size, indices, values.copyOf()) else this
    is DenseVector -> stableFor(destination)
    else -> DenseVector.wrap(toDoubleArray())
}

/**
 * Read [from] at this vector's stored positions into it (Sparse BLAS `usga`), the inverse of [copyFrom] from
 * a sparse source. This vector keeps its pattern, so a nonzero of [from] at an unstored position is not read.
 *
 * [from] is the adjacent shape rather than any dense one. A pattern indexes its operand directly, which is
 * what makes these routines worth having, and there is no increment to hand a stored position; the type says
 * so instead of a check that would refuse a caller at run time.
 */
public fun SparseVector.gather(from: ContiguousVector) {
    requireSameSize(size, from.size, "gather")
    koblas.gather(this, from.values)
}

/** [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`). */
public fun SparseVector.gatherZero(from: ContiguousVector) {
    requireSameSize(size, from.size, "gatherZero")
    koblas.gatherZero(this, from.values)
}

/**
 * Exchange the contents of this vector and [other] (BLAS `dswap`), including rows and columns of a dense
 * matrix.
 *
 * Identical mappings are a no-op. Other overlapping operands are snapshotted; at a shared physical entry
 * the final write is from [other].
 */
public fun DenseVector.swap(other: DenseVector): Unit = koblas.swap(this, other)

/**
 * Applies a plane rotation to this vector `x` and [y]: `(x(i), y(i)) = (c*x(i) + s*y(i), c*y(i) - s*x(i))`
 * (BLAS `drot`).
 * Negative strides are supported. Overlaps are snapshotted; at a shared entry the final write is from [y].
 */
public fun DenseVector.rot(y: DenseVector, c: Double, s: Double): Unit = koblas.rot(this, y, c, s)

/**
 * `y = y + alpha * x`. A sparse `x` touches only the positions it stores. A dense [x] with an identical
 * mapping is read directly; other dense overlaps, or sparse values sharing the destination buffer, are
 * snapshotted before writing.
 * Custom [Vector] implementations are also snapshotted, since their backing storage is opaque.
 */
public fun DenseVector.axpy(alpha: Double, x: Vector) {
    if (x is DenseVector) return koblas.axpy(alpha, x, this)
    requireSameSize(size, x.size, "axpy")
    if (alpha == 0.0) return
    when (val source = x.stableFor(this)) {
        is DenseVector -> koblas.axpy(alpha, source, this)

        // SparseVector uses indexed kernels; foreign Vector implementations use the generic loop.
        is SparseVector -> if (isWholeArray) {
            koblas.axpy(values, alpha, source)
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
    koblas.scale(values, offset, alpha, size, stride)
}
