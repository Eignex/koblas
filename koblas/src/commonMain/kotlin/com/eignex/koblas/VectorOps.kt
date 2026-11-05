@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.sparse.*

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
public infix fun Vector.dot(other: Vector): Double = koblas.dotVector(this, other)

/**
 * Euclidean norm (BLAS `dnrm2`). Rescales when the sum of squares would overflow or underflow, so any
 * finite input gives the correct norm.
 */
public fun Vector.norm2(): Double = koblas.normVector(this)

/**
 * Plain sum of the entries. Distinct from [asum], which sums their absolute values.
 *
 * Dense and sparse storage reach [KoblasEngine.sum], with sparse vectors summing only their packed stored
 * entries. Other implementations are read through [Vector.get].
 */
public fun Vector.sum(): Double = koblas.sumVector(this)

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun Vector.asum(): Double = koblas.absoluteSumVector(this)

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
public fun Vector.iamax(): Int = koblas.maxAbsVector(this)

/**
 * `this = src` (BLAS `dcopy`). A sparse source zero-fills this vector first, so nothing survives.
 * Identical dense mappings are a no-op. Portable contiguous copies use an overlap-safe block move;
 * other overlapping sources are snapshotted before writing.
 * Custom [Vector] implementations are also snapshotted, since their backing storage is opaque.
 *
 * One entry point for both dense spacings: this vector is written through its own origin and step, so a
 * contiguous destination and a borrowed slice of a longer buffer are the same call.
 */
public fun DenseVector.copyFrom(src: Vector): Unit = koblas.copyVector(src, this)

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
public fun DenseVector.axpy(alpha: Double, x: Vector): Unit = koblas.axpyVector(alpha, x, this)

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
