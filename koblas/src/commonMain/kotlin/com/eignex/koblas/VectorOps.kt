@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.storage.copyVectorStorage

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
            for (k in offset until offset + nnz) block(idx[k], vals[k])
        }

        else -> for (i in 0 until size) block(i, this[i])
    }
}

/**
 * `aT * b` through the default [koblas] engine's typed [KoblasEngine.dot].
 * Sparse operands visit stored entries only; custom vectors use portable readers.
 */
public infix fun Vector.dot(other: Vector): Double = koblas.dotVector(this, other)

/**
 * Vector norm of [type]: [Norm.One] sums absolute values, [Norm.Two] is the Euclidean norm, and
 * [Norm.Infinity] and [Norm.MaxAbs] select the maximum absolute value. An empty vector has norm zero.
 *
 * Uses the default [koblas] engine for dense and packed sparse storage, preserving dense offsets and
 * strides. Sparse vectors visit only stored values; custom vectors use portable fallbacks. [Norm.Two]
 * rescales to avoid unnecessary overflow and underflow. Rounding and non-finite behavior follow the
 * selected implementation, including [KoblasEngine.iamax] for the infinity norm over built-in storage.
 *
 * @throws IllegalArgumentException if [type] is [Norm.Frobenius], which is a matrix norm.
 */
public fun Vector.norm(type: Norm = Norm.Two): Double = koblas.normVector(this, type)

/**
 * Plain sum of the entries. Distinct from [asum], which sums their absolute values.
 *
 * Dense and sparse storage reach [KoblasEngine.sum], with sparse vectors summing only their packed stored
 * entries. Other implementations are read through [Vector.get].
 */
public fun Vector.sum(): Double = koblas.sumVector(this)

/**
 * Sum of absolute values (BLAS `dasum`) through the default [koblas] engine's [KoblasEngine.asum].
 * Sparse vectors sum over stored entries only; custom vectors use portable readers.
 */
public fun Vector.asum(): Double = koblas.absoluteSumVector(this)

/**
 * An independent copy of the logical entries. Sparse storage keeps its pattern, including stored zeros;
 * dense views and custom vectors produce owning dense storage. No backing arrays are shared.
 */
public fun Vector.copy(): Vector = copyVectorStorage(this)

/**
 * Copies [source] into this vector. Built-in containers use the default [koblas] engine's [KoblasEngine.copy].
 * Requires equal sizes before writing. Inputs may share storage; overlapping or opaque sources are
 * snapshotted first. Dense views retain their offsets and strides.
 *
 * Sparse destinations retain a sparse source's pattern and stored zeros. Other sources store nonzero
 * entries and negative zeros. Structural changes replace sparse buffers; equal patterns reuse them.
 */
public fun Vector.copyFrom(source: Vector): Unit = koblas.copyVector(source, this)
