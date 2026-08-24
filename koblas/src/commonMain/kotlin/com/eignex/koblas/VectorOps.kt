@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.requireSameSize
import com.eignex.koblas.sparse.F64SparseVectorKernels
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

/**
 * The entries as a flat array, the backing itself when the vector already is one and a densified copy
 * otherwise. For a routine that reads every position and so cannot walk stored entries.
 *
 * The result may be the caller's own buffer, and may therefore alias another argument, so a caller must
 * neither write through it nor read it after writing whatever else it was handed.
 */
internal fun F64VectorLike.denseEntries(): DoubleArray = if (this is F64DenseVector) data else toDoubleArray()

/** `aT * b`. Any sparse operand goes through [F64SparseVectorKernels], walking the stored entries only. */
public infix fun F64VectorLike.dot(other: F64VectorLike): Double {
    requireSameSize(size, other.size)
    if (this is F64DenseVector && other is F64DenseVector) {
        return koblas.vectorKernels.dot(data, 0, other.data, 0, size)
    }
    if (this is F64SparseVector && other is F64SparseVector) return koblas.sparseVectorKernels.dot(this, other)
    if (this is F64SparseVector && other is F64DenseVector) return koblas.sparseVectorKernels.dot(this, other.data)
    if (this is F64DenseVector && other is F64SparseVector) return koblas.sparseVectorKernels.dot(other, data)
    var s = 0.0
    for (i in 0 until size) s += this[i] * other[i]
    return s
}

/**
 * Euclidean norm (BLAS `dnrm2`). Rescales when the sum of squares would overflow or underflow, so any
 * finite input gives the correct norm.
 */
public fun F64VectorLike.norm2(): Double = when (this) {
    is F64DenseVector -> koblas.vectorKernels.nrm2(data, 0, size)
    is F64SparseVector -> koblas.sparseVectorKernels.nrm2(this)
    else -> euclideanNorm(toDoubleArray(), 0, size)
}

/** Sum of absolute values (BLAS `dasum`). Sparse vectors sum over stored entries only. */
public fun F64VectorLike.asum(): Double = when (this) {
    is F64DenseVector -> koblas.vectorKernels.asum(data, 0, size)

    is F64SparseVector -> koblas.sparseVectorKernels.asum(this)

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
            koblas.sparseVectorKernels.scatter(src, dst.data)
        }

        else -> {
            dst.data.fill(0.0)
            src.forEachStored { i, v -> dst.data[i] = v }
        }
    }
}

/** Exchange the contents of [a] and [b] (BLAS `dswap`). */
public fun swap(a: F64DenseVector, b: F64DenseVector) {
    requireSameSize(a.size, b.size)
    val ad = a.data
    val bd = b.data
    for (i in ad.indices) {
        val t = ad[i]
        ad[i] = bd[i]
        bd[i] = t
    }
}

/** `y = y + alpha * x`. A sparse `x` touches only the positions it stores. */
public fun F64DenseVector.axpy(alpha: Double, x: F64VectorLike) {
    requireSameSize(size, x.size)
    if (alpha == 0.0) return
    when (x) {
        is F64DenseVector -> koblas.vectorKernels.axpy(data, 0, alpha, x.data, 0, size)
        is F64SparseVector -> koblas.sparseVectorKernels.axpy(data, alpha, x)
        else -> x.forEachStored { i, v -> data[i] += alpha * v }
    }
}

/** `v = alpha * v`. */
public fun F64DenseVector.scale(alpha: Double) {
    if (alpha == 1.0) return
    koblas.vectorKernels.scale(data, 0, alpha, size)
}
