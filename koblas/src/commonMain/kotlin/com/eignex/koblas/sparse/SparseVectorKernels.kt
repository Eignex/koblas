package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseVector
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The sparse level-1 kernels: a sparse vector against a dense one, or against another sparse one.
 *
 * The sparse counterpart of `VectorKernels`, and a real standard rather than an invention — the BLAS
 * Technical Forum's Sparse BLAS defines this tier (`usdot`, `usaxpy`, gather and scatter), and it is the
 * shape a revised simplex prices in: one sparse column against a dense reduced-cost vector.
 *
 * Unlike `VectorKernels` there is no length threshold, and the difference is structural rather than an
 * oversight. Dense level-1 kernels are compiled per target, so consulting a backend has to beat a
 * *compiled-in primitive* and only pays above a length. These have no compile-time leaf to protect: the
 * default is an object either way, so dispatch is unconditional and the [sparseVectorKernels] accessor
 * never returns null.
 *
 * Defaults implement every routine over the ascending index arrays, so a backend overrides only what it
 * accelerates.
 */
interface SparseVectorKernels : Backend {
    /** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
    fun dot(x: SparseVector, y: DoubleArray): Double {
        require(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) s += vals[k] * y[idx[k]]
        return s
    }

    /**
     * `xᵀ·y` for two sparse vectors, merging their index lists in one pass — `O(nnz_x + nnz_y)`.
     *
     * Gathering instead, looking each stored position of one up in the other, would be
     * `O(nnz_x · log nnz_y)`. Both operands are strictly ascending, which is what makes the merge possible;
     * `SparseVector` validates that.
     */
    fun dot(x: SparseVector, y: SparseVector): Double {
        require(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        var a = 0
        var b = 0
        while (a < x.indices.size && b < y.indices.size) {
            val ia = x.indices[a]
            val ib = y.indices[b]
            when {
                ia < ib -> a++

                ia > ib -> b++

                else -> {
                    s += x.values[a] * y.values[b]
                    a++
                    b++
                }
            }
        }
        return s
    }

    /** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`); touches only `x`'s
     *  positions, which is the reason to pass a sparse operand at all. */
    fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        require(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) y[idx[k]] += alpha * vals[k]
    }

    /** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
     *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
    fun scatter(x: SparseVector, out: DoubleArray) {
        require(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) out[idx[k]] = vals[k]
    }

    /**
     * Euclidean norm over the stored entries.
     *
     * Rescales when the plain sum of squares overflows or underflows, as the dense kernel does — a stored
     * entry near `1e±150` is no less likely for being sparse.
     */
    fun nrm2(x: SparseVector): Double {
        val vals = x.values
        var s = 0.0
        for (v in vals) s += v * v
        if (s.isFinite() && s >= MIN_NORMAL) return sqrt(s)
        var amax = 0.0
        for (v in vals) {
            val a = abs(v)
            if (a > amax) amax = a
        }
        if (amax == 0.0 || amax.isInfinite()) return sqrt(s)
        var t = 0.0
        for (v in vals) {
            val r = v / amax
            t += r * r
        }
        return amax * sqrt(t)
    }

    /** `Sum |x_i|` over the stored entries. */
    fun asum(x: SparseVector): Double {
        var s = 0.0
        for (v in x.values) s += abs(v)
        return s
    }
}

/** Smallest normal double; a squares-sum below this has lost precision to underflow. */
private const val MIN_NORMAL = 2.2250738585072014e-308
