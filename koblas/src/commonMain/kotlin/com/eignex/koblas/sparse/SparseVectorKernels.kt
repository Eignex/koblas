package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseVector
import com.eignex.koblas.euclideanNorm
import com.eignex.koblas.requireShape
import kotlin.math.abs

/** Sparse vector-vector routines as a backend half. */
public interface SparseVectorKernels : Backend {
    /** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
    public fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) s += vals[k] * y[idx[k]]
        return s
    }

    /**
     * `xᵀ·y` for two sparse vectors, merging their index lists in one pass. Relies on both operands being
     * strictly ascending, which `SparseVector` validates.
     */
    public fun dot(x: SparseVector, y: SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
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

    /** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`), touching only x's stored
     *  positions. */
    public fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) y[idx[k]] += alpha * vals[k]
    }

    /** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
     *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
    public fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) out[idx[k]] = vals[k]
    }

    /** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
    public fun nrm2(x: SparseVector): Double = euclideanNorm(x.values, 0, x.values.size)

    /** `Sum |x_i|` over the stored entries. */
    public fun asum(x: SparseVector): Double {
        var s = 0.0
        for (v in x.values) s += abs(v)
        return s
    }
}
