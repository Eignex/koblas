package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.euclideanNorm

/** Sparse vector-vector routines as a backend half. */
public interface F64SparseVectorKernels : Backend {
    /** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
    public fun dot(x: F64SparseVector, y: DoubleArray): Double

    /**
     * `xᵀ·y` for two sparse vectors, merging their index lists in one pass. Relies on both operands being
     * strictly ascending, which `F64SparseVector` validates.
     */
    public fun dot(x: F64SparseVector, y: F64SparseVector): Double

    /** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`), touching only x's stored
     *  positions. */
    public fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector)

    /** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
     *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
    public fun scatter(x: F64SparseVector, out: DoubleArray)

    /** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
    public fun nrm2(x: F64SparseVector): Double

    /** `Sum |x_i|` over the stored entries. */
    public fun asum(x: F64SparseVector): Double
}
