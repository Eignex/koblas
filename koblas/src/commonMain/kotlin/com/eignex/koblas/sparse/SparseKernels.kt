package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.internal.numeric.euclideanNorm

/** Sparse vector-vector kernel contract. */
public interface SparseKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
    public fun dot(x: SparseVector, y: DoubleArray): Double

    /**
     * `xᵀ·y` for two sparse vectors, merging their index lists in one pass. Relies on both operands being
     * strictly ascending, which `SparseVector` validates.
     */
    public fun dot(x: SparseVector, y: SparseVector): Double

    /** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`), touching only x's stored
     *  positions. */
    public fun axpy(y: DoubleArray, alpha: Double, x: SparseVector)

    /** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
     *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
    public fun scatter(x: SparseVector, out: DoubleArray)

    /**
     * Read [from] at [x]'s stored positions into [x]'s values (Sparse BLAS `usga`), the inverse of
     * [scatter]. The pattern is what [x] already stores and does not change; a position of [from] that is
     * nonzero and unstored stays unread, so this narrows a dense vector to a pattern rather than sparsifying
     * it.
     */
    public fun gather(x: SparseVector, from: DoubleArray)

    /**
     * [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`), leaving the rest of [from]
     * alone. The pair is one pass rather than two because the caller of a gather-and-zero wants [from]
     * emptied of exactly what it took.
     */
    public fun gatherZero(x: SparseVector, from: DoubleArray)

    /** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
    public fun nrm2(x: SparseVector): Double

    /** `Sum |x_i|` over the stored entries. */
    public fun asum(x: SparseVector): Double
}
