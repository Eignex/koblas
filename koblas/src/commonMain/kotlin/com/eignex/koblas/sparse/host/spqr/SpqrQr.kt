package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseQrFactorization

/**
 * SPQR's sparse QR, the routine a SuiteSparse backend fills its QR half with rather than a backend of its
 * own: SPQR has no LU and no Cholesky, so one offering it alone would take the seam from one offering all.
 *
 * @param config policy for this instance, including where to look for the library.
 */
public expect class SpqrQr(config: SpqrConfig = SpqrConfig()) {
    /** Whether both libraries opened and every symbol resolved. */
    public val isAvailable: Boolean

    /** Why SPQR is unusable, or null when it is usable. */
    public val unavailableReason: String?

    /**
     * Factor [a], or null when the library is unusable.
     *
     * @throws IllegalArgumentException if [a] has fewer rows than columns.
     */
    public fun factor(a: F64SparseMatrix): F64SparseQrFactorization?
}
