package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization

/**
 * CHOLMOD's sparse Cholesky, as the routine a SuiteSparse backend fills its Cholesky half with.
 *
 * Not a backend itself. CHOLMOD has no LU, and a backend offering only half of what the seam carries would
 * take the seam from one that offers all of it; the SuiteSparse bindings hold this instead, so the
 * collection answers both routines natively however the seam resolves.
 *
 * Each target binds CHOLMOD its own way, through `java.lang.foreign` or through cinterop, and what they
 * agree on is declared here so a caller and the bindings that hold one see the same routine either way.
 *
 * @param config policy for this instance, including where to look for the library.
 */
public expect class CholmodCholesky(config: CholmodConfig = CholmodConfig()) {
    /** Whether libcholmod opened and every symbol this needs bound. */
    public val isAvailable: Boolean

    /**
     * Factor [a]'s lower triangle, or null when the library is unusable, which lets the caller fall back
     * rather than fail.
     *
     * @throws NotPositiveDefinite at the column CHOLMOD stopped at, matching the portable Cholesky.
     */
    public fun factor(a: F64SparseMatrix): F64SparseCholeskyFactorization?

    /**
     * Factor [a]'s lower triangle into `L·D·Lᵀ`, or null when the library is unusable.
     *
     * A zero pivot comes back as a factorization reporting `singular` at its column rather than raising,
     * since an `L·D·Lᵀ` failing means the matrix is singular where an `L·Lᵀ` failing only means it was not
     * positive definite.
     */
    public fun factorQuasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization?
}
