package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NOT_SINGULAR
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
 */
public class CholmodCholesky(config: CholmodConfig = CholmodConfig()) {
    private val calls = CholmodCalls(config)

    /** Whether libcholmod opened and every symbol this needs bound. */
    public val isAvailable: Boolean get() = calls.available

    /** Why CHOLMOD is unusable, or null when it is usable. For diagnostics, not control flow. */
    public val unavailableReason: String? get() = calls.unavailableReason

    /**
     * Factor [a]'s lower triangle, or null when the library is unusable, which lets the caller fall back
     * rather than fail.
     *
     * @throws NotPositiveDefinite at the column CHOLMOD stopped at, matching the portable Cholesky.
     */
    public fun factor(a: F64SparseMatrix): F64SparseCholeskyFactorization? {
        val factor = CholmodMatrix.lowerTriangleOf(a).use { calls.factorize(it) } ?: return null
        if (factor.minor < factor.n) {
            val column = factor.minor
            calls.free(factor)
            throw NotPositiveDefinite(column, 0.0, "cholesky: CHOLMOD stopped at column $column, which is not positive")
        }
        return CholmodCholeskyFactorization(CholmodFactorization(factor, calls))
    }

    /**
     * Factor [a]'s lower triangle into `L·D·Lᵀ`, or null when the library is unusable.
     *
     * A zero pivot comes back as a factorization reporting `singular` at its column rather than raising,
     * since an `L·D·Lᵀ` failing means the matrix is singular where an `L·Lᵀ` failing only means it was not
     * positive definite.
     */
    public fun factorQuasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization? {
        val factor = CholmodMatrix.lowerTriangleOf(a).use { calls.factorize(it, ldl = true) } ?: return null
        val failedAt = if (factor.minor < factor.n) factor.minor else NOT_SINGULAR
        return CholmodLdlFactorization(CholmodFactorization(factor, calls, failedAt))
    }
}
