@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseFactorization
import kotlinx.cinterop.*

/**
 * CHOLMOD's sparse Cholesky, as the routine a SuiteSparse backend fills its Cholesky half with.
 *
 * Not a backend itself. CHOLMOD has no LU, and a backend offering only half of what the seam carries would
 * take the seam from one that offers all of it; the SuiteSparse bindings hold this instead, so the
 * collection answers both routines natively however the seam resolves.
 */
public class CholmodCholesky(config: CholmodConfig = CholmodConfig()) {
    private val loader = CholmodLoader(config)

    /** Whether libcholmod opened and every symbol this needs resolved. */
    public val isAvailable: Boolean get() = loader.available

    /**
     * Factor [a]'s lower triangle, or null when the library is unusable, which lets the caller fall back
     * rather than fail.
     *
     * @throws NotPositiveDefinite at the column CHOLMOD stopped at, matching the portable Cholesky.
     */
    public fun factor(a: F64SparseMatrix): F64SparseFactorization? = build(a, CHOLMOD_TRUE) { minor, n ->
        if (minor < n) {
            throw NotPositiveDefinite(minor, 0.0, "cholesky: CHOLMOD stopped at column $minor, which is not positive")
        }
        NOT_SINGULAR
    }

    /**
     * Factor [a]'s lower triangle into `L·D·Lᵀ`, or null when the library is unusable.
     *
     * A zero pivot comes back as a factorization reporting `singular` at its column rather than raising,
     * since an `L·D·Lᵀ` failing means the matrix is singular where an `L·Lᵀ` failing only means it was not
     * positive definite.
     */
    public fun factorLdl(a: F64SparseMatrix): F64SparseFactorization? =
        build(a, finalLl = 0) { minor, n -> if (minor < n) minor else NOT_SINGULAR }

    private inline fun build(
        a: F64SparseMatrix,
        finalLl: Int,
        verdict: (minor: Int, n: Int) -> Int,
    ): F64SparseFactorization? {
        val functions = loader.functions ?: return null
        val common = loader.common(finalLl)
        val sparse = describeLowerTriangle(a)
        val factor = try {
            functions.analyze(
                sparse,
                common,
            ).also { if (it != null) functions.factorize(sparse, it.reinterpret(), common) }
        } finally {
            freeMatrix(sparse)
        }
        if (factor == null) {
            nativeHeap.free(common)
            return null
        }
        val block = factor.reinterpret<ByteVar>()
        val n = sizeAt(block, CHOLMOD_FACTOR_N).toInt()
        val minor = sizeAt(block, CHOLMOD_FACTOR_MINOR).toInt()
        val handle = CholmodFactorization.CholmodHandle(factor, common, functions, n)
        val failedAt = try {
            verdict(minor, n)
        } catch (raised: NotPositiveDefinite) {
            handle.release()
            throw raised
        }
        return CholmodFactorization(handle, functions, n, failedAt)
    }
}

/** The CHOLMOD that ships beside a SuiteSparse library already loaded from [libraryPath]. */
internal fun suiteSparseCholesky(libraryPath: String?): CholmodCholesky =
    CholmodCholesky(CholmodConfig(searchDirectory = libraryPath?.parentDirectory()))

/** The directory part of a path, or null when it names no directory to look in. */
private fun String.parentDirectory(): String? = substringBeforeLast('/', missingDelimiterValue = "").ifEmpty { null }
