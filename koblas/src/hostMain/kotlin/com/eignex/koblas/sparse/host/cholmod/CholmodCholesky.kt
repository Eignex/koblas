@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

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
    public fun factor(a: F64SparseMatrix): F64SparseFactorization? {
        val functions = loader.functions ?: return null
        val common = loader.common()
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
        if (minor < n) {
            CholmodFactorization.CholmodHandle(factor, common, functions).release()
            throw NotPositiveDefinite(minor, 0.0, "cholesky: CHOLMOD stopped at column $minor, which is not positive")
        }
        return CholmodFactorization(CholmodFactorization.CholmodHandle(factor, common, functions), functions, n)
    }
}

/** The CHOLMOD that ships beside a SuiteSparse library already loaded from [libraryPath]. */
internal fun suiteSparseCholesky(libraryPath: String?, factorizeMin: Int?): CholmodCholesky = CholmodCholesky(
    CholmodConfig(searchDirectory = libraryPath?.parentDirectory(), factorizeMin = factorizeMin),
)

/** The directory part of a path, or null when it names no directory to look in. */
private fun String.parentDirectory(): String? = substringBeforeLast('/', missingDelimiterValue = "").ifEmpty { null }
