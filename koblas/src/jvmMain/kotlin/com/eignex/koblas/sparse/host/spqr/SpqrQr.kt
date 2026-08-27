package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.cholmod.CholmodMatrix
import java.io.File

/**
 * SPQR's sparse QR, the routine a SuiteSparse backend fills its QR half with rather than a backend of its
 * own: SPQR has no LU and no Cholesky, so one offering it alone would take the seam from one offering all.
 */
public class SpqrQr(config: SpqrConfig = SpqrConfig()) {
    private val calls = SpqrCalls(config)
    private val ordering = config.options.ordering.name.lowercase()

    /** Whether both libraries opened and every symbol resolved. */
    public val isAvailable: Boolean get() = calls.available

    /** Why SPQR is unusable, or null when it is usable. */
    public val unavailableReason: String? get() = calls.unavailableReason

    /**
     * Factor [a], or null when the library is unusable.
     *
     * @throws IllegalArgumentException if [a] has fewer rows than columns.
     */
    public fun factor(a: F64SparseMatrix): F64SparseQrFactorization? {
        requireShape(a.rows >= a.cols) {
            "qr: A is ${a.rows}x${a.cols}, which is wider than it is tall; factor its transpose instead"
        }
        val factor = CholmodMatrix.generalOf(a).use { calls.factorize(it, a.rows, a.cols) } ?: return null
        return SpqrQrFactorization(factor, a.rows, a.cols, ordering)
    }
}

internal fun suiteSparseQr(libraryPath: String?): SpqrQr =
    SpqrQr(SpqrConfig(searchDirectory = libraryPath?.let { File(it).parent }))
