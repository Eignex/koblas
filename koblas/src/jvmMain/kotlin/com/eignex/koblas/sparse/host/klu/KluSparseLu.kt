package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64HostSparseLuAdapter

/** Sparse LU factorizations backed by KLU 2. */
public class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64HostSparseLuAdapter() {
    private val calls = KluCalls(config)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val nativeAvailable: Boolean get() = calls.available

    override fun factorWhenUnavailable(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization = error("KLU 2 is not available")

    override fun factorWithDropTolerance(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization = throw IllegalArgumentException("KLU does not support drop tolerance")

    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization {
        val factor = calls.factorize(a.rows, a.colPtr, a.rowIdx, a.values, equilibrate)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return KluFactorization(NOT_SINGULAR, factor, calls)
    }
}
