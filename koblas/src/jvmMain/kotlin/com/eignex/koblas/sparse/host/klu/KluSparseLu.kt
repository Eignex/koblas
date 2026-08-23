package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization
import com.eignex.koblas.sparse.factorization.lu.NO_DROP

/** Sparse LU factorizations backed by KLU 2. */
public class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64SparseLu {
    private val calls = KluCalls(config.libraryPath)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val isAvailable: Boolean get() = calls.available
    override val isPortable: Boolean get() = false

    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization {
        requireSquare(a, "factor")
        require(dropTolerance == NO_DROP) { "KLU does not support drop tolerance" }
        if (a.rows == 0 || a.nnz == 0) return F64SparseLuFactorization.factorCsc(a, equilibrate)
        check(calls.available) { "KLU 2 is not available" }
        val factor = calls.factorize(a.rows, a.colPtr, a.rowIdx, a.values, equilibrate)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return KluFactorization(NOT_SINGULAR, factor, calls)
    }
}
