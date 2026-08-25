package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import com.eignex.koblas.sparse.host.F64SparseLuAdapter

/** Sparse LU factorizations backed by KLU 2. */
public class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64SparseLuAdapter(config.factorizeMin) {
    private val calls = KluCalls(config)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val nativeAvailable: Boolean get() = calls.available

    override fun refactor(
        previous: F64SparseFactorization,
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization {
        requireSquare(a, "refactor")
        require(dropTolerance >= 0.0) { "dropTolerance must not be negative" }
        if (dropTolerance != NO_DROP || !nativeAvailable) return factor(a, equilibrate, dropTolerance)
        val reusable = previous as? KluFactorization ?: return factor(a, equilibrate, dropTolerance)
        return when (reusable.refactor(a, equilibrate)) {
            KluRefactorResult.Success -> reusable
            KluRefactorResult.Incompatible -> factor(a, equilibrate, dropTolerance)
            KluRefactorResult.Singular -> F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
    }

    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization {
        val factor = calls.factorize(a.rows, a.colPtr, a.rowIdx, a.values, equilibrate)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return KluFactorization(NOT_SINGULAR, factor, calls)
    }
}
