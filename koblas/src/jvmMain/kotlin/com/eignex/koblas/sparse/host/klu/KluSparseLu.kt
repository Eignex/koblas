package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64SparseLuAdapter

/** Sparse LU factorizations backed by KLU 2. */
public class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64SparseLuAdapter(config.factorizeMin, config.equilibrate) {
    private val calls = KluCalls(config)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val nativeAvailable: Boolean get() = calls.available

    /**
     * Factor [a] reusing the symbolic analysis behind [previous], which supersedes it and must not be
     * solved after this call. Reuse is what KLU is for: a circuit keeps one sparsity pattern and factors it
     * again for every operating point, and the analysis is the expensive half.
     *
     * A [previous] from another backend, or one whose pattern does not match, is answered as [factor]
     * would. This is KLU's own routine rather than a seam method, since no other sparse LU koblas binds
     * carries an analysis worth reusing.
     */
    public fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization {
        requireSquare(a, "refactor")
        if (!nativeAvailable) return factor(a)
        val reusable = previous as? KluFactorization ?: return factor(a)
        return when (reusable.refactor(a, equilibrate)) {
            KluRefactorResult.Success -> reusable
            KluRefactorResult.Incompatible -> factor(a)
            KluRefactorResult.Singular -> F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
    }

    override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        val factor = calls.factorize(a.rows, a.colPtr, a.rowIdx, a.values, equilibrate)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return KluFactorization(NOT_SINGULAR, factor, calls)
    }
}
