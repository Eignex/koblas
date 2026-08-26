package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter

/**
 * Sparse LU, hypersparse solves, and Forrest-Tomlin basis updates backed by HiGHS's HFactor.
 *
 * What HFactor is for is [basisSolver]: a basis named by index into a matrix that outlives it, solved
 * through vectors that stay sparse. [factor] is the plainer surface, HFactor taking a square matrix as a
 * basis of its own columns.
 *
 * Open so a bundled provider is one of these under another name rather than a wrapper around one, as the
 * other host bindings are. Only the name and the ranking are meant to vary.
 */
public open class HfactorSparseLu(
    /** Policy for this backend instance. */
    public val config: HfactorConfig = HfactorConfig(),
) : F64SparseDecompositionsAdapter(config.factorizeMin),
    F64BasisSolvers {
    private val calls = HfactorCalls(config)

    override val name: String get() = BackendNames.HFACTOR

    /**
     * Below the general sparse LUs, which are better at the factorization this also offers, and it does not
     * need to outrank them to win the half it is here for: no other backend offers [basisSolver].
     */
    override val priority: Int get() = HOST_BACKEND_PRIORITY - 2
    final override val nativeAvailable: Boolean get() = calls.available

    /**
     * HFactor offers no row scaling, so a backend set to equilibrate stays with the portable factorization
     * rather than reproducing it here by scaling the values on the way in.
     */
    final override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        if (equilibrate) return portable.factor(a)
        val handle = calls.create(a.rows, a.cols, a.copyColumnPointers(), a.copyRowIndices(), a.values)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        // A square matrix is its own basis, slot t holding column t.
        if (calls.build(handle, IntArray(a.rows) { it }) != 0) {
            calls.free(handle)
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        return HfactorFactorization(a.rows, calls, handle)
    }

    /**
     * A basis solver over the columns of [a], at any size: the gate the general factorization answers to
     * weighs one factorization against a crossing into the library, which has nothing to say about a
     * caller that will pivot through the same factors thousands of times.
     */
    override fun basisSolver(a: F64SparseMatrix): F64BasisSolver {
        requireShape(a.rows <= a.cols) { "a basis needs ${a.rows} columns to choose from; a has ${a.cols}" }
        if (!nativeAvailable) return F64ProductFormBasisSolver(a, this)
        val handle = calls.create(a.rows, a.cols, a.copyColumnPointers(), a.copyRowIndices(), a.values)
            ?: return F64ProductFormBasisSolver(a, this)
        return HfactorBasisSolver(a, calls, handle)
    }
}
