package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.SingularSparseFactorization
import com.eignex.koblas.sparse.SparseLuFactorization
import com.eignex.koblas.sparse.basis.BasisSolver
import com.eignex.koblas.sparse.basis.BasisSolvers
import com.eignex.koblas.sparse.basis.ProductFormBasisSolver
import com.eignex.koblas.sparse.host.SparseDecompositionsAdapter
import com.eignex.koblas.sparse.host.f64EquilibrationScale
import com.eignex.koblas.sparse.host.f64ScaledValues

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
) : SparseDecompositionsAdapter(
    equilibrate = config.equilibrate,
    metadata = BackendMetadata(options = config.options.metadataOptions()),
),
    GeneralSparseLu,
    BasisSolvers {
    private val calls = HfactorCalls(config)

    override val name: String get() = BackendNames.HFACTOR

    /**
     * Below the general sparse LUs, which are better at the factorization this also offers, and it does not
     * need to outrank them to win the half it is here for: no other backend offers [basisSolver].
     */
    override val priority: Int get() = HOST_BACKEND_PRIORITY - 2
    final override val libraryScalesRows: Boolean get() = false

    final override val nativeAvailable: Boolean get() = calls.available

    /**
     * HFactor offers no row scaling of its own, which [libraryScalesRows] is what says: HiGHS scales the
     * model before the simplex ever reaches HFactor and hands it an already-scaled matrix, and the adapter
     * does the same here. Every call reaches HFactor whatever the flag says.
     */
    final override fun factorNative(a: SparseMatrix): SparseLuFactorization {
        val handle = calls.create(a.rows, a.cols, a.copyColumnPointers(), a.copyRowIndices(), a.values)
            ?: return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        // A square matrix is its own basis, slot t holding column t.
        if (calls.build(handle, IntArray(a.rows) { it }) != 0) {
            calls.free(handle)
            return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        return HfactorFactorization(a.rows, calls, handle)
    }

    /** The row factors this backend equilibrates with, or null when it was not asked to. */
    @OptIn(UnsafeKoblasApi::class)
    private fun equilibrationOf(a: SparseMatrix): DoubleArray? =
        if (equilibrate) f64EquilibrationScale(a.rows, a.rowIdx, a.values) else null

    /** [a]'s values under [scale], or its own array when there is nothing to apply. */
    @OptIn(UnsafeKoblasApi::class)
    private fun scaledValues(a: SparseMatrix, scale: DoubleArray?): DoubleArray =
        if (scale == null) a.values else f64ScaledValues(a.rowIdx, a.values, scale)

    /**
     * A basis solver over the columns of [a], at any size: the gate the general factorization answers to
     * weighs one factorization against a crossing into the library, which has nothing to say about a
     * caller that will pivot through the same factors thousands of times.
     */
    override fun basisSolver(a: SparseMatrix): BasisSolver {
        requireShape(a.rows <= a.cols) { "a basis needs ${a.rows} columns to choose from; a has ${a.cols}" }
        if (!nativeAvailable) return ProductFormBasisSolver(a, this)
        val scale = equilibrationOf(a)
        val handle = calls.create(a.rows, a.cols, a.copyColumnPointers(), a.copyRowIndices(), scaledValues(a, scale))
            ?: return ProductFormBasisSolver(a, this)
        return HfactorBasisSolver(a, calls, handle, scale)
    }
}
