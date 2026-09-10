package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.requireHfactorShape
import com.eignex.koblas.sparse.SingularSparseFactorization
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.host.EquilibratedSparseLu
import com.eignex.koblas.sparse.host.f64EquilibrationScale
import com.eignex.koblas.sparse.host.f64ScaledValues

/**
 * Sparse LU, hypersparse solves, and Forrest-Tomlin basis updates backed by HiGHS's HFactor.
 *
 * What HFactor is for is [basisSolver]: a basis named by index into a matrix that outlives it, solved
 * through vectors that stay sparse. [factor] is the plainer surface, HFactor taking a square matrix as a
 * basis of its own columns.
 *
 * Construction is direct and does not register or alter koblas's BLAS engine. A null library path uses the
 * platform lookup chain; [com.eignex.koblas.hfactor.BundledHfactor] selects this same implementation from
 * this module's bundled resources.
 */
public open class HfactorSparseLu(
    /** Policy for this backend instance. */
    public val config: HfactorConfig = HfactorConfig(),
) {
    private val calls = HfactorCalls(config)

    /** The result of lazily loading and binding this implementation. */
    public val availability: HfactorAvailability
        get() = HfactorAvailability(calls.available, calls.unavailableReason)

    /**
     * HFactor offers no row scaling of its own. HiGHS scales the model before the simplex reaches HFactor,
     * and this adapter applies the same policy before building the native factors.
     */
    public fun factor(a: SparseMatrix): SparseFactorization {
        requireHfactorShape(a.rows == a.cols) { "factor: A is ${a.rows}x${a.cols}, expected square" }
        requireAvailable()
        val scale = equilibrationOf(a)
        val factored = factorNative(
            if (scale == null) {
                a
            } else {
                SparseMatrix.wrap(
                    a.rows,
                    a.cols,
                    a.copyColumnPointers(),
                    a.copyRowIndices(),
                    scaledValues(a, scale),
                )
            },
        )
        return if (scale == null || factored.singular) factored else EquilibratedSparseLu(factored, scale)
    }

    private fun factorNative(a: SparseMatrix): SparseFactorization {
        val handle = calls.create(a.rows, a.cols, a.copyColumnPointers(), a.copyRowIndices(), a.values)
            ?: return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        if (calls.build(handle, IntArray(a.rows) { it }) != 0) {
            calls.free(handle)
            return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        return HfactorFactorization(a.rows, calls, handle)
    }

    @OptIn(UnsafeKoblasApi::class)
    private fun equilibrationOf(a: SparseMatrix): DoubleArray? =
        if (config.equilibrate) f64EquilibrationScale(a.rows, a.rowIdx, a.values) else null

    @OptIn(UnsafeKoblasApi::class)
    private fun scaledValues(a: SparseMatrix, scale: DoubleArray?): DoubleArray =
        if (scale == null) a.values else f64ScaledValues(a.rowIdx, a.values, scale)

    /**
     * A basis solver over the columns of [a], at any size: the gate the general factorization answers to
     * weighs one factorization against a crossing into the library, which has nothing to say about a
     * caller that will pivot through the same factors thousands of times.
     */
    public fun basisSolver(a: SparseMatrix): HfactorBasisSolver {
        requireHfactorShape(a.rows <= a.cols) { "a basis needs ${a.rows} columns to choose from; a has ${a.cols}" }
        requireAvailable()
        val scale = equilibrationOf(a)
        val handle = checkNotNull(
            calls.create(a.rows, a.cols, a.copyColumnPointers(), a.copyRowIndices(), scaledValues(a, scale)),
        ) { "HFactor could not create a basis solver" }
        return HfactorBasisSolver(a, calls, handle, scale)
    }

    private fun requireAvailable() {
        val result = availability
        check(result.available) { "HFactor is unavailable: ${result.reason ?: "unknown reason"}" }
    }
}
