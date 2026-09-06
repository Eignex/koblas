package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.noSizeDependentManagedAllocation
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.FactorsNotExposed
import kotlin.math.sqrt

/**
 * The half of a CHOLMOD factorization that touches no FFI.
 *
 * Both platforms convert the factor the same way, read the same three things out of it, and answer [nnz] and
 * [rcond] in the same shape around a call that differs. Only those calls and the solves are platform work,
 * so those are what a platform supplies; everything derived from them is written here once.
 */
internal abstract class CholmodFactorizationBase : F64SparseFactorization {
    /** Whether the factor is an `L·Lᵀ` rather than an `L·D·Lᵀ`, read once so a copy converts the same way. */
    protected abstract val isLl: Boolean

    /** The owner every native call is anchored against; [NativeOwnership] says what that guarantees. */
    protected abstract val ownership: NativeOwnership

    /** CHOLMOD's converted factors, or null where the library would not produce them. */
    protected abstract fun extractFactors(): CholmodFactors?

    /** The stored entry count CHOLMOD reports for the factor, read inside [ownership]. */
    protected abstract fun readNzmax(): Int

    /** CHOLMOD's own reciprocal-condition estimate for the factor, read inside [ownership]. */
    protected abstract fun readRcond(): Double

    /**
     * `L` and the ordering, converted on the first read. CHOLMOD copies the factor to convert it, so a
     * caller who only solves never pays for this.
     */
    private val extracted: CholmodFactors by lazy {
        extractFactors() ?: throw FactorsNotExposed("native factors")
    }

    private val factors: CholmodFactors get() = ownership.anchoring { extracted }

    /** The lower triangular factor, as [CholmodFactors.lower] documents it for each kind. */
    internal val lowerFactor: F64SparseMatrix
        get() {
            requireCholmodFactors("l")
            return factors.lower(isLl)
        }

    /** The diagonal factor of an `L·D·Lᵀ`, which CHOLMOD stores as the diagonal of `L`. */
    internal val diagonalFactor: DoubleArray
        get() {
            requireCholmodFactors("d")
            return factors.diagonal()
        }

    /** The fill-reducing ordering CHOLMOD chose. */
    internal val ordering: IntArray
        get() {
            requireCholmodFactors("order")
            return factors.permutation.copyOf()
        }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noSizeDependentManagedAllocation

    override val nnz: Int get() = ownership.anchoring { if (singular) 0 else readNzmax() }

    /**
     * CHOLMOD's own estimate, which is the ratio of the smallest factor diagonal to the largest, squared for
     * an `L·Lᵀ`. The square root brings it back to the ratio this seam documents, so the number means the
     * same thing whichever backend produced it.
     */
    override val rcond: Double get() = ownership.anchoring {
        if (singular) {
            0.0
        } else {
            val estimate = readRcond()
            if (isLl) sqrt(estimate) else estimate
        }
    }

    override fun close(): Unit = ownership.close()
}
