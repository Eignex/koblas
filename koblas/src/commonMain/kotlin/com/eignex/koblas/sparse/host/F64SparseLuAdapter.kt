package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.hostDispatchThresholds
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.NO_DROP

/**
 * Shared routing for a host sparse LU binding, the counterpart of the dense host adapters.
 *
 * Below the gate a request is answered by the portable factorization, since crossing into a native library
 * costs more than the work saves on a small problem. The gate is the shared factorization threshold, read
 * here as a count of stored entries rather than as a dimension, because that is what sparse work scales
 * with: an `n` of a thousand with a diagonal and little else is smaller work than a dense hundred.
 *
 * @param factorizeMin stored entries from which this binding factorizes natively, or null for the platform
 *   default.
 */
public abstract class F64SparseLuAdapter protected constructor(factorizeMin: Int? = null) : F64SparseLu {
    /** Whether the binding resolved every symbol needed to factor and solve. */
    protected abstract val nativeAvailable: Boolean

    private val factorizeGate = hostDispatchThresholds(factorize = factorizeMin).factorize

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    final override fun factor(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization {
        requireSquare(a, "factor")
        require(dropTolerance >= 0.0) { "dropTolerance must not be negative" }
        // Rejected before the gate is read, so which exception a caller gets does not turn on the size.
        require(dropTolerance == NO_DROP) { "$name does not support drop tolerance" }
        if (a.nnz < factorizeGate) return F64ReferenceSparseLinearAlgebra.factor(a, equilibrate, dropTolerance)
        check(nativeAvailable) { "$name is not available" }
        return factorNative(a, equilibrate)
    }

    /** Factorizes a matrix without a drop tolerance through the native library. */
    protected abstract fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization
}
