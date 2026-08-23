package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.NO_DROP

/**
 * Shared routing for a host sparse LU binding. A selected host backend never silently delegates a request to
 * the portable implementation.
 */
public abstract class F64HostSparseLuAdapter internal constructor() : F64SparseLu {
    /** Whether the binding resolved every symbol needed to factor and solve. */
    protected abstract val nativeAvailable: Boolean

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    final override fun factor(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization {
        requireSquare(a, "factor")
        require(dropTolerance >= 0.0) { "dropTolerance must not be negative" }
        require(dropTolerance == NO_DROP) { "$name does not support drop tolerance" }
        check(nativeAvailable) { "$name is not available" }
        return factorNative(a, equilibrate)
    }

    /** Factorizes a matrix without a drop tolerance through the native library. */
    protected abstract fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization
}
