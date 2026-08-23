package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization
import com.eignex.koblas.sparse.factorization.lu.NO_DROP

/**
 * Shared routing for a host sparse LU binding. Unsupported requests and an absent host library fall back to
 * koblas's portable factorization; a binding that must reject either condition overrides the corresponding
 * hook.
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
        if (dropTolerance != NO_DROP) return factorWithDropTolerance(a, equilibrate, dropTolerance)
        if (!nativeAvailable) return factorWhenUnavailable(a, equilibrate, dropTolerance)
        if (a.rows == 0 || a.nnz == 0) return portableFactor(a, equilibrate, dropTolerance)
        return factorNative(a, equilibrate)
    }

    /** Handles an absent native binding. */
    protected open fun factorWhenUnavailable(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization = portableFactor(a, equilibrate, dropTolerance)

    /** Handles a drop-tolerance request when the binding has no equivalent native control. */
    protected open fun factorWithDropTolerance(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double,
    ): F64SparseFactorization = portableFactor(a, equilibrate, dropTolerance)

    /** Factorizes a nonempty matrix without a drop tolerance through the native library. */
    protected abstract fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization

    /** The portable semantic fallback. */
    protected fun portableFactor(
        a: F64SparseMatrix,
        equilibrate: Boolean,
        dropTolerance: Double = NO_DROP,
    ): F64SparseFactorization = F64SparseLuFactorization.factorCsc(a, equilibrate, dropTolerance)
}
