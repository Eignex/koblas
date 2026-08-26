package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.hostDispatchThresholds
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.*

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
 * @property equilibrate whether this backend scales rows before factorizing and undoes it in the solves. It
 *   is settled once here rather than per call, since it is policy of a piece with the scaling each
 *   library's own settings already choose.
 */
public abstract class F64SparseDecompositionsAdapter protected constructor(
    factorizeMin: Int? = null,
    protected val equilibrate: Boolean = false,
) : F64SparseDecompositions {
    /** Whether the binding resolved every symbol needed to factor and solve. */
    protected abstract val nativeAvailable: Boolean

    private val thresholds = hostDispatchThresholds(factorize = factorizeMin, symmetricFactorize = factorizeMin)
    private val factorizeGate = thresholds.factorize

    /**
     * The gate the two symmetric factorizations use, which is later than [factorizeGate]: they need no
     * numerical pivoting and the portable ones stay ahead of a library well past where the general
     * factorization has crossed. A backend naming its own `factorizeMin` moves both, since a caller asking
     * for one gate means it.
     */
    private val symmetricGate = thresholds.symmetricFactorize

    /** The portable factorization at this backend's own policy, for everything the library will not take. */
    protected val portable: F64ReferenceSparseDecompositions =
        F64ReferenceSparseDecompositions(equilibrate = equilibrate)

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    final override fun factor(a: F64SparseMatrix): F64SparseFactorization {
        requireSquare(a, "factor")
        // A binding whose library is absent answers portably rather than throwing, so a caller reaching a
        // configured backend on a host without it gets the portable answer instead of an error.
        if (a.nnz < factorizeGate || !nativeAvailable) {
            return portable.factor(a)
        }
        return factorNative(a)
    }

    /** Factorizes a matrix through the native library, equilibrating when this backend is set to. */
    protected abstract fun factorNative(a: F64SparseMatrix): F64SparseFactorization

    final override fun cholesky(a: F64SparseMatrix): F64SparseFactorization {
        requireSquare(a, "cholesky")
        if (a.nnz < symmetricGate || !nativeAvailable) {
            return portable.cholesky(a)
        }
        return choleskyNative(a)
    }

    /**
     * Factorizes a symmetric positive-definite matrix through the native library. Most of these libraries are
     * unsymmetric LU and have none, so the default is the portable factorization: the seam carries every
     * sparse factorization, and a binding filling one half of it does not have to offer the rest.
     */
    protected open fun choleskyNative(a: F64SparseMatrix): F64SparseFactorization = portable.cholesky(a)

    final override fun ldl(a: F64SparseMatrix): F64SparseFactorization {
        requireSquare(a, "ldl")
        if (a.nnz < symmetricGate || !nativeAvailable) {
            return portable.ldl(a)
        }
        return ldlNative(a)
    }

    /** Factorizes a symmetric matrix into `L·D·Lᵀ` through the native library, portably by default. */
    protected open fun ldlNative(a: F64SparseMatrix): F64SparseFactorization = portable.ldl(a)
}
