package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.hostDispatchThresholds
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
 * @param metadata effective provider options exposed through structured diagnostics.
 */
public abstract class F64SparseDecompositionsAdapter protected constructor(
    factorizeMin: Int? = null,
    protected val equilibrate: Boolean = false,
    private val metadata: BackendMetadata = BackendMetadata(),
) : F64SparseDecompositions,
    F64RoutingBackend,
    BackendMetadataProvider {
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

    /**
     * The QR rides the general gate. Its own crossing is later, fitted at 368 stored entries on the JVM and
     * 296 on Kotlin/Native against gates of 128 and 0; the band between is given up.
     */
    private val qrGate = factorizeGate

    /** The portable factorization at this backend's own policy, for everything the library will not take. */
    protected val portable: F64ReferenceSparseDecompositions =
        F64ReferenceSparseDecompositions(equilibrate = equilibrate)

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    override val backendMetadata: BackendMetadata get() = metadata

    override fun route(query: F64RouteQuery): BackendRoute? = when (query) {
        is F64RouteQuery.SparseLu -> thresholdRoute(
            query,
            this,
            portable.name,
            DispatchGate(DispatchMetric.STORED_ENTRIES, query.storedEntries.toLong(), factorizeGate.toLong()),
            query.storedEntries >= factorizeGate,
        )

        is F64RouteQuery.SparseQr -> thresholdRoute(
            query,
            this,
            portable.name,
            DispatchGate(DispatchMetric.STORED_ENTRIES, query.storedEntries.toLong(), qrGate.toLong()),
            query.storedEntries >= qrGate,
        )

        else -> null
    }

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

    final override fun qr(a: F64SparseMatrix): F64SparseQrFactorization {
        requireShape(a.rows >= a.cols) {
            "qr: A is ${a.rows}x${a.cols}, which is wider than it is tall; factor its transpose instead"
        }
        if (a.nnz < qrGate || !nativeAvailable) {
            return portable.qr(a)
        }
        return qrNative(a)
    }

    /** Factorizes into `Q·R` natively, portably by default: SPQR is the one library here with a sparse QR. */
    protected open fun qrNative(a: F64SparseMatrix): F64SparseQrFactorization = portable.qr(a)
}
