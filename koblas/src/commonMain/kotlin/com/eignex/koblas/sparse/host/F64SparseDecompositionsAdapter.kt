package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.*

/**
 * Shared routing for a host sparse LU binding, the counterpart of the dense host adapters.
 * @property equilibrate whether this backend scales rows before factorizing and undoes it in the solves. It
 *   is settled once here rather than per call, since it is policy of a piece with the scaling each
 *   library's own settings already choose.
 * @param metadata effective provider options exposed through structured diagnostics.
 */
public abstract class F64SparseDecompositionsAdapter protected constructor(
    protected val equilibrate: Boolean = false,
    private val metadata: BackendMetadata = BackendMetadata(),
) : F64SparseDecompositions,
    F64RoutingBackend,
    BackendMetadataProvider {
    /** Whether the binding resolved every symbol needed to factor and solve. */
    protected abstract val nativeAvailable: Boolean

    /** The portable factorization at this backend's own policy, for everything the library will not take. */
    protected val portable: F64ReferenceSparseDecompositions =
        F64ReferenceSparseDecompositions(equilibrate = equilibrate)

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    override val backendMetadata: BackendMetadata get() = metadata

    override fun route(query: F64RouteQuery): BackendRoute? = when (query) {
        is F64RouteQuery.SparseLu, is F64RouteQuery.SparseQr ->
            nativeRoute(query, this, portable.name, available = nativeAvailableFor(query))

        else -> null
    }

    /**
     * Whether the library that answers [query] resolved.
     *
     * [nativeAvailable] is this backend's own library, which is the right answer wherever one library
     * carries every routine. A backend drawing a routine from a sibling overrides this, so a route names the
     * library that will actually run rather than the one the backend is named for.
     */
    protected open fun nativeAvailableFor(query: F64RouteQuery): Boolean = nativeAvailable

    final override fun factor(a: F64SparseMatrix): F64SparseLuFactorization {
        requireSquare(a, "factor")
        // A binding whose library is absent answers portably rather than throwing, so a caller reaching a
        // configured backend on a host without it gets the portable answer instead of an error.
        if (!nativeAvailable) {
            return portable.factor(a)
        }
        if (!equilibrate || libraryScalesRows) return factorNative(a)
        val scale = f64EquilibrationScale(a.rows, a.rowIdx, a.values)
        return EquilibratedSparseLu(factorNative(scaledBy(a, scale)), scale)
    }

    /**
     * Whether the library equilibrates for itself when [equilibrate] is set.
     *
     * A library that takes a scaling flag or control entry scales inside itself. A binding whose library
     * offers no scaling sets this false and is handed values already scaled, with the solves undone around
     * it here rather than in each binding.
     */
    protected open val libraryScalesRows: Boolean get() = true

    /** [a] with its values scaled, sharing the pattern it was already validated against. */
    @OptIn(UnsafeKoblasApi::class)
    private fun scaledBy(a: F64SparseMatrix, scale: DoubleArray): F64SparseMatrix = F64SparseMatrix.wrapTrusted(
        a.rows,
        a.cols,
        a.colPtr,
        a.rowIdx,
        f64ScaledValues(a.rowIdx, a.values, scale),
    )

    /** Factorizes a matrix through the native library, equilibrating when this backend is set to. */
    protected abstract fun factorNative(a: F64SparseMatrix): F64SparseLuFactorization

    final override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization {
        requireSquare(a, "cholesky")
        if (!nativeAvailable) {
            return portable.cholesky(a)
        }
        return choleskyNative(a)
    }

    /**
     * Factorizes a symmetric positive-definite matrix through the native library. Most of these libraries are
     * unsymmetric LU and have none, so the default is the portable factorization: the seam carries every
     * sparse factorization, and a binding filling one half of it does not have to offer the rest.
     */
    protected open fun choleskyNative(a: F64SparseMatrix): F64SparseCholeskyFactorization = portable.cholesky(a)

    final override fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization {
        requireSquare(a, "quasiDefiniteLdl")
        if (!nativeAvailable) {
            return portable.quasiDefiniteLdl(a)
        }
        return quasiDefiniteLdlNative(a)
    }

    /** Factorizes a symmetric matrix into `L·D·Lᵀ` through the native library, portably by default. */
    protected open fun quasiDefiniteLdlNative(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization =
        portable.quasiDefiniteLdl(a)

    final override fun qr(a: F64SparseMatrix): F64SparseQrFactorization {
        requireShape(a.rows >= a.cols) {
            "qr: A is ${a.rows}x${a.cols}, which is wider than it is tall; factor its transpose instead"
        }
        if (!nativeAvailable) {
            return portable.qr(a)
        }
        return qrNative(a)
    }

    /** Factorizes into `Q·R` natively, portably by default for a library without a sparse QR. */
    protected open fun qrNative(a: F64SparseMatrix): F64SparseQrFactorization = portable.qr(a)
}
