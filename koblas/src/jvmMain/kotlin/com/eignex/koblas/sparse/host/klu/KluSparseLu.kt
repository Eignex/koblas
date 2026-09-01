package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter
import com.eignex.koblas.sparse.host.cholmod.suiteSparseCholesky
import com.eignex.koblas.sparse.host.spqr.suiteSparseQr

/**
 * Sparse LU factorizations backed by KLU 2.
 *
 * Open so a bundled provider is one of these under another name rather than a wrapper around one. What a
 * caller reaches this backend by name for is the routines it carries outside the seam, and a wrapper
 * would hide their type. Only the name and the ranking are meant to vary.
 */
public open class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64SparseDecompositionsAdapter(
    equilibrate = config.equilibrate,
    metadata = BackendMetadata(options = config.options.metadataOptions()),
),
    com.eignex.koblas.sparse.F64GeneralSparseLu,
    com.eignex.koblas.sparse.F64RepeatedSparseLu,
    com.eignex.koblas.sparse.F64SparseCholesky,
    com.eignex.koblas.sparse.F64QuasiDefiniteLdl,
    com.eignex.koblas.sparse.F64SparseQr {
    private val calls = KluCalls(config)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    final override val nativeAvailable: Boolean get() = calls.available

    private val cholmod by lazy { suiteSparseCholesky(config.libraryPath) }

    private val spqr by lazy { suiteSparseQr(config.libraryPath) }

    /**
     * CHOLMOD ships beside KLU in the same collection, so this backend answers the seam's Cholesky
     * natively too rather than leaving it to the portable factorization. A machine carrying KLU without
     * CHOLMOD falls back, which is what the null answer is for.
     */
    final override fun choleskyNative(a: F64SparseMatrix): F64SparseCholeskyFactorization =
        cholmod.factor(a) ?: portable.cholesky(a)

    /** CHOLMOD's quasi-definite factorization, retaining the fill-reducing ordering. */
    final override fun quasiDefiniteLdlNative(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization = cholmod.factorQuasiDefiniteLdl(
        a,
    ) ?: portable.quasiDefiniteLdl(a)

    /** SPQR ships in the same collection, so the seam's QR is answered natively too where it is installed. */
    final override fun qrNative(a: F64SparseMatrix): F64SparseQrFactorization = spqr.factor(a) ?: portable.qr(a)

    /**
     * Factor [a] reusing the symbolic analysis behind [previous], which supersedes it and must not be
     * solved after this call. Reuse is what KLU is for: a circuit keeps one sparsity pattern and factors it
     * again for every operating point, and the analysis is the expensive half.
     *
     * A [previous] from another backend, or one whose pattern does not match, is answered as [factor]
     * would. This is KLU's own routine rather than a seam method, since no other sparse LU koblas binds
     * carries an analysis worth reusing.
     */
    final override fun refactor(previous: F64SparseLuFactorization, a: F64SparseMatrix): F64SparseLuFactorization {
        requireSquare(a, "refactor")
        if (!nativeAvailable) return factor(a)
        val reusable = previous as? KluFactorization ?: return factor(a)
        return when (reusable.refactor(a, equilibrate)) {
            KluRefactorResult.Success -> reusable

            KluRefactorResult.Incompatible -> factor(a).also { reusable.close() }

            KluRefactorResult.Singular -> F64SingularSparseFactorization(
                a.rows,
                SINGULAR_POSITION_UNKNOWN,
            ).also { reusable.close() }
        }
    }

    final override fun factorNative(a: F64SparseMatrix): F64SparseLuFactorization {
        val factor = calls.factorize(a.rows, a.colPtr, a.rowIdx, a.values, equilibrate)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return KluFactorization(NOT_SINGULAR, factor, calls)
    }
}
