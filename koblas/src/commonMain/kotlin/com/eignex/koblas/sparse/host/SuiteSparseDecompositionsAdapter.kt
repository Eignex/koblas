package com.eignex.koblas.sparse.host

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.cholmod.suiteSparseCholesky
import com.eignex.koblas.sparse.host.spqr.suiteSparseQr

/**
 * A binding to a SuiteSparse LU, which answers the seam's symmetric and least-squares routines from the
 * libraries shipping beside it rather than leaving them to the portable factorization.
 *
 * CHOLMOD and SPQR are part of the same collection as KLU and UMFPACK, so a machine or a bundle carrying one
 * usually carries all of them. Where a sibling is missing the routine falls back, which is what each
 * library's null answer is for, and every binding that sits here made the same three choices.
 *
 * @param libraryPath the library this binding loaded, which is where its siblings are looked for.
 * @param equilibrate whether to scale rows before factorizing, as [F64SparseDecompositionsAdapter] documents.
 * @param metadata effective provider options exposed through structured diagnostics.
 */
public abstract class SuiteSparseDecompositionsAdapter protected constructor(
    libraryPath: String?,
    equilibrate: Boolean = false,
    metadata: BackendMetadata = BackendMetadata(),
) : F64SparseDecompositionsAdapter(equilibrate = equilibrate, metadata = metadata) {

    private val cholmod by lazy { suiteSparseCholesky(libraryPath) }

    private val spqr by lazy { suiteSparseQr(libraryPath) }

    /**
     * CHOLMOD's Cholesky, so this backend answers the seam's Cholesky natively too. A machine carrying the
     * LU without CHOLMOD falls back, which is what the null answer is for.
     */
    final override fun choleskyNative(a: F64SparseMatrix): F64SparseCholeskyFactorization =
        cholmod.factor(a) ?: portable.cholesky(a)

    /** CHOLMOD's quasi-definite factorization, retaining the fill-reducing ordering. */
    final override fun quasiDefiniteLdlNative(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization =
        cholmod.factorQuasiDefiniteLdl(a) ?: portable.quasiDefiniteLdl(a)

    /** SPQR's QR, answered natively wherever that library is installed beside this one. */
    final override fun qrNative(a: F64SparseMatrix): F64SparseQrFactorization = spqr.factor(a) ?: portable.qr(a)
}
