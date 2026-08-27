@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter
import com.eignex.koblas.sparse.host.cholmod.suiteSparseCholesky
import com.eignex.koblas.sparse.host.spqr.suiteSparseQr
import kotlinx.cinterop.*

/**
 * Sparse LU factorizations backed by a host KLU 2.
 *
 * Open so a bundled provider is one of these under another name rather than a wrapper around one. What a
 * caller reaches this backend by name for is the routines it carries outside the seam, and a wrapper
 * would hide their type. Only the name and the ranking are meant to vary.
 */
public open class KluSparseLu(
    /** Policy for this backend instance. */
    public val config: KluConfig = KluConfig(),
) : F64SparseDecompositionsAdapter(
    config.factorizeMin,
    config.equilibrate,
    BackendMetadata(options = config.options.metadataOptions()),
    qrFactorizeMin = config.qrFactorizeMin,
),
    com.eignex.koblas.sparse.F64GeneralSparseLu,
    com.eignex.koblas.sparse.F64RepeatedSparseLu,
    com.eignex.koblas.sparse.F64SparseCholesky,
    com.eignex.koblas.sparse.F64SparseLdl,
    com.eignex.koblas.sparse.F64SparseQr {
    private val loader = KluLoader(config)

    override val name: String get() = BackendNames.KLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    final override val nativeAvailable: Boolean get() = loader.available

    private val cholmod by lazy { suiteSparseCholesky(config.libraryPath) }

    private val spqr by lazy { suiteSparseQr(config.libraryPath) }

    /**
     * CHOLMOD ships beside KLU in the same collection, so this backend answers the seam's Cholesky
     * natively too rather than leaving it to the portable factorization. A machine carrying KLU without
     * CHOLMOD falls back, which is what the null answer is for.
     */
    final override fun choleskyNative(a: F64SparseMatrix): F64SparseFactorization =
        cholmod.factor(a) ?: portable.cholesky(a)

    /** CHOLMOD's own default factorization, which is the one this seam's `ldl` asks for. */
    final override fun ldlNative(a: F64SparseMatrix): F64SparseFactorization = cholmod.factorLdl(a) ?: portable.ldl(a)

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
    final override fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization {
        requireSquare(a, "refactor")
        if (!nativeAvailable) return factor(a)
        val reusable = previous as? KluFactorization ?: return factor(a)
        return when (reusable.refactor(a)) {
            KluRefactorResult.Success -> reusable

            KluRefactorResult.Incompatible -> factor(a).also { reusable.close() }

            KluRefactorResult.Singular -> F64SingularSparseFactorization(
                a.rows,
                SINGULAR_POSITION_UNKNOWN,
            ).also { reusable.close() }
        }
    }

    final override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        val functions = loader.functions ?: error("KLU 2 is not available")
        val common = loader.common(equilibrate)
        val symbolic = nativeHeap.alloc<COpaquePointerVar>().also { it.value = null }
        val numeric = nativeHeap.alloc<COpaquePointerVar>().also { it.value = null }
        val colPtr = a.copyColumnPointers()
        val rowIdx = a.copyRowIndices()

        symbolic.value = colPtr.usePinned { ap ->
            rowIdx.usePinned { ai -> functions.analyze(a.rows, ap.addressOf(0), ai.addressOf(0), common) }
        }
        if (symbolic.value == null) {
            release(functions, symbolic, numeric, common)
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }

        numeric.value = colPtr.usePinned { ap ->
            rowIdx.usePinned { ai ->
                a.values.usePinned { ax ->
                    functions.factor(ap.addressOf(0), ai.addressOf(0), ax.addressOf(0), symbolic.value, common)
                }
            }
        }
        if (numeric.value == null) {
            release(functions, symbolic, numeric, common)
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        return KluFactorization(
            KluFactorization.KluHandle(symbolic.ptr, numeric.ptr, common, functions),
            functions,
            a.rows,
            colPtr,
            rowIdx,
        )
    }

    /** Frees what a failed factorization allocated, since no cleaner has been attached to it yet. */
    private fun release(
        functions: KluFunctions,
        symbolic: COpaquePointerVar,
        numeric: COpaquePointerVar,
        common: CPointer<ByteVar>,
    ) {
        functions.freeNumeric(numeric.ptr, common)
        functions.freeSymbolic(symbolic.ptr, common)
        nativeHeap.free(numeric)
        nativeHeap.free(symbolic)
        nativeHeap.free(common)
    }
}
