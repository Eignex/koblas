@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter
import com.eignex.koblas.sparse.host.cholmod.suiteSparseCholesky
import kotlinx.cinterop.*

/**
 * The sparse half backed by SuiteSparse's UMFPACK.
 *
 * Open so a bundled provider is one of these under another name rather than a wrapper around one. What a
 * caller reaches this backend by name for is the routines it carries outside the seam, and a wrapper
 * would hide their type. Only the name and the ranking are meant to vary.
 */
public open class UmfpackSparseLu(
    /** Policy for this backend instance and the factors it produces. */
    public val config: UmfpackConfig = UmfpackConfig(),
) : F64SparseDecompositionsAdapter(config.factorizeMin, config.equilibrate),
    com.eignex.koblas.sparse.F64GeneralSparseLu,
    com.eignex.koblas.sparse.F64SparseCholesky,
    com.eignex.koblas.sparse.F64SparseLdl {
    private val loader = UmfpackLoader(config)

    override val name: String get() = BackendNames.UMFPACK

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** UMFPACK ships separately from OpenBLAS, so a host can have one without the other. */
    final override val nativeAvailable: Boolean get() = loader.available

    private val cholmod by lazy { suiteSparseCholesky(config.libraryPath) }

    /**
     * CHOLMOD ships beside UMFPACK in the same collection, so this backend answers the seam's Cholesky
     * natively too rather than leaving it to the portable factorization. A machine carrying UMFPACK without
     * CHOLMOD falls back, which is what the null answer is for.
     */
    final override fun choleskyNative(a: F64SparseMatrix): F64SparseFactorization =
        cholmod.factor(a) ?: portable.cholesky(a)

    /** CHOLMOD's own default factorization, which is the one this seam's `ldl` asks for. */
    final override fun ldlNative(a: F64SparseMatrix): F64SparseFactorization = cholmod.factorLdl(a) ?: portable.ldl(a)

    internal val refinementSteps: Double? get() = loader.refinementSteps

    internal val pivotTolerance: Double? get() = loader.pivotTolerance

    final override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        val f = loader.functions ?: error("UMFPACK is not available")

        val info = DoubleArray(INFO)
        val control = loader.control(equilibrate)
        // Nulled for the reason [UmfpackFactorization.allocateHandle] nulls its own holder: nativeHeap.alloc
        // does not clear, and the free below runs whatever the analysis returned. UMFPACK nulls this itself
        // on the error paths it has today, so the guard is against the ones it might not.
        val symbolicHolder = nativeHeap.alloc<COpaquePointerVar>()
        symbolicHolder.value = null
        val handle = UmfpackFactorization.allocateHandle(f)

        // Until a factorization holds the handle, this call owns it, and nothing else would free it: the
        // cleaner is created with the factorization, so an escape before that point strands the factors for
        // the life of the process.
        @Suppress("TooGenericExceptionCaught") // rethrown, and the handle has no other owner yet
        val status = try {
            analyzeAndFactor(a, symbolicHolder, handle, info, control, f)
        } catch (t: Throwable) {
            handle.release()
            throw t
        } finally {
            // The symbolic analysis is scratch, UMFPACK keeps what it needs inside Numeric.
            f.freeSymbolic(symbolicHolder.ptr)
            nativeHeap.free(symbolicHolder.ptr)
        }

        if (status != OK && status != WARNING_SINGULAR) {
            handle.release()
            error("umfpack_di_numeric failed with status $status")
        }
        if (status == WARNING_SINGULAR) {
            // Solving a singular factorization is forbidden, so holding UMFPACK's partial factors leaks.
            handle.release()
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        val factorization = UmfpackFactorization(a, NOT_SINGULAR, handle, f, control)
        val (lnz, unz) = UmfpackFactorization.fillOf(info)
        factorization.lnz = lnz
        factorization.unz = unz
        factorization.rcond = UmfpackFactorization.rcondOf(info)
        return factorization
    }

    /** The symbolic analysis followed by the numeric factorization, sharing one set of pinned arrays. */
    @Suppress("NestedBlockDepth") // one nesting level per pinned array
    private fun analyzeAndFactor(
        a: F64SparseMatrix,
        symbolicHolder: COpaquePointerVar,
        handle: UmfpackFactorization.NumericHandle,
        info: DoubleArray,
        control: DoubleArray?,
        f: UmfpackFunctions,
    ): Int = a.colPtr.usePinned { ap ->
        a.rowIdx.usePinned { ai ->
            a.values.usePinned { ax ->
                info.usePinned { ip ->
                    withControl(control) { cp ->
                        val symbolic = f.symbolic(
                            a.rows,
                            a.rows,
                            ap.addressOf(0),
                            ai.addressOf(0),
                            ax.addressOf(0),
                            symbolicHolder.ptr,
                            cp,
                            ip.addressOf(0),
                        )
                        if (symbolic != OK) {
                            symbolic
                        } else {
                            f.numeric(
                                ap.addressOf(0),
                                ai.addressOf(0),
                                ax.addressOf(0),
                                symbolicHolder.value,
                                handle.pointer,
                                cp,
                                ip.addressOf(0),
                            )
                        }
                    }
                }
            }
        }
    }
}
