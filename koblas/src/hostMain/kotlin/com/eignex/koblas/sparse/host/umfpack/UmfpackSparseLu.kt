@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.free
import kotlinx.cinterop.invoke
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value

/**
 * The sparse half backed by SuiteSparse's UMFPACK. Equilibration, a non-default drop tolerance and any
 * UMFPACK failure fall back to [F64SparseLuFactorization]; a singular result becomes [F64SingularSparseFactorization].
 */
public class UmfpackSparseLu internal constructor(private val f: UmfpackFunctions) : F64SparseLu {
    override val name: String get() = BackendNames.UMFPACK

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** UMFPACK ships separately from OpenBLAS, so a host can have one without the other. */
    override val isAvailable: Boolean get() = UmfpackLoader.available

    /** Stated rather than delegated: the delegate is portable and this calls out to UMFPACK. */
    override val isPortable: Boolean get() = false

    @Suppress("ReturnCount") // one early return per condition UMFPACK cannot serve
    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization {
        requireSquare(a, "factor")
        if (equilibrate || dropTolerance != NO_DROP) {
            return F64SparseLuFactorization.factorCsc(
                a,
                equilibrate,
                dropTolerance,
            )
        }
        // An empty matrix and an all-zero one have nothing to pin, and `usePinned` yields no address.
        if (a.rows == 0 || a.nnz == 0) return F64SparseLuFactorization.factorCsc(a)

        val info = DoubleArray(INFO)
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
            analyzeAndFactor(a, symbolicHolder, handle, info)
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
            return F64SparseLuFactorization.factorCsc(a)
        }
        if (status == WARNING_SINGULAR) {
            // Solving a singular factorization is forbidden, so holding UMFPACK's partial factors leaks.
            handle.release()
            return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        val factorization = UmfpackFactorization(a, NOT_SINGULAR, handle, f)
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
    ): Int = a.colPtr.usePinned { ap ->
        a.rowIdx.usePinned { ai ->
            a.values.usePinned { ax ->
                info.usePinned { ip ->
                    withControl(UmfpackLoader.control) { cp ->
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

    // UMFPACK provides no symmetric factorization, so these run the portable versions. Forwarded explicitly
    // rather than by class delegation, which would route the convenience overloads to the portable factor
    // instead of this one, since a delegated member calls back into the delegate.
}
