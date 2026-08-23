@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64HostSparseLuAdapter
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

/** The sparse half backed by SuiteSparse's UMFPACK. */
public class UmfpackSparseLu(
    /** Policy for this backend instance and the factors it produces. */
    public val config: UmfpackConfig = UmfpackConfig(),
) : F64HostSparseLuAdapter() {
    private val loader = UmfpackLoader(config)

    override val name: String get() = BackendNames.UMFPACK

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** UMFPACK ships separately from OpenBLAS, so a host can have one without the other. */
    override val nativeAvailable: Boolean get() = loader.available

    internal val refinementSteps: Double? get() = loader.refinementSteps

    internal val pivotTolerance: Double? get() = loader.pivotTolerance

    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization {
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
