@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.umfpack

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.NO_DROP
import com.eignex.koblas.sparse.SingularSparseFactorization
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLu
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
 * The sparse factorization half backed by SuiteSparse's UMFPACK, on the native targets.
 *
 * Behaviourally identical to the JVM binding, down to the fallbacks: `equilibrate` and any non-default
 * `dropTolerance` go to the portable factorization because UMFPACK cannot honor either, a singular result
 * becomes koblas's canonical [SingularSparseFactorization], and any UMFPACK failure falls back rather than
 * propagating. The conformance test asserts against the portable factorization on both platforms for exactly
 * that reason — a host backend that answered differently would be a bug wherever it ran.
 *
 * Registered at 90, the priority the native host backends use, so it wins over `SparseLu` where SuiteSparse
 * is installed. The JVM bindings use 100 for the same role; the two values only ever compete within one half,
 * so the difference is cosmetic, but the platform's own convention is the one worth matching.
 */
class UmfpackSparseLapack internal constructor(private val f: UmfpackFunctions) : SparseLapack {
    override val name: String get() = "umfpack"

    override val priority: Int get() = 90

    @Suppress("ReturnCount") // one early return per condition UMFPACK cannot serve
    override fun factor(a: SparseMatrix, equilibrate: Boolean, dropTolerance: Double): SparseFactorization {
        require(a.rows == a.cols) { "factor requires a square matrix; got ${a.rows}x${a.cols}" }
        if (equilibrate || dropTolerance != NO_DROP) return SparseLu.factorCsc(a, equilibrate, dropTolerance)
        // An empty matrix and an all-zero one both have nothing to pin: `usePinned` on an empty array cannot
        // produce an address. The portable path handles both, the second by reporting singular.
        if (a.rows == 0 || a.nnz == 0) return SparseLu.factorCsc(a)

        val info = DoubleArray(INFO)
        val symbolicHolder = nativeHeap.alloc<COpaquePointerVar>()
        val handle = UmfpackFactorization.allocateHandle(f)
        val status = analyzeAndFactor(a, symbolicHolder, handle, info)
        // The symbolic analysis is scratch: UMFPACK keeps what it needs inside Numeric.
        f.freeSymbolic(symbolicHolder.ptr)
        nativeHeap.free(symbolicHolder.ptr)

        if (status != OK && status != WARNING_SINGULAR) {
            handle.release()
            return SparseLu.factorCsc(a)
        }
        if (status == WARNING_SINGULAR) {
            // Solving a singular factorization is forbidden, so UMFPACK's partial factors can never be used
            // and holding them would leak. The canonical result also keeps `nnz == 0` true of every singular
            // factorization whatever produced it.
            handle.release()
            return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        val factorization = UmfpackFactorization(a, NOT_SINGULAR, handle, f)
        val (lnz, unz) = UmfpackFactorization.fillOf(info)
        factorization.lnz = lnz
        factorization.unz = unz
        return factorization
    }

    /**
     * The symbolic analysis followed by the numeric factorization, sharing one set of pinned arrays.
     *
     * Split out because pinning five arrays around two calls is already four levels of nesting, and because
     * a symbolic failure has to skip the numeric step while still freeing what was allocated.
     */
    @Suppress("NestedBlockDepth") // one nesting level per pinned array
    private fun analyzeAndFactor(
        a: SparseMatrix,
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
}
