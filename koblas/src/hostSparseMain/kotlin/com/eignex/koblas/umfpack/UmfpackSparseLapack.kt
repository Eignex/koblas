@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.requireShape
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
 * The sparse half backed by SuiteSparse's UMFPACK. Equilibration, a non-default drop tolerance and any
 * UMFPACK failure fall back to [SparseLu]; a singular result becomes [SingularSparseFactorization].
 */
public class UmfpackSparseLapack internal constructor(private val f: UmfpackFunctions) : SparseLapack {
    override val name: String get() = "umfpack"

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    @Suppress("ReturnCount") // one early return per condition UMFPACK cannot serve
    override fun factor(a: SparseMatrix, equilibrate: Boolean, dropTolerance: Double): SparseFactorization {
        requireShape(a.rows == a.cols) { "factor requires a square matrix; got ${a.rows}x${a.cols}" }
        if (equilibrate || dropTolerance != NO_DROP) return SparseLu.factorCsc(a, equilibrate, dropTolerance)
        // An empty matrix and an all-zero one have nothing to pin, and `usePinned` yields no address.
        if (a.rows == 0 || a.nnz == 0) return SparseLu.factorCsc(a)

        val info = DoubleArray(INFO)
        val symbolicHolder = nativeHeap.alloc<COpaquePointerVar>()
        val handle = UmfpackFactorization.allocateHandle(f)
        val status = analyzeAndFactor(a, symbolicHolder, handle, info)
        // The symbolic analysis is scratch, UMFPACK keeps what it needs inside Numeric.
        f.freeSymbolic(symbolicHolder.ptr)
        nativeHeap.free(symbolicHolder.ptr)

        if (status != OK && status != WARNING_SINGULAR) {
            handle.release()
            return SparseLu.factorCsc(a)
        }
        if (status == WARNING_SINGULAR) {
            // Solving a singular factorization is forbidden, so holding UMFPACK's partial factors leaks.
            handle.release()
            return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        }
        val factorization = UmfpackFactorization(a, NOT_SINGULAR, handle, f)
        val (lnz, unz) = UmfpackFactorization.fillOf(info)
        factorization.lnz = lnz
        factorization.unz = unz
        return factorization
    }

    /** The symbolic analysis followed by the numeric factorization, sharing one set of pinned arrays. */
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
