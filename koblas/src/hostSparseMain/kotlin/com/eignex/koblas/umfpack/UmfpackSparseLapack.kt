@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.umfpack

import com.eignex.koblas.BackendNames
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.NO_DROP
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SingularSparseFactorization
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLdlPolicy
import com.eignex.koblas.sparse.SparseLu
import com.eignex.koblas.sparse.SparseOrdering
import com.eignex.koblas.sparse.SparseSymbolic
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
    override val name: String get() = BackendNames.UMFPACK

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** UMFPACK ships separately from OpenBLAS, so a host can have one without the other. */
    override val isAvailable: Boolean get() = UmfpackLoader.available

    /** Stated rather than delegated: the delegate is portable and this calls out to UMFPACK. */
    override val isPortable: Boolean get() = false

    @Suppress("ReturnCount") // one early return per condition UMFPACK cannot serve
    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): SparseFactorization {
        requireSquare(a, "factor")
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
    override fun analyze(a: F64SparseMatrix, ordering: SparseOrdering): SparseSymbolic =
        ReferenceSparseLinearAlgebra.analyze(a, ordering)

    override fun ldl(a: F64SparseMatrix, policy: SparseLdlPolicy, ordering: SparseOrdering): SparseFactorization =
        ReferenceSparseLinearAlgebra.ldl(a, policy, ordering)
}
