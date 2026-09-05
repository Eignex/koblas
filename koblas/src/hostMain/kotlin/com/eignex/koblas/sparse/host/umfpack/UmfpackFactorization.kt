@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.FactorsNotExposed
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi

/**
 * UMFPACK's numeric factors behind koblas's [F64SparseLuFactorization]. Holds its [matrix] alive because
 * `umfpack_di_solve` takes Ap, Ai and Ax alongside the factors.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: F64SparseMatrix,
    override val failedAt: Int,
    private val handle: NumericHandle,
    private val f: UmfpackFunctions,
    private val control: DoubleArray?,
) : F64SparseLuFactorization {

    /** The `void **Numeric` holder, its own object so the cleaner captures it and not the factorization. */
    internal class NumericHandle(val pointer: CPointer<COpaquePointerVar>, private val f: UmfpackFunctions) {
        fun release() {
            f.freeNumeric(pointer)
            nativeHeap.free(pointer)
        }
    }

    private val ownership = NativeOwnership(this, "UMFPACK factorization", handle::release)

    /** The factors, extracted on the first read; UMFPACK copies both out of its numeric object. */
    private val extracted: UmfpackFactors by lazy {
        extractUmfpackFactors(f, handle.pointer.pointed.value, matrix.rows) ?: throw FactorsNotExposed("native factors")
    }

    private val factors: UmfpackFactors get() = ownership.anchoring { extracted }

    override val l: F64SparseMatrix get() = factors.lower

    override val u: F64SparseMatrix get() = factors.upper

    override val rowOrder: IntArray get() = factors.rowOrder.copyOf()

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override val rowScaling: DoubleArray get() = factors.rowScaling.copyOf()

    private val emptyOffDiagonal: F64SparseMatrix by lazy {
        F64SparseMatrix.wrap(n, n, IntArray(n + 1), IntArray(0), DoubleArray(0))
    }

    override val offDiagonal: F64SparseMatrix get() = anchoring { emptyOffDiagonal }

    override val n: Int get() = matrix.rows

    override val nnz: Int get() = anchoring { lnz + unz }

    /** Fill in L and U, read out of UMFPACK's Info array at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    override var rcond: Double = 0.0
        get() = anchoring { field }
        internal set

    /** Reused because a caller sharing a factorization serializes its calls; see [NativeOwnership]. */
    private val solveInfo = DoubleArray(INFO)
    private val aliasedSolveAllocation = AllocationCapability(
        AllocationGuarantee.NO_MANAGED,
        listOf(ScratchRequirement(ScratchKind.F64, n)),
    )

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        if (aliasing) aliasedSolveAllocation else noManagedAllocation

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireSolveShapes(n, n, b, out)
        // umfpack_di_solve declares X an output and B an input and says nothing about them overlapping, so an
        // aliased pair needs a separate right-hand side, borrowed when there is a workspace to lend it.
        if (out !== b) return solveDistinct(b, out, transpose)
        return workspace.borrow(n) { rhs ->
            b.copyInto(rhs)
            solveDistinct(rhs, out, transpose)
        }
    }

    /** The call itself, over a right-hand side that does not alias [out]. */
    @Suppress("NestedBlockDepth") // one nesting level per pinned array, the alternative is copying them
    private fun solveDistinct(rhs: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        val status = anchoring {
            matrix.colPtr.usePinned { ap ->
                matrix.rowIdx.usePinned { ai ->
                    matrix.values.usePinned { ax ->
                        out.usePinned { x ->
                            rhs.usePinned { rp ->
                                solveInfo.usePinned { ip ->
                                    withControl(control) { cp ->
                                        f.solve(
                                            if (transpose) SYS_AT else SYS_A,
                                            ap.addressOf(0),
                                            ai.addressOf(0),
                                            ax.addressOf(0),
                                            x.addressOf(0),
                                            rp.addressOf(0),
                                            handle.pointer.pointed.value,
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
        }
        check(status == OK) { "umfpack_di_solve failed with status $status" }
        return out
    }

    override fun close(): Unit = ownership.close()

    /** Every native call goes through here; [NativeOwnership] says what that guarantees. */
    private fun <R> anchoring(body: () -> R): R = ownership.anchoring(body)

    internal companion object {
        /**
         * Allocates a `void **Numeric` holder, owned by the factorization that will hold it.
         *
         * Nulled here because `nativeHeap.alloc` is `malloc` and does not zero. A failed symbolic analysis
         * returns before `umfpack_di_numeric` writes the holder, and freeing it then would hand UMFPACK
         * whatever the heap happened to contain.
         */
        fun allocateHandle(f: UmfpackFunctions): NumericHandle {
            val holder = nativeHeap.alloc<COpaquePointerVar>()
            holder.value = null
            return NumericHandle(holder.ptr, f)
        }

        /** Reads the fill counts out of a completed factorization's Info array. */
        fun fillOf(info: DoubleArray): Pair<Int, Int> = info[INFO_LNZ].toInt() to info[INFO_UNZ].toInt()

        fun rcondOf(info: DoubleArray): Double = info[INFO_RCOND]
    }
}

/** Runs [body] with a pointer to [control], or with null, which means UMFPACK's own defaults. */
internal inline fun <R> withControl(control: DoubleArray?, body: (CPointer<DoubleVar>?) -> R): R =
    if (control == null) body(null) else control.usePinned { body(it.addressOf(0)) }
