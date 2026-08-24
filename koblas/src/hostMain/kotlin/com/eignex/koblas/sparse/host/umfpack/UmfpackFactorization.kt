@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireFactored
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.free
import kotlinx.cinterop.invoke
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.pointed
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/**
 * UMFPACK's numeric factors behind koblas's [F64SparseFactorization]. Holds its [matrix] alive because
 * `umfpack_di_solve` takes Ap, Ai and Ax alongside the factors.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: F64SparseMatrix,
    override val failedAt: Int,
    private val handle: NumericHandle,
    private val f: UmfpackFunctions,
    private val control: DoubleArray?,
) : F64SparseFactorization {

    /** The `void **Numeric` holder, its own object so the cleaner captures it and not the factorization. */
    internal class NumericHandle(val pointer: CPointer<COpaquePointerVar>, private val f: UmfpackFunctions) {
        fun release() {
            f.freeNumeric(pointer)
            nativeHeap.free(pointer)
        }
    }

    /** Frees the factors once this object goes away; [AnchoredFactorization] covers the calls in flight. */
    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(handle) { it.release() }

    override val n: Int get() = matrix.rows

    override val nnz: Int get() = lnz + unz

    /** Fill in L and U, read out of UMFPACK's Info array at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    override var rcond: Double = 0.0
        internal set

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireSolveShapes(n, b, out)
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
        val info = DoubleArray(INFO)
        val status = anchoring {
            matrix.colPtr.usePinned { ap ->
                matrix.rowIdx.usePinned { ai ->
                    matrix.values.usePinned { ax ->
                        out.usePinned { x ->
                            rhs.usePinned { rp ->
                                info.usePinned { ip ->
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

    /**
     * Runs [body] with this factorization reachable from a global, so the cleaner cannot free the factors
     * while the native call inside it is reading them. See [AnchoredFactorization].
     */
    private inline fun <R> anchoring(body: () -> R): R {
        val previous = AnchoredFactorization.held
        AnchoredFactorization.held = this
        try {
            return body()
        } finally {
            AnchoredFactorization.held = previous
        }
    }

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

/**
 * Keeps a factorization reachable from a global while its own native call runs.
 *
 * The calls in [UmfpackFactorization] reach the factors as a raw address, and the cleaner that frees them is
 * one of its own fields, so a receiver nothing else refers to can have its factors freed underneath a call in
 * flight. `koblas.factor(a).solve(b)` is that shape. The JVM binding uses `Reference.reachabilityFence`;
 * Kotlin/Native has no equivalent, and pinning the handle is no substitute, since the cleaner fires on its own
 * unreachability rather than the resource's. Global variables are GC roots, which is what makes this a fence.
 *
 * One slot per thread, since a solve does not re-enter another. Saved and restored rather than cleared, so a
 * nested call could not strand the outer one.
 */
@ThreadLocal
internal object AnchoredFactorization {
    var held: UmfpackFactorization? = null
}

/** Runs [body] with a pointer to [control], or with null, which means UMFPACK's own defaults. */
internal inline fun <R> withControl(control: DoubleArray?, body: (CPointer<kotlinx.cinterop.DoubleVar>?) -> R): R =
    if (control == null) body(null) else control.usePinned { body(it.addressOf(0)) }
