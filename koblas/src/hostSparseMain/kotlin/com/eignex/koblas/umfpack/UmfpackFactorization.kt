@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.umfpack

import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.SparseFactorization
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
import kotlin.math.pow
import kotlin.native.ref.createCleaner

/**
 * UMFPACK's numeric factors behind koblas's [SparseFactorization]. Holds its [matrix] alive because
 * `umfpack_di_solve` takes Ap, Ai and Ax alongside the factors.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: F64SparseMatrix,
    override val failedAt: Int,
    private val handle: NumericHandle,
    private val f: UmfpackFunctions,
) : SparseFactorization {

    /** The `void **Numeric` holder, its own object so the cleaner captures it and not the factorization. */
    internal class NumericHandle(val pointer: CPointer<COpaquePointerVar>, private val f: UmfpackFunctions) {
        fun release() {
            f.freeNumeric(pointer)
            nativeHeap.free(pointer)
        }
    }

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(handle) { it.release() }

    override val n: Int get() = matrix.rows

    override val nnz: Int get() = lnz + unz

    /** Fill in L and U, read out of UMFPACK's Info array at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    @Suppress("NestedBlockDepth") // one nesting level per pinned array, the alternative is copying them
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // umfpack_di_solve declares X an output and B an input and says nothing about them overlapping.
        val rhs = if (out === b) b.copyOf() else b
        val info = DoubleArray(INFO)
        val control = UmfpackLoader.control
        val status = matrix.colPtr.usePinned { ap ->
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
        check(status == OK) { "umfpack_di_solve failed with status $status" }
        return out
    }

    /**
     * UMFPACK reports the determinant as a mantissa and a base-10 exponent. Recombining them here can
     * overflow to infinity, which is the honest answer for a value a double cannot hold.
     */
    override fun determinant(): Double {
        if (singular) return 0.0
        val mantissa = DoubleArray(1)
        val exponent = DoubleArray(1)
        val info = DoubleArray(INFO)
        val status = mantissa.usePinned { mx ->
            exponent.usePinned { ex ->
                info.usePinned { ip ->
                    f.determinant(mx.addressOf(0), ex.addressOf(0), handle.pointer.pointed.value, ip.addressOf(0))
                }
            }
        }
        check(status == OK) { "umfpack_di_get_determinant failed with status $status" }
        return mantissa[0] * 10.0.pow(exponent[0])
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
    }
}

/** Runs [body] with a pointer to [control], or with null, which means UMFPACK's own defaults. */
internal inline fun <R> withControl(control: DoubleArray?, body: (CPointer<kotlinx.cinterop.DoubleVar>?) -> R): R =
    if (control == null) body(null) else control.usePinned { body(it.addressOf(0)) }
