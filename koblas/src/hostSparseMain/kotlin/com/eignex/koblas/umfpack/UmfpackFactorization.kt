@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.umfpack

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
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
 * UMFPACK's numeric factors behind koblas's [SparseFactorization], on the native targets.
 *
 * The JVM twin of this class, with `Cleaner` swapped for [createCleaner] and `MemorySegment` for pinned
 * Kotlin arrays. The reason both exist rather than one shared implementation is that nothing about the two
 * foreign-function layers is common: there is no `java.lang.foreign` on native and no `usePinned` on the JVM.
 * What *is* shared is everything that matters to a caller — the seam, the packed input format, the singular
 * contract and the fill accounting all live in `commonMain`.
 *
 * Holds its [matrix] alive because `umfpack_di_solve` takes `Ap`, `Ai` and `Ax` alongside the factors and the
 * signature has no way to omit them.
 */
class UmfpackFactorization internal constructor(
    private val matrix: SparseMatrix,
    override val failedAt: Int,
    private val handle: NumericHandle,
    private val f: UmfpackFunctions,
) : SparseFactorization {

    /**
     * The `void **Numeric` holder, in its own object so the cleaner action captures *it* and not the
     * factorization.
     *
     * A cleaner lambda closing over `UmfpackFactorization` would keep it reachable for as long as the cleaner
     * lives, which is forever, and the factors would never be freed. `createCleaner` is stricter than the
     * JVM's `Cleaner` about it — it rejects capturing anything not shareable — but the trap is the same one.
     */
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

    /** Fill in `L` and `U`, read out of UMFPACK's `Info` at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    @Suppress("NestedBlockDepth") // one nesting level per pinned array; the alternative is copying them
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        check(!singular) {
            "cannot solve against a singular factorization: umfpack reported the matrix singular. " +
                "Check `singular` before solving; repair the matrix and factor again."
        }
        require(b.size == n) { "solve: b size ${b.size}, expected $n" }
        require(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // umfpack_di_solve declares X an output and B an input and says nothing about them overlapping, so
        // handing it one array as both is outside what the library promises. Removing this copy does not
        // currently fail any test — with refinement off UMFPACK happens to tolerate the aliasing — which is
        // an observation about this version, not a guarantee to build a public `solveInto(x, x)` on.
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
     * `det(B)` from UMFPACK, reported as a mantissa and a base-10 exponent so the value survives a range the
     * product of `n` pivots leaves a double long before `n` is large.
     *
     * Recombining them here can still overflow to infinity, which is the honest answer for a value that does
     * not fit.
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
        /** A freshly allocated `void **Numeric` holder, owned by the factorization that will hold it. */
        fun allocateHandle(f: UmfpackFunctions): NumericHandle =
            NumericHandle(nativeHeap.alloc<COpaquePointerVar>().ptr, f)

        /** Reads the fill counts out of a completed factorization's `Info` array. */
        fun fillOf(info: DoubleArray): Pair<Int, Int> = info[INFO_LNZ].toInt() to info[INFO_UNZ].toInt()
    }
}

/**
 * Runs [body] with a pointer to [control], or with null when there is none.
 *
 * A null `Control` means UMFPACK's defaults, refinement included — the same degradation the JVM binding
 * takes when `umfpack_di_defaults` is missing.
 */
internal inline fun <R> withControl(control: DoubleArray?, body: (CPointer<kotlinx.cinterop.DoubleVar>?) -> R): R =
    if (control == null) body(null) else control.usePinned { body(it.addressOf(0)) }
