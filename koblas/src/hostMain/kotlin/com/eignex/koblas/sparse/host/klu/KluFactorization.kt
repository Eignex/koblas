@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.Workspace
import com.eignex.koblas.requireFactored
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/** KLU's symbolic and numeric objects behind koblas's [F64SparseFactorization]. */
public class KluFactorization internal constructor(
    private val handle: KluHandle,
    private val functions: KluFunctions,
    override val n: Int,
) : F64SparseFactorization {

    /** The two holders and the common block, one object so the cleaner captures it and not the factorization. */
    internal class KluHandle(
        val symbolic: CPointer<COpaquePointerVar>,
        val numeric: CPointer<COpaquePointerVar>,
        val common: CPointer<ByteVar>,
        private val functions: KluFunctions,
    ) {
        fun release() {
            functions.freeNumeric(numeric, common)
            functions.freeSymbolic(symbolic, common)
            nativeHeap.free(numeric)
            nativeHeap.free(symbolic)
            nativeHeap.free(common)
        }
    }

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(handle) { it.release() }

    override var failedAt: Int = NOT_SINGULAR
        internal set

    override val nnz: Int get() = anchoring {
        val numeric = handle.numeric.pointed.value?.reinterpret<ByteVar>() ?: return@anchoring 0
        intAt(numeric, KLU_NUMERIC_LNZ) + intAt(numeric, KLU_NUMERIC_UNZ)
    }

    override val rcond: Double get() = anchoring {
        val computed = functions.rcond(handle.symbolic.pointed.value, handle.numeric.pointed.value, handle.common)
        if (computed != 1) 0.0 else doubleAt(handle.common, KLU_COMMON_RCOND)
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireSolveShapes(n, b, out)
        // KLU solves in place over the right-hand side it is given, so the destination carries it in.
        if (out !== b) b.copyInto(out)
        val solve = if (transpose) functions.transposedSolve else functions.solve
        val status = anchoring {
            out.usePinned { rhs ->
                solve(
                    handle.symbolic.pointed.value,
                    handle.numeric.pointed.value,
                    n,
                    1,
                    rhs.addressOf(0),
                    handle.common,
                )
            }
        }
        check(status == 1) { "klu_solve failed with status ${intAt(handle.common, KLU_COMMON_STATUS)}" }
        return out
    }

    /** Refactorizes onto this factorization's pattern, reporting whether KLU kept it. */
    internal fun refactor(colPtr: IntArray, rowIdx: IntArray, values: DoubleArray): Boolean = anchoring {
        val status = colPtr.usePinned { ap ->
            rowIdx.usePinned { ai ->
                values.usePinned { ax ->
                    functions.refactor(
                        ap.addressOf(0),
                        ai.addressOf(0),
                        ax.addressOf(0),
                        handle.symbolic.pointed.value,
                        handle.numeric.pointed.value,
                        handle.common,
                    )
                }
            }
        }
        val singular = intAt(handle.common, KLU_COMMON_STATUS) == KLU_SINGULAR
        failedAt = if (singular) SINGULAR_POSITION_UNKNOWN else NOT_SINGULAR
        status == 1 && !singular
    }

    /**
     * Runs [body] with this factorization reachable from a global, so the cleaner cannot free the objects
     * while the native call inside it is reading them.
     */
    private inline fun <R> anchoring(body: () -> R): R {
        val previous = AnchoredKlu.held
        AnchoredKlu.held = this
        try {
            return body()
        } finally {
            AnchoredKlu.held = previous
        }
    }
}

/** Holds the factorization a native call is reading, the counterpart of UMFPACK's own anchor. */
@ThreadLocal
internal object AnchoredKlu {
    var held: KluFactorization? = null
}
