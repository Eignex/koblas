@file:OptIn(
    ExperimentalForeignApi::class,
    ExperimentalNativeApi::class,
    com.eignex.koblas.UnsafeKoblasApi::class,
)

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.requireFactored
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.basicReport
import com.eignex.koblas.sparse.requireBlockSolveShapes
import com.eignex.koblas.sparse.requireSolveShapes
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/** KLU's symbolic and numeric objects behind koblas's [F64SparseLuFactorization]. */
public class KluFactorization internal constructor(
    private val handle: KluHandle,
    private val functions: KluFunctions,
    override val n: Int,
    private val columnPointers: IntArray,
    private val rowIndices: IntArray,
) : F64SparseLuFactorization {

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = noManagedAllocation

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

    private val lifecycle = NativeResourceLifecycle("KLU factorization", handle::release)

    /** The factors, extracted on the first read; KLU copies them out of its numeric object. */
    private val extracted: KluFactors by lazy {
        checkNotNull(
            extractKluFactors(functions, handle.symbolic.pointed.value, handle.numeric.pointed.value, handle.common, n),
        ) {
            "this libklu does not expose klu_extract, so its factors cannot be read"
        }
    }

    private val factors: KluFactors get() = anchoring { extracted }

    override val l: F64SparseMatrix get() = factors.lower

    override val u: F64SparseMatrix get() = factors.upper

    override val offDiagonal: F64SparseMatrix get() = factors.offDiagonal

    override val rowOrder: IntArray get() = factors.rowOrder.copyOf()

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override val rowScaling: DoubleArray get() = factors.rowScaling.copyOf()

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(lifecycle) { it.close() }

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

    override fun solveInto(
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(failedAt, "solve")
        requireBlockSolveShapes(n, b, out)
        if (b.cols == 0) return out
        if (out.data !== b.data) b.data.copyInto(out.data)
        val solve = if (transpose) functions.transposedSolve else functions.solve
        val status = anchoring {
            out.data.usePinned { rhs ->
                solve(
                    handle.symbolic.pointed.value,
                    handle.numeric.pointed.value,
                    n,
                    b.cols,
                    rhs.addressOf(0),
                    handle.common,
                )
            }
        }
        check(status == 1) { "klu_solve failed with status ${intAt(handle.common, KLU_COMMON_STATUS)}" }
        return out
    }

    override fun report(): F64SparseFactorizationReport = basicReport("klu").copy(
        fillRatio = if (rowIndices.isEmpty()) 0.0 else nnz.toDouble() / rowIndices.size,
        details = mapOf("blockSolve" to "native"),
    )

    /** Refactorizes onto this factorization's pattern, rejecting another structure before calling KLU. */
    internal fun refactor(a: F64SparseMatrix): KluRefactorResult = anchoring {
        if (a.rows != n || !columnPointers.contentEquals(a.colPtr) || !rowIndices.contentEquals(a.rowIdx)) {
            return@anchoring KluRefactorResult.Incompatible
        }
        val status = a.colPtr.usePinned { ap ->
            a.rowIdx.usePinned { ai ->
                a.values.usePinned { ax ->
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
        when {
            singular -> KluRefactorResult.Singular
            status == 1 -> KluRefactorResult.Success
            else -> error("klu_refactor failed with status ${intAt(handle.common, KLU_COMMON_STATUS)}")
        }
    }

    override fun close(): Unit = lifecycle.close()

    /**
     * Runs [body] with this factorization reachable from a global, so the cleaner cannot free the objects
     * while the native call inside it is reading them.
     */
    private fun <R> anchoring(body: () -> R): R = lifecycle.withResource {
        val previous = AnchoredKlu.held
        AnchoredKlu.held = this
        try {
            body()
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
