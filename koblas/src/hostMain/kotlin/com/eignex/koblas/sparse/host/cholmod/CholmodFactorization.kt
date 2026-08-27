@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.math.sqrt
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/** A CHOLMOD factorization with deterministic close and cleaner fallback for its native factor. */
public class CholmodFactorization internal constructor(
    private val handle: CholmodHandle,
    private val functions: CholmodFunctions,
    override val n: Int,
    /** The column a zero pivot stopped an `L·D·Lᵀ` at, or [NOT_SINGULAR]; a Cholesky raises instead. */
    override val failedAt: Int = NOT_SINGULAR,
) : F64SparseFactorization {

    /** The factor and the common it must be freed against, one object so the cleaner captures only these. */
    internal class CholmodHandle(
        val factor: COpaquePointer,
        val common: CPointer<ByteVar>,
        private val functions: CholmodFunctions,
        n: Int,
    ) {
        val solveRhs: CPointer<DoubleVar> = nativeHeap.allocArray(maxOf(n, 1))
        val solveDense: CPointer<ByteVar> = nativeHeap.allocArray(CHOLMOD_DENSE_BYTES)
        val solveSlot: COpaquePointerVar = nativeHeap.alloc()

        init {
            sizeAt(solveDense, CHOLMOD_DENSE_NROW, n.toLong())
            sizeAt(solveDense, CHOLMOD_DENSE_NCOL, 1L)
            sizeAt(solveDense, CHOLMOD_DENSE_NZMAX, n.toLong())
            sizeAt(solveDense, CHOLMOD_DENSE_D, n.toLong())
            pointerAt(solveDense, CHOLMOD_DENSE_X, solveRhs)
            pointerAt(solveDense, CHOLMOD_DENSE_Z, null)
            intAt(solveDense, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
            intAt(solveDense, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
        }

        fun release() {
            solveSlot.value = factor
            functions.freeFactor(solveSlot.ptr, common)
            nativeHeap.free(solveSlot)
            nativeHeap.free(solveDense)
            nativeHeap.free(solveRhs)
            nativeHeap.free(common)
        }
    }

    private val lifecycle = NativeResourceLifecycle("CHOLMOD factorization", handle::release)

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(lifecycle) { it.close() }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = noManagedAllocation

    override val nnz: Int get() = anchoring {
        if (singular) {
            0
        } else {
            sizeAt(handle.factor.reinterpret(), CHOLMOD_FACTOR_NZMAX).toInt()
        }
    }

    /**
     * CHOLMOD's own estimate, which is the ratio of the smallest factor diagonal to the largest, squared for
     * an `L·Lᵀ`. The square root brings it back to the ratio this seam documents, so the number means the
     * same thing whichever backend produced it.
     */
    override val rcond: Double get() = anchoring {
        if (singular) {
            0.0
        } else {
            val estimate = functions.rcond(handle.factor.reinterpret(), handle.common)
            if (intAt(handle.factor.reinterpret(), CHOLMOD_FACTOR_IS_LL) == CHOLMOD_TRUE) sqrt(estimate) else estimate
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        if (singular) throw singularFailure(failedAt, "solve")
        requireSolveShapes(n, b, out)
        if (out !== b) b.copyInto(out)
        anchoring {
            for (i in 0 until n) handle.solveRhs[i] = out[i]

            val solved = functions.solve(CHOLMOD_A, handle.factor.reinterpret(), handle.solveDense, handle.common)
            check(solved != null) { "cholmod_solve failed on a factorization it produced" }
            val answer = solved.reinterpret<ByteVar>()
            val values = pointerAt(answer, CHOLMOD_DENSE_X)!!.reinterpret<DoubleVar>()
            for (i in 0 until n) out[i] = values[i]
            handle.solveSlot.value = solved
            functions.freeDense(handle.solveSlot.ptr, handle.common)
        }
        return out
    }

    override fun close(): Unit = lifecycle.close()

    /**
     * Holds this factorization in a thread-local for the length of a native call, so the cleaner cannot run
     * against the factor a call still has a pointer to. The same fence the other native bindings take.
     */
    private fun <R> anchoring(body: () -> R): R = lifecycle.withResource {
        val previous = AnchoredCholmod.held
        AnchoredCholmod.held = this
        try {
            body()
        } finally {
            AnchoredCholmod.held = previous
        }
    }
}

@ThreadLocal
private object AnchoredCholmod {
    var held: CholmodFactorization? = null
}
