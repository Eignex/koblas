@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.singularFailure
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.math.sqrt
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/** A CHOLMOD factorization whose native factor is reclaimed when it becomes unreachable. */
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
    ) {
        fun release() {
            val slot = nativeHeap.alloc<COpaquePointerVar>()
            slot.value = factor
            functions.freeFactor(slot.ptr, common)
            nativeHeap.free(slot)
            nativeHeap.free(common)
        }
    }

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(handle) { it.release() }

    override val nnz: Int get() = if (singular) {
        0
    } else {
        anchoring {
            sizeAt(handle.factor.reinterpret(), CHOLMOD_FACTOR_NZMAX).toInt()
        }
    }

    /**
     * CHOLMOD's own estimate, which is the ratio of the smallest factor diagonal to the largest, squared for
     * an `L·Lᵀ`. The square root brings it back to the ratio this seam documents, so the number means the
     * same thing whichever backend produced it.
     */
    override val rcond: Double get() = if (singular) {
        0.0
    } else {
        anchoring {
            val estimate = functions.rcond(handle.factor.reinterpret(), handle.common)
            if (intAt(handle.factor.reinterpret(), CHOLMOD_FACTOR_IS_LL) == CHOLMOD_TRUE) sqrt(estimate) else estimate
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        if (singular) throw singularFailure(failedAt, "solve")
        requireSolveShapes(n, b, out)
        if (out !== b) b.copyInto(out)
        anchoring {
            val rhs = nativeHeap.allocArray<DoubleVar>(n)
            for (i in 0 until n) rhs[i] = out[i]
            val dense = nativeHeap.allocArray<ByteVar>(CHOLMOD_DENSE_BYTES)
            sizeAt(dense, CHOLMOD_DENSE_NROW, n.toLong())
            sizeAt(dense, CHOLMOD_DENSE_NCOL, 1L)
            sizeAt(dense, CHOLMOD_DENSE_NZMAX, n.toLong())
            sizeAt(dense, CHOLMOD_DENSE_D, n.toLong())
            pointerAt(dense, CHOLMOD_DENSE_X, rhs)
            pointerAt(dense, CHOLMOD_DENSE_Z, null)
            intAt(dense, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
            intAt(dense, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)

            val solved = functions.solve(CHOLMOD_A, handle.factor.reinterpret(), dense, handle.common)
            check(solved != null) { "cholmod_solve failed on a factorization it produced" }
            val answer = solved.reinterpret<ByteVar>()
            val values = pointerAt(answer, CHOLMOD_DENSE_X)!!.reinterpret<DoubleVar>()
            for (i in 0 until n) out[i] = values[i]
            val slot = nativeHeap.alloc<COpaquePointerVar>()
            slot.value = solved
            functions.freeDense(slot.ptr, handle.common)
            nativeHeap.free(slot)
            nativeHeap.free(dense)
            nativeHeap.free(rhs)
        }
        return out
    }

    /**
     * Holds this factorization in a thread-local for the length of a native call, so the cleaner cannot run
     * against the factor a call still has a pointer to. The same fence the other native bindings take.
     */
    private inline fun <R> anchoring(body: () -> R): R {
        val previous = AnchoredCholmod.held
        AnchoredCholmod.held = this
        try {
            return body()
        } finally {
            AnchoredCholmod.held = previous
        }
    }
}

@ThreadLocal
private object AnchoredCholmod {
    var held: CholmodFactorization? = null
}
