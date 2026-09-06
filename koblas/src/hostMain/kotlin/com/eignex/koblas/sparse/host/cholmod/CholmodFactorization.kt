@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.host.NativeBlock
import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.requireSolveShapes
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi

/** A CHOLMOD factorization with deterministic close and cleaner fallback for its native factor. */
internal class CholmodFactorization(
    private val handle: CholmodHandle,
    private val functions: CholmodFunctions,
    override val n: Int,
    /** The column a zero pivot stopped an `L·D·Lᵀ` at, or [NOT_SINGULAR]; a Cholesky raises instead. */
    override val failedAt: Int = NOT_SINGULAR,
) : CholmodFactorizationBase() {

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
            NativeBlock(solveDense).asCholmodDense(NativeBlock(solveRhs.reinterpret()), n, 1)
        }

        fun release() {
            solveSlot.value = factor
            functions.freeFactor(solveSlot.ptr, common)
            nativeHeap.free(solveSlot)
            nativeHeap.free(solveDense)
            nativeHeap.free(solveRhs)
            functions.finish(common)
            nativeHeap.free(common)
        }
    }

    override val isLl: Boolean =
        NativeBlock(handle.factor.reinterpret()).getInt(CHOLMOD_FACTOR_IS_LL) == CHOLMOD_TRUE

    override val ownership = NativeOwnership(this, "CHOLMOD factorization", handle::release)

    override fun extractFactors(): CholmodFactors? = extractCholmodFactor(functions, handle.factor, handle.common, isLl)

    override fun readNzmax(): Int = NativeBlock(handle.factor.reinterpret()).getSize(CHOLMOD_FACTOR_NZMAX).toInt()

    override fun readRcond(): Double = functions.rcond(handle.factor.reinterpret(), handle.common)

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        if (singular) throw singularFailure(failedAt, "solve")
        requireSolveShapes(n, n, b, out)
        if (out !== b) b.copyInto(out)
        anchoring {
            for (i in 0 until n) handle.solveRhs[i] = out[i]

            val solved = functions.solve(CHOLMOD_A, handle.factor.reinterpret(), handle.solveDense, handle.common)
            check(solved != null) { "cholmod_solve failed on a factorization it produced" }
            val answer = NativeBlock(solved.reinterpret())
            val values = checkNotNull(answer.getPointer(CHOLMOD_DENSE_X, n.toLong() * Double.SIZE_BYTES)) {
                "cholmod_solve answered with a dense block holding no values"
            }
            values.readDoublesInto(out, n)
            handle.solveSlot.value = solved
            functions.freeDense(handle.solveSlot.ptr, handle.common)
        }
        return out
    }

    override fun solveInto(
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix {
        if (singular) throw singularFailure(failedAt, "solve")
        requireSolveShapes(n, n, b, out)
        if (b.cols == 0) return out
        if (out.data !== b.data) b.data.copyInto(out.data)
        anchoring {
            memScoped {
                val entryCount = out.data.size
                val rhs = allocArray<DoubleVar>(maxOf(entryCount, 1))
                val dense = allocArray<ByteVar>(CHOLMOD_DENSE_BYTES)
                val slot = alloc<COpaquePointerVar>()
                NativeBlock(dense).asCholmodDense(NativeBlock(rhs.reinterpret()), n, b.cols)
                for (i in 0 until entryCount) rhs[i] = out.data[i]

                val solved = functions.solve(CHOLMOD_A, handle.factor.reinterpret(), dense, handle.common)
                check(solved != null) { "cholmod_solve failed on a factorization it produced" }
                try {
                    val answer = NativeBlock(solved.reinterpret())
                    val values = checkNotNull(
                        answer.getPointer(CHOLMOD_DENSE_X, entryCount.toLong() * Double.SIZE_BYTES),
                    ) { "cholmod_solve answered with a dense block holding no values" }
                    values.readDoublesInto(out.data, entryCount)
                } finally {
                    slot.value = solved
                    functions.freeDense(slot.ptr, handle.common)
                }
            }
        }
        return out
    }

    /** Every native call goes through here; [NativeOwnership] says what that guarantees. */
    private fun <R> anchoring(body: () -> R): R = ownership.anchoring(body)
}
