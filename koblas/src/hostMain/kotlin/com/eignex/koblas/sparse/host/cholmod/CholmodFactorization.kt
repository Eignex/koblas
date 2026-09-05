@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeBlock
import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.FactorsNotExposed
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.math.sqrt

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

    /** Which factorization this holds, so converting a copy asks for the same one. */
    private val isLl: Boolean =
        NativeBlock(handle.factor.reinterpret()).getInt(CHOLMOD_FACTOR_IS_LL) == CHOLMOD_TRUE

    private val ownership = NativeOwnership(this, "CHOLMOD factorization", handle::release)

    /** `L` and the ordering, converted on the first read; CHOLMOD copies the factor to convert it. */
    private val extracted: CholmodFactors by lazy {
        extractCholmodFactor(functions, handle.factor, handle.common, isLl) ?: throw FactorsNotExposed("native factors")
    }

    private val factors: CholmodFactors get() = anchoring { extracted }

    /** The lower triangular factor, as [CholmodFactors.lower] documents it for each kind. */
    internal val lowerFactor: F64SparseMatrix
        get() {
            requireCholmodFactors("l")
            return factors.lower(isLl)
        }

    /** The diagonal factor of an `L·D·Lᵀ`, which CHOLMOD stores as the diagonal of `L`. */
    internal val diagonalFactor: DoubleArray
        get() {
            requireCholmodFactors("d")
            return factors.diagonal()
        }

    /** The fill-reducing ordering CHOLMOD chose. */
    internal val ordering: IntArray
        get() {
            requireCholmodFactors("order")
            return factors.permutation.copyOf()
        }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = noManagedAllocation

    override val nnz: Int get() = anchoring {
        if (singular) {
            0
        } else {
            NativeBlock(handle.factor.reinterpret()).getSize(CHOLMOD_FACTOR_NZMAX).toInt()
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
            if (isLl) sqrt(estimate) else estimate
        }
    }

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
            values.readDoubles(n).copyInto(out)
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
                    values.readDoubles(entryCount).copyInto(out.data)
                } finally {
                    slot.value = solved
                    functions.freeDense(slot.ptr, handle.common)
                }
            }
        }
        return out
    }

    override fun close(): Unit = ownership.close()

    /** Every native call goes through here; [NativeOwnership] says what that guarantees. */
    private fun <R> anchoring(body: () -> R): R = ownership.anchoring(body)
}
